from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import os
import shutil
from datetime import datetime
import logging
from typing import Dict, Any, Optional

from utils.file_parser import extract_text_from_file, FileParsingError
from config import get_settings
from agents.parts_research.tasks import run_parts_research_task
from celery.result import AsyncResult
from celery_app import celery_app

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# Ensure uploads directory exists (might be redundant if upload.py does it)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define the URL for the internal parser service
# In a real microservice setup, this would be configurable (e.g., via env vars)
PARSER_SERVICE_URL = "http://localhost:8000/api/v1/parse" # Assuming running in the same process for now

async def save_uploaded_file(file: UploadFile) -> str:
    """Saves the uploaded file locally and returns the path."""
    try:
        # Generate unique filename (similar to upload.py)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize filename slightly
        base_filename = os.path.basename(file.filename or "unknown_file")
        safe_base_filename = "".join(c if c.isalnum() or c in ('.', '-', '_') else '_' for c in base_filename)
        unique_filename = f"{timestamp}_{safe_base_filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        logger.info(f"Saving uploaded file to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return file_path
    except Exception as e:
        logger.exception(f"Failed to save uploaded file: {file.filename}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        await file.close() # Ensure file is closed

async def call_parser_service(text: str) -> Dict[str, Any]:
    """Calls the Quote Parser service using a dedicated client."""
    logger.info(f"Calling Quote Parser Agent.")
    async with httpx.AsyncClient() as client:
        try:
            parser_payload = {"text": text}
            response = await client.post(PARSER_SERVICE_URL, json=parser_payload, timeout=60.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Quote Parser Agent returned error: {e.response.status_code} - {e.response.text}")
            detail = f"Failed to parse quote: {e.response.json().get('detail', e.response.text) if e.response.content else 'Parser service error'}"
            raise HTTPException(status_code=e.response.status_code, detail=detail)
        except httpx.RequestError as e:
            logger.error(f"Could not connect to Quote Parser Agent: {e}")
            raise HTTPException(status_code=503, detail="Quote parsing service is unavailable.")

class AuditStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None

@router.post("/audit", status_code=202)
async def perform_audit_from_file(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Orchestrates the car repair audit process:
    1. Receives an uploaded file (quote).
    2. Saves the file.
    3. Extracts text from the file.
    4. Calls the Quote Parser Agent (synchronously).
    5. Triggers a background task for Parts Research.
    6. Returns a task ID for polling results.
    """
    file_path = None
    try:
        # 1 & 2: Save the uploaded file
        file_path = await save_uploaded_file(file)

        # 3: Extract text from the file
        logger.info(f"Extracting text from saved file: {file_path}")
        extracted_text = extract_text_from_file(file_path)
        
        if not extracted_text:
             # Handle cases where PDF has no text (e.g., image-based PDF)
             logger.warning(f"No text extracted from {file.filename} at {file_path}. Cannot proceed with parsing.")
             raise HTTPException(status_code=400, detail="Could not extract text from the uploaded file. It might be an image or empty.")

        # 4: Call Parser (still sync for now as it's faster)
        parsed_data = await call_parser_service(extracted_text)
        
        # Prepare data for the background task
        all_parts_to_research = [item for item in parsed_data.get("line_items", []) if item.get("item_type") == "part"]
        vehicle_info = {
            "year": parsed_data.get("year"),
            "make": parsed_data.get("make"),
            "model": parsed_data.get("model")
        }

        # --- ADDED: Limit parts sent to background task for DEBUGGING ---
        MAX_PARTS_FOR_BACKGROUND_RESEARCH = 1
        parts_to_research_limited = all_parts_to_research[:MAX_PARTS_FOR_BACKGROUND_RESEARCH]
        # --- END ADDED --- 

        # Check if there are parts AFTER limiting
        if not parts_to_research_limited:
             logger.info("No parts found to research (or none left after limit), skipping background task.")
             return {"message": "Quote parsed, but no parts found for research.", "quote_analysis": parsed_data}

        # 5: Trigger background task for Parts Research (using the limited list)
        logger.info(f"Sending {len(parts_to_research_limited)} part(s) (limited from {len(all_parts_to_research)}) to background task for research.")
        task = run_parts_research_task.delay(parts_to_research=parts_to_research_limited, vehicle_info=vehicle_info)
        logger.info(f"Background task {task.id} created for parts research.")

        # 6: Return Task ID
        return {"task_id": task.id, "status": "PENDING", "message": "Parts research task submitted."}

    except FileNotFoundError as e:
         logger.error(f"File not found during audit process: {e}")
         raise HTTPException(status_code=404, detail=str(e))
    except FileParsingError as e:
         logger.error(f"Failed to parse file content: {e}")
         raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        # Re-raise known HTTP exceptions
        raise e
    except Exception as e:
        # Catch-all for unexpected errors during orchestration
        logger.exception(f"An unexpected error occurred during the audit submission: {file.filename if file else 'N/A'}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during audit submission.")
    finally:
        # Optional: Clean up the saved file?
        # if file_path and os.path.exists(file_path):
        #     logger.info(f"Cleaning up temporary file: {file_path}")
        #     os.remove(file_path)
        pass # Decide on cleanup strategy later 

@router.get("/audit/results/{task_id}", response_model=AuditStatusResponse)
async def get_audit_results(task_id: str) -> AuditStatusResponse:
    """Poll this endpoint with the task_id received from /audit to get results."""
    logger.debug(f"Checking status for task ID: {task_id}")
    task_result = AsyncResult(task_id, app=celery_app)
    
    response_data = {
        "task_id": task_id,
        "status": task_result.status, # PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
        "result": None,
        "error": None
    }
    
    if task_result.successful():
        result_data = task_result.get()
        # Assuming our task returns {"status": "SUCCESS|FAILURE", "result|error": ...}
        if isinstance(result_data, dict) and result_data.get("status") == "SUCCESS":
             response_data["result"] = result_data.get("result")
             logger.info(f"Task {task_id} completed successfully.")
        else:
             response_data["status"] = "FAILURE" # Override Celery status if our task reported failure
             response_data["error"] = result_data.get("error", "Task execution failed internally.")
             logger.warning(f"Task {task_id} finished but reported failure: {response_data['error']}")
             
    elif task_result.failed():
        # Get traceback if stored
        tb = task_result.traceback
        logger.error(f"Task {task_id} failed. Traceback: {tb}")
        response_data["error"] = "Task failed during execution. Check server logs."
        # Or: response_data["error"] = str(task_result.result) # If exception stored in result
        
    elif task_result.status == 'PENDING':
        logger.debug(f"Task {task_id} is pending.")
    elif task_result.status == 'STARTED':
        logger.debug(f"Task {task_id} has started.")
    else:
        logger.warning(f"Task {task_id} has unhandled status: {task_result.status}")

    return AuditStatusResponse(**response_data) 