from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
import logging

# Use OpenAI library
from openai import OpenAI, AsyncOpenAI, RateLimitError, APIError
from config import get_settings

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# Initialize Async OpenAI Client (best practice for async FastAPI)
if not settings.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in settings. Quote Parser will not function.")
    # Optionally raise an error at startup if key is absolutely required
    # raise ValueError("OPENAI_API_KEY is not configured.")
    async_openai_client = None
else:
    async_openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def construct_extraction_prompt(text: str) -> List[Dict[str, str]]:
    """Constructs the messages list for the OpenAI API call."""
    system_prompt = f"""
You are an expert assistant specialized in extracting structured information from car repair quotes.
Analyze the provided quote text and extract the following details:

1.  **Vehicle Information:**
    *   `vin`: Vehicle Identification Number (17-character alphanumeric string, return null if not found).
    *   `make`: Vehicle Manufacturer (e.g., "Mercedes-Benz", "Honda", "Ford", return null if not found).
    *   `model`: Specific Vehicle Model (e.g., "G55 AMG", "Civic EX", return null if not found).
    *   `year`: Vehicle Model Year (integer, e.g., 2011, return null if not found).

2.  **Repair Summary:**
    *   `repair_type_summary`: A brief summary of the main service(s) being performed (e.g., "Brake Repair", "Engine Noise Diagnosis and Valve Cover Replacement", "Oil Change", "Diagnostic").

3.  **Line Items:** A list of all parts, labor charges, fees, and discounts mentioned. For each item, extract:
    *   `item_type`: Should be one of "part", "labor", "fee", "discount", or "other".
    *   `description`: The name or description of the item.
    *   `quantity`: The quantity (e.g., number of parts, hours of labor). Default to 1 if not explicitly specified.
    *   `unit_price`: The price per single unit or per hour. Calculate if possible (total_price / quantity), otherwise return null.
    *   `total_price`: The total price for that line item (quantity * unit_price). If a total line price is explicitly stated, use that value. If only unit price and quantity are available, calculate it.
    *   `part_number`: The part number, if available (return null otherwise).
    *   **IMPORTANT Price Logic:** If a line shows quantity, unit price, AND total price, prioritize the stated total price for the `total_price` field. Then, if possible, verify or calculate the `unit_price` (total_price / quantity). If only quantity and total price are given, calculate `unit_price`. If only quantity and unit price are given, calculate `total_price`.

4.  **Totals:** Extract any summary totals provided at the end of the quote, such as:
    *   `subtotal`: The total before tax and potentially discounts.
    *   `tax`: The amount of sales tax.
    *   `discounts_total`: Total amount of explicitly listed discounts (sum of line items with type 'discount').
    *   `grand_total`: The final amount due stated on the quote.

Respond ONLY with a valid JSON object containing these fields:
`vin` (string|null),
`make` (string|null),
`model` (string|null),
`year` (integer|null),
`repair_type_summary` (string),
`line_items` (list of objects with `item_type`, `description`, `quantity`, `unit_price`, `total_price`, `part_number`),
`totals` (object with `subtotal`, `tax`, `discounts_total`, `grand_total` - use null for fields not found).

Do not include any explanations or introductory text outside the JSON object.
Ensure all monetary values are floats.
"""

    user_prompt = f"""
Please extract the information from the following car repair quote text:

```text
{text}
```

Respond ONLY with the JSON object as described in the system instructions.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

async def call_llm_for_quote_extraction(text: str) -> Dict[str, Any]:
    """
    Calls the OpenAI API asynchronously to extract structured data from quote text.

    Args:
        text: The raw text of the car repair quote.

    Returns:
        A dictionary with the extracted information.

    Raises:
        HTTPException: If the API key is missing, the call fails, or the response is invalid.
    """
    if not async_openai_client:
         logger.error("OpenAI client not initialized due to missing API key.")
         raise HTTPException(status_code=501, detail="Quote parsing feature is not configured (Missing API Key)")

    logger.info(f"Requesting OpenAI ({settings.OPENAI_MODEL}) to parse text starting with: {text[:100]}...")
    messages = construct_extraction_prompt(text)

    try:
        response = await async_openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=0.1, # Lower temperature for more deterministic extraction
            response_format={"type": "json_object"} # Request JSON output
        )

        response_content = response.choices[0].message.content
        if not response_content:
             logger.error("OpenAI response content is empty.")
             raise HTTPException(status_code=500, detail="Failed to parse quote: LLM returned empty response.")

        # Attempt to parse the JSON string from the LLM response
        try:
            extracted_data = json.loads(response_content)
            # Basic validation (can be enhanced with Pydantic models later)
            if not isinstance(extracted_data, dict) or 'line_items' not in extracted_data:
                 logger.error(f"LLM response is not valid JSON or missing required fields: {response_content[:500]}")
                 raise HTTPException(status_code=500, detail="Failed to parse quote: Invalid format received from LLM.")
            
            logger.info("Successfully received and parsed JSON response from OpenAI.")
            return extracted_data
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from OpenAI response: {response_content[:500]}")
            raise HTTPException(status_code=500, detail="Failed to parse quote: Could not decode LLM response.")

    except RateLimitError as e:
        logger.error(f"OpenAI Rate Limit Error: {e}")
        raise HTTPException(status_code=429, detail="Quote parsing service is currently overloaded. Please try again later.")
    except APIError as e:
        logger.error(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=503, detail="Quote parsing service (LLM) unavailable or encountered an error.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during OpenAI API call: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while contacting the quote parsing service.")

@router.post("/parse", response_model=Optional[Dict[str, Any]])
async def parse_quote_endpoint(payload: Dict[str, str] = Body(...)) -> JSONResponse:
    """
    Receives quote text, sends it to OpenAI for parsing, and returns the structured result.
    """
    if "text" not in payload or not payload["text"]:
        raise HTTPException(status_code=400, detail="Missing or empty 'text' field in request body")

    text_to_parse = payload["text"]

    try:
        # Call the function that interacts with the LLM
        parsing_result = await call_llm_for_quote_extraction(text_to_parse)

        # Add metadata (keep this)
        parsing_result["metadata"] = {
            "parsed_at": datetime.now().isoformat(),
            "source": "openai_parser", # Updated source
            "model_used": settings.OPENAI_MODEL,
            # "original_text_preview": text_to_parse[:200] + "..." # Optional
        }

        return JSONResponse(
            status_code=200,
            content=parsing_result
        )
    except HTTPException as http_err:
        # Re-raise HTTP exceptions (e.g., from LLM call failure)
        raise http_err
    except Exception as e:
        # Catch unexpected errors during the process
        logger.exception(f"Unexpected error in /parse endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing quote."
        )

# Example Usage with curl (remember server needs to be running):
# curl -X POST "http://localhost:8000/api/v1/parse" \\
#      -H "Content-Type: application/json" \\
#      -d '{"text": "Customer: Jane Doe\\nVIN: 1FAFP4AU6J4EXAMPLE\\nOil Change Service $49.95\\nSynthetic Oil $20.00\\nFilter 123-ABC $15.50\\nLabor $14.45\\nTotal: $85.45"}'

# parser = QuoteParser()

# @router.post("/parse")
# async def parse_quote_endpoint(data: Dict[str, str]) -> JSONResponse:
#     """
#     Parse a quote text and extract specific car repair information.
#
#     Args:
#         data: Dictionary containing the quote text, e.g., {"text": "..."}
#
#     Returns:
#         JSONResponse with parsed quote information
#     """
#     if "text" not in data:
#          raise HTTPException(status_code=400, detail="Missing 'text' field in request body")
#     text = data["text"]
#
#     try:
#         result = parser.parse_quote(text)
#         return JSONResponse(
#             status_code=200,
#             content=result
#         )
#     except HTTPException as e:
#          # Re-raise HTTP exceptions from the parser
#          raise e
#     except Exception as e:
#         # Catch unexpected errors
#         print(f"Unexpected error in /parse endpoint: {e}") # Log error
#         raise HTTPException(
#             status_code=500,
#             detail=f"Internal server error while processing quote."
#         ) 