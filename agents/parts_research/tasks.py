import logging
import httpx
from typing import Dict, List, Any

from celery_app import celery_app # Import the Celery app instance
from .researcher import research_parts_workflow # Import the existing workflow logic

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='tasks.run_parts_research') # `bind=True` allows access to `self` (the task instance)
def run_parts_research_task(self, parts_to_research: List[Dict[str, Any]], vehicle_info: Dict[str, Any]) -> Dict[str, Any]:
    """Celery task to run the parts research workflow asynchronously."""
    logger.info(f"Celery task {self.request.id} started: Running parts research for {len(parts_to_research)} parts.")
    
    try:
        # Need an HTTP client within the task
        # Note: httpx.AsyncClient might require an event loop running.
        # For simplicity in Celery tasks, often synchronous httpx.Client is used,
        # or the async workflow is run within asyncio.run(). Let's try sync first.
        # However, our workflow uses async functions, so we need an async context.
        import asyncio
        
        async def main_async_wrapper():
             # Create httpx client within the async context
             async with httpx.AsyncClient() as client:
                 results = await research_parts_workflow(parts_to_research, client, vehicle_info)
             return results

        # Run the async workflow within the synchronous Celery task
        task_result = asyncio.run(main_async_wrapper())
        
        logger.info(f"Celery task {self.request.id} completed successfully.")
        return {"status": "SUCCESS", "result": task_result}
        
    except Exception as e:
        logger.exception(f"Celery task {self.request.id} failed: {e}")
        # Update task state to FAILURE
        # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # Depending on Celery version and setup, raising might be enough or preferred
        # raise Ignore() # Or just return an error structure
        return {"status": "FAILURE", "error": str(e)} 