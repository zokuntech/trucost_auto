from celery import Celery
from config import get_settings

settings = get_settings()

# Initialize Celery
celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['agents.parts_research.tasks'] # Important: Points to where tasks are defined
)

# Optional Celery configuration (timeouts, etc.)
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_track_started=True,
    # Add other relevant configurations if needed
    # result_expires=3600, # Optional: expire results after 1 hour
)

if __name__ == "__main__":
    # This allows running celery directly using `python celery_app.py worker` 
    # although `celery -A celery_app.celery_app worker` is more common.
    celery_app.start() 