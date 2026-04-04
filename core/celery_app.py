from celery import Celery
from core.config import settings

celery_app = Celery(
    "sdr_celery",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["ingestion.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True
)

@celery_app.task(name="demo_task")
def demo_task(job_id: str):
    return {"job_id": job_id, "status": "completed"}
