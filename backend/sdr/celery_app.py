from celery import Celery
from celery.signals import worker_process_init
from sdr.core.config import settings

# -----------------------------------------------------------------------------
# Celery Initialization
# -----------------------------------------------------------------------------
# We use REDIS_URL for both the broker and the result backend based on your config
celery_app = Celery(
    "sdr",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "sdr.apps.ai.tasks",
        "sdr.apps.designs.tasks",
        "sdr.apps.reviews.tasks",
        "sdr.apps.standards.tasks",
    ]
)

# -----------------------------------------------------------------------------
# Celery Configuration
# -----------------------------------------------------------------------------
celery_app.conf.update(
    # Serialization (Strict JSON for security and compatibility)
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # Timeouts & Resource Management
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    
    # Task Tracking
    task_track_started=True,
    
    # Timezone settings (Replaces CELERY_TIMEZONE)
    timezone="UTC",
    enable_utc=True,
    
    # Worker Optimization Settings
    worker_prefetch_multiplier=1,  # Best practice for long-running AI tasks
    worker_max_tasks_per_child=50, # Prevents memory leaks in heavy ML processing
)


@worker_process_init.connect
def _prewarm_openai_submodules(**kwargs):
    """Import openai's lazily-loaded submodules once, single-threaded, before
    the worker accepts tasks. Without this, the first concurrent chat +
    embeddings calls in a fresh worker process can race on the same first
    import and deadlock (_ModuleLock deadlock between openai.resources.chat
    and openai.resources.embeddings)."""
    import openai.resources.chat  # noqa: F401
    import openai.resources.embeddings  # noqa: F401