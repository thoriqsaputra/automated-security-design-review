import logging
from typing import Any, Dict, List
from celery import shared_task
from sqlalchemy import select, update

from sdr.core.database import SessionLocal
from .models import Review
from .models.choices import ReviewStatus

logger = logging.getLogger(__name__)


def _default_preparation_builders() -> List[Dict[str, Any]]:
    return [
        {"id": "doc_extraction", "label": "Extracting text", "status": "pending"},
        {"id": "diagram_extraction", "label": "Extracting diagrams", "status": "pending"},
    ]


def _review_is_cancelled(review: Review | None) -> bool:
    return bool(review and getattr(review, "status", None) == ReviewStatus.CANCELLED.value)


@shared_task(bind=True, max_retries=3)
def dispatch_review_analysis_task(self, review_id: int):
    """
    Background task to run review analysis pipeline.
    """
    logger.info(f"Starting analysis for review {review_id}")
    
    try:
        with SessionLocal() as db:
            from sqlalchemy.orm import joinedload
            from sdr.apps.standards.models import StandardIngestionJob
            
            review = db.execute(
                select(Review)
                .options(
                    joinedload(Review.design),
                    joinedload(Review.selected_categories),
                    joinedload(Review.ingestion_job).joinedload(StandardIngestionJob.category)
                )
                .where(Review.id == review_id)
            ).scalars().first()
            
            if not review:
                logger.error(f"Review {review_id} not found.")
                return
            db.expunge_all()
            
        from sdr.apps.ai.services.analysis import run_tsd_analysis
        run_tsd_analysis(review)
        logger.info(f"Analysis for review {review_id} completed successfully.")
            
    except Exception as exc:
        logger.exception(f"Error during review analysis for review {review_id}: {exc}")
        try:
            with SessionLocal() as db:
                review = db.get(Review, review_id)
                if _review_is_cancelled(review):
                    logger.warning(
                        "dispatch_review_analysis_task: review_id=%s already cancelled; suppressing failure/retry",
                        review_id,
                    )
                    return
                if review:
                    review.status = ReviewStatus.FAILED.value
                    review.error_message = str(exc)
                    db.commit()
        except Exception as inner_exc:
            logger.error(f"Failed to record failure for review {review_id}: {inner_exc}")
        try:
            with SessionLocal() as db:
                review = db.get(Review, review_id)
                if _review_is_cancelled(review):
                    logger.warning(
                        "dispatch_review_analysis_task: review_id=%s cancelled after exception; suppressing retry",
                        review_id,
                    )
                    return
        except Exception as inner_exc:
            logger.error(f"Failed to re-check cancellation for review {review_id}: {inner_exc}")
        raise self.retry(exc=exc, countdown=60)


def dispatch_review_analysis(review_id: int) -> Dict[str, Any]:
    try:
        task = dispatch_review_analysis_task.delay(review_id)
        return {"mode": "async", "task_id": task.id}
    except Exception as e:
        logger.error(f"Failed to dispatch review analysis task: {e}")
        raise
