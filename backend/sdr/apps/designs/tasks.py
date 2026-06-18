"""
Celery tasks for design document processing.
"""
import os
import tempfile
import logging
from celery import shared_task

from sdr.core.database import SessionLocal
from .models import Design
from sdr.apps.workspace.document_processing import get_document_content

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_design_document(self, design_id: int):
    """
    Async task to process a design document.
    Updates design status to 'ready' on success or 'error' on failure.
    """
    try:
        with SessionLocal() as db:
            design = db.get(Design, design_id)
            if not design:
                logger.error(f"Design {design_id} not found in database.")
                return

            design.status = Design.STATUS_PROCESSING
            db.commit()
            
            logger.info(f"Processing design {design_id}: {design.name}")
            
            # For PDF, just mark as ready
            _process_pdf_design(design)
            
            design.status = Design.STATUS_READY
            design.processing_error = None
            db.commit()
            logger.info(f"Design {design_id} processing completed successfully")
            
    except Exception as exc:
        logger.exception(f"Error processing design {design_id}: {exc}")
        try:
            with SessionLocal() as db:
                design = db.get(Design, design_id)
                if design:
                    design.status = Design.STATUS_ERROR
                    design.processing_error = str(exc)
                    db.commit()
        except Exception as inner_exc:
            logger.error(f"Failed to record processing error for design {design_id}: {inner_exc}")
        
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


def _process_pdf_design(design: Design):
    """Process PDF design (e.g., validate, index, or extract content if needed)."""
    logger.info(f"Processing PDF design {design.id}: {design.name}")
    # Placeholder for future PDF processing logic
    # e.g., OCR, indexing, content extraction
    pass
