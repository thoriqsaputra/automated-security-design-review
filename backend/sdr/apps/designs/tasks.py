from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task

from sdr.core.database import SessionLocal

from .models import Design
from .preparation_store import DesignPreparationStore

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2)
def prepare_design_for_analysis_task(self, design_id: int, preparation_id: int):
    store = DesignPreparationStore()
    try:
        with SessionLocal() as db:
            design = db.get(Design, design_id)
            if not design:
                logger.error("prepare_design_for_analysis_task: design %s not found", design_id)
                return
            preparation = store.get_active_preparation(db, design_id)
            if not preparation or preparation.id != preparation_id:
                logger.warning(
                    "prepare_design_for_analysis_task: design %s preparation %s is no longer active",
                    design_id,
                    preparation_id,
                )
                return
            store.run_preparation(db, design=design, preparation=preparation)
            logger.info(
                "prepare_design_for_analysis_task: design %s preparation %s completed",
                design_id,
                preparation_id,
            )
    except Exception as exc:
        logger.exception(
            "prepare_design_for_analysis_task: design %s preparation %s failed: %s",
            design_id,
            preparation_id,
            exc,
        )
        try:
            with SessionLocal() as db:
                design = db.get(Design, design_id)
                if design:
                    preparation = store.get_active_preparation(db, design_id)
                    if preparation and preparation.id == preparation_id:
                        store.mark_failed(design, preparation, str(exc))
                        db.commit()
        except Exception as inner_exc:
            logger.error(
                "prepare_design_for_analysis_task: failed to persist error for design %s: %s",
                design_id,
                inner_exc,
            )
        raise self.retry(exc=exc, countdown=30)


def dispatch_design_preparation(design_id: int, preparation_id: int) -> Dict[str, Any]:
    task = prepare_design_for_analysis_task.delay(design_id, preparation_id)
    return {"mode": "async", "task_id": task.id}
