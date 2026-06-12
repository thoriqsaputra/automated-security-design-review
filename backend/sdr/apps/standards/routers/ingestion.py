import hashlib
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from sdr.core.database import get_db
from sdr.core.config import settings
from ..utils import infer_category_code, get_category
from ..models import ASVSLevelDefinition, StandardIngestionJob, StandardSourceDocument
from ..schemas import ASVSLevelDefinitionSchema, StandardIngestionJobSchema, CategoryCodeEnum, CategoryCodeWithAutoEnum
from ..tasks import dispatch_standard_ingestion
from ..validators import validate_standard_file
from sdr.apps.workspace.services.storage import storage_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingestion")

AUTO_CATEGORY_CODE = "auto"


@router.get("/", response_model=List[StandardIngestionJobSchema])
def list_ingestion_jobs(
    category_code: Optional[CategoryCodeEnum] = Query(None, description="Filter by category code"),
    db: Session = Depends(get_db)
):
    """List all ingestion jobs, optionally filtered by category."""
    stmt = select(StandardIngestionJob).order_by(StandardIngestionJob.created_at.desc())
    
    if category_code:
        category_code_str = category_code.value.strip().lower()
        stmt = stmt.join(StandardIngestionJob.category).where(StandardIngestionJob.category.has(code=category_code_str))
        
    jobs = db.execute(stmt).scalars().all()
    return jobs


from typing import Annotated

@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_model=StandardIngestionJobSchema)
def create_ingestion_job(
    category_code: Annotated[CategoryCodeWithAutoEnum, Form(...)],
    document: UploadFile = File(...),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None),
    level_definition_start_page: Optional[int] = Form(None),
    level_definition_end_page: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    """Create a new ingestion job with a document upload."""
    category_code_str = category_code.value.strip().lower()
    if not category_code_str:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="category_code is required.")

    is_auto_mode = category_code_str == AUTO_CATEGORY_CODE
    category = None if is_auto_mode else get_category(category_code_str, db)
    
    if not is_auto_mode and not category:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid category_code.")

    if not document:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A document is required.")

    try:
        validate_standard_file(document)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if is_auto_mode:
        inferred_code = infer_category_code(document.filename)
        category = get_category(inferred_code, db)
        if not category:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not resolve category from auto mode. Please select a category explicitly."
            )

    resolved_categories = {category.code: 1}

    try:
        # Get max version number
        max_version = db.execute(
            select(func.max(StandardIngestionJob.version_no))
            .where(StandardIngestionJob.category_id == category.id)
        ).scalar() or 0
        next_version = max_version + 1

        job = StandardIngestionJob(
            category_id=category.id,
            status=StandardIngestionJob.STATUS_PENDING,
            version_no=next_version,
            is_active=False,
            summary_json={
                "mode": "auto" if is_auto_mode else "manual",
                "resolved_categories": resolved_categories,
                "start_page": start_page,
                "end_page": end_page,
                "level_definition_start_page": level_definition_start_page,
                "level_definition_end_page": level_definition_end_page,
            }
        )
        db.add(job)
        db.flush()  # To get job.id

        # Process and save the document
        content = document.file.read()
        file_hash = hashlib.sha256(content).hexdigest()
        document.file.seek(0)
        
        doc_path_rel = f"standards/{document.filename}"
        try:
            storage_service.upload_file(content, doc_path_rel, document.content_type)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload document to storage.")

        source_doc = StandardSourceDocument(
            ingestion_job_id=job.id,
            name=document.filename,
            document=doc_path_rel,
            content_hash=file_hash,
            status=StandardSourceDocument.STATUS_UPLOADED
        )
        db.add(source_doc)
        
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {exc}")

    try:
        dispatch = dispatch_standard_ingestion(str(job.id))
        if dispatch.get("mode") == "async":
            job.summary_json["celery_task_id"] = dispatch.get("task_id")
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(job, "summary_json")
            db.commit()
            return job
        else:
            return job
    except Exception as exc:
        logger.exception("Failed to enqueue job_id=%s", job.id)
        job.status = StandardIngestionJob.STATUS_FAILED
        job.error_message = f"Failed to enqueue ingestion job: {exc}"
        db.commit()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Failed to enqueue ingestion job.")


@router.get("/{job_id}", response_model=StandardIngestionJobSchema)
def get_ingestion_job(job_id: int, db: Session = Depends(get_db)):
    """Get details of a specific ingestion job."""
    job = db.get(StandardIngestionJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job


@router.get("/{job_id}/asvs-level-definitions", response_model=List[ASVSLevelDefinitionSchema])
def get_ingestion_job_asvs_level_definitions(job_id: int, db: Session = Depends(get_db)):
    """Get ASVS level definitions extracted for a specific ingestion job."""
    job = db.get(StandardIngestionJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return db.execute(
        select(ASVSLevelDefinition)
        .where(ASVSLevelDefinition.ingestion_job_id == job.id)
        .order_by(ASVSLevelDefinition.level)
    ).scalars().all()


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ingestion_job(job_id: int, db: Session = Depends(get_db)):
    """Delete a specific ingestion job when safe to do so."""
    job = db.get(StandardIngestionJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    if job.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Active ingestion version cannot be deleted.")
    
    if job.status in (StandardIngestionJob.STATUS_PENDING, StandardIngestionJob.STATUS_RUNNING):
        celery_task_id = job.summary_json.get("celery_task_id") if job.summary_json else None
        if celery_task_id:
            from sdr.celery_app import celery_app
            celery_app.control.revoke(celery_task_id, terminate=True, signal="SIGTERM")

    db.delete(job)
    db.commit()
    return None


@router.post("/{job_id}/cancel", response_model=StandardIngestionJobSchema)
def cancel_ingestion_job(job_id: int, db: Session = Depends(get_db)):
    """Cancel a pending or running ingestion job."""
    job = db.get(StandardIngestionJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    if job.status not in (StandardIngestionJob.STATUS_PENDING, StandardIngestionJob.STATUS_RUNNING):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only pending or running jobs can be cancelled.")

    celery_task_id = job.summary_json.get("celery_task_id") if job.summary_json else None
    if celery_task_id:
        from sdr.celery_app import celery_app
        celery_app.control.revoke(celery_task_id, terminate=True, signal="SIGTERM")

    job.status = StandardIngestionJob.STATUS_CANCELLED
    job.error_message = (job.error_message or "") + " [Job manually cancelled by user]"
    db.commit()
    db.refresh(job)
    return job


@router.post("/{job_id}/activate", response_model=StandardIngestionJobSchema)
def activate_ingestion_job(job_id: int, db: Session = Depends(get_db)):
    """Activate a completed ingestion job."""
    job = db.get(StandardIngestionJob, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    if job.status != StandardIngestionJob.STATUS_COMPLETED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only completed jobs can be activated.")

    if not job.is_active:
        from datetime import datetime
        # Deactivate others
        db.execute(
            StandardIngestionJob.__table__.update()
            .where(StandardIngestionJob.category_id == job.category_id)
            .where(StandardIngestionJob.is_active == True)
            .where(StandardIngestionJob.id != job.id)
            .values(is_active=False)
        )
        job.is_active = True
        job.activated_at = datetime.utcnow()
        db.commit()
        db.refresh(job)

    return job
