from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import select

from sdr.core.database import get_db
from sdr.apps.workspace.services.storage import storage_service
from ..models import Design, DesignPreparation
from ..preparation_store import DesignPreparationStore, compute_sha256_bytes
from ..schemas import DesignSchema, DesignDetailSchema
from ..tasks import dispatch_design_preparation
from ..validators import validate_design_document_file
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
preparation_store = DesignPreparationStore()


@router.get("/", response_model=List[DesignSchema])
def list_designs(db: Session = Depends(get_db)):
    """List all designs."""
    designs = db.execute(
        select(Design).order_by(Design.created_at.desc())
    ).scalars().all()
    return designs


@router.post("/", response_model=DesignSchema, status_code=status.HTTP_201_CREATED)
def create_design(
    name: str = Form(...),
    document: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Create a new design."""
    try:
        validate_design_document_file(document)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    content = document.file.read()
    document.file.seek(0)
    document_sha256 = compute_sha256_bytes(content)
    doc_path = f"designs/{document_sha256}-{document.filename}"
    
    try:
        storage_service.upload_file(content, doc_path, document.content_type)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload document to storage.")

    design = Design(
        name=name,
        document=doc_path,
        source_format=Design.SOURCE_FORMAT_PDF,
        original_filename=document.filename,
        status=Design.STATUS_READY,
        document_sha256=document_sha256,
        preparation_status=DesignPreparation.STATUS_QUEUED,
    )
    db.add(design)
    db.flush()
    preparation = preparation_store.ensure_preparation(
        db,
        design=design,
        document_sha256=document_sha256,
        force_rebuild=True,
    )
    db.commit()
    db.refresh(design)
    dispatch_res = dispatch_design_preparation(design.id, preparation.id)
    preparation.celery_task_id = dispatch_res.get("task_id")
    db.commit()
    return design


@router.get("/{design_id}", response_model=DesignDetailSchema)
def get_design(design_id: int, db: Session = Depends(get_db)):
    """Retrieve details of a specific design."""
    design = db.get(Design, design_id)
    if not design:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Design not found.")

    return _build_design_detail_payload(design)


@router.put("/{design_id}", response_model=DesignDetailSchema)
def update_design(
    design_id: int,
    name: Optional[str] = Form(None),
    document: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Update a design."""
    design = db.get(Design, design_id)
    if not design:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Design not found.")

    if name is not None:
        design.name = name

    if document is not None:
        from sdr.apps.reviews.models import Review

        active_review = db.execute(
            select(Review.id).where(
                Review.design_id == design.id,
                Review.status.in_([Review.STATUS_PENDING, Review.STATUS_RUNNING]),
            )
        ).scalars().first()
        if active_review:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot reingest while a review is pending or running for this design.",
            )

        try:
            validate_design_document_file(document)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        content = document.file.read()
        document.file.seek(0)
        document_sha256 = compute_sha256_bytes(content)
        doc_path = f"designs/{document_sha256}-{document.filename}"
        
        try:
            storage_service.upload_file(content, doc_path, document.content_type)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload document to storage.")

        design.document = doc_path
        design.source_format = Design.SOURCE_FORMAT_PDF
        design.original_filename = document.filename
        design.status = Design.STATUS_READY
        design.processing_error = None
        design.document_sha256 = document_sha256
        design.prepared_document_sha256 = None
        design.prepared_at = None
        design.preparation_error = None
        design.preparation_snapshot_json = None
        preparation = preparation_store.ensure_preparation(
            db,
            design=design,
            document_sha256=document_sha256,
            force_rebuild=True,
        )
    else:
        preparation = None

    db.commit()
    db.refresh(design)
    if preparation is not None:
        dispatch_res = dispatch_design_preparation(design.id, preparation.id)
        preparation.celery_task_id = dispatch_res.get("task_id")
        db.commit()
    return _build_design_detail_payload(design)


@router.post("/{design_id}/prepare", response_model=DesignDetailSchema)
def prepare_design(design_id: int, db: Session = Depends(get_db)):
    design = db.get(Design, design_id)
    if not design:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Design not found.")
    if not design.document_sha256:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Design is missing a document hash.")

    preparation = preparation_store.ensure_preparation(
        db,
        design=design,
        document_sha256=design.document_sha256,
        force_rebuild=True,
    )
    db.commit()
    db.refresh(design)
    dispatch_res = dispatch_design_preparation(design.id, preparation.id)
    preparation.celery_task_id = dispatch_res.get("task_id")
    db.commit()
    return _build_design_detail_payload(design)


@router.delete("/{design_id}", status_code=status.HTTP_200_OK)
def delete_design(design_id: int, db: Session = Depends(get_db)):
    """Delete a design."""
    design = db.get(Design, design_id)
    if not design:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Design not found.")
    

    
    design_name = design.name
    doc_path = design.document
    
    db.delete(design)
    db.commit()

    if doc_path:
        try:
            storage_service.delete_file(doc_path)
        except Exception as e:
            logger.warning("Failed to delete document from storage: %s", e)

    return {"message": f'Design "{design_name}" has been successfully deleted.'}


@router.post("/{design_id}/cancel-preparation", response_model=DesignDetailSchema)
def cancel_design_preparation(design_id: int, db: Session = Depends(get_db)):
    """Cancel a design preparation that is currently running or queued."""
    from sdr.celery_app import celery_app
    from ..models import DesignPreparationStatus

    design = db.get(Design, design_id)
    if not design:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Design not found.")
        
    preparation = preparation_store.get_active_preparation(db, design.id)
    if not preparation:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No active preparation found for this design.")
        
    if preparation.status not in [DesignPreparationStatus.QUEUED, DesignPreparationStatus.RUNNING]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot cancel a preparation that is not queued or running.")

    if preparation.celery_task_id:
        celery_app.control.revoke(preparation.celery_task_id, terminate=True, signal="SIGTERM")

    preparation_store.mark_failed(design, preparation, "Preparation was cancelled by user.")
    preparation.status = DesignPreparationStatus.CANCELLED
    design.preparation_status = DesignPreparationStatus.CANCELLED
    
    db.commit()
    db.refresh(design)

    return _build_design_detail_payload(design)


def _build_design_detail_payload(design: Design) -> dict:
    preparation = design.active_preparation
    preparation_duration_seconds = None
    if preparation and preparation.started_at and preparation.completed_at:
        preparation_duration_seconds = (
            preparation.completed_at - preparation.started_at
        ).total_seconds()

    return {
        "id": design.id,
        "name": design.name,
        "document": design.document,
        "source_format": design.source_format,
        "original_filename": design.original_filename,
        "created_at": design.created_at,
        "updated_at": design.updated_at,
        "status": design.status,
        "processing_error": design.processing_error,
        "document_sha256": design.document_sha256,
        "prepared_document_sha256": design.prepared_document_sha256,
        "preparation_status": design.preparation_status,
        "preparation_error": design.preparation_error,
        "prepared_at": design.prepared_at,
        "active_preparation_id": design.active_preparation_id,
        "preparation_snapshot_json": design.preparation_snapshot_json,
        "preparation_progress": design.preparation_progress_json,
        "preparation_stats_json": preparation.stats_json if preparation else None,
        "preparation_started_at": preparation.started_at if preparation else None,
        "preparation_completed_at": preparation.completed_at if preparation else None,
        "preparation_duration_seconds": preparation_duration_seconds,
        "can_start_analysis": design.can_start_analysis,
        "review_status": "no_review",
        "review_id": None,
        "review_has_unmet_findings": False,
        "review_queue_position": None,
        "review_queue_size": None,
        "review_queue_state": "none",
        "has_review": False,
        "review": None,
    }
