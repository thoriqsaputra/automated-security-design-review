import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import select

from sdr.core.database import get_db
from sdr.apps.workspace.services.storage import storage_service
from ..models import Design
from ..schemas import DesignSchema, DesignDetailSchema
from ..validators import validate_design_document_file
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


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

    doc_path = f"designs/{document.filename}"
    
    content = document.file.read()
    document.file.seek(0)
    
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
    )
    db.add(design)
    db.commit()
    db.refresh(design)
    return design


@router.get("/{design_id}", response_model=DesignDetailSchema)
def get_design(design_id: int, db: Session = Depends(get_db)):
    """Retrieve details of a specific design."""
    design = db.get(Design, design_id)
    if not design:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Design not found.")

    design_dict = {
        "id": design.id,
        "name": design.name,
        "document": design.document,
        "source_format": design.source_format,
        "original_filename": design.original_filename,
        "created_at": design.created_at,
        "updated_at": design.updated_at,
        "status": design.status,
        "processing_error": design.processing_error,
        "review_status": "no_review",
        "review_id": None,
        "review_has_unmet_findings": False,
        "review_queue_position": None,
        "review_queue_size": None,
        "review_queue_state": "none",
        "has_review": False,
        "review": None,
    }
    return design_dict


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
        try:
            validate_design_document_file(document)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
        doc_path = f"designs/{document.filename}"
        
        content = document.file.read()
        document.file.seek(0)
        
        try:
            storage_service.upload_file(content, doc_path, document.content_type)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload document to storage.")

        design.document = doc_path
        design.source_format = Design.SOURCE_FORMAT_PDF
        design.original_filename = document.filename
        design.status = Design.STATUS_READY
        design.processing_error = None

    db.commit()
    db.refresh(design)

    # Return a mocked DesignDetailSchema for now
    design_dict = {
        "id": design.id,
        "name": design.name,
        "document": design.document,
        "source_format": design.source_format,
        "original_filename": design.original_filename,
        "created_at": design.created_at,
        "updated_at": design.updated_at,
        "status": design.status,
        "processing_error": design.processing_error,
        "review_status": "no_review",
        "review_id": None,
        "review_has_unmet_findings": False,
        "review_queue_position": None,
        "review_queue_size": None,
        "review_queue_state": "none",
        "has_review": False,
        "review": None,
    }
    return design_dict


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
