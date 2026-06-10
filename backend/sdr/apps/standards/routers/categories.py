from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select, delete, func

from sdr.core.database import get_db
from ..models import (
    CategoryParameterChild,
    CategoryParameterParent,
    StandardCategory,
    StandardIngestionJob,
)
from ..schemas import (
    CategoryParameterParentSchema,
    StandardCategorySchema,
    CategoryCodeEnum,
)

router = APIRouter(prefix="/categories")


@router.get("/", response_model=List[StandardCategorySchema])
def list_categories(db: Session = Depends(get_db)):
    """List all active standard categories."""
    categories = db.execute(
        select(StandardCategory)
        .where(StandardCategory.is_active == True)
        .order_by(StandardCategory.name)
    ).scalars().all()
    
    results = []
    for cat in categories:
        active_job = db.execute(
            select(StandardIngestionJob)
            .where(StandardIngestionJob.category_id == cat.id, StandardIngestionJob.is_active == True)
        ).scalars().first()
        
        active_job_version = active_job.version_no if active_job else None
        active_parameters_count = 0
        if active_job:
            active_parameters_count = db.execute(
                select(func.count(CategoryParameterChild.id))
                .join(CategoryParameterParent, CategoryParameterChild.parent_id == CategoryParameterParent.id)
                .where(
                    CategoryParameterParent.category_id == cat.id,
                    CategoryParameterParent.ingestion_job_id == active_job.id
                )
            ).scalar() or 0

        results.append(StandardCategorySchema(
            id=cat.id,
            code=cat.code,
            name=cat.name,
            description=cat.description,
            is_active=cat.is_active,
            active_parameters_count=active_parameters_count,
            active_job_version=active_job_version,
            created_at=cat.created_at,
            updated_at=cat.updated_at
        ))
    return results


@router.get("/{code}/parameters")
def get_category_parameters(
    code: CategoryCodeEnum,
    job_id: int = Query(None, description="Optional job ID to filter parameters by"),
    db: Session = Depends(get_db)
):
    """Get parameters for a specific category, optionally filtered by job_id."""
    category = db.execute(
        select(StandardCategory)
        .where(StandardCategory.code == code.value, StandardCategory.is_active == True)
    ).scalar_one_or_none()
    
    if not category:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Category not found.")

    if job_id:
        job = db.get(StandardIngestionJob, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
        if job.category_id != category.id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Selected job does not belong to this category.")
    else:
        job = db.execute(
            select(StandardIngestionJob)
            .where(StandardIngestionJob.category_id == category.id, StandardIngestionJob.is_active == True)
            .order_by(StandardIngestionJob.created_at.desc())
        ).scalars().first()

    if not job:
        return {
            "category": StandardCategorySchema.model_validate(category),
            "parameters": []
        }

    parents = db.execute(
        select(CategoryParameterParent)
        .where(
            CategoryParameterParent.category_id == category.id,
            CategoryParameterParent.ingestion_job_id == job.id
        )
    ).scalars().all()

    return {
        "category": StandardCategorySchema.model_validate(category),
        "parameters": [CategoryParameterParentSchema.model_validate(p) for p in parents]
    }


@router.delete("/parameters/parent/{parent_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_parameter_parent(parent_id: int, db: Session = Depends(get_db)):
    """Delete a parent parameter."""
    parent = db.get(CategoryParameterParent, parent_id)
    if not parent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Parent parameter not found.")
    
    db.delete(parent)
    db.commit()
    return None


@router.delete("/parameters/child/{child_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_parameter_child(child_id: int, db: Session = Depends(get_db)):
    """Delete a child parameter."""
    child = db.get(CategoryParameterChild, child_id)
    if not child:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Child parameter not found.")
    
    db.delete(child)
    db.commit()
    return None
