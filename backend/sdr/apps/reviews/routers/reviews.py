from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from sdr.core.database import get_db
from ..models import Review, Finding
from ..models.choices import ReviewStatus
from ..schemas import ReviewSchema, ReviewCreateSchema, FindingSchema
from ..tasks import dispatch_review_analysis

router = APIRouter()


@router.get("/", response_model=List[ReviewSchema])
def list_reviews(
    skip: int = 0, 
    limit: int = 20, 
    design_id: Optional[int] = Query(None, description="Filter by design ID"),
    db: Session = Depends(get_db)
):
    stmt = select(Review).order_by(Review.created_at.desc())
    if design_id:
        stmt = stmt.where(Review.design_id == design_id)
        
    reviews = db.execute(stmt.offset(skip).limit(limit)).scalars().all()
    return reviews


@router.post("/", response_model=ReviewSchema, status_code=status.HTTP_201_CREATED)
def create_review(payload: ReviewCreateSchema, db: Session = Depends(get_db)):
    from sdr.apps.standards.models import StandardIngestionJob, StandardCategory
    
    # Check category active job
    category = db.get(StandardCategory, payload.category_id)
    if not category:
        raise HTTPException(status_code=400, detail="Category not found.")
        
    job = db.execute(
        select(StandardIngestionJob)
        .where(StandardIngestionJob.category_id == category.id, StandardIngestionJob.is_active == True)
    ).scalars().first()
    
    if not job:
        raise HTTPException(
            status_code=400, 
            detail=f"Category '{category.name}' has no active parameter baseline. Please complete an ingestion job before creating a review."
        )

    review = Review(
        design_id=payload.design_id,
        ingestion_job_id=job.id,
        status=Review.STATUS_PENDING,
        asvs_level_override=payload.asvs_level_override,
    )
    db.add(review)
    db.flush()

    review.selected_categories.append(category)

    db.commit()
    db.refresh(review)
    return review


@router.get("/{review_id}", response_model=ReviewSchema)
def get_review(review_id: int, db: Session = Depends(get_db)):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
    return review


@router.get("/{review_id}/retrieval-visualization")
def get_review_retrieval_visualization(review_id: int, db: Session = Depends(get_db)):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")

    snapshot = review.retrieval_snapshot_json
    if snapshot:
        return snapshot
    return {"status": "pending", "generated_at": None, "raptor": None, "graph": None}


@router.get("/{review_id}/findings", response_model=List[FindingSchema])
def get_review_findings(
    review_id: int, 
    skip: int = 0, 
    limit: int = 50,
    db: Session = Depends(get_db)
):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
        
    findings = db.execute(
        select(Finding)
        .where(Finding.review_id == review_id)
        .order_by(Finding.created_at.desc())
        .offset(skip)
        .limit(limit)
    ).scalars().all()
    
    return findings


@router.post("/{review_id}/trigger", response_model=ReviewSchema)
def trigger_review(review_id: int, db: Session = Depends(get_db)):
    from sqlalchemy import delete
    
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
        
    if review.status in [Review.STATUS_RUNNING, Review.STATUS_COMPLETED_CLEAN, Review.STATUS_COMPLETED_WITH_FINDINGS]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Review cannot be triggered in its current state")
        
    try:
        # Clear old state if re-triggering a cancelled or failed review
        db.execute(delete(Finding).where(Finding.review_id == review.id))
        review.error_message = None
        review.completed_at = None
        review.summary_json = {}
        review.retrieval_snapshot_json = None

        review.status = Review.STATUS_RUNNING
        dispatch_res = dispatch_review_analysis(review.id)
        review.celery_task_id = dispatch_res.get("task_id")
        db.commit()
        db.refresh(review)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to trigger review: {e}")
        
    return review

@router.post("/{review_id}/cancel", response_model=ReviewSchema)
def cancel_review(review_id: int, db: Session = Depends(get_db)):
    from datetime import datetime, timezone
    from sdr.celery_app import celery_app
    
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
        
    # Prevent cancelling already finished reviews
    terminal_states = [
        ReviewStatus.COMPLETED_CLEAN.value, 
        ReviewStatus.COMPLETED_WITH_FINDINGS.value, 
        ReviewStatus.APPROVED.value, 
        ReviewStatus.REJECTED.value,
        ReviewStatus.CANCELLED.value,
        ReviewStatus.FAILED.value
    ]
    if review.status in terminal_states:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot cancel a review that is already completed or terminated")
        
    # Kill the background task if it exists
    if review.celery_task_id:
        celery_app.control.revoke(review.celery_task_id, terminate=True, signal="SIGTERM")
        
    review.status = ReviewStatus.CANCELLED.value
    review.completed_at = datetime.now(timezone.utc)
    review.error_message = "Analysis was cancelled by user."
    db.commit()
    db.refresh(review)
    return review
