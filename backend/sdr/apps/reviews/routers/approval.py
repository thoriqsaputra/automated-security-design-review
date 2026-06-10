from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from sdr.core.database import get_db
from ..models import Review
from ..models.choices import ReviewStatus
from ..schemas import ReviewSchema

router = APIRouter()


@router.post("/{review_id}/approve", response_model=ReviewSchema)
def approve_review(review_id: int, db: Session = Depends(get_db)):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
        
    if review.status not in [ReviewStatus.COMPLETED_CLEAN.value, ReviewStatus.COMPLETED_WITH_FINDINGS.value, ReviewStatus.REJECTED.value]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only completed reviews can be approved")
        
    review.status = ReviewStatus.APPROVED.value
    db.commit()
    db.refresh(review)
    return review


@router.post("/{review_id}/reject", response_model=ReviewSchema)
def reject_review(review_id: int, db: Session = Depends(get_db)):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
        
    if review.status not in [ReviewStatus.COMPLETED_CLEAN.value, ReviewStatus.COMPLETED_WITH_FINDINGS.value, ReviewStatus.APPROVED.value]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only completed reviews can be rejected")
        
    review.status = ReviewStatus.REJECTED.value
    db.commit()
    db.refresh(review)
    return review
