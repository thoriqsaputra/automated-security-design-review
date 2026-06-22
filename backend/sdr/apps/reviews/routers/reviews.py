import asyncio
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from sdr.core.database import get_db
from sdr.apps.workspace.services.storage import storage_service
from sdr.apps.reviews.services import review_debate_event_store
from ..models import Review, Finding
from ..models.choices import ReviewStatus
from ..schemas import (
    ReviewSchema,
    ReviewCreateSchema,
    ReviewTriggerSchema,
    FindingSchema,
    PaginatedResponse,
)
from ..tasks import dispatch_review_analysis

router = APIRouter()

_TERMINAL_REVIEW_STATES = {
    ReviewStatus.COMPLETED_CLEAN.value,
    ReviewStatus.COMPLETED_WITH_FINDINGS.value,
    ReviewStatus.APPROVED.value,
    ReviewStatus.REJECTED.value,
    ReviewStatus.CANCELLED.value,
    ReviewStatus.FAILED.value,
}


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
    from sdr.apps.designs.models import Design
    from sdr.apps.standards.models import StandardIngestionJob, StandardCategory
    
    design = db.get(Design, payload.design_id)
    if not design:
        raise HTTPException(status_code=400, detail="Design not found.")
    if not getattr(design, "can_start_analysis", False):
        raise HTTPException(
            status_code=409,
            detail="Design preparation is not ready. Wait for upload preparation to finish before starting analysis.",
        )

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
        design_id=design.id,
        category_id=category.id,
        ingestion_job_id=job.id,
        status=Review.STATUS_PENDING,
        analysis_mode=payload.analysis_mode.value,
    )
    db.add(review)
    db.flush()

    db.commit()
    db.refresh(review)
    return review


@router.get("/{review_id}", response_model=ReviewSchema)
def get_review(review_id: int, db: Session = Depends(get_db)):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
    return review


@router.get("/{review_id}/debates/stream")
async def stream_review_debates(
    review_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")

    snapshot = review_debate_event_store.load_snapshot(review_id)
    if snapshot is None and review.status in _TERMINAL_REVIEW_STATES:
        findings = db.execute(
            select(Finding)
            .where(Finding.review_id == review_id)
            .order_by(Finding.created_at.asc(), Finding.id.asc())
        ).scalars().all()
        snapshot = review_debate_event_store.build_completed_snapshot(
            review_id=review_id,
            review_status=review.status,
            findings=findings,
            error_message=review.error_message,
        )
        review_debate_event_store.save_snapshot(review_id, snapshot)
    elif snapshot is None:
        snapshot = {
            "review_id": review_id,
            "review_status": review.status,
            "error_message": review.error_message,
            "updated_at": None,
            "last_event_id": None,
            "debates": [],
        }

    async def event_generator():
        yield _format_sse_event("snapshot", snapshot)
        last_event_id = request.headers.get("last-event-id") or snapshot.get("last_event_id") or "0-0"

        while True:
            if await request.is_disconnected():
                break
            events = await asyncio.to_thread(
                review_debate_event_store.read_events,
                review_id,
                last_event_id=last_event_id,
                block_ms=15000,
            )
            if not events:
                yield ": keepalive\n\n"
                continue
            for event_id, payload in events:
                last_event_id = event_id
                yield _format_sse_event(
                    str(payload.get("type") or "message"),
                    payload,
                    event_id=event_id,
                )

    return StreamingResponse(event_generator(), media_type="text/event-stream")



@router.get("/{review_id}/document")
def get_review_document(review_id: int, db: Session = Depends(get_db)):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")

    design = getattr(review, "design", None)
    object_name = getattr(design, "document", None)
    if not object_name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review document not found")

    try:
        content = storage_service.download_bytes(str(object_name))
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review document not found")

    filename = getattr(design, "original_filename", None) or f"review-{review.id}.pdf"
    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )

@router.get("/{review_id}/retrieval-visualization")
def get_review_retrieval_visualization(review_id: int, db: Session = Depends(get_db)):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")

    snapshot = review.retrieval_snapshot_json
    if snapshot:
        return snapshot
    return {"status": "pending", "generated_at": None, "raptor": None, "graph": None}


@router.get("/{review_id}/findings", response_model=PaginatedResponse[FindingSchema])
def get_review_findings(
    review_id: int, 
    search: Optional[str] = Query(None, description="Search by title or description"),
    met_status: Optional[str] = Query(None, description="Comma-separated statuses"),
    severity: Optional[str] = Query(None, description="Comma-separated severities"),
    finding_type: Optional[str] = Query(None, description="Comma-separated types"),
    page: int = Query(1, ge=1), 
    size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    from sqlalchemy import and_, func
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
        
    where_clauses = [Finding.review_id == review_id]

    if search:
        where_clauses.append(Finding.title.ilike(f"%{search}%") | Finding.description.ilike(f"%{search}%"))
        
    if met_status:
        statuses = [s.strip() for s in met_status.split(",") if s.strip()]
        if statuses:
            where_clauses.append(Finding.met_status.in_(statuses))
            
    if severity:
        severities = [s.strip() for s in severity.split(",") if s.strip()]
        if severities:
            where_clauses.append(Finding.severity.in_(severities))
            
    if finding_type:
        types = [t.strip() for t in finding_type.split(",") if t.strip()]
        if types:
            where_clauses.append(Finding.finding_type.in_(types))

    total = db.execute(select(func.count(Finding.id)).where(and_(*where_clauses))).scalar() or 0

    findings = db.execute(
        select(Finding)
        .where(and_(*where_clauses))
        .order_by(Finding.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
    ).scalars().all()
    
    total_pages = (total + size - 1) // size if total > 0 else 1
    
    return {
        "items": findings,
        "total": total,
        "page": page,
        "size": size,
        "total_pages": total_pages
    }


@router.post("/{review_id}/trigger", response_model=ReviewSchema)
def trigger_review(
    review_id: int,
    payload: Optional[ReviewTriggerSchema] = None,
    db: Session = Depends(get_db),
):
    from sqlalchemy import delete
    
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
        
    if review.status in [Review.STATUS_RUNNING, Review.STATUS_COMPLETED_CLEAN, Review.STATUS_COMPLETED_WITH_FINDINGS]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Review cannot be triggered in its current state")
    if not getattr(review.design, "can_start_analysis", False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Design preparation is not ready. Retry once the design is prepared.",
        )
        
    try:
        # Clear old state if re-triggering a cancelled or failed review
        db.execute(delete(Finding).where(Finding.review_id == review.id))
        review.error_message = None
        review.completed_at = None
        review.summary_json = {}
        review.retrieval_snapshot_json = None
        review_debate_event_store.reset_review(review.id)
        if payload and payload.analysis_mode is not None:
            review.analysis_mode = payload.analysis_mode.value

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
    terminal_states = list(_TERMINAL_REVIEW_STATES)
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
    review_debate_event_store.mark_debates_cancelled(review.id, error_message=review.error_message)
    return review


def _format_sse_event(event_type: str, payload: dict, *, event_id: Optional[str] = None) -> str:
    parts = []
    if event_id:
        parts.append(f"id: {event_id}")
    parts.append(f"event: {event_type}")
    encoded = json.dumps(payload, default=str)
    for line in encoded.splitlines():
        parts.append(f"data: {line}")
    parts.append("")
    parts.append("")
    return "\n".join(parts)
