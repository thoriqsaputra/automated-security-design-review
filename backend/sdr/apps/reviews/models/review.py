from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, ForeignKey, DateTime, Index, func, Table, Column, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from sdr.core.database import Base
from .choices import ReviewAnalysisMode, ReviewStatus
from sdr.apps.reviews.services.debate_events import review_debate_event_store



class Review(Base):
    """
    Top-level record linking a TSD (Design) to the security standards
    it is being reviewed against.
    """
    __tablename__ = "reviews_review"

    STATUS_PENDING = ReviewStatus.PENDING.value
    STATUS_RUNNING = ReviewStatus.RUNNING.value
    STATUS_COMPLETED = ReviewStatus.COMPLETED_WITH_FINDINGS.value
    STATUS_COMPLETED_CLEAN = ReviewStatus.COMPLETED_CLEAN.value
    STATUS_COMPLETED_WITH_FINDINGS = ReviewStatus.COMPLETED_WITH_FINDINGS.value
    STATUS_FAILED = ReviewStatus.FAILED.value
    STATUS_APPROVED = ReviewStatus.APPROVED.value
    STATUS_REJECTED = ReviewStatus.REJECTED.value
    ANALYSIS_MODE_DEFAULT = ReviewAnalysisMode.DEFAULT.value
    ANALYSIS_MODE_TEXT_ONLY = ReviewAnalysisMode.TEXT_ONLY.value
    ANALYSIS_MODE_DIAGRAM_ONLY = ReviewAnalysisMode.DIAGRAM_ONLY.value

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    design_id: Mapped[int] = mapped_column(ForeignKey("designs_design.id", ondelete="CASCADE"), index=True)
    # Removing 'requested_by' and 'reviewer' as AUTH_USER_MODEL has been removed in the project.
    
    ingestion_job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("standards_standingestionjob.id", ondelete="SET NULL"), nullable=True, index=True)
    category_id: Mapped[Optional[int]] = mapped_column(ForeignKey("standards_standardcategory.id", ondelete="SET NULL"), nullable=True, index=True)
    
    status: Mapped[str] = mapped_column(String(24), default=ReviewStatus.PENDING.value, index=True)
    analysis_mode: Mapped[str] = mapped_column(
        String(24),
        default=ReviewAnalysisMode.DEFAULT.value,
        server_default=ReviewAnalysisMode.DEFAULT.value,
        nullable=False,
    )
    overview: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    summary_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    retrieval_snapshot_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Snapshot of the design's document at the time this review was created, so
    # that reingesting a design with a different PDF doesn't shift the document
    # (and misalign citation bboxes) underneath already-completed reviews.
    document_object_key: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    document_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    document_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Relationships
    design = relationship("Design", backref=backref("reviews", cascade="all, delete-orphan", passive_deletes=True))
    category = relationship("StandardCategory")
    ingestion_job = relationship("StandardIngestionJob")
    findings = relationship("Finding", back_populates="review", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_review_design_status", "design_id", "status"),
        CheckConstraint(
            "analysis_mode IN ('default', 'text_only', 'diagram_only')",
            name="ck_review_analysis_mode_valid",
        ),
    )

    def __str__(self):
        # We try to get design.name safely if joined
        design_name = getattr(self.design, "name", "Unknown Design") if self.design else "Unknown Design"
        return f"Review for {design_name} — {self.status}"

    @property
    def progress(self) -> Optional[Dict[str, Any]]:
        summary = self.summary_json or {}
        live_snapshot = review_debate_event_store.load_snapshot(self.id) if getattr(self, "id", None) else None
        debate_total = int(summary.get("debate_total_parameters") or summary.get("analysis_total_parameters") or 0)
        debate_completed = int(summary.get("debate_completed_parameters") or summary.get("analysis_processed_parameters") or 0)
        debate_remaining = int(summary.get("debate_remaining_parameters") or summary.get("analysis_remaining_parameters") or 0)
        persistence_total = int(summary.get("persistence_total_parameters") or debate_total or 0)
        persistence_completed = int(summary.get("persistence_completed_parameters") or 0)
        persistence_remaining = int(summary.get("persistence_remaining_parameters") or 0)
        error_count = int(summary.get("error_count") or 0)

        if (
            self.status not in {ReviewStatus.RUNNING.value}
            and debate_total == 0
            and persistence_total == 0
        ):
            return None

        if debate_remaining > 0:
            stage = "debate"
            total_items = debate_total
            completed_items = debate_completed
            remaining_items = debate_remaining
        elif persistence_remaining > 0:
            stage = "persistence"
            total_items = persistence_total
            completed_items = persistence_completed
            remaining_items = persistence_remaining
        elif self.status == ReviewStatus.RUNNING.value:
            stage = "preparation"
            total_items = max(debate_total, persistence_total)
            completed_items = debate_completed if debate_total else persistence_completed
            remaining_items = max(debate_remaining, persistence_remaining)
        else:
            stage = "completed"
            total_items = max(debate_total, persistence_total)
            completed_items = max(debate_completed, persistence_completed, total_items)
            remaining_items = 0

        progress_percent = int(round((completed_items / total_items) * 100)) if total_items > 0 else 0
        current_debate = None
        if isinstance(live_snapshot, dict):
            debates = live_snapshot.get("debates") or []
            for debate in debates:
                if str(debate.get("status") or "").lower() == "running":
                    current_debate = debate
                    break
        label = (
            f"Debate {debate_completed}/{debate_total} · "
            f"Persistence {persistence_completed}/{persistence_total}"
        )
        return {
            "stage": stage,
            "label": label,
            "total_items": total_items,
            "completed_items": completed_items,
            "failed_items": error_count,
            "remaining_items": remaining_items,
            "progress_percent": progress_percent,
            "current_parameter_reference": current_debate.get("requirement_reference") if isinstance(current_debate, dict) else None,
            "current_parameter_title": current_debate.get("requirement_text") if isinstance(current_debate, dict) else None,
            "preparation": {
                "debate": {
                    "total": debate_total,
                    "completed": debate_completed,
                    "remaining": debate_remaining,
                },
                "persistence": {
                    "total": persistence_total,
                    "completed": persistence_completed,
                    "remaining": persistence_remaining,
                },
                "categories": summary.get("category_stats", {}) if isinstance(summary.get("category_stats"), dict) else {},
            },
        }
