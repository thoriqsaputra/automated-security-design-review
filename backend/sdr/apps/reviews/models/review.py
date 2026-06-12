from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, Integer, ForeignKey, DateTime, Index, func, Table, Column, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from sdr.core.database import Base
from .choices import ReviewStatus

review_category_association = Table(
    "reviews_review_categories",
    Base.metadata,
    Column("review_id", Integer, ForeignKey("reviews_review.id", ondelete="CASCADE"), primary_key=True),
    Column("category_id", Integer, ForeignKey("standards_standardcategory.id", ondelete="CASCADE"), primary_key=True),
)

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

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    design_id: Mapped[int] = mapped_column(ForeignKey("designs_design.id", ondelete="CASCADE"), index=True)
    # Removing 'requested_by' and 'reviewer' as AUTH_USER_MODEL has been removed in the project.
    
    ingestion_job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("standards_standingestionjob.id", ondelete="SET NULL"), nullable=True, index=True)
    
    status: Mapped[str] = mapped_column(String(24), default=ReviewStatus.PENDING.value, index=True)
    overview: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    asvs_level_override: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    summary_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    retrieval_snapshot_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Relationships
    design = relationship("Design", backref=backref("reviews", cascade="all, delete-orphan", passive_deletes=True))
    ingestion_job = relationship("StandardIngestionJob")
    findings = relationship("Finding", back_populates="review", cascade="all, delete-orphan")
    selected_categories = relationship("StandardCategory", secondary=review_category_association)

    __table_args__ = (
        Index("idx_review_design_status", "design_id", "status"),
        CheckConstraint("asvs_level_override IS NULL OR asvs_level_override IN (1, 2, 3)", name="ck_review_asvs_level_override_range"),
    )

    def __str__(self):
        # We try to get design.name safely if joined
        design_name = getattr(self.design, "name", "Unknown Design") if self.design else "Unknown Design"
        return f"Review for {design_name} — {self.status}"
