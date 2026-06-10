import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import String, Integer, DateTime, ForeignKey, Index, func, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sdr.core.database import Base


class ReviewAnalysisWorkPack(Base):
    """
    Groups analysis work items for a specific review.
    """
    __tablename__ = "ai_reviewanalysisworkpack"

    STATE_PENDING = "pending"
    STATE_RUNNING = "running"
    STATE_FINALIZING = "finalizing"
    STATE_VISION_POSTPASS = "vision_postpass"
    STATE_COMPLETED = "completed"
    STATE_FAILED = "failed"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    review_id: Mapped[int] = mapped_column(ForeignKey("reviews_review.id", ondelete="CASCADE"), index=True)
    state: Mapped[str] = mapped_column(String(24), default=STATE_PENDING, index=True)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    completed_items: Mapped[int] = mapped_column(Integer, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, default=0)
    chunk_size: Mapped[int] = mapped_column(Integer, default=4)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str] = mapped_column(String, default="")
    progress_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("idx_ai_workpack_review_state", "review_id", "state"),
        Index("idx_ai_workpack_created", "created_at"),
    )


class ReviewAnalysisWorkItem(Base):
    """
    A single analysis task for a specific parameter.
    """
    __tablename__ = "ai_reviewanalysisworkitem"

    STATE_PENDING = "pending"
    STATE_RUNNING = "running"
    STATE_DONE = "done"
    STATE_FAILED = "failed"
    STATE_SKIPPED = "skipped"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    workpack_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("ai_reviewanalysisworkpack.id", ondelete="CASCADE"), index=True)
    parameter_id: Mapped[Optional[int]] = mapped_column(ForeignKey("standards_categoryparameterchild.id", ondelete="SET NULL"), nullable=True, index=True)
    category_id: Mapped[Optional[int]] = mapped_column(ForeignKey("standards_standardcategory.id", ondelete="SET NULL"), nullable=True, index=True)
    ingestion_job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("standards_standingestionjob.id", ondelete="SET NULL"), nullable=True, index=True)
    
    domain: Mapped[str] = mapped_column(String(64), default="")
    payload_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    state: Mapped[str] = mapped_column(String(24), default=STATE_PENDING, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(String, default="")

    # Relationships
    workpack = relationship("ReviewAnalysisWorkPack", backref="items")
    # For parameters, etc., relationships can be defined or queried directly.

    __table_args__ = (
        Index("idx_ai_workitem_pack_state", "workpack_id", "state"),
    )
