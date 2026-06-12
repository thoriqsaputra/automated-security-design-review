from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, Index, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sdr.core.database import Base
from .choices import FindingStatus, FindingType, MetStatus, Severity


class Finding(Base):
    """
    One Finding = one security parameter evaluated against the TSD.

    Produced by the Multi-Agent Debate pipeline:
      - Hunter agent identifies evidence (or lack thereof)
      - Critic agent challenges the Hunter's finding
      - Mediator agent produces the final binding verdict
    """
    __tablename__ = "reviews_finding"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews_review.id", ondelete="CASCADE"), index=True)
    category_id: Mapped[Optional[int]] = mapped_column(ForeignKey("standards_standardcategory.id", ondelete="SET NULL"), nullable=True, index=True)
    parent_parameter_id: Mapped[Optional[int]] = mapped_column(ForeignKey("standards_categoryparameterparent.id", ondelete="SET NULL"), nullable=True, index=True)
    child_parameter_id: Mapped[Optional[int]] = mapped_column(ForeignKey("standards_categoryparameterchild.id", ondelete="SET NULL"), nullable=True, index=True)

    # Common fields
    finding_type: Mapped[str] = mapped_column(String(24), default=FindingType.REQUIREMENT.value, index=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(String)

    # Verdict fields
    met_status: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    severity: Mapped[Optional[str]] = mapped_column(String(24), nullable=True, index=True)
    severity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    severity_analysis: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # Agent reasoning audit trail
    hunter_reasoning: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    critic_reasoning: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mediator_reasoning: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    hunter_thought_process: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    critic_thought_process: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mediator_thought_process: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Requirement traceability
    reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    recommendation: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    requirement_reference: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    requirement_text: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    requirement_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    diagram_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    diagram_caption: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    vision_reasoning: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    vision_thought_process: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # ORM relationships
    review = relationship("Review", back_populates="findings")
    citations = relationship("CitationAnchor", back_populates="finding", cascade="all, delete-orphan")
    category = relationship("StandardCategory")
    parent_parameter = relationship("CategoryParameterParent")
    child_parameter = relationship("CategoryParameterChild")

    __table_args__ = (
        Index("idx_finding_review_type", "review_id", "finding_type"),
        Index("idx_finding_review_met", "review_id", "met_status"),
        Index("idx_finding_severity", "severity"),
        Index("idx_finding_met_conf", "met_status", "confidence_score"),
        Index("idx_finding_cat_created", "category_id", "created_at"),
    )

    def __str__(self):
        return (
            f"Finding: {self.title} "
            f"[{self.met_status or 'pending'}] "
            f"({self.finding_type})"
        )

    @property
    def is_actionable(self) -> bool:
        return self.met_status == MetStatus.NOT_MET.value
