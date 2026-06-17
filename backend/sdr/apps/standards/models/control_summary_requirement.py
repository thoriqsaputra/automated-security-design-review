from typing import Optional

from sqlalchemy import String, Integer, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from sdr.core.database import Base
from .base import StandardsBigIntBase


class CategoryControlSummaryRequirement(Base, StandardsBigIntBase):
    """
    Distilled control-family summary requirements generated during standard ingestion.

    Each CategoryParameterParent (control family, e.g. "V1 Architecture") can have
    10-30 raw children.  This table stores 3-5 synthesized requirements per parent
    per ASVS level, purpose-built for text-based TSD debate.

    Using these as debate units instead of raw children reduces debate iterations
    from 200-400 to ~50-90 per review while preserving full traceability via
    covered_child_keys.
    """

    __tablename__ = "standards_categorycontrolsummaryrequirement"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("standards_standardcategory.id", ondelete="CASCADE"),
        index=True,
    )
    ingestion_job_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("standards_standingestionjob.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    parent_id: Mapped[int] = mapped_column(
        ForeignKey("standards_categoryparameterparent.id", ondelete="CASCADE"),
        index=True,
    )
    asvs_level: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )

    # Unique identifier for this summary requirement (e.g. "job5-CFSR-V1-L1-1")
    stable_key: Mapped[str] = mapped_column(String(255), index=True)

    # Synthesized requirement text for TSD debate, max ~150 chars
    requirement_text: Mapped[str] = mapped_column(String)

    # 2-3 sentences describing what TSD content satisfies this requirement
    analysis_hint: Mapped[str] = mapped_column(String)

    # JSON array of the raw child stable_keys this CFSR covers (for cascade marking)
    covered_child_keys: Mapped[list] = mapped_column(JSONB, default=list)

    # Ordering within (parent, asvs_level)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    category = relationship("StandardCategory", backref="control_summary_requirements")
    ingestion_job = relationship(
        "StandardIngestionJob",
        backref=backref("control_summary_requirements", cascade="all, delete-orphan"),
    )
    parent = relationship("CategoryParameterParent", backref="control_summary_requirements")

    @property
    def details(self) -> str:
        """Duck-type compatibility with CategoryParameterChild for DebateInputFactory."""
        return self.analysis_hint

    def __str__(self):
        return f"[CFSR-L{self.asvs_level}] {self.requirement_text[:80]}"

    __table_args__ = (
        UniqueConstraint(
            "category_id",
            "stable_key",
            name="unique_cfsr_key_per_category",
        ),
        CheckConstraint(
            "asvs_level IN (1, 2, 3)",
            name="ck_cfsr_asvs_level_range",
        ),
    )
