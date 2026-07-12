from typing import Optional

from sqlalchemy import String, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from sdr.core.database import Base
from .base import StandardsBigIntBase


class CategoryDiagramRequirement(Base, StandardsBigIntBase):
    """
    Diagram-specific security requirements extracted during standard ingestion.

    Unlike CategoryParameterChild (text parameters written for code/config
    verification), these requirements are purpose-built for visual verification
    — they describe security controls that CAN be seen in an architecture
    diagram.

    Generated from text requirements for visual verification during review.
    """

    __tablename__ = "standards_categorydiagramrequirement"

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
    stable_key: Mapped[str] = mapped_column(String(255), index=True)

    source_requirement_key: Mapped[str] = mapped_column(String(255))

    requirement_text: Mapped[str] = mapped_column(String)

    verification_hint: Mapped[str] = mapped_column(String)

    parent_section: Mapped[str] = mapped_column(String(255))

    diagram_type: Mapped[str] = mapped_column(String(64), server_default="", default="")

    ordinal: Mapped[int] = mapped_column(Integer, default=0)

    category = relationship("StandardCategory", backref="diagram_requirements")
    ingestion_job = relationship(
        "StandardIngestionJob",
        backref=backref("diagram_requirements", cascade="all, delete-orphan"),
    )


    __table_args__ = (
        UniqueConstraint(
            "category_id",
            "stable_key",
            name="unique_diagram_req_key_per_category",
        ),
    )

    def __str__(self):
        return f"[D] {self.requirement_text[:80]}"
