from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, Index, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sdr.core.database import Base
from .choices import AnchorType


class CitationAnchor(Base):
    """
    Stores precise source location metadata for a Finding so the frontend
    can implement "click-to-source" navigation in the PDF viewer.

    A single Finding may have multiple CitationAnchors — the Mediator
    agent may cite 3 different text blocks across 2 pages as evidence
    for a single verdict. Each anchor maps to one block_id from the
    TSD ingestion pipeline.

    One Finding → Many CitationAnchors.
    """
    __tablename__ = "reviews_citationanchor"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    finding_id: Mapped[int] = mapped_column(ForeignKey("reviews_finding.id", ondelete="CASCADE"), index=True)
    anchor_type: Mapped[str] = mapped_column(String(16), default=AnchorType.TEXT.value)
    
    # Matches TextBlock.block_id or DiagramBlock.diagram_id from TSD ingestor
    # Format: "p{page_number}_b{block_idx}" or "p{page_number}_d{diagram_idx}"
    block_id: Mapped[str] = mapped_column(String(64), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    
    # Bounding box in PDF coordinate space (points from bottom-left)
    bbox_x0: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_y0: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_x1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_y1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Verbatim snippet from the source block that supports the verdict
    quoted_text: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Relationships
    finding = relationship("Finding", back_populates="citations")

    __table_args__ = (
        Index("idx_citationanchor_finding_page", "finding_id", "page_number"),
    )

    def __str__(self):
        return f"Citation [{self.block_id}] p.{self.page_number} → Finding {self.finding_id}"