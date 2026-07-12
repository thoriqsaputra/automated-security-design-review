from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import String, Boolean, DateTime, Integer, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from sdr.core.database import Base
from .base import StandardsBigIntBase


class StandardIngestionJob(Base, StandardsBigIntBase):
    __tablename__ = "standards_standingestionjob"

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("standards_standardcategory.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(24), default=STATUS_PENDING)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    summary_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    version_no: Mapped[int] = mapped_column(Integer, default=1)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    activated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    category = relationship("StandardCategory", backref="ingestion_jobs")
    # requested_by = relationship("User", backref="standard_ingestion_jobs") # Assuming User model exists
    
    @property
    def summary(self) -> Dict[str, Any]:
        return self.summary_json or {}

    @property
    def progress(self) -> Dict[str, Any]:
        docs = self.source_documents if getattr(self, "source_documents", None) else []
        detailed_progress = self.summary_json.get("detailed_progress", {})
        
        status_label = detailed_progress.get("label") or self.status.capitalize()
        stored_percentage = detailed_progress.get("percentage")
        if stored_percentage is not None:
            percentage = stored_percentage
        else:
            percentage = 100 if self.status in ["completed", "failed"] else 0
        
        return {
            "phase": self.status,
            "percentage": percentage,
            "total_document": len(docs),
            "uploaded_document": sum(1 for d in docs if d.status != "failed"),
            "parsed_document": sum(1 for d in docs if d.status in ["parsed", "processed"]),
            "processed_document": sum(1 for d in docs if d.status == "processed"),
            "failed_document": sum(1 for d in docs if d.status == "failed"),
            "status_label": status_label
        }

    __table_args__ = (
        Index("unique_active_ingestion_job_per_category", "category_id", unique=True, postgresql_where=(is_active == True)),
    )


class StandardSourceDocument(Base, StandardsBigIntBase):
    __tablename__ = "standards_standardsourcedocument"

    STATUS_UPLOADED = "uploaded"
    STATUS_PARSED = "parsed"
    STATUS_PROCESSED = "processed"
    STATUS_FAILED = "failed"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ingestion_job_id: Mapped[int] = mapped_column(ForeignKey("standards_standingestionjob.id", ondelete="CASCADE"), index=True)
    
    name: Mapped[str] = mapped_column(String(255))
    document: Mapped[str] = mapped_column(String(1024))
    content_hash: Mapped[str] = mapped_column(String(128), index=True)
    document_version_label: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    
    status: Mapped[str] = mapped_column(String(24), default=STATUS_UPLOADED)

    # Relationships
    ingestion_job = relationship("StandardIngestionJob", backref=backref("source_documents", cascade="all, delete-orphan"))

import os
from sqlalchemy import event

@event.listens_for(StandardSourceDocument, 'after_delete')
def delete_standard_file_on_delete(mapper, connection, target):
    """
    Deletes the associated file from storage when a StandardSourceDocument object is deleted.
    """
    if target.document:
        print(f"Deleting file: {target.document}")
        try:
            if os.path.exists(target.document):
                os.remove(target.document)
                print(f"Successfully deleted file: {target.document}")
        except Exception as e:
            print(f"Failed to delete file {target.document}: {e}")
