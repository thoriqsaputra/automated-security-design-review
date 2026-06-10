import os
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, func, event
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sdr.core.database import Base


class Design(Base):
    __tablename__ = "designs_design"

    SOURCE_FORMAT_PDF = "pdf"
    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_READY = "ready"
    STATUS_ERROR = "error"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    name: Mapped[str] = mapped_column(String(255))
    document: Mapped[str] = mapped_column(String(1024))
    source_format: Mapped[str] = mapped_column(String(10), default=SOURCE_FORMAT_PDF)
    original_filename: Mapped[str] = mapped_column(String(255), default="")
    status: Mapped[str] = mapped_column(String(20), default=STATUS_READY)
    processing_error: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Note: 'reviews' relationship is defined in apps.reviews.models using a backref or back_populates.
    # We will assume `reviews` will be accessible if `Review` defines the relationship.
    
    def __str__(self):
        return self.name


@event.listens_for(Design, 'after_delete')
def delete_design_file_on_delete(mapper, connection, target):
    """
    Deletes the associated file from storage when a Design object is deleted.
    """
    if target.document:
        try:
            if os.path.exists(target.document):
                os.remove(target.document)
        except Exception:
            pass


@event.listens_for(Design, 'before_update')
def delete_old_file_on_update(mapper, connection, target):
    """
    Remove the previous file when a design document is replaced.
    """
    # Note: Using SQLAlchemy's Session to query the old object state can be done
    # via the object's history, but since we are updating the file directly via FastAPI,
    # the old state can be retrieved from `target`'s history.
    state = getattr(target, "_sa_instance_state", None)
    if not state:
        return

    history = state.attrs.document.history
    if history.has_changes():
        old_file = history.deleted[0] if history.deleted else None
        if old_file and old_file != target.document:
            try:
                if os.path.exists(old_file):
                    os.remove(old_file)
            except Exception:
                pass