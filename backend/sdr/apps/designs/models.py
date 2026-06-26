from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, event, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sdr.core.database import Base


class DesignPreparationStatus:
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"
    STALE = "stale"


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

    document_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    prepared_document_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    preparation_status: Mapped[str] = mapped_column(
        String(24),
        default=DesignPreparationStatus.QUEUED,
        server_default=DesignPreparationStatus.QUEUED,
    )
    preparation_error: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    prepared_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    preparation_snapshot_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    preparation_progress_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    active_preparation_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("designs_designpreparation.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    active_preparation = relationship(
        "DesignPreparation",
        foreign_keys=[active_preparation_id],
        post_update=True,
    )
    preparations = relationship(
        "DesignPreparation",
        back_populates="design",
        foreign_keys="DesignPreparation.design_id",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __str__(self):
        return self.name

    @property
    def can_start_analysis(self) -> bool:
        return (
            self.preparation_status == DesignPreparationStatus.READY
            and bool(self.active_preparation_id)
            and bool(self.document_sha256)
            and self.document_sha256 == self.prepared_document_sha256
        )


class DesignPreparation(Base):
    __tablename__ = "designs_designpreparation"

    STATUS_QUEUED = DesignPreparationStatus.QUEUED
    STATUS_RUNNING = DesignPreparationStatus.RUNNING
    STATUS_READY = DesignPreparationStatus.READY
    STATUS_FAILED = DesignPreparationStatus.FAILED
    STATUS_STALE = DesignPreparationStatus.STALE

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    design_id: Mapped[int] = mapped_column(ForeignKey("designs_design.id", ondelete="CASCADE"), index=True)
    document_sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(24), default=STATUS_QUEUED, server_default=STATUS_QUEUED)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    pipeline_schema_version: Mapped[int] = mapped_column(Integer, default=1)
    embedding_model_name: Mapped[str] = mapped_column(String(128), default="")
    embedding_model_dim: Mapped[int] = mapped_column(Integer, default=1024)
    tsd_document_object_key: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    raptor_artifact_object_key: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    retrieval_snapshot_object_key: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    stats_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    progress_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    prepared_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    design = relationship("Design", back_populates="preparations", foreign_keys=[design_id])
    raptor_nodes = relationship(
        "DesignPreparationRaptorNode",
        back_populates="preparation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    __table_args__ = (
        Index("idx_designpreparation_design_active", "design_id", "is_active"),
        Index(
            "idx_designpreparation_design_hash_version",
            "design_id",
            "document_sha256",
            "pipeline_schema_version",
            "embedding_model_name",
            "embedding_model_dim",
        ),
    )


class DesignPreparationRaptorNode(Base):
    __tablename__ = "designs_designpreparationraptornode"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    preparation_id: Mapped[int] = mapped_column(
        ForeignKey("designs_designpreparation.id", ondelete="CASCADE"),
        index=True,
    )
    node_id: Mapped[str] = mapped_column(String(255))
    parent_node_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    level: Mapped[int] = mapped_column(Integer, default=0)
    section_heading: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(String)
    source_block_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)
    page_numbers: Mapped[list[int]] = mapped_column(JSONB, default=list)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    embedding = mapped_column(Vector(1024), nullable=True)
    has_embedding: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    preparation = relationship("DesignPreparation", back_populates="raptor_nodes")

    __table_args__ = (
        Index("idx_designprep_raptor_preparation_level", "preparation_id", "level"),
        Index(
            "idx_designprep_raptor_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("idx_designprep_raptor_preparation_node", "preparation_id", "node_id", unique=True),
    )


@event.listens_for(Design, "after_delete")
def delete_design_file_on_delete(mapper, connection, target):
    if target.document:
        try:
            if os.path.exists(target.document):
                os.remove(target.document)
        except Exception:
            pass


@event.listens_for(Design, "before_update")
def delete_old_file_on_update(mapper, connection, target):
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
