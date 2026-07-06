from __future__ import annotations

import hashlib
import json
import logging
import gzip
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from sdr.apps.ai.client import usage_tracker
from sdr.apps.ai.client.session import build_tsd_ingestion_session_id, job_session_context
from sdr.apps.ai.engine.dto import IngestionOutput, RetrievalIndexes
from sdr.apps.ai.engine.preparation.ingestion_service import IngestionService
from sdr.apps.ai.engine.preparation.retrieval_service import RetrievalService
from sdr.apps.ai.engine.reporting.retrieval_snapshot_builder import RetrievalSnapshotBuilder
from sdr.apps.ai.tsd_processing.document_models import DiagramBlock, TSDDocument, TSDPage, TextBlock
from sdr.apps.ai.tsd_processing.raptor import RAPTORNode, RAPTORTree
from sdr.apps.workspace.services.storage import storage_service
from sdr.core.config import settings

from .models import (
    Design,
    DesignPreparation,
    DesignPreparationRaptorNode,
)

logger = logging.getLogger(__name__)

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - runtime fallback when dependency is absent
    zstd = None


class PreparationNotReadyError(RuntimeError):
    pass


class PreparationArtifactError(RuntimeError):
    pass


def default_preparation_progress() -> Dict[str, Any]:
    return {
        "phase": "queued",
        "percentage": 0,
        "status_label": "Queued",
        "current_step": "Waiting to start TSD preparation",
        "updated_at": None,
        "steps": {
            "document_ingestion": {"status": "pending", "progress_percent": 0, "label": "Queued"},
            "tsd_screening": {"status": "pending", "progress_percent": 0, "label": "Queued"},
            "raptor_index": {"status": "pending", "progress_percent": 0, "label": "Queued"},
            "artifact_persistence": {"status": "pending", "progress_percent": 0, "label": "Queued"},
        },
    }


def compute_sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compress_json_bytes(payload: Dict[str, Any]) -> bytes:
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if zstd is None:
        return gzip.compress(raw, compresslevel=6)
    compressor = zstd.ZstdCompressor(level=6)
    return compressor.compress(raw)


def decompress_json_bytes(content: bytes) -> Dict[str, Any]:
    if zstd is None:
        raw = gzip.decompress(content)
    else:
        decompressor = zstd.ZstdDecompressor()
        raw = decompressor.decompress(content)
    return json.loads(raw.decode("utf-8"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PreparationProgressTracker:
    def __init__(self, write_progress: Callable[[Dict[str, Any]], None]) -> None:
        self.write_progress = write_progress
        self.payload: Dict[str, Any] = default_preparation_progress()
        self._lock = threading.Lock()
        self._last_persisted_percent = -1
        self._last_persisted_step = ""
        self._last_persisted_at = 0.0
        self._max_overall_percent = 0
        self._max_step_percent: Dict[str, int] = {}

    def start(self) -> None:
        self._update_phase(
            phase="ingesting",
            percentage=0,
            status_label="Preparing document",
            current_step="Downloading and parsing uploaded TSD",
        )
        self._update_step("document_ingestion", status="running", progress_percent=0, label="Parsing uploaded PDF")
        self._persist(force=True)

    def mark_ingestion_complete(self) -> None:
        self._update_step("document_ingestion", status="completed", progress_percent=100, label="Document parsed")
        self._update_phase(
            phase="screening",
            percentage=20,
            status_label="Screening document",
            current_step="Validating that the upload is a TSD",
        )
        self._update_step("tsd_screening", status="running", progress_percent=0, label="Checking TSD eligibility")
        self._persist(force=True)

    def mark_screening_complete(self) -> None:
        self._update_step("tsd_screening", status="completed", progress_percent=100, label="TSD screening passed")
        self._update_phase(
            phase="building_indexes",
            percentage=25,
            status_label="Building retrieval indexes",
            current_step="Starting RAPTOR indexing",
        )
        self._update_step("raptor_index", status="running", progress_percent=0, label="Starting RAPTOR indexing")
        self._persist(force=True)

    def update_raptor(self, payload: Dict[str, Any]) -> None:
        self._update_parallel_step("raptor_index", payload, default_label="Building RAPTOR index")

    def mark_persisting(self) -> None:
        self._update_phase(
            phase="persisting",
            percentage=90,
            status_label="Persisting artifacts",
            current_step="Saving prepared indexes and vectors",
        )
        self._update_step("artifact_persistence", status="running", progress_percent=0, label="Saving artifacts")
        self._persist(force=True)

    def mark_persisting_complete(self) -> None:
        self._update_step("artifact_persistence", status="completed", progress_percent=100, label="Artifacts saved")
        self._update_phase(
            phase="ready",
            percentage=100,
            status_label="Ready",
            current_step="TSD preparation completed",
        )
        self._persist(force=True)

    def mark_failed(self, message: str) -> None:
        with self._lock:
            self.payload["phase"] = "failed"
            self.payload["status_label"] = "Preparation failed"
            self.payload["current_step"] = message
            self.payload["updated_at"] = _utc_now().isoformat()
            self._mark_running_step_failed(message)
        self._persist(force=True)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.payload))

    def _mark_running_step_failed(self, message: str) -> None:
        steps = self.payload.get("steps", {})
        for key, step in steps.items():
            if step.get("status") == "running":
                step["status"] = "failed"
                step["label"] = message
                return

    def _update_parallel_step(self, key: str, incoming: Dict[str, Any], *, default_label: str) -> None:
        with self._lock:
            mapped_status = self._map_status(incoming.get("status"))
            label = str(incoming.get("current_step") or default_label)
            incoming_percent = incoming.get("progress_percent")
            if incoming_percent is None:
                progress_percent = int(self.payload["steps"].get(key, {}).get("progress_percent") or 0)
            else:
                progress_percent = max(0, min(100, int(incoming_percent)))
            self._update_step(key, status=mapped_status, progress_percent=progress_percent, label=label)

            raptor_progress = int(self.payload["steps"]["raptor_index"]["progress_percent"] or 0)
            overall = min(90, 25 + int(round(0.65 * raptor_progress)))
            overall = max(overall, self._max_overall_percent)
            self._max_overall_percent = overall
            self.payload["phase"] = "building_indexes"
            self.payload["percentage"] = overall
            self.payload["status_label"] = "Building retrieval indexes"
            self.payload["current_step"] = self.payload["steps"]["raptor_index"].get("label") or "Building RAPTOR index"
            self.payload["updated_at"] = _utc_now().isoformat()
        self._persist()

    def _update_phase(self, *, phase: str, percentage: int, status_label: str, current_step: str) -> None:
        with self._lock:
            percentage = max(percentage, self._max_overall_percent)
            self._max_overall_percent = percentage
            self.payload["phase"] = phase
            self.payload["percentage"] = percentage
            self.payload["status_label"] = status_label
            self.payload["current_step"] = current_step
            self.payload["updated_at"] = _utc_now().isoformat()

    def _update_step(self, key: str, *, status: str, progress_percent: int, label: str) -> None:
        step = self.payload.setdefault("steps", {}).setdefault(key, {})
        prev_max = self._max_step_percent.get(key, 0)
        if status == "completed":
            clamped = 100
        else:
            clamped = max(progress_percent, prev_max)
        self._max_step_percent[key] = clamped
        step["status"] = status
        step["progress_percent"] = clamped
        step["label"] = label
        self.payload["updated_at"] = _utc_now().isoformat()

    def _persist(self, *, force: bool = False) -> None:
        snapshot = self.snapshot()
        percent = int(snapshot.get("percentage") or 0)
        current_step = str(snapshot.get("current_step") or "")
        now = time.monotonic()
        if not force:
            if abs(percent - self._last_persisted_percent) < 2 and current_step == self._last_persisted_step and (now - self._last_persisted_at) < 0.8:
                return
        self.write_progress(snapshot)
        self._last_persisted_percent = percent
        self._last_persisted_step = current_step
        self._last_persisted_at = now

    def _map_status(self, status: Any) -> str:
        normalized = str(status or "").strip().lower()
        if normalized in {"completed", "complete"}:
            return "completed"
        if normalized in {"failed", "error"}:
            return "failed"
        if normalized in {"skipped"}:
            return "completed"
        return "running"


class DesignPreparationStore:
    PIPELINE_SCHEMA_VERSION = int(getattr(settings, "AI_TSD_PREPARATION_SCHEMA_VERSION", 1))

    def __init__(
        self,
        *,
        ingestion_service: Optional[IngestionService] = None,
        retrieval_service: Optional[RetrievalService] = None,
    ) -> None:
        self.ingestion_service = ingestion_service or IngestionService()
        self.retrieval_service = retrieval_service or RetrievalService()
        self.snapshot_builder = RetrievalSnapshotBuilder(workflow_repository=_NoopWorkflowRepository())

    def ensure_preparation(
        self,
        db: Session,
        *,
        design: Design,
        document_sha256: str,
        force_rebuild: bool = False,
    ) -> DesignPreparation:
        active = self.get_active_preparation(db, design.id)
        if (
            active
            and not force_rebuild
            and active.document_sha256 == document_sha256
            and active.pipeline_schema_version == self.PIPELINE_SCHEMA_VERSION
            and active.embedding_model_name == settings.AI_MODEL_EMBEDDING
            and active.embedding_model_dim == self.embedding_dimensions
            and active.status in {
                DesignPreparation.STATUS_QUEUED,
                DesignPreparation.STATUS_RUNNING,
                DesignPreparation.STATUS_READY,
            }
        ):
            self._sync_design_with_preparation(design, active)
            db.flush()
            return active

        for prep in list(getattr(design, "preparations", []) or []):
            if prep.is_active:
                prep.is_active = False
                if prep.status != DesignPreparation.STATUS_FAILED:
                    prep.status = DesignPreparation.STATUS_STALE

        preparation = DesignPreparation(
            design=design,
            document_sha256=document_sha256,
            status=DesignPreparation.STATUS_QUEUED,
            is_active=True,
            pipeline_schema_version=self.PIPELINE_SCHEMA_VERSION,
            embedding_model_name=settings.AI_MODEL_EMBEDDING,
            embedding_model_dim=self.embedding_dimensions,
            stats_json={},
            progress_json=default_preparation_progress(),
        )
        db.add(preparation)
        db.flush()
        self._sync_design_with_preparation(design, preparation)
        db.flush()
        return preparation

    @property
    def embedding_dimensions(self) -> int:
        return 1024

    def get_active_preparation(self, db: Session, design_id: int) -> Optional[DesignPreparation]:
        return db.execute(
            select(DesignPreparation)
            .where(
                DesignPreparation.design_id == design_id,
                DesignPreparation.is_active == True,
            )
            .order_by(DesignPreparation.created_at.desc())
        ).scalars().first()

    def mark_running(self, design: Design, preparation: DesignPreparation) -> None:
        preparation.status = DesignPreparation.STATUS_RUNNING
        preparation.error_message = None
        preparation.started_at = _utc_now()
        design.preparation_status = DesignPreparation.STATUS_RUNNING
        design.preparation_error = None
        progress = default_preparation_progress()
        preparation.progress_json = progress
        design.preparation_progress_json = progress

    def mark_failed(self, design: Design, preparation: DesignPreparation, message: str) -> None:
        preparation.status = DesignPreparation.STATUS_FAILED
        preparation.error_message = message
        preparation.completed_at = _utc_now()
        design.preparation_status = DesignPreparation.STATUS_FAILED
        design.preparation_error = message
        progress = dict(preparation.progress_json or default_preparation_progress())
        progress["phase"] = "failed"
        progress["status_label"] = "Preparation failed"
        progress["current_step"] = message
        progress["updated_at"] = _utc_now().isoformat()
        preparation.progress_json = progress
        design.preparation_progress_json = progress

    def mark_ready(
        self,
        design: Design,
        preparation: DesignPreparation,
        *,
        snapshot: Optional[Dict[str, Any]],
        artifact_keys: Dict[str, Optional[str]],
        stats_json: Dict[str, Any],
    ) -> None:
        preparation.status = DesignPreparation.STATUS_READY
        preparation.error_message = None
        preparation.completed_at = _utc_now()
        preparation.prepared_at = preparation.completed_at
        preparation.tsd_document_object_key = artifact_keys.get("tsd_document")
        preparation.raptor_artifact_object_key = artifact_keys.get("raptor")
        preparation.retrieval_snapshot_object_key = artifact_keys.get("snapshot")
        preparation.stats_json = stats_json

        design.preparation_status = DesignPreparation.STATUS_READY
        design.preparation_error = None
        design.prepared_at = preparation.prepared_at
        design.prepared_document_sha256 = preparation.document_sha256
        design.preparation_snapshot_json = snapshot
        design.preparation_progress_json = preparation.progress_json
        design.active_preparation = preparation

    def load_prepared_assets(self, db: Session, design: Design) -> Tuple[DesignPreparation, TSDDocument, RetrievalIndexes]:
        preparation = self.get_active_preparation(db, design.id)
        if not preparation or preparation.status != DesignPreparation.STATUS_READY:
            raise PreparationNotReadyError(f"Design {design.id} preparation is not ready.")
        if design.prepared_document_sha256 and preparation.document_sha256 != design.prepared_document_sha256:
            raise PreparationArtifactError("Prepared artifact hash does not match design hash.")

        tsd_document_payload = self._load_artifact(preparation.tsd_document_object_key)
        raptor_payload = self._load_artifact(preparation.raptor_artifact_object_key)
        tsd_document = deserialize_tsd_document(tsd_document_payload)
        raptor_tree = deserialize_raptor_tree(raptor_payload)
        return preparation, tsd_document, RetrievalIndexes(raptor_tree=raptor_tree)

    def run_preparation(self, db: Session, *, design: Design, preparation: DesignPreparation) -> Dict[str, Any]:
        self.mark_running(design, preparation)
        db.commit()
        db.refresh(design)
        db.refresh(preparation)
        progress_tracker = PreparationProgressTracker(
            write_progress=lambda payload: self.persist_progress(
                design_id=design.id,
                preparation_id=preparation.id,
                payload=payload,
            )
        )
        progress_tracker.start()
        output: Optional[IngestionOutput] = None
        session_id = build_tsd_ingestion_session_id(preparation.id)
        try:
            with job_session_context(session_id=session_id, job_type="tsd_ingestion", job_id=preparation.id):
                output = self.ingestion_service.ingest_design(design)
                if output is None:
                    raise PreparationArtifactError("Failed to ingest the uploaded TSD document.")
                progress_tracker.mark_ingestion_complete()
                if not output.is_valid_tsd:
                    progress_tracker.mark_failed(
                        output.screening_message
                        or "Document failed TSD screening and is not eligible for analysis."
                    )
                    raise PreparationArtifactError(
                        output.screening_message
                        or "Document failed TSD screening and is not eligible for analysis."
                    )
                progress_tracker.mark_screening_complete()

                indexes = self.retrieval_service.build_indexes(
                    output.tsd_document,
                    progress_callbacks={
                        "raptor": progress_tracker.update_raptor,
                    },
                )
                snapshot = self.snapshot_builder.build_snapshot(indexes)
                progress_tracker.mark_persisting()
                artifact_keys = self._persist_artifacts(preparation, output, indexes, snapshot)
                self._replace_vector_rows(db, preparation, indexes)

                stats_json = {
                    "document_name": output.tsd_document.document_name,
                    "total_pages": output.tsd_document.total_pages,
                    "total_text_blocks": output.tsd_document.total_text_blocks,
                    "total_diagrams": output.tsd_document.total_diagrams,
                    "raptor_total_nodes": int(getattr(indexes.raptor_tree, "total_nodes", 0) or 0),
                    "llm_usage": usage_tracker.snapshot(session_id),
                }
                self.mark_ready(
                    design,
                    preparation,
                    snapshot=snapshot,
                    artifact_keys=artifact_keys,
                    stats_json=stats_json,
                )
                progress_tracker.mark_persisting_complete()
                preparation.progress_json = progress_tracker.snapshot()
                design.preparation_progress_json = preparation.progress_json
                db.commit()
                return {"snapshot": snapshot, "stats_json": stats_json}
        except Exception as exc:
            progress_tracker.mark_failed(str(exc))
            raise
        finally:
            usage_tracker.clear(session_id)
            if output is not None and getattr(output, "tsd_document", None) is not None:
                output.tsd_document.cleanup_temporary_artifacts()

    def _persist_artifacts(
        self,
        preparation: DesignPreparation,
        output: IngestionOutput,
        indexes: RetrievalIndexes,
        snapshot: Optional[Dict[str, Any]],
    ) -> Dict[str, Optional[str]]:
        base_prefix = (
            f"tsd-preparations/design-{preparation.design_id}/"
            f"sha256-{preparation.document_sha256}/v{preparation.pipeline_schema_version}"
        )
        artifact_keys = {
            "tsd_document": f"{base_prefix}/tsd_document.json.zst",
            "raptor": f"{base_prefix}/raptor_tree.json.zst",
            "snapshot": f"{base_prefix}/retrieval_snapshot.json.zst",
        }
        _eager_load_diagram_images(output.tsd_document)
        storage_service.upload_file(
            compress_json_bytes(serialize_tsd_document(output.tsd_document)),
            artifact_keys["tsd_document"],
            "application/zstd",
        )
        storage_service.upload_file(
            compress_json_bytes(serialize_raptor_tree(indexes.raptor_tree)),
            artifact_keys["raptor"],
            "application/zstd",
        )
        if snapshot is not None:
            storage_service.upload_file(
                compress_json_bytes(snapshot),
                artifact_keys["snapshot"],
                "application/zstd",
            )
        return artifact_keys

    def _replace_vector_rows(self, db: Session, preparation: DesignPreparation, indexes: RetrievalIndexes) -> None:
        db.execute(delete(DesignPreparationRaptorNode).where(DesignPreparationRaptorNode.preparation_id == preparation.id))

        for node in (indexes.raptor_tree.get_all_nodes() if indexes.raptor_tree else []):
            db.add(
                DesignPreparationRaptorNode(
                    preparation_id=preparation.id,
                    node_id=node.node_id,
                    parent_node_id=self._find_raptor_parent_id(indexes.raptor_tree, node.node_id),
                    level=node.level,
                    section_heading=node.section_heading,
                    text=node.text,
                    source_block_ids=list(node.source_block_ids or []),
                    page_numbers=list(node.page_numbers or []),
                    token_estimate=node.token_estimate,
                    content_hash=self._content_hash(node.text),
                    embedding=(list(node.embedding) if node.embedding else None),
                    has_embedding=bool(node.has_embedding),
                )
            )
        db.flush()

    def _load_artifact(self, object_key: Optional[str]) -> Dict[str, Any]:
        if not object_key:
            raise PreparationArtifactError("Preparation artifact key is missing.")
        return decompress_json_bytes(storage_service.download_bytes(object_key))

    def _sync_design_with_preparation(self, design: Design, preparation: DesignPreparation) -> None:
        design.preparation_status = preparation.status
        design.preparation_error = preparation.error_message
        design.active_preparation = preparation
        design.prepared_at = preparation.prepared_at
        design.preparation_progress_json = preparation.progress_json
        if preparation.is_active:
            design.prepared_document_sha256 = preparation.document_sha256

    def persist_progress(self, *, design_id: int, preparation_id: int, payload: Dict[str, Any]) -> None:
        from sdr.core.database import SessionLocal

        with SessionLocal() as progress_db:
            preparation = progress_db.get(DesignPreparation, preparation_id)
            design = progress_db.get(Design, design_id)
            if not preparation or not design:
                return
            preparation.progress_json = payload
            design.preparation_progress_json = payload
            if payload.get("phase") in {"running", "ingesting", "screening", "building_indexes", "persisting"}:
                design.preparation_status = DesignPreparation.STATUS_RUNNING
            progress_db.commit()

    def _find_raptor_parent_id(self, tree: Optional[RAPTORTree], node_id: str) -> Optional[str]:
        if not tree:
            return None
        for candidate in tree.get_all_nodes():
            for child in candidate.children:
                if child.node_id == node_id:
                    return candidate.node_id
        return None

    def _content_hash(self, value: str) -> str:
        return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


class _NoopWorkflowRepository:
    def save_retrieval_snapshot(self, *_args, **_kwargs) -> None:
        return None


def _eager_load_diagram_images(document: TSDDocument) -> None:
    """
    DiagramBlock.image_b64 is populated lazily from `source_pdf_path` (see
    TSDIngestor), which lives in a temp directory that gets deleted by
    cleanup_temporary_artifacts() right after preparation persists artifacts.
    serialize_tsd_document only carries `image_b64` (not `source_pdf_path`/
    `image_xref`), so any diagram not resolved before serialization is
    permanently unrecoverable once the cached artifact is the only copy.
    """
    for page in document.pages:
        for diagram in page.diagrams:
            try:
                diagram.ensure_image_loaded(document.min_diagram_bytes)
            except Exception:
                logger.warning(
                    "_eager_load_diagram_images: failed to resolve diagram_id=%s before persisting artifacts",
                    diagram.diagram_id,
                    exc_info=True,
                )


def serialize_tsd_document(document: TSDDocument) -> Dict[str, Any]:
    return {
        "file_path": document.file_path,
        "document_name": document.document_name,
        "total_pages": document.total_pages,
        "total_text_blocks": document.total_text_blocks,
        "total_diagrams": document.total_diagrams,
        "metadata": dict(document.metadata or {}),
        "min_diagram_bytes": document.min_diagram_bytes,
        "pages": [
            {
                "page_number": page.page_number,
                "section_heading": page.section_heading,
                "raw_text": page.raw_text,
                "markdown_text": page.markdown_text,
                "width_pt": page.width_pt,
                "height_pt": page.height_pt,
                "metadata": dict(page.metadata or {}),
                "text_blocks": [block.to_dict() | {"section_heading": block.section_heading} for block in page.text_blocks],
                "diagrams": [
                    {
                        "diagram_id": diagram.diagram_id,
                        "page_number": diagram.page_number,
                        "bbox": {
                            "x0": diagram.bbox_x0,
                            "y0": diagram.bbox_y0,
                            "x1": diagram.bbox_x1,
                            "y1": diagram.bbox_y1,
                        },
                        "image_b64": diagram.image_b64,
                        "image_format": diagram.image_format,
                        "caption": diagram.caption,
                        "surrounding_text": diagram.surrounding_text,
                        "width_pt": diagram.width_pt,
                        "height_pt": diagram.height_pt,
                    }
                    for diagram in page.diagrams
                ],
            }
            for page in document.pages
        ],
    }


def deserialize_tsd_document(payload: Dict[str, Any]) -> TSDDocument:
    pages: List[TSDPage] = []
    for page_payload in payload.get("pages", []) or []:
        text_blocks = [
            TextBlock(
                block_id=str(item.get("block_id") or ""),
                text=str(item.get("text") or ""),
                page_number=int(item.get("page_number") or page_payload.get("page_number") or 0),
                bbox_x0=float((item.get("bbox") or {}).get("x0") or 0.0),
                bbox_y0=float((item.get("bbox") or {}).get("y0") or 0.0),
                bbox_x1=float((item.get("bbox") or {}).get("x1") or 0.0),
                bbox_y1=float((item.get("bbox") or {}).get("y1") or 0.0),
                font_size=float(item.get("font_size") or 0.0),
                is_bold=bool(item.get("is_bold")),
                is_heading=bool(item.get("is_heading")),
                section_heading=item.get("section_heading"),
            )
            for item in (page_payload.get("text_blocks") or [])
        ]
        diagrams = [
            DiagramBlock(
                diagram_id=str(item.get("diagram_id") or ""),
                page_number=int(item.get("page_number") or page_payload.get("page_number") or 0),
                bbox_x0=float((item.get("bbox") or {}).get("x0") or 0.0),
                bbox_y0=float((item.get("bbox") or {}).get("y0") or 0.0),
                bbox_x1=float((item.get("bbox") or {}).get("x1") or 0.0),
                bbox_y1=float((item.get("bbox") or {}).get("y1") or 0.0),
                image_b64=str(item.get("image_b64") or ""),
                image_format=str(item.get("image_format") or "png"),
                caption=item.get("caption"),
                surrounding_text=item.get("surrounding_text"),
                width_pt=float(item.get("width_pt") or 0.0),
                height_pt=float(item.get("height_pt") or 0.0),
            )
            for item in (page_payload.get("diagrams") or [])
        ]
        pages.append(
            TSDPage(
                page_number=int(page_payload.get("page_number") or 0),
                text_blocks=text_blocks,
                diagrams=diagrams,
                section_heading=page_payload.get("section_heading"),
                raw_text=str(page_payload.get("raw_text") or ""),
                markdown_text=str(page_payload.get("markdown_text") or ""),
                width_pt=float(page_payload.get("width_pt") or 0.0),
                height_pt=float(page_payload.get("height_pt") or 0.0),
                metadata=dict(page_payload.get("metadata") or {}),
            )
        )
    return TSDDocument(
        file_path=str(payload.get("file_path") or ""),
        document_name=str(payload.get("document_name") or ""),
        pages=pages,
        total_pages=int(payload.get("total_pages") or len(pages)),
        total_text_blocks=int(payload.get("total_text_blocks") or 0),
        total_diagrams=int(payload.get("total_diagrams") or 0),
        metadata=dict(payload.get("metadata") or {}),
        temp_directories=[],
        min_diagram_bytes=int(payload.get("min_diagram_bytes") or 512),
    )


def serialize_raptor_tree(tree: Optional[RAPTORTree]) -> Dict[str, Any]:
    if not tree:
        return {"document_name": "", "levels": [], "root_node_id": None, "total_nodes": 0, "max_level": 0, "build_stats": {}}
    all_nodes = tree.get_all_nodes()
    return {
        "document_name": tree.document_name,
        "levels": [
            [
                {
                    "node_id": node.node_id,
                    "level": node.level,
                    "text": node.text,
                    "embedding": list(node.embedding or []),
                    "source_block_ids": list(node.source_block_ids or []),
                    "page_numbers": list(node.page_numbers or []),
                    "section_heading": node.section_heading,
                    "has_embedding": bool(node.has_embedding),
                    "child_ids": [child.node_id for child in node.children],
                }
                for node in level_nodes
            ]
            for level_nodes in tree.levels
        ],
        "root_node_id": getattr(tree.root_node, "node_id", None),
        "total_nodes": tree.total_nodes,
        "max_level": tree.max_level,
        "build_stats": dict(tree.build_stats or {}),
    }


def deserialize_raptor_tree(payload: Dict[str, Any]) -> RAPTORTree:
    levels: List[List[RAPTORNode]] = []
    node_map: Dict[str, RAPTORNode] = {}
    child_lookup: Dict[str, List[str]] = {}
    for level_payload in payload.get("levels", []) or []:
        level_nodes: List[RAPTORNode] = []
        for item in level_payload:
            node = RAPTORNode(
                node_id=str(item.get("node_id") or ""),
                level=int(item.get("level") or 0),
                text=str(item.get("text") or ""),
                embedding=list(item.get("embedding") or []),
                source_block_ids=list(item.get("source_block_ids") or []),
                children=[],
                page_numbers=list(item.get("page_numbers") or []),
                section_heading=item.get("section_heading"),
                has_embedding=bool(item.get("has_embedding")),
            )
            node_map[node.node_id] = node
            child_lookup[node.node_id] = list(item.get("child_ids") or [])
            level_nodes.append(node)
        levels.append(level_nodes)
    for node_id, child_ids in child_lookup.items():
        node_map[node_id].children = [node_map[child_id] for child_id in child_ids if child_id in node_map]
    root_id = payload.get("root_node_id")
    return RAPTORTree(
        document_name=str(payload.get("document_name") or ""),
        levels=levels,
        root_node=node_map.get(root_id) if root_id else None,
        total_nodes=int(payload.get("total_nodes") or 0),
        max_level=int(payload.get("max_level") or 0),
        build_stats=dict(payload.get("build_stats") or {}),
    )
