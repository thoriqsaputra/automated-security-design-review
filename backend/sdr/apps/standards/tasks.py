import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from celery import shared_task
from sqlalchemy import delete, select, update
from sqlalchemy.orm import joinedload
from sqlalchemy.orm.attributes import flag_modified

from sdr.core.database import SessionLocal
from sdr.apps.standards.utils import (
    build_diagram_requirement_analysis_text,
    stable_key,
    normalize_requirement_text,
)
from sdr.apps.ai.client import usage_tracker
from sdr.apps.ai.client.session import (
    build_standard_ingestion_session_id,
    capture_current_context,
    job_session_context,
)
from sdr.apps.ai.utils.embedding import (
    generate_and_store_diagram_requirement_embeddings,
)
from .models import (
    CategoryParameterChild,
    CategoryParameterParent,
    CategoryDiagramRequirement,
    StandardIngestionJob,
    StandardSourceDocument,
)
from .validators import validate_standard_name

import re

logger = logging.getLogger(__name__)


_OWASP_TOP_LEVEL_RE = re.compile(
    r"^(V\d+)(?:\s+\S|$)",
    re.IGNORECASE,
)
_DOCUMENT_TITLE_SECTION_RE = re.compile(
    r"^\s*application\s+security\s+verification\s+standard\s*$",
    re.IGNORECASE,
)


def _flatten_owasp_sections(
    requirements_by_section: Dict[str, Any],
    all_keys: List[str],
) -> Dict[str, Any]:
    """
    Safety-net post-processor:
    1. Folds sub-section requirements (e.g. "V1.1 Secure SDLC") into their
       matching top-level parent (e.g. "V1 Architecture").
    2. Canonicalizes variations of the same top-level parent (e.g.
       "V9 Communications" and "V9 Communication" both map to whichever
       variation was encountered first) to prevent duplicate parents.
    """
    # Build an index: top-level prefix (e.g. "V1") -> canonical parent key
    top_level_map: Dict[str, str] = {}
    for key in all_keys:
        m = _OWASP_TOP_LEVEL_RE.match(key.strip())
        if m:
            # Only register if the key itself doesn't have a dot-sub-number
            is_sub = re.match(r"^V\d+\.\d+", key.strip(), re.IGNORECASE)
            if not is_sub:
                prefix = m.group(1).upper()
                if prefix not in top_level_map:
                    top_level_map[prefix] = key

    if not top_level_map:
        # No OWASP-style structure detected — return as-is
        return requirements_by_section

    for key in all_keys:
        is_sub = re.match(r"^V(\d+)\.\d+", key.strip(), re.IGNORECASE)
        if is_sub:
            top_prefix = f"V{is_sub.group(1)}".upper()
            if top_prefix not in top_level_map:
                top_level_map[top_prefix] = top_prefix  # e.g. "V7"

    merged: Dict[str, Any] = {}
    for key, reqs in requirements_by_section.items():
        if _DOCUMENT_TITLE_SECTION_RE.match(key.strip()):
            logger.info(
                "_flatten_owasp_sections: dropping document-title section '%s' (%d reqs)",
                key, len(reqs or []),
            )
            continue
        is_sub = re.match(r"^V(\d+)\.\d+", key.strip(), re.IGNORECASE)
        if is_sub:
            top_prefix = f"V{is_sub.group(1)}".upper()
            parent_key = top_level_map.get(top_prefix)
            if parent_key:
                sub_label = key.strip().split()[0]
                tagged = []
                for req in (reqs or []):
                    if isinstance(req, dict):
                        orig = req.get("requirement", "")
                        if not orig.startswith(sub_label):
                            req = dict(req, requirement=f"{sub_label} - {orig}")
                    tagged.append(req)
                merged.setdefault(parent_key, []).extend(tagged)
                logger.info(
                    "_flatten_owasp_sections: merged sub-section '%s' into parent '%s' (%d reqs)",
                    key, parent_key, len(tagged),
                )
                continue
        else:
            m = _OWASP_TOP_LEVEL_RE.match(key.strip())
            if m:
                prefix = m.group(1).upper()
                parent_key = top_level_map.get(prefix)
                if parent_key and parent_key != key:
                    merged.setdefault(parent_key, []).extend(reqs or [])
                    logger.info(
                        "_flatten_owasp_sections: canonicalized parent '%s' -> '%s' (%d reqs)",
                        key, parent_key, len(reqs or []),
                    )
                    continue

        merged.setdefault(key, []).extend(reqs or [])

    return merged


def _atoi(text: str):
    return int(text) if text.isdigit() else text.lower()


def _natural_keys(text: str):
    """
    Sort key function for natural sorting (e.g. V9 comes before V10).
    """
    return [_atoi(c) for c in re.split(r'(\d+)', text)]


_VALID_REQUIREMENT_CATEGORIES = {"design", "code", "infrastructure", "process"}


def _strip_null_bytes(value: str) -> str:
    if "\x00" not in value:
        return value
    logger.warning("tasks._strip_null_bytes: removed null byte(s) from extracted text")
    return value.replace("\x00", "")


def _to_naive_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


def _coerce_requirement_text(requirement_item: Any) -> str:
    """
    Backward-compatible requirement text extraction.
    """
    if isinstance(requirement_item, str):
        return requirement_item.strip()
    if isinstance(requirement_item, dict):
        return str(requirement_item.get("requirement", "")).strip()
    return str(requirement_item or "").strip()


def _build_initial_summary(
    *,
    job: StandardIngestionJob,
    requested_ranges: Dict[str, Any],
    mode: str,
    celery_task_id: Optional[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "inserted": 0,
        "sections": 0,
        "errors": 0,
        "diagram_requirement_embeddings_created": 0,
        "diagram_requirement_embeddings_failed": 0,
        "diagram_requirements": 0,
        "diagram_extraction_status": "pending",
        "diagram_extraction_error": None,
        "mode": mode,
        "resolved_categories": {job.category.code if job.category else "unknown": 0},
        "version_no": getattr(job, "version_no", 1),
        "start_page": requested_ranges["start_page"],
        "end_page": requested_ranges["end_page"],
        "page_detection": {
            "source": "pending",
            "matched_anchors": {},
            "detected": {},
            "requested_overrides": requested_ranges,
            "field_sources": {},
        },
    }
    if celery_task_id:
        summary["celery_task_id"] = celery_task_id
    return summary


def _persist_summary(db, job: StandardIngestionJob, summary: Dict[str, Any]) -> None:
    job.summary_json = summary
    flag_modified(job, "summary_json")
    db.commit()


def _reset_existing_generated_rows(db, job: StandardIngestionJob) -> None:
    stale_parent_ids = db.execute(
        select(CategoryParameterParent.id)
        .where(CategoryParameterParent.ingestion_job_id == job.id)
    ).scalars().all()
    if stale_parent_ids:
        logger.warning(
            "run_standard_ingestion_job_sync: clearing stale generated rows for job_id=%s parent_count=%d",
            job.id,
            len(stale_parent_ids),
        )
    db.execute(
        delete(CategoryDiagramRequirement)
        .where(CategoryDiagramRequirement.ingestion_job_id == job.id)
    )
    if stale_parent_ids:
        db.execute(
            delete(CategoryParameterChild)
            .where(CategoryParameterChild.parent_id.in_(stale_parent_ids))
        )
    db.execute(
        delete(CategoryParameterParent)
        .where(CategoryParameterParent.ingestion_job_id == job.id)
    )
    db.commit()


def _mark_source_doc_failed(
    db,
    source_doc: StandardSourceDocument,
    *,
    summary: Dict[str, Any],
    exc: Exception,
    screening_error: bool = False,
) -> None:
    summary["errors"] += 1
    if screening_error:
        summary["screening_error"] = str(exc)
        logger.error("Standard screening rejected source_doc=%s: %s", source_doc.id, exc)
    else:
        logger.exception("Exception processing source_doc=%s", source_doc.id)
    db.rollback()
    try:
        source_doc.status = StandardSourceDocument.STATUS_FAILED
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to record error for source_doc=%s", source_doc.id)


def _get_or_create_parent(
    db,
    *,
    job: StandardIngestionJob,
    section_name: str,
) -> CategoryParameterParent:
    section_key = stable_key(section_name)
    parent_key = f"section:{section_key}:{job.id}"
    parent = db.execute(
        select(CategoryParameterParent)
        .where(
            CategoryParameterParent.category_id == job.category_id,
            CategoryParameterParent.stable_key == parent_key,
        )
    ).scalars().first()
    if parent:
        return parent
    parent = CategoryParameterParent(
        category_id=job.category_id,
        ingestion_job_id=job.id,
        stable_key=parent_key,
        title=section_name,
        title_normalized=normalize_requirement_text(section_name),
        description=f'Controls extracted for baseline v{getattr(job, "version_no", 1)}.',
    )
    db.add(parent)
    db.flush()
    return parent


def _persist_extracted_requirements(
    db,
    *,
    job: StandardIngestionJob,
    source_doc: StandardSourceDocument,
    requirements_by_section: Dict[str, Any],
    canonicalize_requirement_items,
    summary: Dict[str, Any],
) -> None:
    parents_by_section: Dict[str, CategoryParameterParent] = {}
    ordinals_by_section: Dict[str, int] = {}

    sorted_section_keys = sorted(
        (requirements_by_section or {}).keys(),
        key=lambda k: _natural_keys(k or ""),
    )
    for section_title in sorted_section_keys:
        requirements = canonicalize_requirement_items(requirements_by_section[section_title])
        section_name = _strip_null_bytes((section_title or "General").strip()) or "General"
        parent = parents_by_section.get(section_name)
        if parent is None:
            parent = _get_or_create_parent(db, job=job, section_name=section_name)
            parents_by_section[section_name] = parent
            ordinals_by_section[section_name] = 0

        sorted_requirements = sorted(
            requirements or [],
            key=lambda req: _natural_keys(_coerce_requirement_text(req)),
        )
        for req in sorted_requirements:
            raw_text = _strip_null_bytes(_coerce_requirement_text(req))
            normalized = normalize_requirement_text(raw_text)
            if not normalized:
                logger.warning(
                    "tasks.ingest: skipping requirement — normalized to empty | section=%s raw_text=%.120r",
                    section_name,
                    raw_text,
                )
                summary["skipped"] = summary.get("skipped", 0) + 1
                continue

            next_ordinal = ordinals_by_section.get(section_name, 0) + 1
            ordinals_by_section[section_name] = next_ordinal
            child_key = f"{stable_key(normalized)}-{str(source_doc.id)[:8]}-{next_ordinal:06d}"
            raw_category = str(req.get("requirement_category", "design")).lower().strip()
            child = CategoryParameterChild(
                parent_id=parent.id,
                stable_key=child_key,
                requirement_text=raw_text,
                requirement_text_normalized=normalized,
                ordinal=next_ordinal,
                requirement_category=(
                    raw_category if raw_category in _VALID_REQUIREMENT_CATEGORIES else "design"
                ),
            )
            db.add(child)
            db.flush()
            summary["inserted"] += 1

    summary["sections"] = len(parents_by_section)
    return None


def _extract_standard_requirements_phase(
    db,
    *,
    job: StandardIngestionJob,
    session_id: str,
    source_doc: StandardSourceDocument,
    requested_ranges: Dict[str, Any],
    summary: Dict[str, Any],
    canonicalize_requirement_items,
    detect_asvs_page_ranges,
    extract_requirements_from_document,
) -> None:
    with job_session_context(
        session_id=session_id,
        job_type="standard_ingestion",
        job_id=job.id,
    ):
        validate_standard_name(source_doc.name.rsplit(".", 1)[0])

        source_doc.status = StandardSourceDocument.STATUS_UPLOADED
        db.commit()

        def _update_progress(label: str, percentage: int):
            db.refresh(job)
            if job.status == StandardIngestionJob.STATUS_CANCELLED:
                raise ValueError("Job was cancelled by the user.")
            summary["detailed_progress"] = {"label": label, "percentage": percentage}
            _persist_summary(db, job, summary)

        if _should_auto_detect_asvs_page_ranges(source_doc):
            detected_ranges = detect_asvs_page_ranges(source_doc)
            resolved_ranges = _resolve_detected_page_ranges(detected_ranges, requested_ranges)
        else:
            resolved_ranges = _resolve_detected_page_ranges(
                {
                    "source": "skipped_non_asvs",
                    "matched_anchors": {},
                    "start_page": None,
                    "end_page": None,
                },
                requested_ranges,
            )
        summary.update(resolved_ranges["effective"])
        summary["page_detection"] = resolved_ranges["page_detection"]
        _persist_summary(db, job, summary)

        extraction_phase_started_at = time.monotonic()
        requirements_by_section = extract_requirements_from_document(
            source_doc,
            start_page=summary.get("start_page"),
            end_page=summary.get("end_page"),
            progress_callback=_update_progress,
        )
        logger.info(
            "run_standard_ingestion_job_sync: [TIMING] requirement extraction phase took %.2fs for job=%s",
            time.monotonic() - extraction_phase_started_at,
            job.id,
        )
        if not requirements_by_section:
            raise ValueError("Extraction returned empty result.")

        all_section_keys = list(requirements_by_section.keys())
        requirements_by_section = _flatten_owasp_sections(requirements_by_section, all_section_keys)
        logger.info(
            "run_standard_ingestion_job_sync: after OWASP flattening: %d section(s)",
            len(requirements_by_section),
        )

        source_doc.status = StandardSourceDocument.STATUS_PARSED
        db.commit()

        _persist_extracted_requirements(
            db,
            job=job,
            source_doc=source_doc,
            requirements_by_section=requirements_by_section,
            canonicalize_requirement_items=canonicalize_requirement_items,
            summary=summary,
        )
        source_doc.status = StandardSourceDocument.STATUS_PROCESSED
        db.commit()
        return None


def _run_diagram_phase(
    db,
    *,
    job: StandardIngestionJob,
    session_id: str,
    summary: Dict[str, Any],
    extract_diagram_requirements,
) -> None:
    with job_session_context(
        session_id=session_id,
        job_type="standard_ingestion",
        job_id=job.id,
    ):
        logger.info("Pre-fetching parameters for diagram requirement extraction, job %s", job.id)
        summary["detailed_progress"] = {
            "label": "Extracting diagram requirements",
            "percentage": 95,
        }
        _persist_summary(db, job, summary)

        diagram_req_params = list(
            db.execute(
                select(CategoryParameterChild)
                .options(joinedload(CategoryParameterChild.parent))
                .join(CategoryParameterParent, CategoryParameterChild.parent_id == CategoryParameterParent.id)
                .where(CategoryParameterParent.category_id == job.category_id)
                .where(CategoryParameterParent.ingestion_job_id == job.id)
                .order_by(CategoryParameterChild.id)
            ).scalars().all()
        )

        diagram_reqs: list = []
        diagram_exc: Optional[Exception] = None
        diagram_phase_started_at = time.monotonic()
        if diagram_req_params:
            try:
                diagram_reqs = capture_current_context(extract_diagram_requirements)(
                    parameters=diagram_req_params,
                    category_id=job.category_id,
                    ingestion_job_id=job.id,
                )
            except Exception as exc:
                diagram_exc = exc
        logger.info(
            "run_standard_ingestion_job_sync: [TIMING] diagram-extraction phase took %.2fs for job=%s",
            time.monotonic() - diagram_phase_started_at,
            job.id,
        )

        logger.info("Persisting diagram requirements for job %s", job.id)
        try:
            if diagram_exc:
                raise diagram_exc

            if not diagram_req_params:
                summary["diagram_requirements"] = 0
                summary["diagram_extraction_status"] = "completed"
                summary["diagram_extraction_error"] = None
                return

            source_params_by_key = {param.stable_key: param for param in diagram_req_params}
            db.execute(
                delete(CategoryDiagramRequirement)
                .where(CategoryDiagramRequirement.ingestion_job_id == job.id)
            )

            persisted_diagram_requirements = []
            for dreq in diagram_reqs:
                persisted = CategoryDiagramRequirement(
                    category_id=dreq["category_id"],
                    ingestion_job_id=dreq["ingestion_job_id"],
                    stable_key=dreq["stable_key"],
                    source_requirement_key=dreq["source_requirement_key"],
                    requirement_text=dreq["requirement_text"],
                    verification_hint=dreq["verification_hint"],
                    parent_section=dreq["parent_section"],
                    diagram_type=dreq.get("diagram_type", ""),
                    ordinal=dreq.get("ordinal", 0),
                )
                db.add(persisted)
                persisted_diagram_requirements.append(persisted)

            db.flush()

            diagram_items_to_embed = []
            for dreq in persisted_diagram_requirements:
                source_parameter = None
                if dreq.source_requirement_key != "composite":
                    source_parameter = source_params_by_key.get(dreq.source_requirement_key)
                retrieval_text = build_diagram_requirement_analysis_text(
                    dreq,
                    source_parameter=source_parameter,
                )
                normalized_text = normalize_requirement_text(retrieval_text)
                if not normalized_text:
                    continue
                diagram_items_to_embed.append(
                    {
                        "diagram_requirement_id": dreq.id,
                        "text": normalized_text,
                        "content_hash": dreq.stable_key,
                    }
                )

            db.commit()
            summary["diagram_requirements"] = len(diagram_reqs)
            if diagram_items_to_embed:
                try:
                    generate_and_store_diagram_requirement_embeddings(
                        diagram_items_to_embed,
                        job.id,
                        summary,
                    )
                except Exception as exc:
                    logger.error("Error generating diagram requirement embeddings: %s", exc)
                    summary["diagram_requirement_embeddings_failed"] = len(diagram_items_to_embed)
            summary["diagram_extraction_status"] = "completed"
            summary["diagram_extraction_error"] = None
            logger.info(
                "run_standard_ingestion_job_sync: extracted %d diagram requirements",
                len(diagram_reqs),
            )
        except Exception as exc:
            db.rollback()
            logger.exception("Error extracting diagram requirements: %s", exc)
            summary["diagram_requirements"] = 0
            summary["diagram_extraction_status"] = "failed"
            summary["diagram_extraction_error"] = str(exc)
            summary["errors"] += 1


def _finalize_ingestion_job(
    db,
    *,
    job: StandardIngestionJob,
    session_id: str,
    summary: Dict[str, Any],
) -> None:
    job_status_new = (
        StandardIngestionJob.STATUS_COMPLETED
        if summary["errors"] == 0 and summary.get("diagram_extraction_status") == "completed"
        else StandardIngestionJob.STATUS_FAILED
    )
    job.status = job_status_new
    summary["detailed_progress"] = {
        "label": "Completed" if job_status_new == StandardIngestionJob.STATUS_COMPLETED else "Failed",
        "percentage": 100,
    }
    summary["llm_usage"] = usage_tracker.snapshot(session_id)
    job.completed_at = datetime.now(timezone.utc)
    if job.started_at:
        summary["job_duration_seconds"] = (
            _to_naive_utc(job.completed_at) - _to_naive_utc(job.started_at)
        ).total_seconds()
    job.summary_json = summary

    if job_status_new == StandardIngestionJob.STATUS_FAILED:
        job.error_message = (
            summary.get("screening_error")
            or summary.get("diagram_extraction_error")
            or "Document failed during ingestion."
        )
        job.is_active = False
    else:
        db.execute(
            update(StandardIngestionJob)
            .where(StandardIngestionJob.category_id == job.category_id)
            .where(StandardIngestionJob.is_active == True)
            .where(StandardIngestionJob.id != job.id)
            .values(is_active=False)
        )
        job.is_active = True
        job.activated_at = datetime.now(timezone.utc)

    try:
        _persist_summary(db, job, summary)
    except Exception as exc:
        logger.warning("Could not persist embedding counters for job=%s: %s", job.id, exc)
    finally:
        usage_tracker.clear(session_id)


def _cleanup_source_documents(source_docs: List[StandardSourceDocument]) -> None:
    from sdr.apps.workspace.services.storage import storage_service

    for source_doc in source_docs:
        if source_doc.document:
            try:
                storage_service.delete_file(source_doc.document)
            except Exception as exc:
                logger.warning(
                    "Failed to delete temporal document %s: %s",
                    source_doc.document,
                    exc,
                )

def _resolve_detected_page_ranges(
    detected_ranges: Optional[Dict[str, Any]],
    requested_ranges: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    detected = dict(detected_ranges or {})
    requested = dict(requested_ranges or {})
    effective: Dict[str, Optional[int]] = {}
    field_sources: Dict[str, str] = {}
    for field_name in (
        "start_page",
        "end_page",
    ):
        override_value = requested.get(field_name)
        if override_value is not None:
            effective[field_name] = override_value
            field_sources[field_name] = "manual_override"
        else:
            effective[field_name] = detected.get(field_name)
            field_sources[field_name] = "auto_detected"

    return {
        "effective": effective,
        "page_detection": {
            "source": detected.get("source", "heuristic"),
            "matched_anchors": detected.get("matched_anchors", {}),
            "detected": {
                key: detected.get(key)
                for key in (
                    "start_page",
                    "end_page",
                )
            },
            "requested_overrides": requested,
            "field_sources": field_sources,
        },
    }


def _should_auto_detect_asvs_page_ranges(source_doc: StandardSourceDocument) -> bool:
    candidate = " ".join(
        part for part in (
            getattr(source_doc, "name", None),
            getattr(source_doc, "document", None),
        )
        if part
    ).lower().replace("_", " ")
    return "asvs" in candidate or "application security verification standard" in candidate


def run_standard_ingestion_job_sync(job_id: str, celery_task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingests all StandardSourceDocuments attached to an ingestion job and
    persists extracted requirement rows plus derived diagram requirements.
    """
    from sdr.apps.ai.engine.extraction import (
        canonicalize_requirement_items,
        detect_asvs_page_ranges,
        extract_requirements_from_document,
        extract_diagram_requirements,
    )
    from sdr.apps.ai.engine.extraction.screening import StandardScreeningError
    
    with SessionLocal() as db:
        job = db.get(StandardIngestionJob, int(job_id))
        if not job:
            raise ValueError(f"Job not found for ID: {job_id}")
        session_id = build_standard_ingestion_session_id(job.id)

        mode = (job.summary_json or {}).get('mode', 'manual')
        if job.category_id is None:
            raise ValueError('Ingestion job has no category assigned.')

        old_summary = job.summary_json or {}
        requested_ranges = {
            "start_page": old_summary.get("start_page"),
            "end_page": old_summary.get("end_page"),
        }
        summary = _build_initial_summary(
            job=job,
            requested_ranges=requested_ranges,
            mode=mode,
            celery_task_id=celery_task_id,
        )
        
        logger.info(
            "run_standard_ingestion_job_sync: summary dict initialized. mode=%s, version_no=%s",
            summary['mode'],
            summary['version_no'],
        )

        logger.info("run_standard_ingestion_job_sync: [STEP 1-INIT] marking job as RUNNING.")
        job.status = StandardIngestionJob.STATUS_RUNNING
        job.started_at = datetime.now(timezone.utc)
        job.completed_at = None
        job.error_message = ''
        _persist_summary(db, job, summary)

        _reset_existing_generated_rows(db, job)

        source_docs = db.execute(
            select(StandardSourceDocument)
            .where(StandardSourceDocument.ingestion_job_id == job.id)
            .order_by(StandardSourceDocument.created_at)
        ).scalars().all()
        
        logger.info(
            "run_standard_ingestion_job_sync: [STEP 1] loaded source documents. count=%d session_id=%s",
            len(source_docs),
            session_id,
        )

        if not source_docs:
            logger.warning("run_standard_ingestion_job_sync: [STEP 1-WARNING] no source documents found. Skipping to finalization.")

        if source_docs:
            source_doc = source_docs[0]
            logger.info("run_standard_ingestion_job_sync: [STEP 1] processing single source_doc.")

            try:
                _extract_standard_requirements_phase(
                    db,
                    job=job,
                    session_id=session_id,
                    source_doc=source_doc,
                    requested_ranges=requested_ranges,
                    summary=summary,
                    canonicalize_requirement_items=canonicalize_requirement_items,
                    detect_asvs_page_ranges=detect_asvs_page_ranges,
                    extract_requirements_from_document=extract_requirements_from_document,
                )

            except StandardScreeningError as exc:
                _mark_source_doc_failed(
                    db,
                    source_doc,
                    summary=summary,
                    exc=exc,
                    screening_error=True,
                )

            except Exception as exc:
                _mark_source_doc_failed(
                    db,
                    source_doc,
                    summary=summary,
                    exc=exc,
                )

        _run_diagram_phase(
            db,
            job=job,
            session_id=session_id,
            summary=summary,
            extract_diagram_requirements=extract_diagram_requirements,
        )
        _finalize_ingestion_job(
            db,
            job=job,
            session_id=session_id,
            summary=summary,
        )
        _cleanup_source_documents(source_docs)

    return summary


# ---------------------------------------------------------------------------
# Celery task + dispatcher
# ---------------------------------------------------------------------------

@shared_task(
    bind=True,
    name="standards.run_standard_ingestion_job_task",
    max_retries=0,
)
def run_standard_ingestion_job_task(self, job_id: str):
    logger.info("run_standard_ingestion_job_task: [CELERY-ENTRY] job_id=%s", job_id)
    try:
        result = run_standard_ingestion_job_sync(
            job_id=str(job_id),
            celery_task_id=getattr(self.request, 'id', None),
        )
        return result
    except Exception as exc:
        logger.error("run_standard_ingestion_job_task: [CELERY-EXIT-FAILURE] Error: %s", exc)
        raise exc


def dispatch_standard_ingestion(job_id: str) -> Dict[str, Any]:
    logger.info("dispatch_standard_ingestion: dispatching job_id=%s", job_id)
    try:
        task_result = run_standard_ingestion_job_task.delay(str(job_id))
        return {
            'mode': 'async',
            'task_id': task_result.id,
        }
    except Exception as exc:
        logger.error("Failed to enqueue task for job_id=%s: %s", job_id, exc)
        raise
