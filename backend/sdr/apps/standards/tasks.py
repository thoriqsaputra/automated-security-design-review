import logging
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from datetime import datetime

from celery import shared_task
from sqlalchemy import delete, select, update

from sdr.core.database import SessionLocal
from sdr.core.config import settings
from sdr.apps.standards.utils import (
    build_parameter_analysis_text,
    stable_key,
    normalize_requirement_text,
)
# Note: assuming extraction and embedding services are refactored to use SQLAlchemy or don't rely on ORM models
from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.client.session import (
    build_standard_ingestion_session_id,
    job_session_context,
)

from sdr.apps.ai.rate_limiter import get_rate_limiter
from sdr.apps.ai.utils.embedding import generate_and_store_embeddings
from .models import (
    ASVSLevelDefinition,
    CategoryParameterChild,
    CategoryParameterParent,
    CategoryDiagramRequirement,
    StandardIngestionJob,
    StandardSourceDocument,
)
from .validators import validate_standard_name

import re

logger = logging.getLogger(__name__)


_OWASP_SUB_SECTION_RE = re.compile(
    r"^(V\d+(?:\s+\S.*?)?)\s*\.\s*(\d+)\b",
    re.IGNORECASE,
)
_OWASP_TOP_LEVEL_RE = re.compile(
    r"^(V\d+)(?:\s+\S|$)",
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

    merged: Dict[str, Any] = {}
    for key, reqs in requirements_by_section.items():
        is_sub = re.match(r"^V(\d+)\.\d+", key.strip(), re.IGNORECASE)
        if is_sub:
            # It's a sub-section (e.g., V1.1) -> fold into parent
            top_prefix = f"V{is_sub.group(1)}".upper()
            parent_key = top_level_map.get(top_prefix)
            if parent_key:
                sub_label = key.strip().split()[0]  # e.g. "V1.1"
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
            # It's a top-level section. Check if it needs canonicalization.
            m = _OWASP_TOP_LEVEL_RE.match(key.strip())
            if m:
                prefix = m.group(1).upper()
                parent_key = top_level_map.get(prefix)
                if parent_key and parent_key != key:
                    # Variation found (e.g. "V9 Communication" != "V9 Communications")
                    merged.setdefault(parent_key, []).extend(reqs or [])
                    logger.info(
                        "_flatten_owasp_sections: canonicalized parent '%s' -> '%s' (%d reqs)",
                        key, parent_key, len(reqs or []),
                    )
                    continue

        # Not an OWASP sub-section and no canonicalization needed: keep as-is
        merged.setdefault(key, []).extend(reqs or [])

    return merged


def _atoi(text: str):
    return int(text) if text.isdigit() else text.lower()


def _natural_keys(text: str):
    """
    Sort key function for natural sorting (e.g. V9 comes before V10).
    """
    return [_atoi(c) for c in re.split(r'(\d+)', text)]


def _coerce_requirement_text(requirement_item: Any) -> str:
    """
    Backward-compatible requirement text extraction.
    """
    if isinstance(requirement_item, str):
        return requirement_item.strip()
    if isinstance(requirement_item, dict):
        return str(requirement_item.get("requirement", "")).strip()
    return str(requirement_item or "").strip()


def _coerce_requirement_details(requirement_item: Any) -> str:
    """
    Extracts the full child-parameter meaning from structured extraction output.
    """
    if isinstance(requirement_item, dict):
        return str(requirement_item.get("details", "")).strip()
    return ""


def _coerce_asvs_level(requirement_item: Any) -> Optional[int]:
    if not isinstance(requirement_item, dict):
        return None
    raw_level = requirement_item.get("asvs_level")
    if raw_level is None:
        return None
    if isinstance(raw_level, int) and raw_level in (1, 2, 3):
        return raw_level
    text = str(raw_level).strip().upper()
    if text.startswith("L"):
        text = text[1:].strip()
    if text in {"1", "2", "3"}:
        return int(text)
    return None


def run_standard_ingestion_job_sync(job_id: str, celery_task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingests all StandardSourceDocuments attached to an ingestion job and
    optionally generates + stores vector embeddings for every extracted
    requirement child.
    """
    from sdr.apps.ai.engine.extraction import (
        extract_asvs_level_definitions_from_document,
        extract_requirements_from_document,
        extract_diagram_requirements,
    )
    
    with SessionLocal() as db:
        job = db.get(StandardIngestionJob, int(job_id))
        if not job:
            raise ValueError(f"Job not found for ID: {job_id}")
        session_id = build_standard_ingestion_session_id(job.id)

        mode = (job.summary_json or {}).get('mode', 'manual')
        if job.category_id is None:
            raise ValueError('Ingestion job has no category assigned.')

        # Preserve configuration from router before overwriting summary
        old_summary = job.summary_json or {}
        start_page = old_summary.get("start_page")
        end_page = old_summary.get("end_page")
        level_definition_start_page = old_summary.get("level_definition_start_page")
        level_definition_end_page = old_summary.get("level_definition_end_page")

        summary: Dict[str, Any] = {
            'inserted': 0,
            'sections': 0,
            'errors': 0,
            'embeddings_created': 0,
            'embeddings_failed': 0,
            'diagram_requirements': 0,
            'diagram_extraction_status': 'pending',
            'diagram_extraction_error': None,
            'mode': mode,
            'resolved_categories': {job.category.code if job.category else "unknown": 0},
            'version_no': getattr(job, 'version_no', 1),
            'start_page': start_page,
            'end_page': end_page,
            'level_definition_start_page': level_definition_start_page,
            'level_definition_end_page': level_definition_end_page,
            'asvs_level_definitions': {
                'status': 'pending',
                'count': 0,
                'source': 'not_started',
            },
        }
        if celery_task_id:
            summary['celery_task_id'] = celery_task_id
        
        logger.info(
            "run_standard_ingestion_job_sync: summary dict initialized. mode=%s, version_no=%s",
            summary['mode'],
            summary['version_no'],
        )

        logger.info("run_standard_ingestion_job_sync: [STEP 1-INIT] marking job as RUNNING.")
        job.status = StandardIngestionJob.STATUS_RUNNING
        job.started_at = datetime.utcnow()
        job.completed_at = None
        job.error_message = ''
        job.summary_json = summary
        db.commit()

        # Make reruns/retries idempotent for the same ingestion job. A previous
        # partial run may have already inserted parents/children/ASVS rows.
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
        db.execute(
            delete(ASVSLevelDefinition)
            .where(ASVSLevelDefinition.ingestion_job_id == job.id)
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

        parents_by_section: Dict[str, CategoryParameterParent] = {}
        ordinals_by_section: Dict[str, int] = {}
        items_to_embed: List[Dict[str, Any]] = []

        if source_docs:
            source_doc = source_docs[0]
            logger.info("run_standard_ingestion_job_sync: [STEP 1] processing single source_doc.")

            try:
                with job_session_context(
                    session_id=session_id,
                    job_type="standard_ingestion",
                    job_id=job.id,
                ):
                    standard_name = validate_standard_name(source_doc.name.rsplit('.', 1)[0])
                    
                    source_doc.status = StandardSourceDocument.STATUS_UPLOADED
                    db.commit()

                    def _update_progress(label: str, percentage: int):
                        db.refresh(job)
                        if job.status == StandardIngestionJob.STATUS_CANCELLED:
                            raise ValueError("Job was cancelled by the user.")

                        summary['detailed_progress'] = {'label': label, 'percentage': percentage}
                        job.summary_json = summary
                        from sqlalchemy.orm.attributes import flag_modified
                        flag_modified(job, "summary_json")
                        db.commit()

                    start_page = summary.get("start_page")
                    end_page = summary.get("end_page")
                    level_definition_start_page = summary.get("level_definition_start_page")
                    level_definition_end_page = summary.get("level_definition_end_page")

                    if level_definition_start_page or level_definition_end_page:
                        _update_progress("Extracting ASVS level definitions", 10)
                        level_definitions = extract_asvs_level_definitions_from_document(
                            source_doc,
                            start_page=level_definition_start_page,
                            end_page=level_definition_end_page,
                        )
                        db.execute(
                            delete(ASVSLevelDefinition)
                            .where(ASVSLevelDefinition.ingestion_job_id == job.id)
                        )
                        for item in level_definitions:
                            db.add(
                                ASVSLevelDefinition(
                                    ingestion_job_id=job.id,
                                    level=item["level"],
                                    code=item["code"],
                                    name=item["name"],
                                    description=item["description"],
                                    classification_guidance=item["classification_guidance"],
                                    source_quote=item.get("source_quote") or None,
                                    context_marker=item.get("context_marker") or None,
                                )
                            )
                        summary['asvs_level_definitions'] = {
                            'status': 'extracted' if level_definitions else 'fallback',
                            'count': len(level_definitions),
                            'source': 'standard_document' if level_definitions else 'static_fallback',
                            'reason': None if level_definitions else 'No ASVS level definitions extracted from configured page range.',
                        }
                        job.summary_json = summary
                        from sqlalchemy.orm.attributes import flag_modified
                        flag_modified(job, "summary_json")
                        db.commit()
                    else:
                        summary['asvs_level_definitions'] = {
                            'status': 'fallback',
                            'count': 0,
                            'source': 'static_fallback',
                            'reason': 'No ASVS level definition page range configured.',
                        }

                    requirements_by_section = extract_requirements_from_document(
                        source_doc,
                        start_page=start_page,
                        end_page=end_page,
                        progress_callback=_update_progress
                    )
                    
                    if not requirements_by_section:
                        raise ValueError("Extraction returned empty result.")

                    # Safety-net: fold OWASP-style sub-sections (V1.1, V1.2...) under their top-level parent (V1...)
                    all_section_keys = list(requirements_by_section.keys())
                    requirements_by_section = _flatten_owasp_sections(requirements_by_section, all_section_keys)
                    logger.info(
                        "run_standard_ingestion_job_sync: after OWASP flattening: %d section(s)",
                        len(requirements_by_section),
                    )

                    source_doc.status = StandardSourceDocument.STATUS_PARSED
                    db.commit()

                    # Sort the parent sections naturally
                    sorted_section_keys = sorted(
                        (requirements_by_section or {}).keys(),
                        key=lambda k: _natural_keys(k or "")
                    )

                    for section_title in sorted_section_keys:
                        requirements = requirements_by_section[section_title]
                        section_name = (section_title or 'General').strip() or 'General'
                        parent = parents_by_section.get(section_name)
                        
                        if parent is None:
                            section_key = stable_key(section_name)
                            parent_key = f'section:{section_key}:{job.id}'
                            
                            parent = db.execute(
                                select(CategoryParameterParent)
                                .where(
                                    CategoryParameterParent.category_id == job.category_id,
                                    CategoryParameterParent.stable_key == parent_key
                                )
                            ).scalars().first()

                            if not parent:
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
                            
                            parents_by_section[section_name] = parent
                            ordinals_by_section[section_name] = 0
                        
                        # Sort the requirements inside this section naturally
                        sorted_requirements = sorted(
                            requirements or [],
                            key=lambda req: _natural_keys(_coerce_requirement_text(req))
                        )

                        for req in sorted_requirements:
                            raw_text = _coerce_requirement_text(req)
                            details = _coerce_requirement_details(req)
                            asvs_level = _coerce_asvs_level(req)
                            analysis_text = build_parameter_analysis_text(raw_text, details)
                            normalized = normalize_requirement_text(analysis_text)
                            
                            if not normalized:
                                continue

                            next_ordinal = ordinals_by_section.get(section_name, 0) + 1
                            ordinals_by_section[section_name] = next_ordinal

                            child_key = f'{stable_key(normalized)}-{str(source_doc.id)[:8]}-{next_ordinal:06d}'
                            
                            child = CategoryParameterChild(
                                parent_id=parent.id,
                                stable_key=child_key,
                                asvs_level=asvs_level,
                                requirement_text=raw_text,
                                details=details,
                                requirement_text_normalized=normalized,
                                ordinal=next_ordinal,
                            )
                            db.add(child)
                            db.flush()
                            
                            summary['inserted'] += 1

                            items_to_embed.append({
                                'child': child,
                                'text': normalized,
                                'content_hash': child_key,
                            })

                    summary['sections'] = len(parents_by_section)
                    source_doc.status = StandardSourceDocument.STATUS_PROCESSED
                    db.commit()

            except Exception as exc:
                summary['errors'] += 1
                db.rollback()
                logger.exception('Exception processing source_doc=%s', source_doc.id)
                try:
                    source_doc.status = StandardSourceDocument.STATUS_FAILED
                    db.commit()
                except Exception as inner_exc:
                    db.rollback()
                    logger.exception('Failed to record error for source_doc=%s', source_doc.id)

        # Step 2 — Embedding + diagram generation
        with job_session_context(
            session_id=session_id,
            job_type="standard_ingestion",
            job_id=job.id,
        ):
            if items_to_embed:
                try:
                    generate_and_store_embeddings(items_to_embed, job.id, summary)
                except Exception as exc:
                    logger.error("Error generating embeddings: %s", exc)
                    summary['embeddings_failed'] = len(items_to_embed)

            # Step 3.5 — Diagram requirement extraction
            logger.info("Extracting diagram requirements for job %s", job.id)
            try:
                summary['detailed_progress'] = {'label': 'Extracting diagram requirements', 'percentage': 95}
                job.summary_json = summary
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(job, "summary_json")
                db.commit()

                diagram_req_params = db.execute(
                    select(CategoryParameterChild)
                    .join(CategoryParameterParent, CategoryParameterChild.parent_id == CategoryParameterParent.id)
                    .where(CategoryParameterParent.category_id == job.category_id)
                    .where(CategoryParameterParent.ingestion_job_id == job.id)
                    .order_by(CategoryParameterChild.id)
                ).scalars().all()

                if diagram_req_params:
                    diagram_reqs = extract_diagram_requirements(
                        parameters=list(diagram_req_params),
                        category_id=job.category_id,
                        ingestion_job_id=job.id,
                    )

                    # Clear previous diagram requirements for this job
                    db.execute(
                        delete(CategoryDiagramRequirement)
                        .where(CategoryDiagramRequirement.ingestion_job_id == job.id)
                    )

                    for dreq in diagram_reqs:
                        db.add(CategoryDiagramRequirement(
                            category_id=dreq["category_id"],
                            ingestion_job_id=dreq["ingestion_job_id"],
                            stable_key=dreq["stable_key"],
                            source_requirement_key=dreq["source_requirement_key"],
                            requirement_text=dreq["requirement_text"],
                            verification_hint=dreq["verification_hint"],
                            asvs_level=dreq.get("asvs_level"),
                            parent_section=dreq["parent_section"],
                            ordinal=dreq.get("ordinal", 0),
                        ))

                    db.commit()
                    summary['diagram_requirements'] = len(diagram_reqs)
                    summary['diagram_extraction_status'] = 'completed'
                    summary['diagram_extraction_error'] = None
                    logger.info(
                        "run_standard_ingestion_job_sync: extracted %d diagram requirements",
                        len(diagram_reqs),
                    )
                else:
                    summary['diagram_requirements'] = 0
                    summary['diagram_extraction_status'] = 'completed'
                    summary['diagram_extraction_error'] = None
            except Exception as exc:
                db.rollback()
                logger.exception("Error extracting diagram requirements: %s", exc)
                summary['diagram_requirements'] = 0
                summary['diagram_extraction_status'] = 'failed'
                summary['diagram_extraction_error'] = str(exc)
                summary['errors'] += 1

        # Step 3 — Finalise the job row only after diagram extraction has completed
        job_status_new = (
            StandardIngestionJob.STATUS_COMPLETED
            if summary['errors'] == 0 and summary.get('diagram_extraction_status') == 'completed'
            else StandardIngestionJob.STATUS_FAILED
        )
        job.status = job_status_new
        summary['detailed_progress'] = {'label': 'Completed' if job_status_new == StandardIngestionJob.STATUS_COMPLETED else 'Failed', 'percentage': 100}
        job.summary_json = summary
        job.completed_at = datetime.utcnow()

        if job_status_new == StandardIngestionJob.STATUS_FAILED:
            job.error_message = (
                summary.get('diagram_extraction_error')
                or 'Document failed during ingestion.'
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
            job.activated_at = datetime.utcnow()

        try:
            job.summary_json = summary
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(job, "summary_json")
            db.commit()
        except Exception as exc:
            logger.warning('Could not persist embedding counters for job=%s: %s', job.id, exc)

        # Step 4 - Clean up temporal documents
        from sdr.apps.workspace.services.storage import storage_service
        for source_doc in source_docs:
            if source_doc.document:
                try:
                    storage_service.delete_file(source_doc.document)
                except Exception as e:
                    logger.warning("Failed to delete temporal document %s: %s", source_doc.document, e)

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
