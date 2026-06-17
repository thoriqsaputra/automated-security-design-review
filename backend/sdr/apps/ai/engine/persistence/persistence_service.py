from __future__ import annotations

import base64
import binascii
import logging
import re
from typing import Any, Optional, List

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from sdr.core.database import SessionLocal
from sdr.apps.reviews.models import Finding, Review, CitationAnchor
from sdr.apps.reviews.models.choices import FindingType, AnchorType
from sdr.apps.workspace.services.storage import storage_service
from sdr.apps.ai.agents.base import Citation
from sdr.apps.standards.models import CategoryControlSummaryRequirement, CategoryParameterChild
from sdr.apps.standards.utils import build_parameter_analysis_text
from sdr.apps.ai.engine.dto import PersistenceInput, AnalysisSummary
from sdr.apps.ai.engine.classification.severity import calculate_deterministic_severity

logger = logging.getLogger(__name__)

_CITATION_BULK_BATCH = 200
_RAW_CITATION_ID_PATTERN = re.compile(r"\b(?:chunk_\d+|p\d+_[bd]\d+|citation_ids?)\b", re.IGNORECASE)
_NULL_BYTE = "\x00"
_DIAGRAM_PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:not\s+completed|could\s+be\s+longer|todo|tbd|draft|placeholder|lorem\s+ipsum|mst_deep_scan)\b",
    re.IGNORECASE,
)
_DIAGRAM_IMAGE_CONTENT_TYPES = {
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


class PersistenceService:
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def persist_finding(
        self,
        review: Review,
        persistence_input: PersistenceInput,
        summary: AnalysisSummary,
    ) -> Optional[Finding]:
        parameter = persistence_input.parameter
        debate_output = persistence_input.debate_output
        mediator = debate_output.mediator_result
        raw_final_verdict = mediator.raw_final_verdict or mediator.final_verdict
        persisted_met_status = (
            "not_met" if raw_final_verdict == "partial" else mediator.final_verdict
        )

        self.logger.info(
            "PersistenceService.persist_finding: [ENTRY] review_id=%s parameter id=%s",
            review.id,
            parameter.id,
        )

        try:
            anchorable_citations, citation_resolution_mode = self._resolve_citations_for_anchoring(
                mediator.final_citations or [],
                debate_output.analysis_trace or {},
            )
            source_map = self._build_citation_source_map(
                anchorable_citations or (mediator.final_citations or []),
                debate_output.analysis_trace or {},
            )
            if persisted_met_status == "met" and not anchorable_citations:
                persisted_met_status = "na"
                raw_final_verdict = "met_without_grounded_citations"
            sanitized_description = self._strip_null_bytes(
                self._sanitize_user_facing_text(mediator.finding_description or mediator.reasoning or "", source_map)
            )
            sanitized_reasoning = self._strip_null_bytes(
                self._sanitize_user_facing_text(mediator.reasoning or "", source_map)
            )
            sanitized_recommendation = self._strip_null_bytes(
                self._sanitize_user_facing_text(mediator.recommendation or "", source_map)
            )
            requirement_text = self._strip_null_bytes(build_parameter_analysis_text(parameter))
            requirement_metadata = self._strip_null_bytes(
                {
                    "section": (
                        parameter.parent.title if parameter.parent else None
                    ),
                    "ordinal": parameter.ordinal,
                    "stable_key": parameter.stable_key,
                    "ingestion_job_version": (
                        parameter.parent.ingestion_job.version_no
                        if parameter.parent
                        and "ingestion_job" in parameter.parent.__dict__
                        and parameter.parent.ingestion_job
                        else None
                    ),
                    "analysis_trace": {
                        **(debate_output.analysis_trace or {}),
                        "raw_final_verdict": raw_final_verdict,
                        "citation_resolution_mode": citation_resolution_mode,
                    },
                    "structured_citations": self._build_structured_citation_metadata(
                        anchorable_citations,
                        source_map,
                    ),
                    **self._build_requirement_provenance(debate_output.analysis_trace or {}),
                }
            )
            severity_payload = self._build_requirement_severity_payload(
                met_status=persisted_met_status,
                confidence_score=mediator.confidence,
                domain=self._resolve_requirement_severity_domain(
                    debate_output.analysis_trace or {},
                    getattr(getattr(persistence_input, "category", None), "code", None),
                ),
                raw_final_verdict=raw_final_verdict,
                requirement_text=requirement_text,
                requirement_metadata=requirement_metadata,
                analysis_trace=debate_output.analysis_trace or {},
                citation_count=len(anchorable_citations),
            )
            sanitized_recommendation = self._normalize_recommendation(
                sanitized_recommendation,
                persisted_met_status,
            )
            finding_title = self._build_finding_title(parameter, mediator)
            
            with SessionLocal() as db:
                try:
                    # CFSRs are not CategoryParameterChild rows — use None for the FK
                    # so the nullable column is set correctly and no FK violation occurs.
                    is_cfsr = isinstance(parameter, CategoryControlSummaryRequirement)
                    finding = Finding(
                        review_id=review.id,
                        category_id=(
                            parameter.parent.category_id if parameter.parent else None
                        ),
                        parent_parameter_id=parameter.parent.id if parameter.parent else None,
                        child_parameter_id=None if is_cfsr else parameter.id,
                        finding_type=FindingType.REQUIREMENT.value if hasattr(FindingType, 'value') else FindingType.REQUIREMENT,
                        title=finding_title,
                        description=sanitized_description,
                        met_status=persisted_met_status,
                        confidence_score=mediator.confidence,
                        severity=severity_payload["severity"],
                        severity_score=severity_payload["severity_score"],
                        severity_analysis=severity_payload["severity_analysis"],
                        recommendation=sanitized_recommendation,
                        hunter_reasoning=self._strip_null_bytes(debate_output.hunter_result.reasoning),
                        critic_reasoning=self._strip_null_bytes(debate_output.critic_result.reasoning),
                        mediator_reasoning=sanitized_reasoning,
                        hunter_thought_process=self._strip_null_bytes(debate_output.hunter_result.cot_trace),
                        critic_thought_process=self._strip_null_bytes(debate_output.critic_result.cot_trace),
                        mediator_thought_process=self._strip_null_bytes(mediator.cot_trace),
                        reason=sanitized_reasoning,
                        requirement_reference=parameter.stable_key,
                        requirement_text=requirement_text,
                        requirement_metadata=requirement_metadata,
                    )
                    db.add(finding)
                    db.flush()

                    if anchorable_citations:
                        anchors = self._build_citation_anchors(
                            finding,
                            anchorable_citations,
                            debate_output.analysis_trace or {},
                        )
                        db.add_all(anchors)
                        summary.citation_count += len(anchors)
                        self.logger.info(
                            "PersistenceService.persist_finding: [CITATIONS] persisted %d anchor(s)",
                            len(anchors),
                        )
                        
                    db.commit()
                    db.refresh(finding)
                except Exception:
                    db.rollback()
                    raise

            if persisted_met_status == "met":
                summary.met_count += 1
            elif persisted_met_status == "not_met":
                summary.not_met_count += 1
                if severity_payload["severity"] == "critical":
                    summary.critical_findings.append(finding_title)
                elif severity_payload["severity"] == "high":
                    summary.high_findings.append(finding_title)
            else:
                summary.na_count += 1

            self.logger.info(
                "PersistenceService.persist_finding: [SUCCESS] finding_id=%s",
                finding.id,
            )
            return finding

        except Exception as exc:
            summary.error_count += 1
            self.logger.exception(
                "PersistenceService.persist_finding: [FAILED] parameter id=%s: %s",
                parameter.id,
                exc,
            )
            return None

    def persist_diagram_debate_finding(
        self,
        *,
        review: Review,
        category,
        diagram_debate_output,
        summary: AnalysisSummary,
    ) -> Optional[Finding]:
        try:
            if getattr(diagram_debate_output, "error", None):
                self.logger.warning(
                    "PersistenceService.persist_diagram_debate_finding: skipping finding creation due to error for diagram_id=%s",
                    getattr(getattr(diagram_debate_output, "diagram", None), "diagram_id", None),
                )
                return None

            diagram_input = diagram_debate_output.diagram
            hunter_result = dict(getattr(diagram_debate_output, "hunter_result", {}) or {})
            critic_result = dict(getattr(diagram_debate_output, "critic_result", {}) or {})
            mediator_result = dict(getattr(diagram_debate_output, "mediator_result", {}) or {})
            assessed_requirements = list(
                mediator_result.get("assessed_requirements")
                or hunter_result.get("requirement_assessments")
                or []
            )
            diagram_scope_verdict = (
                mediator_result.get("diagram_scope_verdict")
                or critic_result.get("diagram_scope_verdict")
                or hunter_result.get("diagram_scope_verdict")
            )
            diagram_scope_reasoning = (
                mediator_result.get("diagram_scope_reasoning")
                or critic_result.get("diagram_scope_reasoning")
                or hunter_result.get("diagram_scope_reasoning")
            )
            met_status = str(mediator_result.get("final_verdict") or "na").strip().lower() or "na"
            confidence_score = mediator_result.get("confidence")
            severity_payload = self._build_diagram_severity_payload(
                met_status=met_status,
                confidence_score=confidence_score,
                domain=getattr(category, "code", None),
                ambiguous_elements=list(critic_result.get("hallucinated_claims") or []),
                missing_information=list(hunter_result.get("missing_controls") or []),
                requirement_text=self._build_diagram_requirement_text(assessed_requirements),
            )
            self.logger.info(
                "PersistenceService.persist_diagram_debate_finding: diagram_id=%s",
                diagram_input.diagram_id,
            )

            diagram_display = self._build_diagram_display_payload(
                diagram_input,
                diagram_debate_output,
            )
            diagram_image_metadata = self._build_diagram_image_metadata(
                review_id=review.id,
                diagram_input=diagram_input,
            )

            finding_type_val = FindingType.DIAGRAM.value if hasattr(FindingType, 'value') else FindingType.DIAGRAM

            with SessionLocal() as db:
                qs = select(Finding).where(
                    Finding.review_id == review.id,
                    Finding.category_id == getattr(category, "id", None),
                    Finding.finding_type == finding_type_val,
                    Finding.diagram_id == diagram_input.diagram_id,
                    Finding.child_parameter_id.is_(None),
                )
                finding = db.execute(qs).scalars().first()
                created = False

                requirement_metadata = self._strip_null_bytes({
                    "source": "diagram_debate_service",
                    "diagram_id": diagram_input.diagram_id,
                    "diagram_caption": diagram_display["caption"],
                    "raw_diagram_caption": diagram_display["raw_caption"],
                    "diagram_image": diagram_image_metadata,
                    "diagram_page_number": diagram_input.page_number,
                    "diagram_bbox": {
                        "x0": diagram_input.bbox_x0,
                        "y0": diagram_input.bbox_y0,
                        "x1": diagram_input.bbox_x1,
                        "y1": diagram_input.bbox_y1,
                    },
                    "assessed_requirements": assessed_requirements,
                    "analysis_trace": {
                        "diagram_scope_verdict": diagram_scope_verdict,
                        "diagram_scope_reasoning": diagram_scope_reasoning,
                        "assessed_requirements": assessed_requirements,
                        "debate_rounds": getattr(diagram_debate_output, "debate_rounds", 1),
                        "hunter_result": hunter_result,
                        "critic_result": critic_result,
                        "mediator_result": mediator_result,
                    },
                })

                if not finding:
                    finding = Finding(review_id=review.id, finding_type=finding_type_val, diagram_id=diagram_input.diagram_id)
                    db.add(finding)
                    created = True

                finding.category_id = getattr(category, "id", None)
                finding.parent_parameter_id = None
                finding.child_parameter_id = None
                finding.title = self._strip_null_bytes(diagram_display["title"])
                finding.description = self._strip_null_bytes(
                    mediator_result.get("finding_description")
                    or mediator_result.get("reasoning")
                    or hunter_result.get("reasoning")
                    or ""
                )
                finding.met_status = met_status
                finding.confidence_score = confidence_score
                finding.severity = severity_payload["severity"]
                finding.severity_score = severity_payload["severity_score"]
                finding.severity_analysis = severity_payload["severity_analysis"]
                finding.recommendation = self._strip_null_bytes(mediator_result.get("recommendation"))
                finding.reason = self._strip_null_bytes(mediator_result.get("reasoning"))
                finding.diagram_caption = self._strip_null_bytes(diagram_display["caption"])
                finding.vision_reasoning = self._strip_null_bytes(hunter_result.get("reasoning"))
                finding.vision_thought_process = None
                finding.hunter_reasoning = self._strip_null_bytes(hunter_result.get("reasoning"))
                finding.critic_reasoning = self._strip_null_bytes(critic_result.get("reasoning"))
                finding.mediator_reasoning = self._strip_null_bytes(mediator_result.get("reasoning"))
                finding.requirement_reference = self._build_diagram_requirement_reference(assessed_requirements)
                finding.requirement_text = self._strip_null_bytes(
                    self._build_diagram_requirement_text(assessed_requirements)
                )
                finding.requirement_metadata = requirement_metadata

                try:
                    db.commit()
                    db.refresh(finding)
                except IntegrityError:
                    db.rollback()
                    finding = db.execute(qs).scalars().first()
                    if finding is None:
                        raise

            if created:
                summary.diagram_findings_count += 1
            if met_status == "met":
                summary.met_count += 1
            elif met_status == "not_met":
                summary.not_met_count += 1
                if severity_payload["severity"] == "critical":
                    summary.critical_findings.append(finding.title)
                elif severity_payload["severity"] == "high":
                    summary.high_findings.append(finding.title)
            else:
                summary.na_count += 1

            self.logger.info(
                "PersistenceService.persist_diagram_debate_finding: [SUCCESS] created=%s finding_id=%s",
                created,
                getattr(finding, "id", None),
            )
            return finding

        except Exception as exc:
            summary.error_count += 1
            self.logger.exception(
                "PersistenceService.persist_diagram_debate_finding: failed: %s",
                exc,
            )
            return None

    def _persist_diagram_finding(
        self,
        review: Review,
        parameter: CategoryParameterChild,
        diagram_input,
        vision_result,
        summary: AnalysisSummary,
    ) -> None:
        self.logger.warning(
            "PersistenceService._persist_diagram_finding: legacy per-parameter vision persistence is deprecated"
        )

    # ---- Private Helpers ----

    def _build_finding_title(self, parameter: CategoryParameterChild, mediator) -> str:
        section = parameter.parent.title if parameter.parent else "General"
        req_snippet = (parameter.requirement_text or "")[:80].strip()
        title = f"{section}: {req_snippet}"
        return title[:255]

    def _build_requirement_provenance(self, analysis_trace: dict) -> dict:
        trace = dict(analysis_trace or {})
        evidence_gate_outcome = trace.get("evidence_gate_outcome")
        verdict_policy = trace.get("verdict_policy") or {}
        retrieval_query_details = trace.get("retrieval_query_details") or {}
        retrieval_metadata = retrieval_query_details.get("retrieval_evidence_metadata") or {}
        outcome_source = "debate"
        synth_mode = (trace.get("contract") or {}).get("synth_mode", "")
        if synth_mode == "cfsr_cascade":
            outcome_source = "cfsr_cascade"
        elif evidence_gate_outcome in {
            "downgraded_to_na_missing_citations",
            "downgraded_to_na_applicability_not_established",
            "downgraded_to_na_no_applicability_signal",
            "downgraded_to_na_no_applicability_signal_after_retry",
        }:
            outcome_source = "evidence_gate_downgrade"
        return {
            "analysis_outcome_source": outcome_source,
            "verdict_policy_source": verdict_policy.get("source"),
            "applicability_established": verdict_policy.get("applicability_established"),
            "not_assessable_reason": verdict_policy.get("not_assessable_reason"),
            "evidence_sufficiency": verdict_policy.get("evidence_sufficiency"),
            "verified_control_evidence_ids": list(verdict_policy.get("verified_control_evidence_ids") or []),
            "prefilter_reason": trace.get("prefilter_reason"),
            "prefilter_confidence": trace.get("prefilter_confidence"),
            "evidence_gate_attempted": bool(trace.get("evidence_gate_attempted", False)),
            "evidence_gate_outcome": evidence_gate_outcome,
            "downgrade_reason": trace.get("downgrade_reason"),
            "retrieval_evidence_quality": retrieval_metadata.get("evidence_quality"),
        }

    def _build_diagram_image_metadata(self, *, review_id: int, diagram_input) -> dict:
        image_format = str(getattr(diagram_input, "image_format", "png") or "png").lower()
        if image_format == "jpg":
            image_format = "jpeg"
        content_type = _DIAGRAM_IMAGE_CONTENT_TYPES.get(image_format, "image/png")
        extension = "jpg" if image_format == "jpeg" else image_format
        object_name = f"reviews/{review_id}/diagrams/{diagram_input.diagram_id}.{extension}"

        try:
            image_b64 = getattr(diagram_input, "image_b64", "") or ""
            image_bytes = base64.b64decode(image_b64, validate=True)
            if not image_bytes:
                raise ValueError("empty image bytes")
            storage_service.upload_file(image_bytes, object_name, content_type)
            return {
                "object_name": object_name,
                "content_type": content_type,
                "image_format": image_format,
                "byte_size": len(image_bytes),
            }
        except (binascii.Error, ValueError) as exc:
            self.logger.warning(
                "PersistenceService._build_diagram_image_metadata: invalid image for review_id=%s diagram_id=%s: %s",
                review_id,
                getattr(diagram_input, "diagram_id", None),
                exc,
            )
            return {"error": f"invalid_image: {exc}"}
        except Exception as exc:
            self.logger.warning(
                "PersistenceService._build_diagram_image_metadata: upload failed for review_id=%s diagram_id=%s: %s",
                review_id,
                getattr(diagram_input, "diagram_id", None),
                exc,
            )
            return {"error": f"upload_failed: {exc}"}

    def _sanitize_user_facing_text(self, text: str, source_map: dict) -> str:
        def _replace(match: re.Match) -> str:
            token = match.group(0)
            source = source_map.get(token)
            if source and source.get("source_label"):
                return source["source_label"]
            return "the cited source excerpt"

        return _RAW_CITATION_ID_PATTERN.sub(_replace, text or "")

    def _strip_null_bytes(self, value: Any) -> Any:
        if isinstance(value, str):
            if _NULL_BYTE not in value:
                return value
            cleaned = value.replace(_NULL_BYTE, "")
            self.logger.warning(
                "PersistenceService._strip_null_bytes: removed null byte(s) from string payload"
            )
            return cleaned
        if isinstance(value, dict):
            return {k: self._strip_null_bytes(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._strip_null_bytes(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._strip_null_bytes(item) for item in value)
        return value

    def _build_citation_source_map(self, citations: List[Citation], analysis_trace: dict) -> dict:
        chunk_map = (analysis_trace or {}).get("context_chunk_map") or {}
        source_map = {}
        for citation in citations:
            block_id = citation.block_id
            if not block_id:
                continue
            source_payload = chunk_map.get(block_id, {})
            section = source_payload.get("section")
            page = citation.page_number
            label_parts = [part for part in [section, f"page {page}" if page else None] if part]
            source_map[block_id] = {
                "source_label": " - ".join(label_parts) if label_parts else "the cited source excerpt",
                "chunk_id": block_id,
                "document": source_payload.get("document"),
                "section": section,
                "page": page,
                "heading": source_payload.get("heading"),
                "source_path": source_payload.get("source_path"),
            }
        return source_map

    def _build_structured_citation_metadata(self, citations: List[Citation], source_map: dict) -> List[dict]:
        output = []
        for citation in citations:
            chunk_id = citation.block_id
            if not chunk_id:
                continue
            payload = source_map.get(chunk_id) or {
                "source_label": "the cited source excerpt",
                "chunk_id": chunk_id,
                "document": None,
                "section": None,
                "page": citation.page_number,
                "heading": None,
                "source_path": None,
            }
            output.append(payload)
        return output

    def _build_citation_anchors(
        self,
        finding: Finding,
        citations: List[Citation],
        analysis_trace: Optional[dict] = None,
    ) -> List[CitationAnchor]:
        anchors: List[CitationAnchor] = []
        chunk_map = (analysis_trace or {}).get("context_chunk_map") or {}

        for citation in citations:
            if not citation.block_id:
                continue
            hydrated = self._hydrate_citation_location(citation, chunk_map)
            if int(hydrated.page_number or 0) < 1:
                self.logger.warning(
                    "PersistenceService._build_citation_anchors: skipping invalid citation anchor block_id=%s page=%s",
                    hydrated.block_id,
                    hydrated.page_number,
                )
                continue

            anchor_type = (
                AnchorType.DIAGRAM.value if hasattr(AnchorType, 'value') else AnchorType.DIAGRAM
                if "_d" in hydrated.block_id
                else AnchorType.TEXT.value if hasattr(AnchorType, 'value') else AnchorType.TEXT
            )

            anchors.append(
                CitationAnchor(
                    finding_id=finding.id,
                    anchor_type=anchor_type,
                    block_id=hydrated.block_id,
                    page_number=hydrated.page_number,
                    quoted_text=hydrated.quoted_text,
                    bbox_x0=hydrated.bbox_x0,
                    bbox_y0=hydrated.bbox_y0,
                    bbox_x1=hydrated.bbox_x1,
                    bbox_y1=hydrated.bbox_y1,
                )
            )

        return anchors

    def _resolve_citations_for_anchoring(
        self,
        citations: List[Citation],
        analysis_trace: Optional[dict],
    ) -> tuple[List[Citation], str]:
        chunk_map = (analysis_trace or {}).get("context_chunk_map") or {}
        quote_matched: List[Citation] = []
        fallback: List[Citation] = []
        seen: set[str] = set()

        for citation in citations:
            if not citation.block_id or citation.block_id in seen:
                continue

            payload = chunk_map.get(citation.block_id) or {}
            if not self._is_citation_grade_payload(payload, citation.block_id):
                continue

            seen.add(citation.block_id)
            hydrated = self._hydrate_citation_location(citation, chunk_map)
            fallback.append(hydrated)
            matched = self._find_quote_matched_citation(hydrated, chunk_map)
            if matched:
                quote_matched.append(matched)

        if quote_matched:
            return quote_matched, "quote_matched"
        if fallback:
            return fallback, "validated_block_fallback"
        return [], "none"

    def _find_quote_matched_citation(self, citation: Citation, chunk_map: dict) -> Optional[Citation]:
        quote = self._normalize_quote_text(citation.quoted_text)
        if not quote:
            return None
        current_payload = chunk_map.get(citation.block_id) or {}
        candidates = []
        if current_payload:
            candidates.append((citation.block_id, current_payload))
        citation_page = self._safe_int(
            citation.page_number or current_payload.get("page_number") or current_payload.get("page")
        )
        for block_id, payload in (chunk_map or {}).items():
            if block_id == citation.block_id:
                continue
            if not self._is_citation_grade_payload(payload, block_id):
                continue
            payload_page = self._safe_int(payload.get("page_number") or payload.get("page"))
            if citation_page and payload_page == citation_page:
                candidates.append((block_id, payload))
        for block_id, payload in (chunk_map or {}).items():
            if any(block_id == existing_id for existing_id, _ in candidates):
                continue
            if not self._is_citation_grade_payload(payload, block_id):
                continue
            candidates.append((block_id, payload))

        for block_id, payload in candidates:
            text = self._normalize_quote_text(payload.get("text") or "")
            if quote and quote in text:
                hydrated = self._hydrate_citation_location(
                    Citation(
                        block_id=block_id,
                        page_number=citation.page_number,
                        quoted_text=citation.quoted_text,
                        bbox_x0=citation.bbox_x0,
                        bbox_y0=citation.bbox_y0,
                        bbox_x1=citation.bbox_x1,
                        bbox_y1=citation.bbox_y1,
                    ),
                    chunk_map,
                )
                return hydrated
        return None

    def _is_citation_grade_payload(self, payload: dict, block_id: str) -> bool:
        if payload.get("citation_grade", True) is False:
            return False
        evidence_kind = str(payload.get("evidence_kind") or "").lower()
        if evidence_kind in {"graph_summary", "baseline_requirement"}:
            return False
        return not str(block_id).startswith("graph_summary_")

    def _normalize_quote_text(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().lower()

    def _hydrate_citation_location(self, citation: Citation, chunk_map: dict) -> Citation:
        payload = (chunk_map or {}).get(citation.block_id) or {}
        page_number = int(citation.page_number or 0)
        source_page = payload.get("page_number") or payload.get("page")
        if page_number < 1 and source_page:
            try:
                page_number = int(source_page)
            except (TypeError, ValueError):
                page_number = 0

        bbox = payload.get("bbox") or {}
        bbox_x0 = citation.bbox_x0 if citation.bbox_x0 is not None else payload.get("bbox_x0", bbox.get("x0"))
        bbox_y0 = citation.bbox_y0 if citation.bbox_y0 is not None else payload.get("bbox_y0", bbox.get("y0"))
        bbox_x1 = citation.bbox_x1 if citation.bbox_x1 is not None else payload.get("bbox_x1", bbox.get("x1"))
        bbox_y1 = citation.bbox_y1 if citation.bbox_y1 is not None else payload.get("bbox_y1", bbox.get("y1"))

        return Citation(
            block_id=citation.block_id,
            page_number=page_number,
            quoted_text=(citation.quoted_text or str(payload.get("quoted_text") or payload.get("text") or "")).strip(),
            bbox_x0=self._safe_float(bbox_x0),
            bbox_y0=self._safe_float(bbox_y0),
            bbox_x1=self._safe_float(bbox_x1),
            bbox_y1=self._safe_float(bbox_y1),
        )

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _build_diagram_display_payload(self, diagram_input, vision_result) -> dict:
        raw_caption = self._normalize_diagram_caption(getattr(diagram_input, "caption", None))
        clean_caption = raw_caption if self._is_credible_diagram_caption(raw_caption) else None

        if clean_caption:
            title_detail = clean_caption
        else:
            architect_summary = self._extract_architect_summary(vision_result)
            title_detail = self._normalize_diagram_caption(
                architect_summary.get("diagram_title")
            )
            if not self._is_credible_diagram_caption(title_detail):
                title_detail = None

        diagram_id = getattr(diagram_input, "diagram_id", None) or "unknown"
        page_number = getattr(diagram_input, "page_number", None)
        if title_detail:
            title = f"Diagram {diagram_id}: {title_detail}"
        elif page_number:
            title = f"Diagram {diagram_id} on page {page_number}"
        else:
            title = f"Diagram {diagram_id}"

        return {
            "title": title,
            "caption": clean_caption,
            "raw_caption": raw_caption,
        }

    def _extract_architect_summary(self, vision_result: Any) -> dict:
        if isinstance(vision_result, dict):
            return dict(vision_result.get("architect_summary") or {})
        mediator = dict(getattr(vision_result, "mediator_result", {}) or {})
        hunter = dict(getattr(vision_result, "hunter_result", {}) or {})
        architect_summary = mediator.get("architect_summary") or hunter.get("architect_summary") or {}
        return dict(architect_summary or {})

    def _build_diagram_requirement_reference(self, assessed_requirements: List[dict]) -> Optional[str]:
        requirement_ids = [
            str(item.get("requirement_id", "")).strip()
            for item in assessed_requirements
            if isinstance(item, dict) and str(item.get("requirement_id", "")).strip()
        ]
        if not requirement_ids:
            return None
        joined = ", ".join(requirement_ids[:12])
        if len(joined) <= 128:
            return joined
        truncated = joined[:120].rstrip(", ")
        return f"{truncated}..."

    def _build_diagram_requirement_text(self, assessed_requirements: List[dict]) -> Optional[str]:
        lines: List[str] = []
        for item in assessed_requirements:
            if not isinstance(item, dict):
                continue
            requirement_id = str(item.get("requirement_id", "")).strip()
            verdict = str(item.get("verdict", "")).strip()
            summary = str(item.get("summary") or item.get("reasoning") or "").strip()
            if requirement_id and summary:
                lines.append(f"[{requirement_id}] {verdict}: {summary}")
            elif requirement_id:
                lines.append(f"[{requirement_id}] {verdict}".strip())
        if not lines:
            return None
        return "\n".join(lines)[:4000]

    def _normalize_diagram_caption(self, caption: Any) -> Optional[str]:
        if caption is None:
            return None
        text = re.sub(r"\s+", " ", str(caption)).strip()
        return text or None

    def _is_credible_diagram_caption(self, caption: Optional[str]) -> bool:
        if not caption:
            return False
        words = caption.split()
        if len(words) > 20 or len(caption) > 160:
            return False
        if _DIAGRAM_PLACEHOLDER_PATTERN.search(caption):
            return False
        if caption.count(",") >= 2 and not re.search(r"\b(?:diagram|architecture|flow|model|overview|sequence|network)\b", caption, re.IGNORECASE):
            return False
        if caption.endswith((",", ";", ":", "-", "(")):
            return False
        return True

    def _build_requirement_severity_payload(
        self,
        *,
        met_status: Optional[str],
        confidence_score: Optional[float],
        domain: Optional[str],
        raw_final_verdict: Optional[str],
        requirement_text: Optional[str],
        requirement_metadata: Optional[dict],
        analysis_trace: Optional[dict],
        citation_count: int,
    ) -> dict:
        severity = calculate_deterministic_severity(
            met_status=met_status,
            confidence_score=confidence_score,
            domain=domain,
            finding_type=FindingType.REQUIREMENT.value if hasattr(FindingType, 'value') else FindingType.REQUIREMENT,
            raw_final_verdict=raw_final_verdict,
            requirement_text=requirement_text,
            requirement_metadata=requirement_metadata,
            analysis_trace=analysis_trace,
            citation_count=citation_count,
        )
        return {
            "severity": severity.severity,
            "severity_score": severity.score,
            "severity_analysis": self._strip_null_bytes(severity.analysis),
        }

    def _resolve_requirement_severity_domain(
        self,
        analysis_trace: dict,
        fallback_domain: Optional[str],
    ) -> Optional[str]:
        retrieval_query_details = (analysis_trace or {}).get("retrieval_query_details") or {}
        for key in ("primary_domain", "domain_signal"):
            value = retrieval_query_details.get(key)
            if isinstance(value, str) and value.strip() and value.strip().lower() != "general":
                return value.strip()
        return fallback_domain

    def _build_diagram_severity_payload(
        self,
        *,
        met_status: Optional[str],
        confidence_score: Optional[float],
        domain: Optional[str],
        ambiguous_elements: List[str],
        missing_information: List[str],
        requirement_text: Optional[str],
    ) -> dict:
        severity = calculate_deterministic_severity(
            met_status=met_status,
            confidence_score=confidence_score,
            domain=domain,
            finding_type=FindingType.DIAGRAM.value if hasattr(FindingType, 'value') else FindingType.DIAGRAM,
            ambiguous_elements=ambiguous_elements,
            missing_information=missing_information,
            requirement_text=requirement_text,
        )
        return {
            "severity": severity.severity,
            "severity_score": severity.score,
            "severity_analysis": self._strip_null_bytes(severity.analysis),
        }

    def _normalize_recommendation(self, recommendation: Optional[str], met_status: Optional[str]) -> Optional[str]:
        if met_status != "not_met":
            return None
        return recommendation
