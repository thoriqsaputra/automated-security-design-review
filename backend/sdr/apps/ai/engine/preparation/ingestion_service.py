from __future__ import annotations

import json
import logging
import re
from typing import Optional

from sdr.apps.workspace.document_processing import get_local_file_path
from sdr.apps.ai.tsd_processing.ingestor import TSDDocument, TSDIngestor
from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.prompts.analysis import (
    TSD_SCREENING_SYSTEM_PROMPT,
    build_tsd_screening_prompt,
)
from sdr.apps.designs.models import Design
from sdr.apps.reviews.models import Review
from sdr.apps.ai.engine.dto import IngestionOutput


logger = logging.getLogger(__name__)

_SCREENING_TEMPERATURE = 0.0
_SCREENING_MAX_TOKENS = 256
_SCREENING_CONFIDENCE_THRESHOLD = 0.8
_SCREENING_SAMPLE_MAX_CHARS = 10000
_SCREENING_PAGE_SNIPPET_MAX_CHARS = 1800
_SCREENING_MAX_SIGNAL_PAGES = 4

_FRONT_MATTER_PATTERNS = (
    "table of contents",
    "contents",
    "revision history",
    "version history",
    "document history",
    "change history",
    "approval",
    "approvals",
    "glossary",
    "definitions",
    "abbreviations",
)

_TSD_SIGNAL_PATTERNS = (
    "architecture",
    "system overview",
    "component",
    "service",
    "api",
    "interface",
    "data flow",
    "sequence diagram",
    "deployment",
    "infrastructure",
    "database",
    "authentication",
    "authorization",
    "identity",
    "session",
    "encryption",
    "security control",
    "threat",
    "trust boundary",
    "network",
    "container",
)

_CLEAR_NON_TSD_PATTERNS = (
    "contract",
    "legal",
    "policy",
    "standard",
    "marketing",
    "brochure",
    "resume",
    "invoice",
    "purchase order",
    "financial report",
    "meeting minutes",
    "presentation",
)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _pattern_count(text: str, patterns: tuple[str, ...]) -> int:
    lowered = (text or "").lower()
    return sum(1 for pattern in patterns if pattern in lowered)


def _looks_like_front_matter(page_number: int, heading: str, text: str) -> bool:
    heading_lowered = (heading or "").strip().lower()
    front_heading = any(pattern in heading_lowered for pattern in _FRONT_MATTER_PATTERNS)
    haystack = f"{heading}\n{text[:1200]}".lower()
    front_matter_hits = _pattern_count(haystack, _FRONT_MATTER_PATTERNS)
    tsd_hits = _pattern_count(haystack, _TSD_SIGNAL_PATTERNS)
    return front_heading or (front_matter_hits > 0 and tsd_hits == 0)


def _trim_page_for_screening(text: str) -> str:
    compact = _compact_text(text)
    if len(compact) <= _SCREENING_PAGE_SNIPPET_MAX_CHARS:
        return compact

    lowered = compact.lower()
    signal_positions = [
        lowered.find(pattern)
        for pattern in _TSD_SIGNAL_PATTERNS
        if lowered.find(pattern) >= 0
    ]
    if signal_positions:
        midpoint = min(signal_positions)
        start = max(0, midpoint - (_SCREENING_PAGE_SNIPPET_MAX_CHARS // 3))
        end = min(len(compact), start + _SCREENING_PAGE_SNIPPET_MAX_CHARS)
        return compact[start:end].strip()

    return compact[:_SCREENING_PAGE_SNIPPET_MAX_CHARS].strip()


def _build_tsd_screening_sample(tsd_document: TSDDocument) -> tuple[str, dict]:
    page_infos = []
    for page in getattr(tsd_document, "pages", []) or []:
        text = getattr(page, "all_text", "") or getattr(page, "raw_text", "") or ""
        text = text.strip()
        if not text:
            continue
        heading = getattr(page, "section_heading", "") or ""
        page_number = int(
            getattr(page, "page_number", len(page_infos) + 1) or len(page_infos) + 1
        )
        score = _pattern_count(f"{heading}\n{text}", _TSD_SIGNAL_PATTERNS)
        page_infos.append(
            {
                "page_number": page_number,
                "heading": heading,
                "text": text,
                "score": score,
                "front_matter": _looks_like_front_matter(page_number, heading, text),
            }
        )

    if not page_infos:
        fallback_text = getattr(tsd_document, "full_text", "") or ""
        return fallback_text[:_SCREENING_SAMPLE_MAX_CHARS], {
            "sampled_pages": [],
            "skipped_front_matter_pages": 0,
            "sample_strategy": "full_text_fallback",
        }

    non_front_pages = [page for page in page_infos if not page["front_matter"]]
    candidate_pages = non_front_pages or page_infos

    selected_by_number: dict[int, dict] = {}

    def add_page(page_info: dict | None) -> None:
        if page_info is not None:
            selected_by_number.setdefault(page_info["page_number"], page_info)

    add_page(candidate_pages[0])
    add_page(candidate_pages[len(candidate_pages) // 2])
    add_page(candidate_pages[-1])

    signal_pages = sorted(
        (page for page in candidate_pages if page["score"] > 0),
        key=lambda page: (-page["score"], page["page_number"]),
    )
    for page in signal_pages[:_SCREENING_MAX_SIGNAL_PAGES]:
        add_page(page)

    selected_pages = sorted(selected_by_number.values(), key=lambda page: page["page_number"])
    sample_parts = []
    current_len = 0
    for page in selected_pages:
        heading = page["heading"] or "Untitled"
        snippet = _trim_page_for_screening(page["text"])
        page_block = f"--- PAGE {page['page_number']}: {heading} ---\n{snippet}"
        remaining = _SCREENING_SAMPLE_MAX_CHARS - current_len
        if remaining <= 0:
            break
        if len(page_block) > remaining:
            page_block = page_block[:remaining].rstrip()
        sample_parts.append(page_block)
        current_len += len(page_block) + 2

    sample_text = "\n\n".join(sample_parts).strip()
    return sample_text, {
        "sampled_pages": [page["page_number"] for page in selected_pages],
        "skipped_front_matter_pages": len(
            [page for page in page_infos if page["front_matter"]]
        ),
        "candidate_pages": len(candidate_pages),
        "sample_strategy": "representative_pages",
    }


def _coerce_bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _should_reject_screening_result(result: dict) -> bool:
    is_tsd = _coerce_bool(result.get("is_tsd", True), default=True)
    try:
        confidence = float(result.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0

    if is_tsd or confidence < _SCREENING_CONFIDENCE_THRESHOLD:
        return False

    document_type = str(result.get("document_type", "") or "").lower()
    reasoning = str(result.get("reasoning", "") or "").lower()
    rejection_text = f"{document_type} {reasoning}"
    return _pattern_count(rejection_text, _CLEAR_NON_TSD_PATTERNS) > 0


class IngestionService:
    """
    Orchestrates TSD document ingestion and validation.
    Pure input → output, no side effects beyond artifact parsing.
    """

    def __init__(self, ingestor: Optional[TSDIngestor] = None) -> None:
        """
        Args:
            ingestor: Injected TSDIngestor instance (for testing).
                     Defaults to a new instance if None.
        """
        self.ingestor = ingestor or TSDIngestor()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def ingest(self, review: Review) -> Optional[IngestionOutput]:
        """
        Ingests the TSD document and screens it.

        Args:
            review: The Review with a linked Design (TSD).

        Returns:
            IngestionOutput with TSDDocument and screening result,
            or None on fatal error.
        """
        self.logger.info(
            "IngestionService.ingest: [ENTRY] review_id=%s design='%s'",
            review.id,
            review.design.name,
        )

        try:
            # Step 1: Parse the PDF
            tsd_document = self._ingest_tsd(review)
            if tsd_document is None:
                self.logger.error(
                    "IngestionService.ingest: [FATAL] ingestion failed for review_id=%s",
                    review.id,
                )
                return None

            # Step 2: Screen the document
            is_valid_tsd, screening_msg = self._screen_tsd(tsd_document)

            output = IngestionOutput(
                tsd_document=tsd_document,
                is_valid_tsd=is_valid_tsd,
                screening_message=screening_msg,
            )

            self.logger.info(
                "IngestionService.ingest: [SUCCESS] review_id=%s is_valid_tsd=%s",
                review.id,
                is_valid_tsd,
            )
            return output

        except Exception as exc:
            self.logger.exception(
                "IngestionService.ingest: [FATAL] review_id=%s: %s",
                review.id,
                exc,
            )
            return None

    def ingest_design(self, design: Design) -> Optional[IngestionOutput]:
        self.logger.info(
            "IngestionService.ingest_design: [ENTRY] design_id=%s design='%s'",
            design.id,
            design.name,
        )
        try:
            tsd_document = self._ingest_design_document(
                design_id=design.id,
                document_name=design.name,
                document_field=design.document,
            )
            if tsd_document is None:
                self.logger.error(
                    "IngestionService.ingest_design: [FATAL] ingestion failed for design_id=%s",
                    design.id,
                )
                return None

            is_valid_tsd, screening_msg = self._screen_tsd(tsd_document)
            output = IngestionOutput(
                tsd_document=tsd_document,
                is_valid_tsd=is_valid_tsd,
                screening_message=screening_msg,
            )
            self.logger.info(
                "IngestionService.ingest_design: [SUCCESS] design_id=%s is_valid_tsd=%s",
                design.id,
                is_valid_tsd,
            )
            return output
        except Exception as exc:
            self.logger.exception(
                "IngestionService.ingest_design: [FATAL] design_id=%s: %s",
                design.id,
                exc,
            )
            return None

    def _ingest_tsd(self, review: Review) -> Optional[TSDDocument]:
        """
        Resolves the TSD file path and runs TSDIngestor.ingest().
        Returns None on any failure.
        """
        try:
            design = review.design
            document_field = design.document

            self.logger.info(
                "IngestionService._ingest_tsd: parsing review_id=%s design='%s'",
                review.id,
                design.name,
            )

            tsd_document = self._ingest_design_document(
                design_id=review.id,
                document_name=design.name,
                document_field=document_field,
            )

            self.logger.info(
                "IngestionService._ingest_tsd: [SUCCESS] parsed '%s' — "
                "%d page(s), %d block(s), %d diagram(s)",
                design.name,
                tsd_document.total_pages,
                tsd_document.total_text_blocks,
                tsd_document.total_diagrams,
            )
            return tsd_document

        except Exception as exc:
            self.logger.exception(
                "IngestionService._ingest_tsd: failed for review_id=%s: %s",
                review.id,
                exc,
            )
            return None

    def _ingest_design_document(
        self,
        *,
        design_id: int,
        document_name: str,
        document_field: str,
    ) -> Optional[TSDDocument]:
        with get_local_file_path(document_field) as tsd_path:
            return self.ingestor.ingest(
                file_path=tsd_path,
                document_name=document_name,
            )

    def _screen_tsd(self, tsd_document: TSDDocument) -> tuple[bool, Optional[str]]:
        """
        Calls build_tsd_screening_prompt() to verify the document is a TSD.

        Returns:
            (is_valid_tsd: bool, screening_message: Optional[str])
            On API failure: (True, None) — screening is best-effort, never blocks.
        """
        sample_text, sample_metadata = _build_tsd_screening_sample(tsd_document)

        self.logger.info(
            "IngestionService._screen_tsd: screening '%s' sample_len=%d chars "
            "sampled_pages=%s skipped_front_matter=%d strategy=%s",
            tsd_document.document_name,
            len(sample_text),
            sample_metadata.get("sampled_pages", []),
            sample_metadata.get("skipped_front_matter_pages", 0),
            sample_metadata.get("sample_strategy"),
        )

        if not sample_text.strip():
            self.logger.warning(
                "IngestionService._screen_tsd: document '%s' has no extractable text",
                tsd_document.document_name,
            )
            return True, None

        prompt = build_tsd_screening_prompt(sample_text)

        try:
            response = chat_completion(
                messages=[
                    {"role": "system", "content": TSD_SCREENING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                component="tsd_ingestion",
                temperature=_SCREENING_TEMPERATURE,
                max_tokens=_SCREENING_MAX_TOKENS,
                response_format={"type": "json_object"},
            )

            if response.error or not response.content:
                self.logger.warning(
                    "IngestionService._screen_tsd: LLM error — allowing document: %s",
                    response.error,
                )
                return True, None

            content_stripped = (response.content or "").strip()
            if not content_stripped:
                self.logger.warning("IngestionService._screen_tsd: Empty content returned.")
                return True, None
                
            try:
                result = json.loads(content_stripped)
            except json.JSONDecodeError:
                self.logger.warning(
                    "IngestionService._screen_tsd: JSON decode error. Attempting LLM repair."
                )
                repair_prompt = (
                    "The following JSON has syntax errors (e.g. missing commas, unescaped quotes). "
                    "Fix it and return ONLY valid JSON without any markdown or conversational text.\n\n"
                    f"{content_stripped}"
                )
                repair_resp = chat_completion(
                    messages=[{"role": "user", "content": repair_prompt}],
                    component="fallback",
                    temperature=0.0,
                    max_tokens=_SCREENING_MAX_TOKENS
                )
                if repair_resp.error:
                    self.logger.warning("IngestionService._screen_tsd: JSON repair API error — allowing document: %s", repair_resp.error)
                    return True, None
                try:
                    result = json.loads((repair_resp.content or "").strip())
                    self.logger.info("IngestionService._screen_tsd: successfully repaired JSON via LLM fallback.")
                except json.JSONDecodeError:
                    self.logger.warning("IngestionService._screen_tsd: JSON repair failed — allowing document.")
                    return True, None
                
            is_tsd = _coerce_bool(result.get("is_tsd", True), default=True)
            try:
                confidence = float(result.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            document_type = str(result.get("document_type", "") or "")
            reasoning = str(result.get("reasoning", "") or "")

            self.logger.info(
                "IngestionService._screen_tsd: is_tsd=%s confidence=%.2f document_type='%s'",
                is_tsd,
                confidence,
                document_type,
            )

            # Only reject clearly wrong uploads. Ambiguous or mixed documents continue.
            if _should_reject_screening_result(result):
                return False, reasoning

            return True, None

        except Exception as exc:
            self.logger.exception(
                "IngestionService._screen_tsd: error screening — allowing document: %s",
                exc,
            )
            return True, None
