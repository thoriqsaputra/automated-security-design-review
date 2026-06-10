# apps/ai/services/ingestion_service.py
"""
Ingestion Service — Handles TSD document parsing, validation, and prep.
Responsibility: Raw document → TSDDocument + screening.
No agent logic. No database writes (except artifact cleanup).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from sdr.apps.workspace.document_processing import get_local_file_path
from sdr.apps.ai.tsd_processing.ingestor import TSDDocument, TSDIngestor
from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.prompts.analysis_prompts import (
    TSD_SCREENING_SYSTEM_PROMPT,
    build_tsd_screening_prompt,
)
from sdr.apps.reviews.models import Review
from .dto import IngestionOutput


logger = logging.getLogger(__name__)

_SCREENING_TEMPERATURE = 0.0
_SCREENING_MAX_TOKENS = 256
_SCREENING_CONFIDENCE_THRESHOLD = 0.6


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

            with get_local_file_path(document_field) as tsd_path:
                tsd_document = self.ingestor.ingest(
                    file_path=tsd_path,
                    document_name=design.name,
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

    def _screen_tsd(self, tsd_document: TSDDocument) -> tuple[bool, Optional[str]]:
        """
        Calls build_tsd_screening_prompt() to verify the document is a TSD.

        Returns:
            (is_valid_tsd: bool, screening_message: Optional[str])
            On API failure: (True, None) — screening is best-effort, never blocks.
        """
        sample_text = tsd_document.full_text[:3000]

        self.logger.info(
            "IngestionService._screen_tsd: screening '%s' sample_len=%d chars",
            tsd_document.document_name,
            len(sample_text),
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
                
            is_tsd = bool(result.get("is_tsd", True))
            confidence = float(result.get("confidence", 1.0))
            reasoning = result.get("reasoning", "")

            self.logger.info(
                "IngestionService._screen_tsd: is_tsd=%s confidence=%.2f",
                is_tsd,
                confidence,
            )

            # Only reject if confident it is NOT a TSD
            if not is_tsd and confidence >= _SCREENING_CONFIDENCE_THRESHOLD:
                return False, reasoning

            return True, None

        except Exception as exc:
            self.logger.exception(
                "IngestionService._screen_tsd: error screening — allowing document: %s",
                exc,
            )
            return True, None
