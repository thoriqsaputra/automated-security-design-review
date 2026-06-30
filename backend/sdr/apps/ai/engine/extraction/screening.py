import logging
import re
from typing import Optional

from sdr.apps.ai.prompts.extraction.standards import (
    STANDARD_SCREENING_SYSTEM_PROMPT,
    build_standard_screening_prompt,
    build_json_repair_prompt,
)
from .llm_client import ExtractionLLMClient
from .normalizers import parse_json_response

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.80
_CONTROL_ID_PATTERN = re.compile(r"\b\d+\.\d+\.\d+\b|REQ-\w+-\d+|PCI-\d")
_CONTROL_ID_FAST_PASS_COUNT = 10
_SAMPLE_SLICE_CHARS = 1500
_SAMPLE_MAX_CHARS = 4500

_NON_STANDARD_KEYWORDS = (
    "whitepaper",
    "white paper",
    "advisory",
    "blog",
    "training",
    "marketing",
    "brochure",
    "legal",
    "contract",
    "resume",
    "invoice",
    "financial report",
    "presentation",
    "meeting minutes",
)


class StandardScreeningError(Exception):
    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.details = details or {}


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


def _has_enough_control_ids(text: str) -> bool:
    """Fast-path: a document dense with X.Y.Z control IDs is almost certainly a standard."""
    return len(_CONTROL_ID_PATTERN.findall(text)) >= _CONTROL_ID_FAST_PASS_COUNT


def _build_standard_screening_sample(text: str) -> str:
    """
    Take representative slices from the beginning, middle, and end of the document
    so the LLM sees actual requirement content rather than just front matter.
    """
    length = len(text)
    if length <= _SAMPLE_MAX_CHARS:
        return text

    mid_start = max(0, length // 2 - _SAMPLE_SLICE_CHARS // 2)
    parts = [
        ("beginning", text[:_SAMPLE_SLICE_CHARS]),
        ("middle", text[mid_start : mid_start + _SAMPLE_SLICE_CHARS]),
        ("end", text[max(0, length - _SAMPLE_SLICE_CHARS) :]),
    ]
    return "\n\n".join(f"--- {label} ---\n{slice_.strip()}" for label, slice_ in parts)


def _should_reject(result: dict) -> bool:
    """
    Only reject when the LLM is confident AND its explanation matches a known non-standard type.
    Uncertain rejections (unknown document_type, no keyword match) pass through — fail-open is
    safer than false rejections for edge cases.
    """
    is_standard = _coerce_bool(result.get("is_security_standard", True), default=True)
    if is_standard:
        return False

    try:
        confidence = float(result.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0

    if confidence < _CONFIDENCE_THRESHOLD:
        return False

    rejection_text = " ".join([
        str(result.get("document_type", "") or ""),
        str(result.get("reasoning", "") or ""),
    ]).lower()
    return any(kw in rejection_text for kw in _NON_STANDARD_KEYWORDS)


class StandardScreeningService:
    def __init__(self, llm_client: ExtractionLLMClient) -> None:
        self.llm_client = llm_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def screen_document(self, text_sample: str) -> None:
        if not text_sample.strip():
            return

        # Fast-path: many X.Y.Z control IDs → confirmed standard, skip LLM call.
        if _has_enough_control_ids(text_sample):
            self.logger.info("StandardScreeningService: fast-pass — sufficient control IDs found.")
            return

        sample = _build_standard_screening_sample(text_sample)

        try:
            response = self.llm_client.complete_json(
                system_prompt=STANDARD_SCREENING_SYSTEM_PROMPT,
                user_prompt=build_standard_screening_prompt(sample),
                component="standard_extraction",
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            if response.error:
                self.logger.error("StandardScreeningService: API error — allowing document: %s", response.error)
                return

            content = response.content or "{}"
            result = parse_json_response(content)

            # Attempt JSON repair if parsing returned empty / incomplete.
            if not result:
                self.logger.warning("StandardScreeningService: empty parse result — attempting JSON repair.")
                repair_response = self.llm_client.complete_json(
                    system_prompt="",
                    user_prompt=build_json_repair_prompt(content),
                    component="standard_extraction",
                    temperature=0.0,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                if repair_response.error:
                    self.logger.warning("StandardScreeningService: JSON repair API error — allowing document.")
                    return
                result = parse_json_response(repair_response.content or "{}")
                if not result:
                    self.logger.warning("StandardScreeningService: JSON repair failed — allowing document.")
                    return

            self.logger.info(
                "StandardScreeningService: is_security_standard=%s confidence=%s document_type='%s'",
                result.get("is_security_standard"),
                result.get("confidence"),
                result.get("document_type"),
            )

            if _should_reject(result):
                reasoning = result.get("reasoning", "Document does not appear to be a security standard.")
                raise StandardScreeningError(
                    f"Document rejected during AI screening: {reasoning}",
                    details=result,
                )

        except StandardScreeningError:
            raise
        except Exception as exc:
            self.logger.error(
                "StandardScreeningService: unexpected error — allowing document: %s (type=%s)",
                exc,
                type(exc).__name__,
            )
