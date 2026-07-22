from __future__ import annotations

import sys
import json
import logging

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from sdr.apps.ai.client import AIResponse, chat_completion
from sdr.apps.ai.client.base import AIProvider
from sdr.apps.ai.utils.parsing import strip_markdown_code_blocks, strip_thinking_block
from sdr.core.config import settings

logger = logging.getLogger(__name__)


VERDICT_MET = "met"
VERDICT_NOT_MET = "not_met"
VERDICT_NA = "na"
VERDICT_PARTIAL = "partial"
VALID_VERDICTS = {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}
VALID_INTERNAL_VERDICTS = {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA, VERDICT_PARTIAL}
APPLICABILITY_ESTABLISHED = "established"
APPLICABILITY_NOT_ESTABLISHED = "not_established"
VALID_APPLICABILITY_STATUSES = {
    APPLICABILITY_ESTABLISHED,
    APPLICABILITY_NOT_ESTABLISHED,
}

OUTCOME_UPHOLD = "UPHOLD"
OUTCOME_OVERTURN = "OVERTURN"
OUTCOME_PARTIAL = "PARTIAL"
VALID_OUTCOMES = {OUTCOME_UPHOLD, OUTCOME_OVERTURN, OUTCOME_PARTIAL}

SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH = "high"
SEVERITY_MEDIUM = "medium"
SEVERITY_LOW = "low"
SEVERITY_INFO = "info"
VALID_SEVERITIES = {
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_INFO,
}

_MAX_ASSUMPTIONS = 8
_MAX_ASSUMPTION_CHARS = 240
_MAX_LOGIC_SUMMARY_CHARS = 2500
_MAX_COT_TRACE_CHARS = 12000


@dataclass
class Citation:
    block_id: str
    page_number: int
    quoted_text: str = ""
    bbox_x0: Optional[float] = None
    bbox_y0: Optional[float] = None
    bbox_x1: Optional[float] = None
    bbox_y1: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "page_number": self.page_number,
            "quoted_text": self.quoted_text,
            "bbox": {
                "x0": self.bbox_x0,
                "y0": self.bbox_y0,
                "x1": self.bbox_x1,
                "y1": self.bbox_y1,
            },
        }


@dataclass
class AgentReasoningMixin:
    reasoning: str = ""
    assumptions: List[str] = field(default_factory=list)
    logic_summary: str = ""
    cot_trace: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def sanitized_logic_summary(self) -> str:
        return self.logic_summary or self.reasoning


@dataclass
class HunterResult(AgentReasoningMixin):
    verdict: str = VERDICT_NOT_MET
    confidence: float = 0.0
    applicability_status: str = APPLICABILITY_ESTABLISHED
    applicability_reason: str = ""
    missing_expected_evidence: List[str] = field(default_factory=list)
    evidence_found: bool = False
    citations: List[Citation] = field(default_factory=list)
    checked_context: str = ""
    evidence_quotes: List[str] = field(default_factory=list)
    evidence_assessment: str = ""


@dataclass
class CriticResult(AgentReasoningMixin):
    outcome: str = OUTCOME_UPHOLD
    revised_verdict: str = VERDICT_NOT_MET
    revised_confidence: float = 0.0
    applicability_status: str = APPLICABILITY_ESTABLISHED
    applicability_reason: str = ""
    missing_expected_evidence: List[str] = field(default_factory=list)
    valid_citations: List[Citation] = field(default_factory=list)
    invalid_citation_ids: List[str] = field(default_factory=list)
    decision: str = "uphold"
    weak_evidence: List[str] = field(default_factory=list)
    missed_evidence: List[str] = field(default_factory=list)
    objections: List[str] = field(default_factory=list)
    requires_rebuttal: bool = False
    requirement_object: str = ""
    requirement_polarity: Optional[str] = None
    evidence_relation: Optional[str] = None
    risk_flags: List[str] = field(default_factory=list)
    clause_coverage: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MediatorResult(AgentReasoningMixin):
    final_verdict: str = VERDICT_NOT_MET
    confidence: float = 0.0
    applicability_status: str = APPLICABILITY_ESTABLISHED
    applicability_reason: str = ""
    missing_expected_evidence: List[str] = field(default_factory=list)
    finding_description: str = ""
    final_citations: List[Citation] = field(default_factory=list)
    severity: Optional[str] = None
    recommendation: Optional[str] = None
    raw_final_verdict: Optional[str] = None
    verified_evidence: List[str] = field(default_factory=list)
    rejected_evidence: List[str] = field(default_factory=list)
    debate_rounds_used: int = 0


@dataclass
class VisionResult(AgentReasoningMixin):
    verdict: str = VERDICT_NOT_MET
    confidence: float = 0.0
    visual_elements_cited: List[str] = field(default_factory=list)
    missing_controls: List[str] = field(default_factory=list)
    visual_evidence: List[Dict[str, Any]] = field(default_factory=list)
    ambiguous_elements: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    ambiguity_reason: str = ""
    architect_summary: Dict[str, Any] = field(default_factory=dict)
    auditor_reasoning: str = ""
    critic_result: Optional[Dict[str, Any]] = None


class BaseAgent:
    system_prompt: str = ""
    model: Optional[str] = None
    model_component: str = "fallback"
    max_tokens: int = 2048
    temperature: float = 0.05
    reasoning_effort: Optional[str] = None
    seed: Optional[int] = None
    request_timeout_seconds: Optional[float] = None
    request_attempts: Optional[int] = None
    transport_retries: Optional[int] = None

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _log_label(self, log_context: str = "") -> str:
        return f"{self.__class__.__name__}[{log_context}]" if log_context else self.__class__.__name__

    def _call_llm(
        self,
        user_prompt: str,
        image_b64: Optional[bytes] = None,
        image_format: str = "png",
        image_payloads: Optional[List[Dict[str, Any]]] = None,
        top_p: Optional[float] = None,
        stream_handler: Optional[Callable[[str], None]] = None,
        max_tokens: Optional[int] = None,
        log_context: str = "",
    ) -> AIResponse:
        request_kwargs = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "model": self.model,
            "component": self.model_component,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "response_format": {"type": "json_object"},
            "image_bytes": image_b64,
            "image_format": image_format,
            "image_payloads": image_payloads,
            "top_p": top_p,
        }
        reasoning_effort = self.reasoning_effort or getattr(settings, "AI_OPENROUTER_REASONING_EFFORT", "") or ""
        if reasoning_effort:
            request_kwargs["reasoning"] = {"effort": reasoning_effort}
        if self.seed is not None:
            request_kwargs["seed"] = self.seed
        if self.request_timeout_seconds is not None:
            request_kwargs["request_timeout_seconds"] = self.request_timeout_seconds
        if self.request_attempts is not None:
            request_kwargs["request_attempts"] = self.request_attempts
        if self.transport_retries is not None:
            request_kwargs["transport_retries"] = self.transport_retries
        if stream_handler:
            stream = chat_completion(stream=True, **request_kwargs)
            content_parts: List[str] = []
            error: Optional[str] = None
            if isinstance(stream, AIResponse):
                response = stream
            else:
                try:
                    for chunk in stream:
                        if not chunk:
                            continue
                        content_parts.append(chunk)
                        try:
                            stream_handler(chunk)
                        except Exception:
                            self.logger.debug("%s._call_llm: stream handler raised.", self._log_label(log_context), exc_info=True)
                except Exception as exc:
                    error = str(exc)
                response = AIResponse(
                    content="".join(content_parts),
                    model=self.model or "",
                    provider=AIProvider.NVIDIA,
                    error=error,
                )
                if error:
                    self.logger.warning("%s._call_llm: streamed request failed: %s", self._log_label(log_context), error)
        else:
            response = chat_completion(**request_kwargs)
        self.logger.info(
            "%s._call_llm: output_chars=%d max_tokens=%d finish_reason=%s error=%s",
            self._log_label(log_context),
            len(response.content or ""),
            request_kwargs["max_tokens"],
            getattr(response, "finish_reason", None),
            bool(response.error),
        )
        return response

    def _call_llm_with_truncation_retry(
        self,
        user_prompt: str,
        **kwargs: Any,
    ) -> AIResponse:
        response = self._call_llm(user_prompt=user_prompt, **kwargs)
        if response.error:
            return response

        is_truncated = getattr(response, "finish_reason", None) == "length"
        is_empty = not (response.content or "").strip()
        if not (is_truncated or is_empty):
            return response

        retry_max_tokens = self.max_tokens
        if is_truncated:
            retry_max_tokens = min(self.max_tokens * 2, 16384)
            self.logger.warning(
                "%s._call_llm_with_truncation_retry: truncated at max_tokens=%d; "
                "retrying once with max_tokens=%d.",
                self._log_label(kwargs.get("log_context", "")),
                self.max_tokens,
                retry_max_tokens,
            )
        else:
            self.logger.warning(
                "%s._call_llm_with_truncation_retry: empty response (finish_reason=%s); "
                "retrying once.",
                self._log_label(kwargs.get("log_context", "")),
                getattr(response, "finish_reason", None),
            )
        retry_kwargs = dict(kwargs)
        retry_kwargs["max_tokens"] = retry_max_tokens
        return self._call_llm(user_prompt=user_prompt, **retry_kwargs)

    def _parse_json_response(self, response: AIResponse, log_context: str = "") -> Optional[Dict[str, Any]]:
        label = self._log_label(log_context)
        if getattr(response, "finish_reason", None) == "length":
            self.logger.warning(
                "%s._parse_json_response: response truncated by max_tokens; skipping JSON repair.",
                label,
            )
            return None

        content = strip_thinking_block(response.content or "")
        content = strip_markdown_code_blocks(content)
        if not content.strip():
            self.logger.warning(
                "%s._parse_json_response: empty response content.",
                label,
            )
            return None

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError) as exc:
            err_pos = getattr(exc, "pos", "unknown")
            self.logger.warning(
                "%s._parse_json_response: json decode/value failed at pos=%s: %s. Attempting repair.",
                label,
                err_pos,
                exc,
            )
            lenient_parsed = None
            try:
                lenient_parsed = json.loads(content, strict=False)
            except (json.JSONDecodeError, ValueError):
                pass
            if lenient_parsed is not None:
                self.logger.info(
                    "%s._parse_json_response: recovered via lenient (strict=False) reparse — "
                    "response contained an unescaped control character.",
                    label,
                )
                parsed = lenient_parsed
                if not isinstance(parsed, dict):
                    self.logger.warning(
                        "%s._parse_json_response: expected dict, got %s.",
                        label,
                        type(parsed).__name__,
                    )
                    return None
                return parsed
            repair_prompt = (
                "The following JSON has syntax errors (e.g. missing commas, unescaped quotes). "
                "Fix it and return ONLY valid JSON without any markdown or conversational text.\n\n"
                f"{content}"
            )
            repair_resp = chat_completion(
                messages=[{"role": "user", "content": repair_prompt}],
                component="fallback",
                temperature=0.0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            if repair_resp.error:
                self.logger.warning("%s._parse_json_response: JSON repair API error.", label)
                return None
            try:
                clean_repair = strip_thinking_block(repair_resp.content or "")
                parsed = json.loads(strip_markdown_code_blocks(clean_repair))
                self.logger.info("%s._parse_json_response: successfully repaired JSON via LLM fallback.", label)
            except (json.JSONDecodeError, ValueError):
                self.logger.warning("%s._parse_json_response: JSON repair failed.", label)
                return None

        if not isinstance(parsed, dict):
            self.logger.warning(
                "%s._parse_json_response: expected dict, got %s.",
                label,
                type(parsed).__name__,
            )
            return None

        return parsed

    def _validate_verdict(self, value: object, fallback: str) -> str:
        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in VALID_VERDICTS:
                return candidate
        return fallback

    def _validate_internal_verdict(self, value: object, fallback: str) -> str:
        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in VALID_INTERNAL_VERDICTS:
                return candidate
        return fallback

    def _validate_severity(self, value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in VALID_SEVERITIES:
                return candidate
        self.logger.debug(
            "%s._validate_severity: invalid severity '%s'.",
            self.__class__.__name__,
            value,
        )
        return None

    def _clamp_confidence(self, value: object, default: float = 0.5) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = float(default)
        return max(0.0, min(1.0, confidence))

    def _extract_text_field(
        self,
        parsed: Dict[str, Any],
        field_name: str,
        *,
        default: str = "",
        max_chars: Optional[int] = None,
    ) -> str:
        raw = parsed.get(field_name)
        if raw is None:
            text = default
        elif isinstance(raw, list):
            text = "\n".join(str(item).strip() for item in raw if str(item).strip()) or default
        else:
            text = str(raw).strip() or default
        if max_chars is not None and len(text) > max_chars:
            self.logger.info(
                "%s._extract_text_field: truncated %s from %d to %d chars.",
                self.__class__.__name__,
                field_name,
                len(text),
                max_chars,
            )
            text = text[:max_chars].rstrip()
        return text

    def _extract_assumptions(self, parsed: Dict[str, Any]) -> List[str]:
        raw = parsed.get("assumptions", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []

        assumptions: List[str] = []
        seen = set()
        for item in raw:
            text = str(item).strip()
            if not text:
                continue
            if len(text) > _MAX_ASSUMPTION_CHARS:
                text = text[:_MAX_ASSUMPTION_CHARS].rstrip()
            if text not in seen:
                assumptions.append(text)
                seen.add(text)
            if len(assumptions) >= _MAX_ASSUMPTIONS:
                break
        return assumptions

    def _extract_reasoning_fields(
        self,
        parsed: Dict[str, Any],
        *,
        reasoning_fallback: str,
    ) -> Dict[str, Any]:
        assumptions = self._extract_assumptions(parsed)
        logic_summary = self._extract_text_field(
            parsed,
            "logic_summary",
            default=self._extract_text_field(
                parsed,
                "reasoning",
                default=reasoning_fallback,
                max_chars=_MAX_LOGIC_SUMMARY_CHARS,
            ),
            max_chars=_MAX_LOGIC_SUMMARY_CHARS,
        )
        cot_trace = self._extract_text_field(
            parsed,
            "cot_trace",
            default="",
            max_chars=_MAX_COT_TRACE_CHARS,
        )
        return {
            "assumptions": assumptions,
            "logic_summary": logic_summary,
            "reasoning": logic_summary,
            "cot_trace": cot_trace or None,
        }

    def _extract_citation_block_id(self, item: object) -> str:
        if isinstance(item, str):
            return item.strip()
        if not isinstance(item, dict):
            return ""

        for key in ("block_id", "id", "citation_id", "chunk_id"):
            value = item.get(key)
            if value is None:
                continue
            block_id = str(value).strip()
            if block_id:
                return block_id
        return ""

    def _extract_citations(
        self,
        raw: object,
        *,
        field_name: str,
    ) -> List[Citation]:
        if not isinstance(raw, list):
            return []

        citations: List[Citation] = []
        seen = set()
        for item in raw:
            block_id = self._extract_citation_block_id(item)
            if not block_id or block_id in seen:
                continue

            bbox = item.get("bbox") if isinstance(item, dict) else {}
            try:
                page_number = int(item.get("page_number") or item.get("page") or 0) if isinstance(item, dict) else 0
            except (TypeError, ValueError):
                page_number = 0

            citations.append(
                Citation(
                    block_id=block_id,
                    page_number=page_number,
                    quoted_text=str(item.get("quoted_text") or item.get("quote") or "").strip() if isinstance(item, dict) else "",
                    bbox_x0=self._safe_float(bbox.get("x0")) if isinstance(bbox, dict) else None,
                    bbox_y0=self._safe_float(bbox.get("y0")) if isinstance(bbox, dict) else None,
                    bbox_x1=self._safe_float(bbox.get("x1")) if isinstance(bbox, dict) else None,
                    bbox_y1=self._safe_float(bbox.get("y1")) if isinstance(bbox, dict) else None,
                )
            )
            seen.add(block_id)

        return citations

    def _extract_string_list(
        self,
        parsed: Dict[str, Any],
        field_name: str,
        *,
        max_items: int = 12,
        max_chars: int = 500,
    ) -> List[str]:
        raw = parsed.get(field_name, [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []

        values: List[str] = []
        seen = set()
        for item in raw:
            text = str(item).strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars].rstrip()
            if text in seen:
                continue
            values.append(text)
            seen.add(text)
            if len(values) >= max_items:
                break
        return values

    def _safe_float(self, value: object) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _extract_applicability_status(
        self,
        parsed: dict,
        *,
        verdict: str,
        default: Optional[str] = None,
    ) -> str:
        value = parsed.get("applicability_status")
        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in VALID_APPLICABILITY_STATUSES:
                return candidate
        if default in VALID_APPLICABILITY_STATUSES:
            return str(default)
        return APPLICABILITY_NOT_ESTABLISHED if verdict == VERDICT_NA else APPLICABILITY_ESTABLISHED

    def _extract_applicability_reason(self, parsed: dict, *, default: str = "") -> str:
        return self._extract_text_field(
            parsed,
            "applicability_reason",
            default=default,
            max_chars=1200,
        )

    def _extract_missing_expected_evidence(self, parsed: dict) -> List[str]:
        return self._extract_string_list(
            parsed,
            "missing_expected_evidence",
            max_items=8,
            max_chars=280,
        )

    def _hunter_error(self, message: str, raw: Optional[str] = None) -> HunterResult:
        return HunterResult(
            verdict=VERDICT_NA,
            confidence=0.0,
            applicability_status=APPLICABILITY_NOT_ESTABLISHED,
            applicability_reason="Hunter failed before applicability could be established.",
            reasoning=message,
            logic_summary=message,
            evidence_found=False,
            citations=[],
            raw_response=raw,
            error=message,
        )

    def _critic_error(self, message: str, raw: Optional[str] = None) -> CriticResult:
        return CriticResult(
            outcome=OUTCOME_UPHOLD,
            revised_verdict=VERDICT_NA,
            revised_confidence=0.0,
            applicability_status=APPLICABILITY_NOT_ESTABLISHED,
            applicability_reason="Critic failed before applicability could be established.",
            reasoning=message,
            logic_summary=message,
            valid_citations=[],
            invalid_citation_ids=[],
            raw_response=raw,
            error=message,
        )

    def _mediator_error(self, message: str, raw: Optional[str] = None) -> MediatorResult:
        return MediatorResult(
            final_verdict=VERDICT_NOT_MET,
            confidence=0.0,
            applicability_status=APPLICABILITY_ESTABLISHED,
            applicability_reason="Mediator failed; preserving conservative applicable verdict.",
            reasoning=message,
            logic_summary=message,
            final_citations=[],
            severity=None,
            recommendation=None,
            raw_response=raw,
            error=message,
        )

    def _vision_error(self, message: str, raw: Optional[str] = None) -> VisionResult:
        return VisionResult(
            verdict=VERDICT_NOT_MET,
            confidence=0.0,
            reasoning=message,
            logic_summary=message,
            visual_elements_cited=[],
            missing_controls=[],
            raw_response=raw,
            error=message,
        )
__all__ = [
    "BaseAgent",
    "Citation",
    "CriticResult",
    "HunterResult",
    "MediatorResult",
    "VisionResult",
    "OUTCOME_PARTIAL",
    "OUTCOME_OVERTURN",
    "OUTCOME_UPHOLD",
    "SEVERITY_CRITICAL",
    "SEVERITY_HIGH",
    "VALID_OUTCOMES",
    "VALID_SEVERITIES",
    "VALID_INTERNAL_VERDICTS",
    "VALID_VERDICTS",
    "VERDICT_MET",
    "VERDICT_NA",
    "VERDICT_NOT_MET",
    "VERDICT_PARTIAL",
]

