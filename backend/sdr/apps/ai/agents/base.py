from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from sdr.apps.ai.client import AIResponse, chat_completion
from sdr.apps.ai.utils.parsing import strip_markdown_code_blocks

logger = logging.getLogger(__name__)


VERDICT_MET = "met"
VERDICT_NOT_MET = "not_met"
VERDICT_NA = "na"
VERDICT_PARTIAL = "partial"
VALID_VERDICTS = {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}
VALID_INTERNAL_VERDICTS = {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA, VERDICT_PARTIAL}

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
    valid_citations: List[Citation] = field(default_factory=list)
    invalid_citation_ids: List[str] = field(default_factory=list)
    decision: str = "uphold"
    weak_evidence: List[str] = field(default_factory=list)
    missed_evidence: List[str] = field(default_factory=list)
    objections: List[str] = field(default_factory=list)
    requires_rebuttal: bool = False


@dataclass
class MediatorResult(AgentReasoningMixin):
    final_verdict: str = VERDICT_NOT_MET
    confidence: float = 0.0
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
    model_component: str = "orchestrator"
    max_tokens: int = 2048
    temperature: float = 0.05

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _call_llm(
        self,
        user_prompt: str,
        image_b64: Optional[bytes] = None,
        image_format: str = "png",
        top_p: Optional[float] = None,
    ) -> AIResponse:
        response = chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.model,
            component=self.model_component,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            image_bytes=image_b64,
            image_format=image_format,
            top_p=top_p,
        )
        self.logger.info(
            "%s._call_llm: output_chars=%d max_tokens=%d error=%s",
            self.__class__.__name__,
            len(response.content or ""),
            self.max_tokens,
            bool(response.error),
        )
        return response

    def _parse_json_response(self, response: AIResponse) -> Optional[Dict[str, Any]]:
        content = strip_markdown_code_blocks(response.content or "")
        if not content.strip():
            self.logger.warning(
                "%s._parse_json_response: empty response content.",
                self.__class__.__name__,
            )
            return None

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            self.logger.warning(
                "%s._parse_json_response: json decode failed at pos=%s: %s. Attempting repair.",
                self.__class__.__name__,
                exc.pos,
                exc,
            )
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
                self.logger.warning("%s._parse_json_response: JSON repair API error.", self.__class__.__name__)
                return None
            try:
                parsed = json.loads(strip_markdown_code_blocks(repair_resp.content or ""))
                self.logger.info("%s._parse_json_response: successfully repaired JSON via LLM fallback.", self.__class__.__name__)
            except json.JSONDecodeError:
                self.logger.warning("%s._parse_json_response: JSON repair failed.", self.__class__.__name__)
                return None

        if not isinstance(parsed, dict):
            self.logger.warning(
                "%s._parse_json_response: expected dict, got %s.",
                self.__class__.__name__,
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
            if not isinstance(item, dict):
                continue
            block_id = str(item.get("block_id") or "").strip()
            if not block_id or block_id in seen:
                continue

            bbox = item.get("bbox") or {}
            try:
                page_number = int(item.get("page_number") or 0)
            except (TypeError, ValueError):
                page_number = 0

            citations.append(
                Citation(
                    block_id=block_id,
                    page_number=page_number,
                    quoted_text=str(item.get("quoted_text") or "").strip(),
                    bbox_x0=self._safe_float(bbox.get("x0")),
                    bbox_y0=self._safe_float(bbox.get("y0")),
                    bbox_x1=self._safe_float(bbox.get("x1")),
                    bbox_y1=self._safe_float(bbox.get("y1")),
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

    def _hunter_error(self, message: str, raw: Optional[str] = None) -> HunterResult:
        return HunterResult(
            verdict=VERDICT_NOT_MET,
            confidence=0.0,
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
            revised_verdict=VERDICT_NOT_MET,
            revised_confidence=0.0,
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


def sanitize_hunter_handoff(result: HunterResult) -> Dict[str, Any]:
        return {
            "verdict": result.verdict,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "logic_summary": result.sanitized_logic_summary(),
            "assumptions": list(result.assumptions),
            "evidence_found": result.evidence_found,
            "citations": [citation.to_dict() for citation in result.citations],
            "checked_context": result.checked_context,
            "evidence_quotes": list(result.evidence_quotes),
            "evidence_assessment": result.evidence_assessment,
            "error": result.error,
        }


def sanitize_critic_handoff(result: CriticResult) -> Dict[str, Any]:
    return {
        "outcome": result.outcome,
        "revised_verdict": result.revised_verdict,
        "revised_confidence": result.revised_confidence,
        "reasoning": result.reasoning,
        "logic_summary": result.sanitized_logic_summary(),
        "assumptions": list(result.assumptions),
        "valid_citations": [citation.to_dict() for citation in result.valid_citations],
        "invalid_citation_ids": list(result.invalid_citation_ids),
        "decision": result.decision,
        "weak_evidence": list(result.weak_evidence),
        "missed_evidence": list(result.missed_evidence),
        "objections": list(result.objections),
        "requires_rebuttal": result.requires_rebuttal,
        "error": result.error,
    }


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
    "sanitize_critic_handoff",
    "sanitize_hunter_handoff",
]
