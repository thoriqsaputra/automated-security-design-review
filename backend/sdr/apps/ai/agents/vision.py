from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sdr.apps.ai.prompts.agents import (
    VISION_CRITIC_DEBATE_SYSTEM_PROMPT,
    VISION_HUNTER_SYSTEM_PROMPT,
    VISION_MEDIATOR_DEBATE_SYSTEM_PROMPT,
    build_vision_critic_debate_prompt,
    build_vision_hunter_prompt,
    build_vision_mediator_debate_prompt,
)
from .base import BaseAgent, VERDICT_MET, VERDICT_NA, VERDICT_NOT_MET
from sdr.core.config import settings

logger = logging.getLogger(__name__)
_DIAGRAM_SCOPE_VALUES = {"architecture_relevant", "non_architecture", "uncertain"}


@dataclass
class DiagramInput:
    diagram_id: str
    image_b64: str
    page_number: int
    caption: Optional[str] = None
    surrounding_text: Optional[str] = None
    image_format: str = "png"
    bbox_x0: Optional[float] = None
    bbox_y0: Optional[float] = None
    bbox_x1: Optional[float] = None
    bbox_y1: Optional[float] = None

    def is_valid(self) -> bool:
        return bool(self.diagram_id) and bool(self.image_b64) and self.page_number > 0

    def decode_image_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.image_b64)
        except Exception as exc:
            raise ValueError(
                f"DiagramInput.decode_image_bytes: failed to decode "
                f"base64 for diagram_id='{self.diagram_id}': {exc}"
            ) from exc


@dataclass
class DiagramDebateOutput:
    """Output of a single diagram debate cycle."""

    diagram: DiagramInput
    hunter_result: Dict[str, Any] = field(default_factory=dict)
    critic_result: Dict[str, Any] = field(default_factory=dict)
    mediator_result: Dict[str, Any] = field(default_factory=dict)
    requirements: List[Any] = field(default_factory=list)
    debate_rounds: int = 1
    error: Optional[str] = None


class VisionAgent(BaseAgent):
    """Thin wrapper around BaseAgent for diagram debate multimodal calls."""
    model_component: str = "vision"  
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 0.9

    def run_multimodal(
        self,
        *,
        user_prompt: str,
        image_bytes: bytes,
        image_format: str = "png",
        system_prompt: str = "",
    ) -> Optional[Dict[str, Any]]:
        original_prompt = self.system_prompt
        if system_prompt:
            self.system_prompt = system_prompt
        try:
            response = self._call_llm_with_truncation_retry(
                user_prompt,
                image_b64=image_bytes,
                image_format=image_format,
                top_p=self.top_p,
            )
        finally:
            self.system_prompt = original_prompt

        if response.error:
            self.logger.error("VisionDebateAgent LLM error: %s", response.error)
            return None

        parsed = self._parse_json_response(response)
        if parsed is None:
            self.logger.error("VisionDebateAgent: failed to parse JSON response")
        return parsed

    def run_text(
        self,
        *,
        user_prompt: str,
        system_prompt: str = "",
    ) -> Optional[Dict[str, Any]]:
        original_prompt = self.system_prompt
        if system_prompt:
            self.system_prompt = system_prompt
        try:
            response = self._call_llm_with_truncation_retry(user_prompt)
        finally:
            self.system_prompt = original_prompt

        if response.error:
            self.logger.error("VisionDebateAgent text LLM error: %s", response.error)
            return None

        parsed = self._parse_json_response(response)
        if parsed is None:
            self.logger.error("VisionDebateAgent: failed to parse JSON text response")
        return parsed


def _format_requirements_compact(requirements: List[Any]) -> str:
    lines = []
    for req in requirements:
        req_id = getattr(req, "stable_key", f"D-{getattr(req, 'ordinal', 0)}")
        text = getattr(req, "requirement_text", "")
        lines.append(f"{req.ordinal}. [{req_id}] {text}")
    return "\n".join(lines)


def _format_requirements_with_hints(requirements: List[Any]) -> str:
    lines = []
    for req in requirements:
        req_id = getattr(req, "stable_key", f"D-{getattr(req, 'ordinal', 0)}")
        text = getattr(req, "requirement_text", "")
        hint = getattr(req, "verification_hint", "")
        lines.append(f"{req.ordinal}. [{req_id}] {text}")
        if hint:
            lines.append(f"   VERIFY: {hint}")
    return "\n".join(lines)


def _apply_diagram_evidence_policy(
    mediator_result: Dict[str, Any],
    critic_result: Dict[str, Any],
    hunter_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hunter_result = dict(hunter_result or {})
    verdict = str(mediator_result.get("final_verdict", VERDICT_NA)).strip().lower()
    assessments = list(
        mediator_result.get("assessed_requirements")
        or hunter_result.get("requirement_assessments")
        or []
    )

    hunter_scope = _normalize_diagram_scope_verdict(hunter_result.get("diagram_scope_verdict"))
    critic_scope = _normalize_diagram_scope_verdict(critic_result.get("diagram_scope_verdict"))
    mediator_scope = _normalize_diagram_scope_verdict(mediator_result.get("diagram_scope_verdict"))
    scope_reasoning = (
        critic_result.get("diagram_scope_reasoning")
        if critic_scope == "non_architecture"
        else hunter_result.get("diagram_scope_reasoning")
        if hunter_scope == "non_architecture"
        else mediator_result.get("diagram_scope_reasoning")
        or critic_result.get("diagram_scope_reasoning")
        or hunter_result.get("diagram_scope_reasoning")
    )

    if "diagram_scope_verdict" not in mediator_result:
        mediator_result["diagram_scope_verdict"] = mediator_scope
    if scope_reasoning:
        mediator_result["diagram_scope_reasoning"] = scope_reasoning

    if hunter_scope == "non_architecture" or critic_scope == "non_architecture":
        mediator_result["diagram_scope_verdict"] = "non_architecture"
        if scope_reasoning:
            mediator_result["diagram_scope_reasoning"] = scope_reasoning
        assessments = _force_assessment_verdicts_na(assessments, default_summary="Image is not an architecture/security-relevant diagram.")
        mediator_result["assessed_requirements"] = assessments
        mediator_result["final_verdict"] = VERDICT_NA
        mediator_result["verdict_policy_source"] = "diagram_non_architecture_image"
        return mediator_result

    if assessments:
        assessments, downgraded = _ground_assessed_requirements_against_critic(assessments, critic_result)
        mediator_result["assessed_requirements"] = assessments
        # The topline verdict is never trusted from the Mediator's own
        # free-standing claim — it's always the deterministic worst-case of
        # the (citation-corrected) individual assessments: not_met > na > met.
        # This is exactly what the mediator prompt itself instructs, but
        # previously nothing enforced it outside the narrow "a citation got
        # downgraded" branch, so a diagram whose top-level claim disagreed
        # with its own assessments (e.g. claimed "met" while one assessment
        # was "not_met") would silently keep the wrong topline verdict.
        aggregated = _worst_case_diagram_verdict(assessments)
        all_na = all(
            str(assessment.get("verdict", "")).strip().lower() == VERDICT_NA
            for assessment in assessments
        )
        if downgraded:
            mediator_result["verdict_policy_source"] = "diagram_requirement_not_corroborated"
        elif all_na and aggregated != verdict:
            mediator_result["verdict_policy_source"] = "diagram_all_requirements_not_applicable"
        elif aggregated != verdict:
            mediator_result["verdict_policy_source"] = "diagram_verdict_aggregated_from_assessments"
        verdict = aggregated
        mediator_result["final_verdict"] = verdict

    # Evidence-quality safety net: a not_met verdict (whether from the
    # aggregate above or the raw claim, if there were no assessments at all)
    # can still be suppressed to "na" when the Critic thinks the underlying
    # claim is a hallucination artifact — this is about trustworthiness of
    # the evidence, not about counting requirements, so it stays an
    # independent layer on top of the aggregation rather than being folded
    # into it.
    hallucinated = critic_result.get("hallucinated_claims") or []
    if hallucinated and verdict == VERDICT_NOT_MET:
        invalidated = critic_result.get("invalidated_requirements") or []
        not_met_in_invalidated = any(
            r.get("verdict", "").lower() == VERDICT_NOT_MET for r in invalidated
        )
        if not_met_in_invalidated or len(hallucinated) >= 2:
            mediator_result["final_verdict"] = VERDICT_NA
            mediator_result["verdict_policy_source"] = "diagram_hallucinated_evidence"
            verdict = VERDICT_NA

    # The remaining special cases only apply when there were no individual
    # assessments to aggregate over at all (the aggregation above already
    # covers every case where `assessments` is non-empty, including "all na"
    # and "not_met claimed but nothing applicable" — both resolve to "na" via
    # _worst_case_diagram_verdict without needing their own branches here).
    validated = critic_result.get("validated_requirements") or []
    if not assessments and verdict == VERDICT_MET and not validated:
        mediator_result["final_verdict"] = VERDICT_NA
        mediator_result["verdict_policy_source"] = "diagram_met_without_validated_evidence"
        verdict = VERDICT_NA

    if not assessments and verdict == VERDICT_NOT_MET:
        mediator_result["final_verdict"] = VERDICT_NA
        mediator_result["verdict_policy_source"] = "diagram_no_applicable_security_scope"

    return mediator_result


def _calibrate_diagram_confidence(
    mediator_result: Dict[str, Any],
    hunter_result: Dict[str, Any],
    critic_result: Dict[str, Any],
) -> Dict[str, Any]:
    raw_confidence = float(mediator_result.get("confidence", 0.5) or 0.5)
    verdict = str(mediator_result.get("final_verdict", VERDICT_NA)).strip().lower()

    hunter_verdict = str(hunter_result.get("overall_verdict", "")).strip().lower()
    critic_outcome = str(critic_result.get("outcome", "")).strip().lower()

    adjustment = 0.0

    if hunter_verdict == verdict and critic_outcome == "uphold":
        adjustment += 0.10
    elif hunter_verdict == verdict:
        adjustment += 0.05

    if critic_outcome == "overturn":
        adjustment -= 0.10

    assessments = hunter_result.get("requirement_assessments") or []
    not_met_count = sum(
        1 for assessment in assessments
        if str(assessment.get("verdict", "")).lower() == VERDICT_NOT_MET
    )
    visual_evidence_count = len(hunter_result.get("visual_elements_cited") or [])

    if verdict == VERDICT_NOT_MET and visual_evidence_count == 0 and not_met_count > 0:
        adjustment -= 0.10
    elif verdict == VERDICT_MET and visual_evidence_count >= 2 and not critic_result.get("hallucinated_claims"):
        adjustment += 0.05

    adjusted = max(0.0, min(raw_confidence + adjustment, 1.0))
    if verdict == VERDICT_NA:
        adjusted = min(adjusted, 0.65)

    mediator_result["confidence"] = round(adjusted, 2)
    return mediator_result


def _ground_assessed_requirements_against_critic(
    assessments: List[Dict[str, Any]],
    critic_result: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], bool]:
    validated_ids = {
        str(r.get("requirement_id")).strip()
        for r in (critic_result.get("validated_requirements") or [])
        if isinstance(r, dict) and r.get("requirement_id")
    }
    invalidated_ids = {
        str(r.get("requirement_id")).strip()
        for r in (critic_result.get("invalidated_requirements") or [])
        if isinstance(r, dict) and r.get("requirement_id")
    }

    grounded: List[Dict[str, Any]] = []
    downgraded = False
    for assessment in assessments:
        item = dict(assessment)
        req_id = str(item.get("requirement_id", "")).strip()
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict == VERDICT_MET and (req_id not in validated_ids or req_id in invalidated_ids):
            item["verdict"] = VERDICT_NA
            item["summary"] = (
                "Downgraded: Critic did not confirm this requirement's evidence "
                "(no corroborating validated_requirements entry)."
            )
            item["reasoning"] = item["summary"]
            downgraded = True
        grounded.append(item)
    return grounded, downgraded


def _worst_case_diagram_verdict(assessments: List[Dict[str, Any]]) -> str:
    verdicts = {str(a.get("verdict", "")).strip().lower() for a in assessments}
    if VERDICT_NOT_MET in verdicts:
        return VERDICT_NOT_MET
    if VERDICT_NA in verdicts:
        return VERDICT_NA
    return VERDICT_MET if verdicts else VERDICT_NA


def _normalize_diagram_scope_verdict(value: Any) -> str:
    normalized = str(value or "uncertain").strip().lower()
    return normalized if normalized in _DIAGRAM_SCOPE_VALUES else "uncertain"


def _force_assessment_verdicts_na(
    assessments: List[Dict[str, Any]],
    *,
    default_summary: str,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for assessment in assessments:
        item = dict(assessment or {})
        item["verdict"] = VERDICT_NA
        if not item.get("summary") and item.get("reasoning"):
            item["summary"] = item.get("reasoning")
        if not item.get("summary") and item.get("visual_evidence"):
            item["summary"] = item.get("visual_evidence")
        item.setdefault("summary", default_summary)
        item.setdefault("reasoning", item["summary"])
        normalized.append(item)
    return normalized

__all__ = [
    "DiagramInput",
    "DiagramDebateOutput",
    "DiagramDebateService",
]


def __getattr__(name: str) -> Any:
    if name == "DiagramDebateService":
        from sdr.apps.ai.engine.debate.diagram_debate_service import DiagramDebateService
        return DiagramDebateService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

