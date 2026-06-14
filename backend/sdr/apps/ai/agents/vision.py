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
            response = self._call_llm(
                user_prompt=user_prompt,
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
            response = self._call_llm(user_prompt=user_prompt)
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
) -> Dict[str, Any]:
    verdict = str(mediator_result.get("final_verdict", VERDICT_NA)).strip().lower()

    hallucinated = critic_result.get("hallucinated_claims") or []
    if hallucinated and verdict == VERDICT_NOT_MET:
        invalidated = critic_result.get("invalidated_requirements") or []
        not_met_in_invalidated = any(
            r.get("verdict", "").lower() == VERDICT_NOT_MET for r in invalidated
        )
        if not_met_in_invalidated or len(hallucinated) >= 2:
            mediator_result["final_verdict"] = VERDICT_NA
            mediator_result["verdict_policy_source"] = "diagram_hallucinated_evidence"

    validated = critic_result.get("validated_requirements") or []
    if verdict == VERDICT_MET and not validated:
        mediator_result["final_verdict"] = VERDICT_NA
        mediator_result["verdict_policy_source"] = "diagram_met_without_validated_evidence"

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
    elif verdict == VERDICT_MET and visual_evidence_count >= 2:
        adjustment += 0.05

    adjusted = max(0.0, min(raw_confidence + adjustment, 1.0))
    if verdict == VERDICT_NA:
        adjusted = min(adjusted, 0.65)

    mediator_result["confidence"] = round(adjusted, 2)
    return mediator_result

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

