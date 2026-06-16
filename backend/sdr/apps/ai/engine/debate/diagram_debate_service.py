from __future__ import annotations

import logging
from typing import Any, List, Optional

from sdr.core.config import settings
from sdr.apps.ai.agents.vision import (
    DiagramInput,
    DiagramDebateOutput,
    VisionAgent,
    _format_requirements_compact,
    _format_requirements_with_hints,
    _apply_diagram_evidence_policy,
    _calibrate_diagram_confidence,
    VERDICT_MET,
    VERDICT_NA,
    VERDICT_NOT_MET,
)
from sdr.apps.ai.prompts.agents import (
    VISION_HUNTER_SYSTEM_PROMPT,
    VISION_CRITIC_DEBATE_SYSTEM_PROMPT,
    VISION_MEDIATOR_DEBATE_SYSTEM_PROMPT,
    build_vision_hunter_prompt,
    build_vision_critic_debate_prompt,
    build_vision_mediator_debate_prompt,
)


class DiagramDebateService:
    """
    Orchestrates Hunter→Critic→Mediator debate for a single diagram.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._agent = VisionAgent()

    def run_diagram_debate(
        self,
        *,
        diagram: DiagramInput,
        requirements: List[Any],
        tsd_context: str = "",
        cancel_check=None,
    ) -> DiagramDebateOutput:
        output = DiagramDebateOutput(
            diagram=diagram,
            requirements=requirements,
        )

        if callable(cancel_check):
            try:
                if cancel_check():
                    output.error = "Analysis was cancelled by user."
                    self.logger.warning(
                        "DiagramDebateService: cancellation detected diagram_id=%s before start",
                        diagram.diagram_id,
                    )
                    return output
            except Exception:
                pass

        if not requirements:
            output.error = "No diagram requirements provided — cannot ground debate."
            self.logger.warning(
                "DiagramDebateService: skipping diagram_id=%s — no requirements",
                diagram.diagram_id,
            )
            return output

        try:
            image_bytes = diagram.decode_image_bytes()
        except ValueError as exc:
            output.error = str(exc)
            return output

        min_bytes = int(getattr(settings, "AI_VISION_MIN_DIAGRAM_BYTES", 512))
        if len(image_bytes) < min_bytes:
            output.error = (
                f"Image too small ({len(image_bytes)} bytes < {min_bytes}) — "
                f"likely icon/logo, not architectural diagram."
            )
            return output

        compact_reqs = _format_requirements_compact(requirements)
        detailed_reqs = _format_requirements_with_hints(requirements)

        caption = diagram.caption or ""
        surrounding = diagram.surrounding_text or ""
        image_format = diagram.image_format or "png"

        self.logger.info(
            "DiagramDebateService: VisionHunter diagram_id=%s requirements=%d",
            diagram.diagram_id,
            len(requirements),
        )
        hunter_prompt = build_vision_hunter_prompt(
            requirements_text=compact_reqs,
            diagram_caption=caption,
            surrounding_text=surrounding,
            tsd_context=tsd_context[:2000],
        )
        hunter_result = self._agent.run_multimodal(
            user_prompt=hunter_prompt,
            image_bytes=image_bytes,
            image_format=image_format,
            system_prompt=VISION_HUNTER_SYSTEM_PROMPT,
        )
        if callable(cancel_check):
            try:
                if cancel_check():
                    output.error = "Analysis was cancelled by user."
                    self.logger.warning(
                        "DiagramDebateService: cancellation detected diagram_id=%s after hunter",
                        diagram.diagram_id,
                    )
                    return output
            except Exception:
                pass
        if hunter_result is None:
            output.error = f"VisionHunter failed for diagram_id={diagram.diagram_id}"
            return output

        hunter_verdict = str(
            hunter_result.get("overall_verdict", VERDICT_NA)
        ).strip().lower()
        if hunter_verdict not in {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}:
            hunter_result["overall_verdict"] = VERDICT_NA
        hunter_result["diagram_scope_verdict"] = self._normalize_scope(
            hunter_result.get("diagram_scope_verdict")
        )
        output.hunter_result = hunter_result

        self.logger.info(
            "DiagramDebateService: VisionCritic diagram_id=%s",
            diagram.diagram_id,
        )
        critic_prompt = build_vision_critic_debate_prompt(
            requirements_with_hints=detailed_reqs,
            hunter_result=hunter_result,
            diagram_caption=caption,
            surrounding_text=surrounding,
        )
        critic_result = self._agent.run_multimodal(
            user_prompt=critic_prompt,
            image_bytes=image_bytes,
            image_format=image_format,
            system_prompt=VISION_CRITIC_DEBATE_SYSTEM_PROMPT,
        )
        if callable(cancel_check):
            try:
                if cancel_check():
                    output.error = "Analysis was cancelled by user."
                    self.logger.warning(
                        "DiagramDebateService: cancellation detected diagram_id=%s after critic",
                        diagram.diagram_id,
                    )
                    return output
            except Exception:
                pass
        if critic_result is None:
            self.logger.warning(
                "DiagramDebateService: VisionCritic failed for diagram_id=%s — "
                "proceeding with Hunter result only",
                diagram.diagram_id,
            )
            critic_result = {
                "outcome": "uphold",
                "reasoning": "Critic call failed; defaulting to uphold.",
                "diagram_scope_verdict": hunter_result.get("diagram_scope_verdict", "uncertain"),
                "diagram_scope_reasoning": "Critic call failed; preserving Hunter scope classification.",
            }

        critic_outcome = str(critic_result.get("outcome", "uphold")).strip().lower()
        if critic_outcome not in {"uphold", "overturn"}:
            critic_result["outcome"] = "uphold"
        critic_result["diagram_scope_verdict"] = self._normalize_scope(
            critic_result.get("diagram_scope_verdict")
        )
        output.critic_result = critic_result

        self.logger.info(
            "DiagramDebateService: VisionMediator diagram_id=%s",
            diagram.diagram_id,
        )
        mediator_prompt = build_vision_mediator_debate_prompt(
            hunter_result=hunter_result,
            critic_result=critic_result,
        )
        mediator_result = self._agent.run_text(
            user_prompt=mediator_prompt,
            system_prompt=VISION_MEDIATOR_DEBATE_SYSTEM_PROMPT,
        )
        if callable(cancel_check):
            try:
                if cancel_check():
                    output.error = "Analysis was cancelled by user."
                    self.logger.warning(
                        "DiagramDebateService: cancellation detected diagram_id=%s after mediator",
                        diagram.diagram_id,
                    )
                    return output
            except Exception:
                pass
        if mediator_result is None:
            self.logger.warning(
                "DiagramDebateService: VisionMediator failed for diagram_id=%s — "
                "falling back to Hunter verdict",
                diagram.diagram_id,
            )
            mediator_result = {
                "final_verdict": hunter_result.get("overall_verdict", VERDICT_NA),
                "confidence": hunter_result.get("confidence", 0.5),
                "finding_description": hunter_result.get("reasoning", ""),
                "recommendation": None,
                "assessed_requirements": hunter_result.get("requirement_assessments", []),
                "diagram_scope_verdict": critic_result.get(
                    "diagram_scope_verdict",
                    hunter_result.get("diagram_scope_verdict", "uncertain"),
                ),
                "diagram_scope_reasoning": critic_result.get("diagram_scope_reasoning")
                or hunter_result.get("diagram_scope_reasoning"),
            }

        final_verdict = str(
            mediator_result.get("final_verdict", VERDICT_NA)
        ).strip().lower()
        if final_verdict not in {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}:
            mediator_result["final_verdict"] = VERDICT_NA
        mediator_result["diagram_scope_verdict"] = self._normalize_scope(
            mediator_result.get("diagram_scope_verdict")
        )

        mediator_result = _apply_diagram_evidence_policy(
            mediator_result,
            critic_result,
            hunter_result,
        )
        mediator_result = _calibrate_diagram_confidence(
            mediator_result, hunter_result, critic_result
        )

        output.mediator_result = mediator_result
        self.logger.info(
            "DiagramDebateService: COMPLETE diagram_id=%s verdict=%s confidence=%.2f",
            diagram.diagram_id,
            mediator_result.get("final_verdict"),
            mediator_result.get("confidence", 0.0),
        )
        return output

    @staticmethod
    def _normalize_scope(value: Any) -> str:
        normalized = str(value or "uncertain").strip().lower()
        if normalized in {"architecture_relevant", "non_architecture", "uncertain"}:
            return normalized
        return "uncertain"


__all__ = ["DiagramDebateService"]
