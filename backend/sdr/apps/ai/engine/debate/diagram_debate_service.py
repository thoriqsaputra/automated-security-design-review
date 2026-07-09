from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from sdr.core.config import settings
from sdr.apps.ai.agents.vision import (
    DiagramInput,
    DiagramDebateOutput,
    VisionHunterAgent,
    VisionCriticAgent,
    VisionMediatorAgent,
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
from sdr.apps.ai.tsd_processing.visual_marker import apply_visual_markers
import base64

class DiagramDebateService:
    """
    Orchestrates Hunter→Critic→Mediator debate for a single diagram.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._hunter_agent = VisionHunterAgent()
        self._critic_agent = VisionCriticAgent()
        self._mediator_agent = VisionMediatorAgent()
        self._skip_mediator_on_uphold = bool(
            getattr(settings, "AI_VISION_SKIP_MEDIATOR_ON_UPHOLD", True)
        )

    def run_diagram_debate(
        self,
        *,
        diagram: DiagramInput,
        requirements: List[Any],
        tsd_context: str = "",
        cancel_check=None,
        apply_markers: bool = True,
        skip_mediator_on_uphold: bool = False,
        agent_started_handler: Optional[Callable[[str], None]] = None,
        agent_completed_handler: Optional[Callable[..., None]] = None,
    ) -> DiagramDebateOutput:
        skip_mediator_on_uphold = skip_mediator_on_uphold or self._skip_mediator_on_uphold
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
                self.logger.warning(
                    "DiagramDebateService: cancel_check() raised for diagram_id=%s — treating as not cancelled",
                    diagram.diagram_id,
                    exc_info=True,
                )

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

        # Apply visual markers (Set-of-Mark) for better LLM grounding
        if apply_markers:
            image_bytes = apply_visual_markers(image_bytes)

            # Update the diagram object so the frontend gets the marked image
            diagram.image_b64 = base64.b64encode(image_bytes).decode("utf-8")

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
        if agent_started_handler:
            agent_started_handler("hunter")
        hunter_prompt = build_vision_hunter_prompt(
            requirements_text=compact_reqs,
            diagram_caption=caption,
            surrounding_text=surrounding,
            tsd_context=tsd_context[:2000],
        )
        hunter_result = self._hunter_agent.run_multimodal(
            user_prompt=hunter_prompt,
            image_bytes=image_bytes,
            image_format=image_format,
            system_prompt=VISION_HUNTER_SYSTEM_PROMPT,
            log_context=f"diagram_id={diagram.diagram_id} agent=hunter",
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
                self.logger.warning(
                    "DiagramDebateService: cancel_check() raised for diagram_id=%s — treating as not cancelled",
                    diagram.diagram_id,
                    exc_info=True,
                )
        if hunter_result is None:
            self.logger.error(
                "DiagramDebateService: VisionHunter failed for diagram_id=%s — "
                "falling back to inconclusive result",
                diagram.diagram_id,
            )
            hunter_result = {
                "overall_verdict": VERDICT_NA,
                "confidence": 0.0,
                "reasoning": "Hunter analysis failed to produce a parseable result; treating as inconclusive.",
                "requirement_assessments": [],
                "diagram_scope_verdict": "uncertain",
                "diagram_scope_reasoning": "Hunter call failed before scope could be assessed.",
            }

        hunter_verdict = str(
            hunter_result.get("overall_verdict", VERDICT_NA)
        ).strip().lower()
        if hunter_verdict not in {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}:
            hunter_result["overall_verdict"] = VERDICT_NA
        hunter_result["diagram_scope_verdict"] = self._normalize_scope(
            hunter_result.get("diagram_scope_verdict")
        )
        output.hunter_result = hunter_result
        if agent_completed_handler:
            agent_completed_handler("hunter", self._reasoning_content(hunter_result))

        self.logger.info(
            "DiagramDebateService: VisionCritic diagram_id=%s",
            diagram.diagram_id,
        )
        if agent_started_handler:
            agent_started_handler("critic")
        critic_prompt = build_vision_critic_debate_prompt(
            requirements_with_hints=detailed_reqs,
            hunter_result=hunter_result,
            diagram_caption=caption,
            surrounding_text=surrounding,
        )
        critic_result = self._critic_agent.run_multimodal(
            user_prompt=critic_prompt,
            image_bytes=image_bytes,
            image_format=image_format,
            system_prompt=VISION_CRITIC_DEBATE_SYSTEM_PROMPT,
            log_context=f"diagram_id={diagram.diagram_id} agent=critic",
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
                self.logger.warning(
                    "DiagramDebateService: cancel_check() raised for diagram_id=%s — treating as not cancelled",
                    diagram.diagram_id,
                    exc_info=True,
                )
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
        if agent_completed_handler:
            # Diagrams only have uphold/overturn (no partial) — normalize to the
            # same uppercase vocabulary text findings use so a single frontend
            # filter covers both finding types.
            normalized_outcome = str(critic_result.get("outcome") or "uphold").strip().upper()
            agent_completed_handler(
                "critic",
                self._reasoning_content(critic_result),
                critic_outcome=normalized_outcome,
                requires_rebuttal=normalized_outcome == "OVERTURN",
            )

        self.logger.info(
            "DiagramDebateService: VisionMediator diagram_id=%s critic_outcome=%s",
            diagram.diagram_id,
            critic_outcome,
        )
        if skip_mediator_on_uphold and critic_outcome != "overturn":
            self.logger.info(
                "DiagramDebateService: skipping mediator LLM for diagram_id=%s because Critic upheld",
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
                "verdict_policy_source": "critic_upheld_skip_mediator",
            }
            if agent_started_handler:
                agent_started_handler("mediator")
            if agent_completed_handler:
                agent_completed_handler("mediator", self._reasoning_content(mediator_result))
        else:
            if agent_started_handler:
                agent_started_handler("mediator")
            mediator_prompt = build_vision_mediator_debate_prompt(
                hunter_result=hunter_result,
                critic_result=critic_result,
            )
            self.logger.info(
                "DiagramDebateService: Critic overturned Hunter — Mediator will see the diagram image diagram_id=%s",
                diagram.diagram_id,
            )
            mediator_result = self._mediator_agent.run_multimodal(
                user_prompt=mediator_prompt,
                image_bytes=image_bytes,
                image_format=image_format,
                system_prompt=VISION_MEDIATOR_DEBATE_SYSTEM_PROMPT,
                log_context=f"diagram_id={diagram.diagram_id} agent=mediator",
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
                self.logger.warning(
                    "DiagramDebateService: cancel_check() raised for diagram_id=%s — treating as not cancelled",
                    diagram.diagram_id,
                    exc_info=True,
                )
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
        if agent_completed_handler and not (skip_mediator_on_uphold and critic_outcome != "overturn"):
            agent_completed_handler("mediator", self._reasoning_content(mediator_result))
        self.logger.info(
            "DiagramDebateService: COMPLETE diagram_id=%s verdict=%s confidence=%.2f",
            diagram.diagram_id,
            mediator_result.get("final_verdict"),
            mediator_result.get("confidence", 0.0),
        )
        return output

    def run_diagram_debate_voted(
        self,
        *,
        diagram: DiagramInput,
        requirements: List[Any],
        tsd_context: str = "",
        votes: int = 3,
        **kwargs: Any,
    ) -> DiagramDebateOutput:
        """
        Runs `run_diagram_debate` `votes` times and returns the run whose
        final_verdict matches the majority — a self-consistency pass to
        suppress per-call LLM non-determinism (confirmed present even at
        temperature=0 with a fixed seed, most likely from OpenRouter's
        automatic provider routing). Ties break toward the run with the
        highest mediator confidence.
        """
        if votes <= 1:
            return self.run_diagram_debate(
                diagram=diagram, requirements=requirements, tsd_context=tsd_context, **kwargs
            )

        outputs: List[DiagramDebateOutput] = []
        for i in range(votes):
            outputs.append(
                self.run_diagram_debate(
                    diagram=diagram, requirements=requirements, tsd_context=tsd_context, **kwargs
                )
            )

        tally: Dict[str, int] = {}
        for output in outputs:
            verdict = str((output.mediator_result or {}).get("final_verdict", VERDICT_NA)).strip().lower()
            tally[verdict] = tally.get(verdict, 0) + 1

        majority_verdict = max(tally.items(), key=lambda kv: kv[1])[0]
        candidates = [
            output for output in outputs
            if str((output.mediator_result or {}).get("final_verdict", VERDICT_NA)).strip().lower() == majority_verdict
        ]
        winner = max(candidates, key=lambda output: float((output.mediator_result or {}).get("confidence", 0.0) or 0.0))

        winner.mediator_result["self_consistency"] = {
            "votes": votes,
            "tally": tally,
            "agreement_rate": round(tally[majority_verdict] / votes, 4),
            # Secondary metrics (marker utilization, invalid-citation rate) are
            # about evidence quality, not the final verdict — reporting them
            # only from the single winning run would understate their sample
            # size under voting, so callers can aggregate over every run here.
            "all_hunter_results": [output.hunter_result for output in outputs],
            "all_critic_results": [output.critic_result for output in outputs],
        }
        self.logger.info(
            "DiagramDebateService: self-consistency vote diagram_id=%s tally=%s winner=%s",
            diagram.diagram_id,
            tally,
            majority_verdict,
        )
        return winner

    @staticmethod
    def _normalize_scope(value: Any) -> str:
        normalized = str(value or "uncertain").strip().lower()
        if normalized in {"architecture_relevant", "non_architecture", "uncertain"}:
            return normalized
        return "uncertain"

    @staticmethod
    def _reasoning_content(result: Any) -> str:
        """Chain-of-thought first: prefer cot_trace over the final-answer
        logic_summary/reasoning, matching the text debate pipeline's preference."""
        if not isinstance(result, dict):
            return ""
        return str(
            result.get("cot_trace") or result.get("logic_summary") or result.get("reasoning") or ""
        ).strip()


__all__ = ["DiagramDebateService"]
