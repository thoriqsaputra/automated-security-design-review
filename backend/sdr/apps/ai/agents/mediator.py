from __future__ import annotations

import logging
import json
from typing import Callable, List, Optional

from sdr.apps.ai.prompts.agents import (
    MEDIATOR_RECOMMENDATION_SYSTEM_PROMPT,
    MEDIATOR_SYSTEM_PROMPT,
    build_mediator_prompt,
    build_mediator_recommendation_prompt,
)
from .base import (
    APPLICABILITY_ESTABLISHED,
    APPLICABILITY_NOT_ESTABLISHED,
    BaseAgent,
    Citation,
    CriticResult,
    HunterResult,
    MediatorResult,
    OUTCOME_UPHOLD,
    OUTCOME_OVERTURN,
    OUTCOME_PARTIAL,
    VERDICT_MET,
    VERDICT_NOT_MET,
    VERDICT_NA,
    VERDICT_PARTIAL,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    VALID_SEVERITIES,
)

logger = logging.getLogger(__name__)


class MediatorAgent(BaseAgent):
    system_prompt: str = MEDIATOR_SYSTEM_PROMPT
    model_component: str = "mediator"
    max_tokens: int = 8192  # Give the structured JSON enough room to finish cleanly
    temperature: float = 0.0
    reasoning_effort: str = "medium"

    def _build_user_prompt(
        self,
        parameter_text: str,
        parameter_section: str,
        hunter_result: HunterResult,
        critic_result: CriticResult,
        debate_history: Optional[List[dict]] = None,
        original_context_chunks: Optional[List[str]] = None,
    ) -> str:
        """
        Delegates to build_mediator_prompt() from agent_prompts.py.
        Serialises CriticResult valid_citations to dicts for the prompt.
        """
        return build_mediator_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            hunter_verdict=hunter_result.verdict,
            hunter_confidence=hunter_result.confidence,
            critic_outcome=critic_result.outcome,
            critic_revised_verdict=critic_result.revised_verdict,
            critic_reasoning=critic_result.reasoning,
            critic_valid_citations=[c.to_dict() for c in critic_result.valid_citations],
            critic_revised_confidence=critic_result.revised_confidence,
            hunter_reasoning=hunter_result.logic_summary or hunter_result.reasoning,
            critic_objections=critic_result.objections,
            critic_weak_evidence=critic_result.weak_evidence,
            critic_missed_evidence=critic_result.missed_evidence,
            hunter_assumptions=hunter_result.assumptions,
            critic_assumptions=critic_result.assumptions,
            debate_history=debate_history or [],
            original_context_chunks=original_context_chunks or [],
        )

    def run(
        self,
        parameter_text: str,
        parameter_section: str,
        hunter_result: HunterResult,
        critic_result: CriticResult,
        debate_history: Optional[List[dict]] = None,
        original_context_chunks: Optional[List[str]] = None,
        stream_handler: Optional[Callable[[str], None]] = None,
    ) -> MediatorResult:
        # ------------------------------------------------------------------
        # 1. Input validation
        # ------------------------------------------------------------------
        if not parameter_text or not parameter_text.strip():
            msg = "parameter_text is empty — cannot mediate a blank requirement."
            self.logger.error("MediatorAgent.run: %s", msg)
            return self._mediator_error(msg)

        # ------------------------------------------------------------------
        # 2. Fast-path cases
        # ------------------------------------------------------------------

        fast_path_result = self._try_fast_path(
            parameter_text=parameter_text,
            hunter_result=hunter_result,
            critic_result=critic_result,
            debate_history=debate_history,
        )
        if fast_path_result is not None:
            return fast_path_result

        # Fast-path B: Both agents errored — nothing meaningful to mediate.
        if hunter_result.error and critic_result.error:
            msg = (
                "Both Hunter and Critic returned errors — "
                "Mediator cannot produce a meaningful verdict."
            )
            self.logger.error("MediatorAgent.run: %s", msg)
            return self._mediator_error(msg)

        # ------------------------------------------------------------------
        # 3. Build prompt
        # ------------------------------------------------------------------
        user_prompt = self._build_user_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            hunter_result=hunter_result,
            critic_result=critic_result,
            debate_history=debate_history,
            original_context_chunks=original_context_chunks,
        )

        self.logger.info(
            "MediatorAgent.run: mediating Hunter verdict='%s' (%.2f) "
            "vs Critic outcome='%s' revised_verdict='%s' (%.2f) "
            "for parameter '%s...'",
            hunter_result.verdict,
            hunter_result.confidence,
            critic_result.outcome,
            critic_result.revised_verdict,
            critic_result.revised_confidence,
            parameter_text[:60],
        )

        # ------------------------------------------------------------------
        # 4. Call LLM
        # ------------------------------------------------------------------
        response = self._call_llm_with_truncation_retry(user_prompt, stream_handler=stream_handler)

        if response.error:
            msg = f"LLM call failed: {response.error}"
            self.logger.error("MediatorAgent.run: %s", msg)
            # Fall back to the Critic's revised verdict rather than
            # returning an uninformative error — the pipeline can still
            # persist a finding with degraded but usable data
            return self._fallback_to_critic(
                critic_result=critic_result,
                error_msg=msg,
                raw=response.content,
                parameter_text=parameter_text,
            )

        if self._response_was_truncated(response):
            msg = (
                "LLM response was truncated by max_tokens; "
                "falling back to Critic output without JSON repair."
            )
            self.logger.error("MediatorAgent.run: %s", msg)
            return self._fallback_to_critic(
                critic_result=critic_result,
                error_msg=msg,
                raw=response.content,
                parameter_text=parameter_text,
            )

        # ------------------------------------------------------------------
        # 5. Parse JSON response
        # ------------------------------------------------------------------
        parsed = self._parse_json_response(response)

        if parsed is None:
            msg = "Failed to parse LLM response as JSON."
            self.logger.error(
                "MediatorAgent.run: %s Raw (first 300 chars): %s",
                msg,
                (response.content or "")[:300],
            )
            return self._fallback_to_critic(
                critic_result=critic_result,
                error_msg=msg,
                raw=response.content,
                parameter_text=parameter_text,
            )

        # ------------------------------------------------------------------
        # 6. Extract and validate all fields
        # ------------------------------------------------------------------
        raw_final_verdict = self._validate_internal_verdict(
            parsed.get("final_verdict"),
            fallback=VERDICT_NOT_MET,
        )
        # "partial" from LLM: Critic found real evidence but incomplete.
        # Resolve direction-aware: if Hunter originally said met AND Critic verified
        # citations exist, keep met. Otherwise not_met (partial on a not_met base
        # means evidence gap narrowed but not closed).
        if raw_final_verdict == VERDICT_PARTIAL:
            hunter_said_met = hunter_result.verdict == VERDICT_MET
            final_verdict = VERDICT_MET if (hunter_said_met and critic_result.valid_citations) else VERDICT_NOT_MET
        else:
            final_verdict = raw_final_verdict
        confidence = self._clamp_confidence(
            parsed.get("confidence"),
            default=0.5,
        )
        reasoning_fields = self._extract_reasoning_fields(
            parsed,
            reasoning_fallback="No reasoning provided by the Mediator agent.",
        )
        severity = self._extract_severity(parsed, final_verdict)
        recommendation = self._extract_recommendation(parsed, final_verdict)
        verified_evidence = self._extract_string_list(parsed, "verified_evidence")
        rejected_evidence = self._extract_string_list(parsed, "rejected_evidence")
        try:
            debate_rounds_used = int(parsed.get("debate_rounds_used") or len(debate_history or []))
        except (TypeError, ValueError):
            debate_rounds_used = len(debate_history or [])

        # Final citations come from the Critic-verified set only —
        # the Mediator must not invent new citations the Critic didn't verify
        final_citations = self._reconcile_final_citations(
            llm_citations=self._extract_citations(
                parsed.get("final_citations", []),
                field_name="final_citations",
            ),
            critic_valid_citations=critic_result.valid_citations,
            parameter_text=parameter_text,
        )

        final_verdict, confidence, recommendation, reasoning_fields = (
            self._enforce_citation_grounding(
                final_verdict=final_verdict,
                confidence=confidence,
                final_citations=final_citations,
                recommendation=recommendation,
                reasoning_fields=reasoning_fields,
                parameter_text=parameter_text,
            )
        )
        applicability_status = self._extract_applicability_status(parsed, verdict=final_verdict)
        applicability_reason = self._extract_applicability_reason(
            parsed,
            default="No applicability reasoning provided by the Mediator agent.",
        )
        missing_expected_evidence = self._extract_missing_expected_evidence(parsed)
        if final_verdict == VERDICT_NA:
            applicability_status = APPLICABILITY_NOT_ESTABLISHED
        elif final_verdict in {VERDICT_MET, VERDICT_NOT_MET}:
            applicability_status = APPLICABILITY_ESTABLISHED
        if final_verdict == VERDICT_NOT_MET and not missing_expected_evidence:
            missing_expected_evidence = list(critic_result.missing_expected_evidence or [])
        if final_verdict != VERDICT_NOT_MET:
            missing_expected_evidence = []

        # ------------------------------------------------------------------
        # 7. Post-parse severity / verdict consistency checks
        # ------------------------------------------------------------------

        # Severity must be null for met/na findings — enforce this regardless
        # of what the LLM returned
        if final_verdict in {VERDICT_MET, VERDICT_NA} and severity is not None:
            self.logger.warning(
                "MediatorAgent.run: final_verdict='%s' but severity='%s' "
                "was set — clearing severity. Parameter: '%s...'",
                final_verdict,
                severity,
                parameter_text[:60],
            )
            severity = None

        # Recommendation must be null for met/na findings
        if final_verdict in {VERDICT_MET, VERDICT_NA} and recommendation:
            self.logger.debug(
                "MediatorAgent.run: clearing recommendation for "
                "final_verdict='%s'. Parameter: '%s...'",
                final_verdict,
                parameter_text[:60],
            )
            recommendation = None

        # Warn if not_met has no severity — Mediator should always assign one
        if final_verdict == VERDICT_NOT_MET and severity is None:
            self.logger.warning(
                "MediatorAgent.run: final_verdict='not_met' but no severity "
                "assigned for parameter '%s...' — defaulting to 'medium'.",
                parameter_text[:60],
            )
            severity = "medium"

        # ------------------------------------------------------------------
        # 8. Log final decision
        # ------------------------------------------------------------------
        self.logger.info(
            "MediatorAgent.run: FINAL verdict='%s' confidence=%.2f "
            "severity=%s citations=%d for parameter '%s...'",
            final_verdict,
            confidence,
            severity,
            len(final_citations),
            parameter_text[:60],
        )

        # Log critical findings explicitly — these need human attention
        if severity == SEVERITY_CRITICAL:
            self.logger.warning(
                "MediatorAgent.run: CRITICAL finding produced for "
                "parameter '%s...' — review requires immediate attention.",
                parameter_text[:60],
            )

        # ------------------------------------------------------------------
        # 9. Return
        # ------------------------------------------------------------------
        finding_description = self._extract_text_field(
            parsed,
            "finding_description",
            default="No specific finding description provided.",
            max_chars=2000,
        )

        return MediatorResult(
            final_verdict=final_verdict,
            confidence=confidence,
            applicability_status=applicability_status,
            applicability_reason=applicability_reason,
            missing_expected_evidence=missing_expected_evidence,
            finding_description=finding_description,
            reasoning=reasoning_fields["reasoning"],
            assumptions=reasoning_fields["assumptions"],
            logic_summary=reasoning_fields["logic_summary"],
            cot_trace=reasoning_fields["cot_trace"],
            final_citations=final_citations,
            severity=severity,
            recommendation=recommendation,
            raw_final_verdict=raw_final_verdict,
            verified_evidence=verified_evidence,
            rejected_evidence=rejected_evidence,
            debate_rounds_used=debate_rounds_used,
            raw_response=response.content,
            error=None,
        )

    # ------------------------------------------------------------------
    # Fast-path logic
    # ------------------------------------------------------------------

    def _try_fast_path(
        self,
        parameter_text: str,
        hunter_result: HunterResult,
        critic_result: CriticResult,
        debate_history: Optional[List[dict]] = None,
    ) -> Optional[MediatorResult]:
        _FAST_PATH_CONFIDENCE_THRESHOLD = 0.75

        if hunter_result.error or critic_result.error:
            return None

        if critic_result.outcome != OUTCOME_UPHOLD:
            return None

        final_rebuttal_converged = (
            bool(debate_history)
            and len(debate_history) > 1
            and critic_result.outcome == OUTCOME_UPHOLD
            and hunter_result.verdict == critic_result.revised_verdict
            and bool(critic_result.valid_citations)
        )

        if critic_result.requires_rebuttal or (
            critic_result.objections and not final_rebuttal_converged
        ):
            return None

        if hunter_result.verdict != critic_result.revised_verdict:
            return None

        agreed_verdict = hunter_result.verdict
        if agreed_verdict == VERDICT_NOT_MET and not critic_result.valid_citations:
            return None
        if agreed_verdict == VERDICT_MET and not critic_result.valid_citations:
            return None
        if agreed_verdict != VERDICT_MET and (
            hunter_result.confidence < _FAST_PATH_CONFIDENCE_THRESHOLD
            or critic_result.revised_confidence < _FAST_PATH_CONFIDENCE_THRESHOLD
        ):
            return None

        # All conditions met — produce a fast-path result

        averaged_confidence = (
            hunter_result.confidence + critic_result.revised_confidence
        ) / 2

        # For fast-path not_met, assign severity from Critic reasoning heuristic
        severity = self._infer_severity_from_confidence(
            verdict=agreed_verdict,
            confidence=averaged_confidence,
        )

        self.logger.info(
            "MediatorAgent._try_fast_path: Hunter and Critic agree "
            "verdict='%s' with averaged_confidence=%.2f — "
            "skipping LLM call for parameter '%s...'",
            agreed_verdict,
            averaged_confidence,
            parameter_text[:60],
        )

        return MediatorResult(
            final_verdict=agreed_verdict,
            confidence=averaged_confidence,
            applicability_status=(
                APPLICABILITY_NOT_ESTABLISHED
                if agreed_verdict == VERDICT_NA
                else APPLICABILITY_ESTABLISHED
            ),
            applicability_reason=(
                "Hunter and Critic agreed that applicability was not established."
                if agreed_verdict == VERDICT_NA
                else "Hunter and Critic agreed the requirement remains applicable."
            ),
            missing_expected_evidence=list(critic_result.missing_expected_evidence or []),
            finding_description=self._build_fast_path_description(agreed_verdict),
            reasoning=self._build_fast_path_reasoning(
                agreed_verdict=agreed_verdict,
                hunter_result=hunter_result,
                critic_result=critic_result,
            ),
            logic_summary=self._build_fast_path_reasoning(
                agreed_verdict=agreed_verdict,
                hunter_result=hunter_result,
                critic_result=critic_result,
            ),
            final_citations=list(critic_result.valid_citations),
            severity=severity,
            recommendation=None,  # Fast-path does not generate recommendations
            raw_final_verdict=agreed_verdict,
            verified_evidence=[c.quoted_text for c in critic_result.valid_citations if c.quoted_text],
            rejected_evidence=[],
            debate_rounds_used=0,
            raw_response=None,
            error=None,
        )

    def _build_fast_path_reasoning(
        self,
        *,
        agreed_verdict: str,
        hunter_result: HunterResult,
        critic_result: CriticResult,
    ) -> str:
        if agreed_verdict == VERDICT_MET:
            return (
                "The cited TSD evidence was verified by the Critic and directly supports the requirement. "
                "The final verdict is met because the accepted citations provide implementation-level support."
            )
        if agreed_verdict == VERDICT_NA:
            return (
                "The retrieved TSD context does not establish that this control applies to the design scope. "
                "The final verdict is not applicable rather than a security failure."
            )
        return (
            "The requirement is applicable, but the verified TSD evidence does not show the required implementation. "
            "The final verdict is not met because the accepted review record identifies missing or insufficient control evidence."
        )

    def _build_fast_path_description(self, agreed_verdict: str) -> str:
        if agreed_verdict == VERDICT_MET:
            return "The TSD contains verified evidence that satisfies this control."
        if agreed_verdict == VERDICT_NA:
            return "This control is not applicable to the documented design scope."
        return "The TSD lacks verified evidence showing this control is implemented."

    # ------------------------------------------------------------------
    # Citation reconciliation
    # ------------------------------------------------------------------

    def _reconcile_final_citations(
        self,
        llm_citations: List[Citation],
        critic_valid_citations: List[Citation],
        parameter_text: str,
    ) -> List[Citation]:
        if not critic_valid_citations:
            # Critic verified nothing — no citations can be considered final
            if llm_citations:
                self.logger.warning(
                    "MediatorAgent._reconcile_final_citations: Mediator "
                    "returned %d citation(s) but Critic verified none — "
                    "discarding all Mediator citations for parameter '%s...'",
                    len(llm_citations),
                    parameter_text[:60],
                )
            return []

        if not llm_citations:
            # Mediator returned no citations — use Critic's verified set directly
            self.logger.debug(
                "MediatorAgent._reconcile_final_citations: Mediator returned "
                "no citations — using %d Critic-verified citation(s) for "
                "parameter '%s...'",
                len(critic_valid_citations),
                parameter_text[:60],
            )
            return list(critic_valid_citations)

        # Build a lookup set of Critic-verified block_ids for O(1) membership check
        critic_verified_ids = {c.block_id for c in critic_valid_citations}

        # Build a lookup dict of Critic citations by block_id so we can return
        # the Critic's version — which carries verified bbox and quoted_text data
        # rather than the Mediator's potentially hallucinated version
        critic_by_block_id = {c.block_id: c for c in critic_valid_citations}

        reconciled: List[Citation] = []
        discarded_ids: List[str] = []
        seen_block_ids: set = set()

        for citation in llm_citations:
            if not citation.block_id:
                continue

            if citation.block_id in seen_block_ids:
                # Deduplicate — Mediator occasionally repeats the same block_id
                continue

            if citation.block_id in critic_verified_ids:
                # Use the Critic's verified version — it has confirmed bbox data
                reconciled.append(critic_by_block_id[citation.block_id])
                seen_block_ids.add(citation.block_id)
            else:
                discarded_ids.append(citation.block_id)

        if discarded_ids:
            self.logger.warning(
                "MediatorAgent._reconcile_final_citations: discarded %d "
                "unverified citation(s) from Mediator output: %s for "
                "parameter '%s...'",
                len(discarded_ids),
                discarded_ids,
                parameter_text[:60],
            )

        if not reconciled:
            # Mediator cited only unverified block_ids — fall back to Critic set
            self.logger.warning(
                "MediatorAgent._reconcile_final_citations: all Mediator "
                "citations were discarded — falling back to %d "
                "Critic-verified citation(s) for parameter '%s...'",
                len(critic_valid_citations),
                parameter_text[:60],
            )
            return list(critic_valid_citations)

        self.logger.debug(
            "MediatorAgent._reconcile_final_citations: reconciled %d "
            "final citation(s) from %d Mediator citation(s) for "
            "parameter '%s...'",
            len(reconciled),
            len(llm_citations),
            parameter_text[:60],
        )

        # Sort by page_number then bbox_y0 for consistent top-to-bottom
        # document order in the frontend PDF viewer [review models]
        reconciled.sort(
            key=lambda c: (
                c.page_number,
                c.bbox_y0 if c.bbox_y0 is not None else 0.0,
            )
        )
        return reconciled

    # ------------------------------------------------------------------
    # Severity helpers
    # ------------------------------------------------------------------

    def _extract_severity(
        self,
        parsed: dict,
        final_verdict: str,
    ) -> Optional[str]:
        raw = parsed.get("severity")

        # Severity is only meaningful for not_met findings
        if final_verdict != VERDICT_NOT_MET:
            if raw is not None:
                self.logger.debug(
                    "MediatorAgent._extract_severity: ignoring severity='%s' "
                    "for non-not_met verdict='%s'.",
                    raw,
                    final_verdict,
                )
            return None

        return self._validate_severity(raw)

    def _infer_severity_from_confidence(
        self,
        verdict: str,
        confidence: float,
    ) -> Optional[str]:
        if verdict != VERDICT_NOT_MET:
            return None

        if confidence >= 0.90:
            return SEVERITY_CRITICAL
        elif confidence >= 0.80:
            return SEVERITY_HIGH
        elif confidence >= 0.70:
            return "medium"
        else:
            return "low"

    # ------------------------------------------------------------------
    # Grounding enforcement
    # ------------------------------------------------------------------

    def _enforce_citation_grounding(
        self,
        *,
        final_verdict: str,
        confidence: float,
        final_citations: List[Citation],
        recommendation: Optional[str],
        reasoning_fields: dict,
        parameter_text: str,
    ) -> tuple:
        """Downgrade an ungrounded 'met' (zero surviving citations) to 'not_met'.

        Unlike severity, final_verdict/final_citations are never recalculated
        downstream — they flow straight into the persisted Finding, so a
        'met' verdict with no surviving evidence must never escape here.
        """
        if final_verdict != VERDICT_MET or final_citations:
            return final_verdict, confidence, recommendation, reasoning_fields

        self.logger.warning(
            "MediatorAgent._enforce_citation_grounding: final_verdict='met' but "
            "zero final_citations survived — downgrading to 'not_met' for "
            "parameter '%s...'.",
            parameter_text[:60],
        )
        downgrade_msg = (
            "Mediator reached 'met' but no citation survived Critic verification; "
            "downgraded to not_met pending explicit grounded evidence."
        )
        reasoning_fields["logic_summary"] = downgrade_msg
        reasoning_fields["reasoning"] = downgrade_msg
        recommendation = recommendation or (
            "Provide explicit, verifiable implementation evidence (e.g. citations "
            "to configuration, code, or architecture) demonstrating that this "
            "control is met."
        )
        return VERDICT_NOT_MET, min(confidence, 0.45), recommendation, reasoning_fields

    # ------------------------------------------------------------------
    # Recommendation helper
    # ------------------------------------------------------------------

    def _extract_recommendation(
        self,
        parsed: dict,
        final_verdict: str,
    ) -> Optional[str]:
        if final_verdict != VERDICT_NOT_MET:
            return None

        raw = parsed.get("recommendation")
        if not raw or not isinstance(raw, str):
            self.logger.debug(
                "MediatorAgent._extract_recommendation: recommendation "
                "field missing or empty in LLM response."
            )
            return None

        recommendation = raw.strip()
        return recommendation if recommendation else None

    def generate_recommendation_for_not_met(
        self,
        *,
        finding_type: str,
        parameter_section: str,
        parameter_text: str,
        finding_description: str,
        reasoning: str,
        severity: Optional[str] = None,
        source: str = "persistence_fallback",
    ) -> Optional[str]:
        prompt = build_mediator_recommendation_prompt(
            finding_type=finding_type,
            parameter_section=parameter_section,
            parameter_text=parameter_text,
            finding_description=finding_description,
            reasoning=reasoning,
            severity=severity,
            source=source,
        )
        try:
            response = self._call_llm_with_truncation_retry(
                prompt,
                max_tokens=400,
            )
            if response.error or self._response_was_truncated(response):
                self.logger.warning(
                    "MediatorAgent.generate_recommendation_for_not_met: recommendation generation failed error=%s finish_reason=%s",
                    response.error,
                    getattr(response, "finish_reason", None),
                )
                return None
            parsed = self._parse_json_response(response)
            if not isinstance(parsed, dict):
                self.logger.warning(
                    "MediatorAgent.generate_recommendation_for_not_met: response was not a JSON object"
                )
                return None
            recommendation = self._extract_text_field(
                parsed,
                "recommendation",
                default="",
                max_chars=500,
            ).strip()
            return recommendation or None
        except Exception:
            self.logger.exception(
                "MediatorAgent.generate_recommendation_for_not_met: failed"
            )
            return None



    # ------------------------------------------------------------------
    # Fallback helper
    # ------------------------------------------------------------------

    def _fallback_to_critic(
        self,
        critic_result: CriticResult,
        error_msg: str,
        raw: Optional[str] = None,
        parameter_text: str = "",
    ) -> MediatorResult:
        self.logger.warning(
            "MediatorAgent._fallback_to_critic: falling back to Critic "
            "revised_verdict='%s' revised_confidence=%.2f due to: %s",
            critic_result.revised_verdict,
            critic_result.revised_confidence,
            error_msg,
        )

        final_verdict = critic_result.revised_verdict
        confidence = critic_result.revised_confidence
        final_citations = list(critic_result.valid_citations)
        degraded_msg = (
            f"[DEGRADED — Mediator LLM failed] "
            f"Verdict derived from Critic output: "
            f"{critic_result.reasoning}"
        )
        reasoning_fields = {"reasoning": degraded_msg, "logic_summary": degraded_msg}

        final_verdict, confidence, _recommendation, reasoning_fields = (
            self._enforce_citation_grounding(
                final_verdict=final_verdict,
                confidence=confidence,
                final_citations=final_citations,
                recommendation=None,
                reasoning_fields=reasoning_fields,
                parameter_text=parameter_text,
            )
        )

        # Infer severity from the (possibly downgraded) verdict/confidence
        # since the LLM didn't provide one
        severity = self._infer_severity_from_confidence(
            verdict=final_verdict,
            confidence=confidence,
        )

        return MediatorResult(
            final_verdict=final_verdict,
            confidence=confidence,
            applicability_status=critic_result.applicability_status,
            applicability_reason=critic_result.applicability_reason,
            missing_expected_evidence=list(critic_result.missing_expected_evidence or []),
            reasoning=reasoning_fields["reasoning"],
            logic_summary=reasoning_fields["logic_summary"],
            final_citations=final_citations,
            severity=severity,
            recommendation=_recommendation,
            raw_response=raw,
            error=error_msg,
        )

    def _response_was_truncated(self, response) -> bool:
        return getattr(response, "finish_reason", None) == "length"


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = ["MediatorAgent"]
