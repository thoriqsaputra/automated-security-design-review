from __future__ import annotations

import logging
import json
from typing import Dict, List, Optional

from sdr.apps.ai.prompts.agent_prompt import (
    MEDIATOR_SYSTEM_PROMPT,
    build_mediator_prompt,
)
from .base import (
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
    """
    Concrete implementation of the Mediator agent.

    The Mediator is the final arbiter in the Multi-Agent Debate pipeline.
    It is called once per security parameter after both the Hunter and
    Critic have produced their results.

    Key responsibilities:
        1. Weigh the Hunter's initial finding against the Critic's challenge.
        2. Produce a single binding final verdict (met / not_met / na).
        3. Select only Critic-verified citations as final evidence.
        4. Assign severity and remediation recommendation for not_met verdicts.
        5. Provide a concise executive-level justification.

    The MediatorResult is persisted directly to the Finding model [3] by
    analysis_service.py — this is the only agent output that touches the DB.

    The Mediator never raises — all errors are captured in MediatorResult.error
    so the pipeline can mark the Finding with an error state and continue
    to the next parameter.
    """

    system_prompt: str = MEDIATOR_SYSTEM_PROMPT
    model_component: str = "mediator"
    max_tokens: int = 2048  # Mediator output is concise by design
    temperature: float = 0.0

    def _build_user_prompt(
        self,
        parameter_text: str,
        parameter_section: str,
        contract: dict,
        hunter_result: HunterResult,
        critic_result: CriticResult,
        debate_history: Optional[List[dict]] = None,
    ) -> str:
        """
        Delegates to build_mediator_prompt() from agent_prompts.py.
        Serialises CriticResult valid_citations to dicts for the prompt.
        """
        return build_mediator_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
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
            debate_history=debate_history or [],
        )

    def run(
        self,
        parameter_text: str,
        parameter_section: str,
        contract: dict,
        hunter_result: HunterResult,
        critic_result: CriticResult,
        debate_history: Optional[List[dict]] = None,
    ) -> MediatorResult:
        """
        Executes the Mediator agent for a single security parameter.

        Pipeline:
            1.  Validate inputs.
            2.  Handle fast-path cases to avoid unnecessary LLM calls.
            3.  Build the user-turn prompt.
            4.  Call the LLM via _call_llm() from BaseAgent.
            5.  Parse the JSON response via _parse_json_response().
            6.  Extract and validate all fields with shared helpers.
            7.  Run post-parse severity/verdict consistency checks.
            8.  Reconcile final citations from Critic-verified set.
            9.  Return a fully populated MediatorResult.

        Args:
            parameter_text:    Full requirement text from CategoryParameterChild [3].
            parameter_section: Parent section title from CategoryParameterParent [3].
            hunter_result:     The HunterResult produced by HunterAgent.run().
            critic_result:     The CriticResult produced by CriticAgent.run().

        Returns:
            MediatorResult — never raises. Check .error field for failures.
        """
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

        # Fast-path A: Both agents agree with high confidence — skip LLM call.
        # Agreement threshold is set conservatively at 0.75 to avoid
        # fast-pathing ambiguous findings. Only UPHOLD (not PARTIAL/OVERTURN)
        # qualifies, and both verdicts must match exactly.
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
            contract=contract,
            hunter_result=hunter_result,
            critic_result=critic_result,
            debate_history=debate_history,
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
        response = self._call_llm(user_prompt)

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
            )

        # ------------------------------------------------------------------
        # 6. Extract and validate all fields
        # ------------------------------------------------------------------
        raw_final_verdict = self._validate_internal_verdict(
            parsed.get("final_verdict"),
            fallback=VERDICT_NOT_MET,
        )
        final_verdict = VERDICT_NOT_MET if raw_final_verdict == VERDICT_PARTIAL else raw_final_verdict
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

    def run_batch(
        self,
        child_inputs: List[dict],
        parameter_section: str,
        hunter_results: Dict[str, HunterResult],
        critic_results: Dict[str, CriticResult],
        debate_history_by_child: Optional[Dict[str, List[dict]]] = None,
    ) -> Dict[str, MediatorResult]:
        if not child_inputs:
            return {}

        response = self._call_llm(
            self._build_batch_user_prompt(
                child_inputs=child_inputs,
                parameter_section=parameter_section,
                hunter_results=hunter_results,
                critic_results=critic_results,
                debate_history_by_child=debate_history_by_child or {},
            )
        )
        if response.error:
            msg = f"LLM call failed: {response.error}"
            self.logger.error("MediatorAgent.run_batch: %s", msg)
            return {
                str(item.get("id")): self._fallback_to_critic(
                    critic_result=critic_results.get(str(item.get("id"))) or CriticResult(),
                    error_msg=msg,
                    raw=response.content,
                )
                for item in child_inputs
                if item.get("id") is not None
            }

        parsed = self._parse_json_response(response)
        if parsed is None:
            self.logger.error("MediatorAgent.run_batch: failed to parse batch JSON.")
            return {}

        allowed_ids = {str(item.get("id")) for item in child_inputs if item.get("id") is not None}
        results: Dict[str, MediatorResult] = {}
        for item in parsed.get("results", []):
            if not isinstance(item, dict):
                continue
            child_id = str(item.get("child_id") or item.get("parameter_id") or "").strip()
            if child_id not in allowed_ids or child_id in results:
                continue
            critic_result = critic_results.get(child_id) or CriticResult()
            raw_final_verdict = self._validate_internal_verdict(
                item.get("final_verdict"),
                fallback=VERDICT_NOT_MET,
            )
            final_verdict = VERDICT_NOT_MET if raw_final_verdict == VERDICT_PARTIAL else raw_final_verdict
            confidence = self._clamp_confidence(item.get("confidence"), default=0.5)
            reasoning_fields = self._extract_reasoning_fields(
                item,
                reasoning_fallback="No reasoning provided by the Mediator agent.",
            )
            severity = self._extract_severity(item, final_verdict)
            recommendation = self._extract_recommendation(item, final_verdict)
            final_citations = self._reconcile_final_citations(
                llm_citations=self._extract_citations(
                    item.get("final_citations", []),
                    field_name="final_citations",
                ),
                critic_valid_citations=critic_result.valid_citations,
                parameter_text=str(child_id),
            )
            if final_verdict in {VERDICT_MET, VERDICT_NA}:
                severity = None
                recommendation = None
            if final_verdict == VERDICT_NOT_MET and severity is None:
                severity = "medium"
            try:
                debate_rounds_used = int(item.get("debate_rounds_used") or 1)
            except (TypeError, ValueError):
                debate_rounds_used = 1
            finding_description = self._extract_text_field(
                item,
                "finding_description",
                default="No specific finding description provided.",
                max_chars=2000,
            )

            results[child_id] = MediatorResult(
                final_verdict=final_verdict,
                confidence=confidence,
                finding_description=finding_description,
                reasoning=reasoning_fields["reasoning"],
                assumptions=reasoning_fields["assumptions"],
                logic_summary=reasoning_fields["logic_summary"],
                cot_trace=reasoning_fields["cot_trace"],
                final_citations=final_citations,
                severity=severity,
                recommendation=recommendation,
                raw_final_verdict=raw_final_verdict,
                verified_evidence=self._extract_string_list(item, "verified_evidence"),
                rejected_evidence=self._extract_string_list(item, "rejected_evidence"),
                debate_rounds_used=debate_rounds_used,
                raw_response=response.content,
                error=None,
            )
        return results

    def _build_batch_user_prompt(
        self,
        *,
        child_inputs: List[dict],
        parameter_section: str,
        hunter_results: Dict[str, HunterResult],
        critic_results: Dict[str, CriticResult],
        debate_history_by_child: Dict[str, List[dict]],
    ) -> str:
        payload = {}
        for child_id, critic in critic_results.items():
            hunter = hunter_results.get(str(child_id)) or HunterResult()
            payload[str(child_id)] = {
                "hunter": {
                    "verdict": hunter.verdict,
                    "confidence": hunter.confidence,
                    "reasoning": hunter.logic_summary or hunter.reasoning,
                },
                "critic": {
                    "outcome": critic.outcome,
                    "revised_verdict": critic.revised_verdict,
                    "revised_confidence": critic.revised_confidence,
                    "reasoning": critic.logic_summary or critic.reasoning,
                    "valid_citations": [citation.to_dict() for citation in critic.valid_citations],
                    "weak_evidence": list(critic.weak_evidence),
                    "missed_evidence": list(critic.missed_evidence),
                    "objections": list(critic.objections),
                },
                "debate_history": debate_history_by_child.get(str(child_id), []),
            }
        return f"""\
## PARENT SECURITY SECTION
Section: {parameter_section}

## CHILD PARAMETERS
{json.dumps(child_inputs, indent=2)}

## DEBATE INPUTS BY CHILD ID
{json.dumps(payload, indent=2)}

Produce one final binding verdict per child independently. Return strict JSON:

{{
  "results": [
    {{
      "child_id": "<id from CHILD PARAMETERS>",
      "assumptions": ["<assumption>", "..."],
      "logic_summary": "<concise final reasoning>",
      "final_verdict": "met" | "not_met" | "na",
      "confidence": <float 0.0-1.0>,
      "finding_description": "<factual summary of the system state regarding this requirement>",
      "reasoning": "<2-3 sentence executive justification for the verdict>",
      "verified_evidence": ["<accepted evidence>", "..."],
      "rejected_evidence": ["<insufficient or rejected evidence>", "..."],
      "debate_rounds_used": <integer>,
      "final_citations": [
        {{"block_id": "<Critic-verified block_id only>", "page_number": <integer>, "quoted_text": "<short quote>", "bbox": {{"x0": null, "y0": null, "x1": null, "y1": null}}}}
      ],
      "severity": "critical" | "high" | "medium" | "low" | "info" | null,
      "recommendation": "<remediation if not_met, else null>"
    }}
  ]
}}

Rules:
- Use child_id exactly as supplied.
- final_citations must be selected only from that child's Critic valid_citations.
- A "met" final verdict requires Critic-verified evidence for that same child.
- If evidence is only partial, final_verdict is "not_met".
"""

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
        """
        Returns a MediatorResult without an LLM call when the debate has
        a clear, high-confidence resolution that does not need arbitration.

        Fast-path conditions (ALL must be true):
            - Critic outcome is UPHOLD (not PARTIAL or OVERTURN)
            - Hunter and Critic verdicts match exactly
            - Both confidence scores are >= 0.75
            - Neither agent errored

        This eliminates LLM calls for the most common and clear-cut cases,
        reducing cost and latency significantly across large parameter sets.

        Returns None if fast-path conditions are not met.
        """
        _FAST_PATH_CONFIDENCE_THRESHOLD = 0.75

        if hunter_result.error or critic_result.error:
            return None

        if critic_result.outcome != OUTCOME_UPHOLD:
            return None

        if debate_history:
            return None

        if critic_result.requires_rebuttal or critic_result.objections:
            return None

        if hunter_result.verdict != critic_result.revised_verdict:
            return None

        if (
            hunter_result.confidence < _FAST_PATH_CONFIDENCE_THRESHOLD
            or critic_result.revised_confidence < _FAST_PATH_CONFIDENCE_THRESHOLD
        ):
            return None

        # All conditions met — produce a fast-path result
        agreed_verdict = hunter_result.verdict
        if agreed_verdict == VERDICT_NOT_MET and not critic_result.valid_citations:
            return None

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
        """
        Ensures the Mediator's final citations are a strict subset of the
        Critic's verified citations.

        The Mediator LLM may return block_ids the Critic never verified —
        a subtle hallucination where the Mediator invents new evidence not
        present in the Critic's verified set. This method filters the
        Mediator's citation list to only include block_ids that appear in
        critic_valid_citations.

        If the LLM returns no citations but the Critic verified some,
        the Critic's verified citations are used directly as the final set
        — they are already the most trustworthy evidence available.

        If neither set has citations, an empty list is returned — this is
        valid for not_met findings where no evidence exists.

        Args:
            llm_citations:          Citations returned by the Mediator LLM.
            critic_valid_citations: Citations the Critic verified as valid.
            parameter_text:         Used only in log messages.

        Returns:
            A deduplicated list of final Citation objects, ordered by
            page_number then bbox_y0 for consistent frontend rendering.
        """
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
        """
        Extracts and validates the severity field from the Mediator's
        parsed response.

        Severity must only be set for not_met findings — the post-parse
        consistency check in run() will clear it for met/na verdicts,
        but we validate here as a first line of defence.

        Args:
            parsed:        The parsed JSON dict from the LLM response.
            final_verdict: The validated final verdict string.

        Returns:
            A valid severity string from VALID_SEVERITIES, or None.
        """
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
        """
        Infers a severity level from the averaged confidence score when
        the fast-path is taken and no LLM severity assignment is available.

        Used exclusively by _try_fast_path() — the LLM is not called in
        fast-path cases, so severity must be derived heuristically.

        Mapping (not_met only):
            confidence >= 0.90  → critical  (very high certainty of gap)
            confidence >= 0.80  → high
            confidence >= 0.70  → medium
            confidence <  0.70  → low       (lower certainty)

        For met/na verdicts, always returns None — severity is not
        applicable to compliant or out-of-scope findings.

        Args:
            verdict:    The agreed fast-path verdict.
            confidence: The averaged confidence from Hunter and Critic.

        Returns:
            A valid severity string, or None for met/na verdicts.
        """
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
    # Recommendation helper
    # ------------------------------------------------------------------

    def _extract_recommendation(
        self,
        parsed: dict,
        final_verdict: str,
    ) -> Optional[str]:
        """
        Extracts the recommendation field from the Mediator's parsed response.

        Recommendations are only meaningful for not_met findings — a met
        or na finding requires no remediation action. Returns None for
        met/na verdicts regardless of what the LLM returned.

        Args:
            parsed:        The parsed JSON dict from the LLM response.
            final_verdict: The validated final verdict string.

        Returns:
            A stripped recommendation string, or None.
        """
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

    # ------------------------------------------------------------------
    # Reasoning helper
    # ------------------------------------------------------------------

    def _extract_mediator_reasoning(self, parsed: dict) -> str:
        """
        Extracts and sanitises the reasoning field from the Mediator's
        parsed response.

        The Mediator's reasoning is the executive-level justification
        stored in Finding.mediator_reasoning [review models]. It should
        be concise — 2 to 3 sentences per the prompt contract.

        Falls back to a generic message if the field is missing or empty
        so downstream code always has a non-null string to persist.

        Args:
            parsed: The parsed JSON dict from the LLM response.

        Returns:
            A non-empty reasoning string.
        """
        raw = parsed.get("reasoning") or ""
        reasoning = str(raw).strip()

        if not reasoning:
            self.logger.debug(
                "MediatorAgent._extract_mediator_reasoning: reasoning "
                "field missing or empty in LLM response."
            )
            return "No reasoning provided by the Mediator agent."

        return reasoning

    # ------------------------------------------------------------------
    # Fallback helper
    # ------------------------------------------------------------------

    def _fallback_to_critic(
        self,
        critic_result: CriticResult,
        error_msg: str,
        raw: Optional[str] = None,
    ) -> MediatorResult:
        """
        Produces a MediatorResult derived from the Critic's output when
        the Mediator's LLM call fails or returns unparseable content.

        This is preferable to returning a bare error result because:
        - The Critic's revised_verdict is already a validated, challenged
          verdict — it is more reliable than VERDICT_NOT_MET as a blind default.
        - The Critic's valid_citations have already been cross-checked —
          they are safe to use as final citations without re-verification.
        - The pipeline can persist a Finding with degraded but real data
          rather than a placeholder, giving the human reviewer something
          meaningful to act on.

        The error is recorded in MediatorResult.error so the analysis
        service can mark the finding with a degraded state indicator
        without blocking the rest of the parameter evaluation loop.

        Args:
            critic_result: The CriticResult to derive the fallback from.
            error_msg:     The error message describing why the fallback
                           was triggered.
            raw:           Optional raw LLM response content for audit.

        Returns:
            A MediatorResult populated from the Critic's output with the
            error field set to indicate degraded mode.
        """
        self.logger.warning(
            "MediatorAgent._fallback_to_critic: falling back to Critic "
            "revised_verdict='%s' revised_confidence=%.2f due to: %s",
            critic_result.revised_verdict,
            critic_result.revised_confidence,
            error_msg,
        )

        # Infer severity from Critic's confidence since the LLM didn't provide one
        severity = self._infer_severity_from_confidence(
            verdict=critic_result.revised_verdict,
            confidence=critic_result.revised_confidence,
        )

        return MediatorResult(
            final_verdict=critic_result.revised_verdict,
            confidence=critic_result.revised_confidence,
            reasoning=(
                f"[DEGRADED — Mediator LLM failed] "
                f"Verdict derived from Critic output: "
                f"{critic_result.reasoning}"
            ),
            logic_summary=(
                f"[DEGRADED — Mediator LLM failed] "
                f"Verdict derived from Critic output: "
                f"{critic_result.reasoning}"
            ),
            final_citations=list(critic_result.valid_citations),
            severity=severity,
            recommendation=None,  # Cannot generate without LLM
            raw_response=raw,
            error=error_msg,
        )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = ["MediatorAgent"]
