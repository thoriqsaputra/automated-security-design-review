# apps/ai/agents/critic.py

"""
Critic Agent — second stage of the Multi-Agent TSD Security Review Pipeline.

Responsibility:
    Receives the Hunter's initial HunterResult and the same TSD context
    chunks, then challenges the finding for hallucinations, over-claims,
    and misinterpretations. Verifies every cited block_id actually contains
    the claimed evidence before passing a CriticResult to the MediatorAgent.

Bias:
    Assume the Hunter has OVER-CLAIMED compliance. Every "met" verdict
    must be verified — citations must be traceable to real content in
    the context window before they are considered valid.

Output:
    CriticResult dataclass — passed directly to MediatorAgent as input.

Dependency chain:
    agent_prompts.py          (pure prompt strings)
         ↓
    base.py                   (BaseAgent, CriticResult, HunterResult, Citation)
         ↓
    hunter.py                 (HunterAgent — produces HunterResult input)
         ↓
    critic.py                 ← YOU ARE HERE
         ↓
    mediator.py
         ↓
    analysis_service.py
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from sdr.core.config import settings

from sdr.apps.ai.prompts.agent_prompt import (
    CRITIC_SYSTEM_PROMPT,
    build_critic_prompt,
)
from .base import (
    BaseAgent,
    Citation,
    CriticResult,
    HunterResult,
    OUTCOME_UPHOLD,
    OUTCOME_OVERTURN,
    OUTCOME_PARTIAL,
    VALID_OUTCOMES,
    VERDICT_MET,
    VERDICT_NOT_MET,
)

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """
    Concrete implementation of the Critic agent.

    The Critic is called once per security parameter, immediately after
    the HunterAgent produces its HunterResult. It receives the same
    context chunks the Hunter saw and independently verifies:

        1. Do the cited block_ids actually exist in the context?
        2. Does the quoted evidence genuinely satisfy the requirement,
           or does it merely mention related concepts?
        3. Is the Hunter's verdict correct, or should it be revised?

    A CriticResult with outcome UPHOLD, OVERTURN, or PARTIAL is passed
    to the MediatorAgent which produces the final binding verdict.

    The Critic never raises — all errors are captured in CriticResult.error
    so the pipeline can continue to the Mediator with degraded input rather
    than failing entirely.
    """

    system_prompt: str = CRITIC_SYSTEM_PROMPT
    model_component: str = "critic"
    max_tokens: int = 4096
    temperature: float = 0.0

    def _build_user_prompt(
        self,
        parameter_text: str,
        parameter_section: str,
        contract: dict,
        context_chunks: List[str],
        hunter_result: HunterResult,
        cited_blocks: List[dict],
    ) -> str:
        """
        Delegates to build_critic_prompt() from agent_prompts.py.
        Serialises HunterResult citations to dicts for the prompt builder.
        """
        return build_critic_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
            context_chunks=context_chunks,
            hunter_verdict=hunter_result.verdict,
            hunter_citation_ids=[c.block_id for c in hunter_result.citations],
            cited_blocks=cited_blocks,
            hunter_confidence=hunter_result.confidence,
            hunter_reasoning=hunter_result.logic_summary or hunter_result.reasoning,
            hunter_checked_context=hunter_result.checked_context,
            hunter_evidence_quotes=hunter_result.evidence_quotes,
            hunter_evidence_assessment=hunter_result.evidence_assessment,
        )

    def run(
        self,
        parameter_text: str,
        parameter_section: str,
        contract: dict,
        context_chunks: List[str],
        hunter_result: HunterResult,
        cited_blocks: List[dict],
    ) -> CriticResult:
        """
        Executes the Critic agent for a single security parameter.

        Pipeline:
            1. Validate inputs — guard against missing parameter text.
            2. Handle degenerate Hunter results before calling the LLM.
            3. Build the user-turn prompt.
            4. Call the LLM via _call_llm() from BaseAgent.
            5. Parse the JSON response via _parse_json_response().
            6. Extract and validate all fields with shared helpers.
            7. Run post-parse citation cross-check against actual context.
            8. Return a fully populated CriticResult.

        Args:
            parameter_text:    Full requirement text from CategoryParameterChild [3].
            parameter_section: Parent section title from CategoryParameterParent [3].
            context_chunks:    Same TSD context chunks given to the Hunter.
                               Each chunk carries a positional banner prepended
                               by chunk_text_with_context() [1].
            hunter_result:     The HunterResult produced by HunterAgent.run().

        Returns:
            CriticResult — never raises. Check .error field for failures.
        """
        # ------------------------------------------------------------------
        # 1. Input validation
        # ------------------------------------------------------------------
        if not parameter_text or not parameter_text.strip():
            msg = "parameter_text is empty — cannot critique a blank requirement."
            self.logger.error("CriticAgent.run: %s", msg)
            return self._critic_error(msg)

        # ------------------------------------------------------------------
        # 2. Handle degenerate Hunter results
        #
        #    If the Hunter itself errored, the Critic has nothing meaningful
        #    to challenge. We uphold the Hunter's (already degraded) result
        #    rather than sending a broken finding to the LLM.
        # ------------------------------------------------------------------
        if hunter_result.error:
            msg = (
                f"Hunter returned an error — upholding degraded result. "
                f"Hunter error: {hunter_result.error}"
            )
            self.logger.warning("CriticAgent.run: %s", msg)
            return CriticResult(
                outcome=OUTCOME_UPHOLD,
                revised_verdict=hunter_result.verdict,
                revised_confidence=hunter_result.confidence,
                reasoning=(
                    "Critic skipped: Hunter agent returned an error. "
                    "No challenge possible — Hunter's result is passed through."
                ),
                logic_summary=(
                    "Critic skipped: Hunter agent returned an error. "
                    "No challenge possible — Hunter's result is passed through."
                ),
                valid_citations=list(hunter_result.citations),
                invalid_citation_ids=[],
                decision="uphold",
                weak_evidence=[],
                missed_evidence=[],
                objections=["Hunter errored before Critic review."],
                requires_rebuttal=False,
                raw_response=None,
                error=None,    # Critic itself did not error — this is expected flow
            )

        # ------------------------------------------------------------------
        # 3. Fast-path: Hunter returned "not_met" with no citations
        #
        #    Nothing to verify — Critic automatically upholds. This avoids
        #    an unnecessary LLM call for the most common compliant outcome.
        # ------------------------------------------------------------------
        if self._can_auto_uphold_strong_not_met(hunter_result):
            self.logger.info(
                "CriticAgent.run: Hunter returned not_met with no citations "
                "for parameter '%s...' — auto-upholding, skipping LLM call.",
                parameter_text[:60],
            )
            return CriticResult(
                outcome=OUTCOME_UPHOLD,
                revised_verdict=VERDICT_NOT_MET,
                revised_confidence=hunter_result.confidence,
                reasoning=(
                    "Critic auto-upheld: Hunter returned not_met with no "
                    "citations or evidence. No LLM call required."
                ),
                logic_summary=(
                    "Critic auto-upheld: Hunter returned not_met with no "
                    "citations or evidence. No LLM call required."
                ),
                valid_citations=[],
                invalid_citation_ids=[],
                decision="uphold",
                weak_evidence=[],
                missed_evidence=[],
                objections=[],
                requires_rebuttal=False,
                raw_response=None,
                error=None,
            )

        # ------------------------------------------------------------------
        # 4. Build prompt
        # ------------------------------------------------------------------
        user_prompt = self._build_user_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
            context_chunks=context_chunks,
            hunter_result=hunter_result,
            cited_blocks=cited_blocks,
        )

        self.logger.info(
            "CriticAgent.run: challenging Hunter verdict='%s' "
            "confidence=%.2f citations=%d for parameter '%s...'",
            hunter_result.verdict,
            hunter_result.confidence,
            len(hunter_result.citations),
            parameter_text[:60],
        )

        # ------------------------------------------------------------------
        # 5. Call LLM
        # ------------------------------------------------------------------
        response = self._call_llm(user_prompt)

        if response.error:
            msg = f"LLM call failed: {response.error}"
            self.logger.error("CriticAgent.run: %s", msg)
            # On LLM failure, fall back to upholding Hunter rather than
            # blocking the pipeline — Mediator will see a degraded Critic
            return self._critic_error(msg, raw=response.content)

        # ------------------------------------------------------------------
        # 6. Parse JSON response
        # ------------------------------------------------------------------
        parsed = self._parse_json_response(response)

        if parsed is None:
            msg = "Failed to parse LLM response as JSON."
            self.logger.error(
                "CriticAgent.run: %s Raw (first 300 chars): %s",
                msg,
                (response.content or "")[:300],
            )
            return self._critic_error(msg, raw=response.content)

        # ------------------------------------------------------------------
        # 7. Extract and validate all fields
        # ------------------------------------------------------------------
        decision = self._validate_decision(parsed.get("decision"))
        outcome = self._validate_outcome(parsed.get("outcome"))
        if "outcome" not in parsed:
            outcome = self._outcome_from_decision(decision)

        revised_verdict = self._validate_verdict(
            parsed.get("revised_verdict"),
            fallback=hunter_result.verdict,  # default to Hunter's verdict
        )
        revised_confidence = self._clamp_confidence(
            parsed.get("revised_confidence"),
            default=hunter_result.confidence,
        )
        reasoning_fields = self._extract_reasoning_fields(
            parsed,
            reasoning_fallback="No reasoning provided by the Critic agent.",
        )

        valid_citations = self._extract_citations(
            parsed.get("valid_citations", []),
            field_name="valid_citations",
        )
        invalid_citation_ids = self._extract_invalid_citation_ids(
            parsed.get("invalid_citation_ids", [])
        )
        weak_evidence = self._extract_string_list(parsed, "weak_evidence")
        missed_evidence = self._extract_string_list(parsed, "missed_evidence")
        objections = self._extract_string_list(parsed, "objections")
        requires_rebuttal = self._extract_bool(parsed.get("requires_rebuttal"), default=False)

        # ------------------------------------------------------------------
        # 8. Post-parse citation cross-check
        #
        #    The LLM may mark a citation as valid that doesn't actually
        #    appear in the context window — a subtle hallucination pattern.
        #    We do a lightweight block_id presence check against the raw
        #    context text as a second line of defence.
        # ------------------------------------------------------------------
        valid_citations, additionally_invalidated = self._cross_check_citations(
            valid_citations=valid_citations,
            context_chunks=context_chunks,
        )

        # Merge any additionally invalidated block_ids into the invalid list
        if additionally_invalidated:
            self.logger.warning(
                "CriticAgent.run: cross-check invalidated %d citation(s) "
                "that the LLM marked as valid for parameter '%s...'",
                len(additionally_invalidated),
                parameter_text[:60],
            )
            invalid_citation_ids = list(
                set(invalid_citation_ids) | set(additionally_invalidated)
            )

        # ------------------------------------------------------------------
        # 9. Consistency check — upgrade outcome if needed
        #
        #    If the Critic said UPHOLD but then gave a different revised
        #    verdict than the Hunter, force PARTIAL to avoid a contradiction
        #    that would confuse the Mediator.
        # ------------------------------------------------------------------
        if outcome == OUTCOME_UPHOLD and revised_verdict != hunter_result.verdict:
            self.logger.warning(
                "CriticAgent.run: outcome=UPHOLD but revised_verdict='%s' "
                "differs from Hunter verdict='%s' — correcting to PARTIAL.",
                revised_verdict,
                hunter_result.verdict,
            )
            outcome = OUTCOME_PARTIAL
            decision = "challenge"

        if decision == "uphold" and outcome in {OUTCOME_OVERTURN, OUTCOME_PARTIAL}:
            decision = "reject" if outcome == OUTCOME_OVERTURN else "challenge"

        # ------------------------------------------------------------------
        # 10. Log and return
        # ------------------------------------------------------------------
        self.logger.info(
            "CriticAgent.run: outcome='%s' revised_verdict='%s' "
            "revised_confidence=%.2f valid_citations=%d "
            "invalid_citation_ids=%d for parameter '%s...'",
            outcome,
            revised_verdict,
            revised_confidence,
            len(valid_citations),
            len(invalid_citation_ids),
            parameter_text[:60],
        )

        # Warn if Critic overturns a "not_met" to "met" — unusual and
        # worth surfacing for audit purposes
        if (
            outcome == OUTCOME_OVERTURN
            and hunter_result.verdict == VERDICT_NOT_MET
            and revised_verdict == VERDICT_MET
        ):
            self.logger.warning(
                "CriticAgent.run: Critic overturned not_met → met "
                "for parameter '%s...' — Mediator will scrutinise carefully.",
                parameter_text[:60],
            )

        return CriticResult(
            outcome=outcome,
            revised_verdict=revised_verdict,
            revised_confidence=revised_confidence,
            reasoning=reasoning_fields["reasoning"],
            assumptions=reasoning_fields["assumptions"],
            logic_summary=reasoning_fields["logic_summary"],
            cot_trace=reasoning_fields["cot_trace"],
            valid_citations=valid_citations,
            invalid_citation_ids=invalid_citation_ids,
            decision=decision,
            weak_evidence=weak_evidence,
            missed_evidence=missed_evidence,
            objections=objections,
            requires_rebuttal=requires_rebuttal,
            raw_response=response.content,
            error=None,
        )

    def run_batch(
        self,
        child_inputs: List[dict],
        parameter_section: str,
        context_chunks: List[str],
        hunter_results: Dict[str, HunterResult],
        cited_blocks_by_child: Optional[Dict[str, List[dict]]] = None,
    ) -> Dict[str, CriticResult]:
        if not child_inputs:
            return {}

        response = self._call_llm(
            self._build_batch_user_prompt(
                child_inputs=child_inputs,
                parameter_section=parameter_section,
                context_chunks=context_chunks,
                hunter_results=hunter_results,
                cited_blocks_by_child=cited_blocks_by_child or {},
            )
        )
        if response.error:
            msg = f"LLM call failed: {response.error}"
            self.logger.error("CriticAgent.run_batch: %s", msg)
            return {
                str(item.get("id")): self._critic_error(msg, raw=response.content)
                for item in child_inputs
                if item.get("id") is not None
            }

        parsed = self._parse_json_response(response)
        if parsed is None:
            self.logger.error("CriticAgent.run_batch: failed to parse batch JSON.")
            return {}

        allowed_ids = {str(item.get("id")) for item in child_inputs if item.get("id") is not None}
        results: Dict[str, CriticResult] = {}
        for item in parsed.get("results", []):
            if not isinstance(item, dict):
                continue
            child_id = str(item.get("child_id") or item.get("parameter_id") or "").strip()
            if child_id not in allowed_ids or child_id in results:
                continue
            hunter_result = hunter_results.get(child_id) or HunterResult()
            decision = self._validate_decision(item.get("decision"))
            outcome = self._validate_outcome(item.get("outcome"))
            if "outcome" not in item:
                outcome = self._outcome_from_decision(decision)
            revised_verdict = self._validate_verdict(
                item.get("revised_verdict"),
                fallback=hunter_result.verdict,
            )
            revised_confidence = self._clamp_confidence(
                item.get("revised_confidence"),
                default=hunter_result.confidence,
            )
            reasoning_fields = self._extract_reasoning_fields(
                item,
                reasoning_fallback="No reasoning provided by the Critic agent.",
            )
            valid_citations = self._extract_citations(
                item.get("valid_citations", []),
                field_name="valid_citations",
            )
            valid_citations, additionally_invalidated = self._cross_check_citations(
                valid_citations=valid_citations,
                context_chunks=context_chunks,
            )
            invalid_citation_ids = self._extract_invalid_citation_ids(
                item.get("invalid_citation_ids", [])
            )
            invalid_citation_ids = list(set(invalid_citation_ids) | set(additionally_invalidated))
            if outcome == OUTCOME_UPHOLD and revised_verdict != hunter_result.verdict:
                outcome = OUTCOME_PARTIAL
                decision = "challenge"
            if decision == "uphold" and outcome in {OUTCOME_OVERTURN, OUTCOME_PARTIAL}:
                decision = "reject" if outcome == OUTCOME_OVERTURN else "challenge"
            results[child_id] = CriticResult(
                outcome=outcome,
                revised_verdict=revised_verdict,
                revised_confidence=revised_confidence,
                reasoning=reasoning_fields["reasoning"],
                assumptions=reasoning_fields["assumptions"],
                logic_summary=reasoning_fields["logic_summary"],
                cot_trace=reasoning_fields["cot_trace"],
                valid_citations=valid_citations,
                invalid_citation_ids=invalid_citation_ids,
                decision=decision,
                weak_evidence=self._extract_string_list(item, "weak_evidence"),
                missed_evidence=self._extract_string_list(item, "missed_evidence"),
                objections=self._extract_string_list(item, "objections"),
                requires_rebuttal=self._extract_bool(item.get("requires_rebuttal"), default=False),
                raw_response=response.content,
                error=None,
            )
        return results

    def _build_batch_user_prompt(
        self,
        *,
        child_inputs: List[dict],
        parameter_section: str,
        context_chunks: List[str],
        hunter_results: Dict[str, HunterResult],
        cited_blocks_by_child: Dict[str, List[dict]],
    ) -> str:
        hunter_payload = {}
        for child_id, result in hunter_results.items():
            hunter_payload[str(child_id)] = {
                "verdict": result.verdict,
                "confidence": result.confidence,
                "reasoning": result.logic_summary or result.reasoning,
                "checked_context": result.checked_context,
                "evidence_quotes": list(result.evidence_quotes),
                "evidence_assessment": result.evidence_assessment,
                "citation_ids": [citation.block_id for citation in result.citations],
                "cited_blocks": cited_blocks_by_child.get(str(child_id), []),
            }
        return f"""\
## PARENT SECURITY SECTION
Section: {parameter_section}

## CHILD PARAMETERS
{json.dumps(child_inputs, indent=2)}

## ORIGINAL TSD CONTEXT
{"\n\n---\n\n".join(context_chunks)}

## HUNTER FINDINGS BY CHILD ID
{json.dumps(hunter_payload, indent=2)}

Challenge or confirm each Hunter finding independently. Return strict JSON:

{{
  "results": [
    {{
      "child_id": "<id from CHILD PARAMETERS>",
      "assumptions": ["<assumption>", "..."],
      "logic_summary": "<concise evidence verification reasoning>",
      "outcome": "UPHOLD" | "OVERTURN" | "PARTIAL",
      "decision": "uphold" | "challenge" | "reject",
      "revised_verdict": "met" | "not_met" | "na",
      "revised_confidence": <float 0.0-1.0>,
      "reasoning": "<one paragraph>",
      "weak_evidence": ["<weakness>", "..."],
      "missed_evidence": ["<missed evidence>", "..."],
      "objections": ["<specific objection>", "..."],
      "requires_rebuttal": <true | false>,
      "valid_citations": [
        {{"block_id": "<verified CONTEXT_CHUNK id>", "page_number": <integer>, "quoted_text": "<short quote>", "bbox": {{"x0": null, "y0": null, "x1": null, "y1": null}}}}
      ],
      "invalid_citation_ids": ["<block_id>", "..."]
    }}
  ]
}}

Rules:
- Use child_id exactly as supplied.
- Verify citations against ORIGINAL TSD CONTEXT for that child only.
- Do not let evidence for one child satisfy a different child.
"""

    # ------------------------------------------------------------------
    # Critic-specific private helpers
    # ------------------------------------------------------------------

    def _validate_outcome(self, value: object) -> str:
        """
        Validates the LLM returned a recognised Critic outcome string.
        Defaults to OUTCOME_UPHOLD on invalid input — safer than
        OVERTURN which would flip a Hunter verdict without justification.

        Args:
            value: The raw outcome value from the parsed JSON.

        Returns:
            A valid outcome string from VALID_OUTCOMES.
        """
        if isinstance(value, str) and value.strip().upper() in VALID_OUTCOMES:
            return value.strip().upper()

        self.logger.warning(
            "CriticAgent._validate_outcome: invalid outcome '%s' — "
            "defaulting to UPHOLD.",
            value,
        )
        return OUTCOME_UPHOLD

    def _validate_decision(self, value: object) -> str:
        if isinstance(value, str):
            candidate = value.strip().lower()
            if candidate in {"uphold", "challenge", "reject"}:
                return candidate
        return "uphold"

    def _outcome_from_decision(self, decision: str) -> str:
        if decision == "reject":
            return OUTCOME_OVERTURN
        if decision == "challenge":
            return OUTCOME_PARTIAL
        return OUTCOME_UPHOLD

    def _extract_bool(self, value: object, *, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return default

    def _can_auto_uphold_strong_not_met(self, hunter_result: HunterResult) -> bool:
        if not bool(getattr(settings, "AI_DEBATE_CRITIC_AUTO_UPHOLD_STRONG_NOT_MET", False)):
            return False
        if hunter_result.verdict != VERDICT_NOT_MET:
            return False
        if hunter_result.citations or hunter_result.evidence_found:
            return False
        if hunter_result.confidence < 0.75:
            return False
        reasoning = (hunter_result.logic_summary or hunter_result.reasoning or "").strip().lower()
        checked_context = (hunter_result.checked_context or "").strip()
        generic_fragments = {
            "no reasoning provided",
            "auto-upheld",
            "default",
            "nothing to verify",
        }
        if not checked_context or len(reasoning) < 80:
            return False
        if any(fragment in reasoning for fragment in generic_fragments):
            return False
        return True

    def _extract_invalid_citation_ids(self, raw: object) -> List[str]:
        """
        Extracts the list of block_ids the LLM identified as invalid.
        Returns an empty list if the value is missing or malformed.

        Args:
            raw: The raw value at "invalid_citation_ids" in the parsed JSON.

        Returns:
            A deduplicated list of block_id strings.
        """
        if not isinstance(raw, list):
            self.logger.debug(
                "CriticAgent._extract_invalid_citation_ids: "
                "'invalid_citation_ids' is not a list — returning []."
            )
            return []

        ids: List[str] = []
        seen: set = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            block_id = item.strip()
            if block_id and block_id not in seen:
                ids.append(block_id)
                seen.add(block_id)

        return ids

    def _cross_check_citations(
        self,
        valid_citations: List[Citation],
        context_chunks: List[str],
    ) -> tuple[List[Citation], List[str]]:
        """
        Performs a lightweight block_id presence check against the raw
        context chunks as a second line of defence against hallucinated
        citations that the LLM incorrectly marked as valid.

        How it works:
            Each chunk produced by chunk_text_with_context() [1] carries a
            positional banner "--- DOCUMENT CHUNK N OF M ---" and the raw
            text. The block_id (e.g. "p3_b12") is embedded inside the chunk
            text by the TSD ingestor. We check that the block_id string
            appears somewhere in the combined context.

            This is intentionally a lightweight string presence check —
            not a semantic verification. Semantic verification is the LLM's
            job. This guard only catches outright hallucinated block_ids that
            could not possibly exist in the document.

        Args:
            valid_citations: Citations the LLM marked as valid.
            context_chunks:  The raw context chunks given to both agents.

        Returns:
            A tuple of:
                - confirmed_citations: Citations whose block_ids were found
                  in the context.
                - additionally_invalidated: block_ids the LLM said were valid
                  but that do not appear anywhere in the context text.
        """
        if not valid_citations:
            return [], []

        if not context_chunks:
            self.logger.debug(
                "CriticAgent._cross_check_citations: no context chunks "
                "to check against — returning all citations as-is."
            )
            return valid_citations, []

        # Join all chunks into one searchable string once — O(n) build,
        # O(1) per block_id lookup via substring search
        combined_context = "\n".join(context_chunks)

        confirmed: List[Citation] = []
        additionally_invalidated: List[str] = []

        for citation in valid_citations:
            if not citation.block_id:
                self.logger.debug(
                    "CriticAgent._cross_check_citations: skipping citation "
                    "with empty block_id."
                )
                continue

            if citation.block_id in combined_context:
                confirmed.append(citation)
            else:
                additionally_invalidated.append(citation.block_id)
                self.logger.debug(
                    "CriticAgent._cross_check_citations: block_id='%s' "
                    "not found in context — marking as additionally invalidated.",
                    citation.block_id,
                )

        return confirmed, additionally_invalidated


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = ["CriticAgent"]
