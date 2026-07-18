from __future__ import annotations

import logging
from typing import Callable, List, Optional

from sdr.apps.ai.prompts.agents import (
    HUNTER_SYSTEM_PROMPT,
    build_hunter_prompt,
)
from .base import (
    APPLICABILITY_ESTABLISHED,
    APPLICABILITY_NOT_ESTABLISHED,
    BaseAgent,
    HunterResult,
    VERDICT_MET,
    VERDICT_NA,
    VERDICT_NOT_MET,
)

logger = logging.getLogger(__name__)


class HunterAgent(BaseAgent):
    system_prompt: str = HUNTER_SYSTEM_PROMPT
    model_component: str = "hunter"
    max_tokens: int = 8192
    temperature: float = 0.0
    reasoning_effort: str = "medium"

    def _build_user_prompt(
        self,
        parameter_text: str,
        parameter_section: str,
        context_chunks: List[str],
        killed_assumptions: Optional[List[dict]] = None,
        available_block_ids: Optional[List[str]] = None,
    ) -> str:
        return build_hunter_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            context_chunks=context_chunks,
            killed_assumptions=killed_assumptions,
            available_block_ids=available_block_ids,
        )

    def run(
        self,
        parameter_text: str,
        parameter_section: str,
        context_chunks: List[str],
        killed_assumptions: Optional[List[dict]] = None,
        available_block_ids: Optional[List[str]] = None,
        stream_handler: Optional[Callable[[str], None]] = None,
    ) -> HunterResult:
        # ------------------------------------------------------------------
        # 1. Input validation
        # ------------------------------------------------------------------
        if not parameter_text or not parameter_text.strip():
            msg = "parameter_text is empty — cannot assess a blank requirement."
            self.logger.error("HunterAgent.run: %s", msg)
            return self._hunter_error(msg)

        if not context_chunks:
            msg = (
                f"No context chunks provided for parameter: "
                f"'{parameter_text[:80]}' — verdict defaults to not_met."
            )
            self.logger.warning("HunterAgent.run: %s", msg)
            # Return a valid not_met result rather than an error — lack of
            # context is a legitimate signal of missing documentation
            return HunterResult(
                verdict=VERDICT_NOT_MET,
                confidence=0.3,
                applicability_status=APPLICABILITY_NOT_ESTABLISHED,
                applicability_reason=(
                    "No relevant context was retrieved, so applicability could "
                    "not be established from the TSD evidence."
                ),
                missing_expected_evidence=[],
                reasoning=(
                    "No relevant context was retrieved from the TSD for this "
                    "parameter. This may indicate the topic is not covered in "
                    "the document."
                ),
                logic_summary=(
                    "No relevant context was retrieved from the TSD for this "
                    "parameter. This may indicate the topic is not covered in "
                    "the document."
                ),
                evidence_found=False,
                citations=[],
                checked_context="No context chunks were retrieved.",
                evidence_quotes=[],
                evidence_assessment="No evidence could be assessed because retrieval returned no context.",
                error=None,
            )

        # ------------------------------------------------------------------
        # 2. Build prompt
        # ------------------------------------------------------------------
        user_prompt = self._build_user_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            context_chunks=context_chunks,
            killed_assumptions=killed_assumptions,
            available_block_ids=available_block_ids,
        )

        self.logger.info(
            "HunterAgent.run: assessing parameter '%s...' "
            "across %d chunk(s).",
            parameter_text[:60],
            len(context_chunks),
        )

        # ------------------------------------------------------------------
        # 3. Call LLM
        # ------------------------------------------------------------------
        response = self._call_llm_with_truncation_retry(user_prompt, stream_handler=stream_handler)

        if response.error:
            msg = f"LLM call failed: {response.error}"
            self.logger.error("HunterAgent.run: %s", msg)
            return self._hunter_error(msg, raw=response.content)

        # ------------------------------------------------------------------
        # 4. Parse JSON response
        # ------------------------------------------------------------------
        parsed = self._parse_json_response(response)

        if parsed is None:
            msg = "Failed to parse LLM response as JSON."
            self.logger.error(
                "HunterAgent.run: %s Raw (first 300 chars): %s",
                msg,
                (response.content or "")[:300],
            )
            return self._hunter_error(msg, raw=response.content)

        # ------------------------------------------------------------------
        # 5. Extract and validate all fields
        # ------------------------------------------------------------------
        verdict = self._validate_verdict(
            parsed.get("verdict"),
            fallback=VERDICT_NOT_MET,
        )
        confidence = self._clamp_confidence(
            parsed.get("confidence"),
            default=0.5,
        )
        reasoning_fields = self._extract_reasoning_fields(
            parsed,
            reasoning_fallback="No reasoning provided by the Hunter agent.",
        )
        evidence_found = self._extract_evidence_found(parsed, verdict)
        citations = self._extract_citations(
            parsed.get("citations", []),
            field_name="citations",
        )
        checked_context = self._extract_text_field(
            parsed,
            "checked_context",
            default=self._fallback_checked_context(context_chunks),
            max_chars=2500,
        )
        evidence_quotes = self._extract_string_list(
            parsed,
            "evidence_quotes",
            max_items=10,
            max_chars=500,
        )
        evidence_assessment = self._extract_text_field(
            parsed,
            "evidence_assessment",
            default=reasoning_fields["logic_summary"],
            max_chars=2500,
        )

        verdict, confidence, evidence_found, reasoning_fields, evidence_assessment = (
            self._enforce_grounding(
                verdict=verdict,
                confidence=confidence,
                evidence_found=evidence_found,
                citations=citations,
                checked_context=checked_context,
                reasoning_fields=reasoning_fields,
                evidence_assessment=evidence_assessment,
                context_chunks=context_chunks,
            )
        )
        applicability_status = self._extract_applicability_status(parsed, verdict=verdict)
        applicability_reason = self._extract_applicability_reason(
            parsed,
            default="No applicability reasoning provided by the Hunter agent.",
        )
        missing_expected_evidence = self._extract_missing_expected_evidence(parsed)
        if verdict == VERDICT_NA:
            applicability_status = APPLICABILITY_NOT_ESTABLISHED
        elif verdict in {VERDICT_MET, VERDICT_NOT_MET}:
            applicability_status = APPLICABILITY_ESTABLISHED
        if verdict == VERDICT_NOT_MET and not missing_expected_evidence:
            missing_expected_evidence = [evidence_assessment[:400]] if evidence_assessment else []
        if verdict != VERDICT_NOT_MET:
            missing_expected_evidence = []

        # ------------------------------------------------------------------
        # 6. Log and return
        # ------------------------------------------------------------------
        self.logger.info(
            "HunterAgent.run: verdict='%s' confidence=%.2f "
            "evidence_found=%s citations=%d for parameter '%s...'",
            verdict,
            confidence,
            evidence_found,
            len(citations),
            parameter_text[:60],
        )

        # Warn if Hunter claims met but has no citations — high hallucination risk
        if verdict == "met" and not citations:
            self.logger.warning(
                "HunterAgent.run: verdict='met' but zero valid citations "
                "for parameter '%s...' — Critic will likely overturn.",
                parameter_text[:60],
            )

        return HunterResult(
            verdict=verdict,
            confidence=confidence,
            applicability_status=applicability_status,
            applicability_reason=applicability_reason,
            missing_expected_evidence=missing_expected_evidence,
            reasoning=reasoning_fields["reasoning"],
            assumptions=reasoning_fields["assumptions"],
            logic_summary=reasoning_fields["logic_summary"],
            cot_trace=reasoning_fields["cot_trace"],
            evidence_found=evidence_found,
            citations=citations,
            checked_context=checked_context,
            evidence_quotes=evidence_quotes,
            evidence_assessment=evidence_assessment,
            raw_response=response.content,
            error=None,
        )

    def _extract_evidence_found(
        self,
        parsed: dict,
        verdict: str,
    ) -> bool:
        raw = parsed.get("evidence_found")

        if isinstance(raw, bool):
            return raw

        # Coerce string representations
        if isinstance(raw, str):
            if raw.lower() in {"true", "yes", "1"}:
                return True
            if raw.lower() in {"false", "no", "0"}:
                return False

        # Derive from verdict as fallback
        self.logger.debug(
            "HunterAgent._extract_evidence_found: 'evidence_found' "
            "missing or unrecognised ('%s') — deriving from verdict '%s'.",
            raw,
            verdict,
        )
        return verdict == "met"

    def _fallback_checked_context(self, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "No context chunks were retrieved."
        return (
            f"Reviewed {len(context_chunks)} retrieved context chunk(s); "
            f"first chunk excerpt: {context_chunks[0][:350].strip()}"
        )

    def _enforce_grounding(
        self,
        verdict: str,
        confidence: float,
        evidence_found: bool,
        citations: List,
        checked_context: str,
        reasoning_fields: dict,
        evidence_assessment: str,
        context_chunks: List[str],
    ) -> tuple[str, float, bool, dict, str]:
        reasoning = (reasoning_fields.get("logic_summary") or "").strip()
        generic_reasoning = not reasoning or reasoning == "No reasoning provided by the Hunter agent."

        if verdict == VERDICT_MET and (not citations or not evidence_found):
            confidence = min(confidence, 0.65)
            evidence_found = True
            repair_required = (
                "Hunter claimed met but did not provide valid supporting evidence; "
                "retaining met for debate citation repair."
            )
            reasoning_fields["logic_summary"] = repair_required
            reasoning_fields["reasoning"] = repair_required
            evidence_assessment = repair_required

        if verdict == VERDICT_NOT_MET and (generic_reasoning or not checked_context.strip()):
            fallback = (
                f"Reviewed {len(context_chunks)} retrieved context chunk(s) and found no explicit "
                "evidence satisfying the requirement in the supplied TSD context."
            )
            reasoning_fields["logic_summary"] = fallback
            reasoning_fields["reasoning"] = fallback
            if not evidence_assessment.strip():
                evidence_assessment = fallback

        return verdict, confidence, evidence_found, reasoning_fields, evidence_assessment


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = ["HunterAgent"]
