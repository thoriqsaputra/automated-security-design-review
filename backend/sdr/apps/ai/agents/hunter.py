"""
Hunter Agent — first stage of the Multi-Agent TSD Security Review Pipeline.

Responsibility:
    Given a security parameter (CategoryParameterChild) and retrieved TSD
    context chunks, the Hunter produces an initial compliance verdict with
    cited source block IDs.

Bias:
    Assume NON-COMPLIANCE unless explicit, unambiguous evidence is found.
    Implicit or assumed compliance does not count.

Output:
    HunterResult dataclass — passed directly to CriticAgent as input.

Dependency chain:
    agent_prompts.py          (pure prompt strings)
         ↓
    base.py                   (BaseAgent, HunterResult, Citation)
         ↓
    hunter.py                 ← YOU ARE HERE
         ↓
    critic.py
         ↓
    mediator.py
         ↓
    analysis_service.py
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from sdr.apps.ai.prompts.agent_prompt import (
    HUNTER_SYSTEM_PROMPT,
    build_hunter_prompt,
)
from .base import (
    BaseAgent,
    HunterResult,
    VERDICT_MET,
    VERDICT_NOT_MET,
    VERDICT_NA,
)

logger = logging.getLogger(__name__)


class HunterAgent(BaseAgent):
    """
    Concrete implementation of the Hunter agent.

    The Hunter is called once per security parameter per TSD analysis run.
    It receives pre-retrieved context chunks from the HybridRetrievalRouter
    and produces an initial HunterResult that is passed to the CriticAgent.

    It never raises — all errors are captured in HunterResult.error so
    the pipeline can continue to the next parameter.
    """

    system_prompt: str = HUNTER_SYSTEM_PROMPT
    model_component: str = "hunter"

    # Hunter uses slightly higher max_tokens than default — reasoning
    # paragraphs and multiple citations can be verbose
    max_tokens: int = 4096
    temperature: float = 0.0

    def _build_user_prompt(
        self,
        parameter_text: str,
        parameter_section: str,
        contract: Optional[dict],
        context_chunks: List[str],
        diagram_captions: Optional[List[str]] = None,
        persona_focus: Optional[str] = None,
        killed_assumptions: Optional[List[dict]] = None,
    ) -> str:
        """
        Delegates to build_hunter_prompt() from agent_prompts.py.
        Kept thin — all prompt logic lives in the prompts layer.
        """
        return build_hunter_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
            context_chunks=context_chunks,
            diagram_captions=diagram_captions,
            persona_focus=persona_focus,
            killed_assumptions=killed_assumptions,
        )

    def run(
        self,
        parameter_text: str,
        parameter_section: str,
        contract: Optional[dict],
        context_chunks: List[str],
        diagram_captions: Optional[List[str]] = None,
        persona_focus: Optional[str] = None,
        killed_assumptions: Optional[List[dict]] = None,
    ) -> HunterResult:
        """
        Executes the Hunter agent for a single security parameter.

        Pipeline:
            1. Validate inputs — guard against empty context.
            2. Build the user-turn prompt.
            3. Call the LLM via _call_llm() from BaseAgent.
            4. Parse the JSON response via _parse_json_response().
            5. Extract and validate all fields with shared helpers.
            6. Return a fully populated HunterResult.

        Args:
            parameter_text:    Full requirement text from CategoryParameterChild.
            parameter_section: Parent section title from CategoryParameterParent.
            context_chunks:    Retrieved TSD text chunks — must be non-empty.
                               Each chunk carries a positional banner
                               "--- DOCUMENT CHUNK N OF M ---" prepended
                               by chunk_text_with_context() [1].
            diagram_captions:  Optional captions of diagrams in the context
                               window — passed to the Vision agent separately
                               but included here as textual signal for the
                               Hunter.

        Returns:
            HunterResult — never raises. Check .error field for failures.
        """
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
            contract=contract,
            context_chunks=context_chunks,
            diagram_captions=diagram_captions,
            persona_focus=persona_focus,
            killed_assumptions=killed_assumptions,
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
        response = self._call_llm(user_prompt)

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

    def run_batch(
        self,
        child_inputs: List[dict],
        parameter_section: str,
        context_chunks: List[str],
        diagram_captions: Optional[List[str]] = None,
        killed_assumptions: Optional[List[dict]] = None,
    ) -> Dict[str, HunterResult]:
        """
        Executes one Hunter call for multiple child parameters.

        Returns a dict keyed by child parameter id. Missing or malformed child
        entries are intentionally omitted so the pipeline validator can rerun
        those children through the single-child path.
        """
        if not child_inputs:
            return {}
        if not context_chunks:
            return {
                str(item.get("id")): HunterResult(
                    verdict=VERDICT_NOT_MET,
                    confidence=0.3,
                    reasoning="No relevant parent context was retrieved from the TSD.",
                    logic_summary="No relevant parent context was retrieved from the TSD.",
                    evidence_found=False,
                    citations=[],
                    checked_context="No context chunks were retrieved.",
                    evidence_quotes=[],
                    evidence_assessment="No evidence could be assessed because retrieval returned no context.",
                )
                for item in child_inputs
                if item.get("id") is not None
            }

        response = self._call_llm(
            self._build_batch_user_prompt(
                child_inputs=child_inputs,
                parameter_section=parameter_section,
                context_chunks=context_chunks,
                diagram_captions=diagram_captions,
                killed_assumptions=killed_assumptions,
            )
        )
        if response.error:
            msg = f"LLM call failed: {response.error}"
            self.logger.error("HunterAgent.run_batch: %s", msg)
            return {
                str(item.get("id")): self._hunter_error(msg, raw=response.content)
                for item in child_inputs
                if item.get("id") is not None
            }

        parsed = self._parse_json_response(response)
        if parsed is None:
            self.logger.error("HunterAgent.run_batch: failed to parse batch JSON.")
            return {}

        allowed_ids = {str(item.get("id")) for item in child_inputs if item.get("id") is not None}
        results: Dict[str, HunterResult] = {}
        for item in parsed.get("results", []):
            if not isinstance(item, dict):
                continue
            child_id = str(item.get("child_id") or item.get("parameter_id") or "").strip()
            if child_id not in allowed_ids or child_id in results:
                continue

            verdict = self._validate_verdict(item.get("verdict"), fallback=VERDICT_NOT_MET)
            confidence = self._clamp_confidence(item.get("confidence"), default=0.5)
            reasoning_fields = self._extract_reasoning_fields(
                item,
                reasoning_fallback="No reasoning provided by the Hunter agent.",
            )
            citations = self._extract_citations(item.get("citations", []), field_name="citations")
            evidence_found = self._extract_evidence_found(item, verdict)
            checked_context = self._extract_text_field(
                item,
                "checked_context",
                default=self._fallback_checked_context(context_chunks),
                max_chars=2500,
            )
            evidence_quotes = self._extract_string_list(
                item,
                "evidence_quotes",
                max_items=10,
                max_chars=500,
            )
            evidence_assessment = self._extract_text_field(
                item,
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
            results[child_id] = HunterResult(
                verdict=verdict,
                confidence=confidence,
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
        return results

    def _build_batch_user_prompt(
        self,
        *,
        child_inputs: List[dict],
        parameter_section: str,
        context_chunks: List[str],
        diagram_captions: Optional[List[str]],
        killed_assumptions: Optional[List[dict]],
    ) -> str:
        chunks_text = "\n\n---\n\n".join(context_chunks)
        diagrams_text = "\n".join(diagram_captions or [])
        killed_text = json.dumps(killed_assumptions or [], indent=2)
        children_text = json.dumps(child_inputs, indent=2)
        return f"""\
## PARENT SECURITY SECTION
Section: {parameter_section}

## CHILD PARAMETERS
{children_text}

## SHARED TSD CONTEXT
{chunks_text}

## DIAGRAM CAPTIONS
{diagrams_text or "[]"}

## INVALIDATED ASSUMPTIONS TO AVOID
{killed_text}

Analyse each child parameter independently. Do not merge child requirements.
Return strict JSON with exactly one result object per child id:

{{
  "results": [
    {{
      "child_id": "<id from CHILD PARAMETERS>",
      "assumptions": ["<assumption>", "..."],
      "logic_summary": "<concise evidence-only reasoning>",
      "verdict": "met" | "not_met" | "na",
      "confidence": <float 0.0-1.0>,
      "reasoning": "<one paragraph explaining this child's verdict>",
      "checked_context": "<what context was checked for this child>",
      "evidence_quotes": ["<short verbatim snippets from context, empty if none>"],
      "evidence_assessment": "<why evidence satisfies or fails this child>",
      "evidence_found": <true | false>,
      "citations": [
        {{"block_id": "<CONTEXT_CHUNK id only>", "page_number": <integer>, "quoted_text": "<short quote>", "bbox": {{"x0": null, "y0": null, "x1": null, "y1": null}}}}
      ]
    }}
  ]
}}

Rules:
- Use child_id exactly as supplied.
- A "met" verdict must include at least one valid citation and evidence quote.
- For "not_met", explain what explicit evidence is missing for that child.
- Cite only block_ids from CONTEXT_CHUNK ids in SHARED TSD CONTEXT.
"""

    def _extract_evidence_found(
        self,
        parsed: dict,
        verdict: str,
    ) -> bool:
        """
        Extracts the evidence_found boolean from the parsed response.

        Falls back to a verdict-derived heuristic if the field is absent:
        - "met" → True
        - "not_met" → False
        - "na"  → False

        This ensures evidence_found is always consistent with the verdict
        even if the LLM omits the field.
        """
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
            verdict = VERDICT_NOT_MET
            confidence = min(confidence, 0.45)
            evidence_found = False
            downgrade = (
                "Hunter claimed met but did not provide valid supporting evidence; "
                "downgraded to not_met pending explicit citations."
            )
            reasoning_fields["logic_summary"] = downgrade
            reasoning_fields["reasoning"] = downgrade
            evidence_assessment = downgrade

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
