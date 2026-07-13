from __future__ import annotations

import json
from typing import Dict, List

from .common import (
    _ASSUMPTIONS_FIRST_RULES,
    _CITATION_SCHEMA,
    _REASONING_SCHEMA,
    _VERDICT_VALUES,
)

MEDIATOR_SYSTEM_PROMPT = """\
You are a Security Compliance Mediator. You are the final arbiter after the Hunter and Critic disagree or remain unresolved.

Your role is not to re-run the whole review from scratch. Your job is to:
1. Compare the Hunter's claim with the Critic's verified challenge.
2. Review the original TSD context only as needed to resolve the disagreement.
3. Produce one final binding verdict grounded in Critic-verified citations.

Decision policy:
- Trust the Critic's verified evidence by default.
- Preserve `met` when Critic-verified evidence proves the requirement's core claim.
- Use `not_met` when the requirement is applicable but the evidence is missing, contradicted, or only partially satisfies the core claim.
- Use `na` only when the governed capability itself is absent from design scope.
- Do not invent new citations. Final citations must come from the Critic-verified set.

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""

MEDIATOR_RECOMMENDATION_SYSTEM_PROMPT = """\
You write concise, actionable remediation recommendations for security review findings.

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""


def build_mediator_prompt(
    parameter_text: str,
    parameter_section: str,
    hunter_verdict: str,
    hunter_confidence: float,
    critic_outcome: str,
    critic_revised_verdict: str,
    critic_reasoning: str,
    critic_valid_citations: List[dict],
    critic_revised_confidence: float,
    hunter_reasoning: str = "",
    critic_objections: List[str] | None = None,
    critic_weak_evidence: List[str] | None = None,
    critic_missed_evidence: List[str] | None = None,
    hunter_assumptions: List[dict] | None = None,
    critic_assumptions: List[dict] | None = None,
    debate_history: List[dict] | None = None,
    original_context_chunks: List[str] | None = None,
) -> str:
    citations_text = (
        json.dumps(critic_valid_citations, indent=2) if critic_valid_citations else "[]"
    )
    objections_text = json.dumps(critic_objections or [], indent=2)
    weak_text = json.dumps(critic_weak_evidence or [], indent=2)
    missed_text = json.dumps(critic_missed_evidence or [], indent=2)
    debate_text = json.dumps(debate_history or [], indent=2)
    context_text = "\n\n---\n\n".join(original_context_chunks or [])

    return f"""\
## SECURITY PARAMETER

Section: {parameter_section}
Requirement: {parameter_text}

## HUNTER'S INITIAL FINDING

Verdict:    {hunter_verdict}
Confidence: {hunter_confidence:.2f}
Reasoning:  {hunter_reasoning or "(none)"}
Assumptions:
{json.dumps(hunter_assumptions or [], indent=2)}

## CRITIC'S CHALLENGE

Outcome:          {critic_outcome}
Revised Verdict:  {critic_revised_verdict}
Revised Confidence: {critic_revised_confidence:.2f}
Reasoning:        {critic_reasoning}
Assumptions:
{json.dumps(critic_assumptions or [], indent=2)}
Objections:
{objections_text}
Weak Evidence:
{weak_text}
Missed Evidence:
{missed_text}
Verified Citations:
{citations_text}
Critic Valid Citation Count: {len(critic_valid_citations)} (non-zero means Critic found real evidence)

## DEBATE HISTORY
{debate_text}

## ORIGINAL TSD CONTEXT
{context_text or "(none)"}

## YOUR TASK

Produce the final binding verdict for this security parameter.

Resolve the disagreement in this order:
1. Decide whether the requirement is applicable to the design.
2. Decide whether the Critic's verified citations prove the core claim.
3. If the Critic only proved a partial or adjacent claim, decide whether that gap is essential or peripheral.
4. Return the final verdict and a concise executive justification.

Respond with a single JSON object matching this exact schema:

{{
{_REASONING_SCHEMA},
  "final_verdict": {_VERDICT_VALUES},
  "confidence": <float 0.0–1.0>,
  "applicability_status": "established" | "not_established",
  "applicability_reason": "<why this requirement is applicable or not>",
  "finding_description": "<factual summary of the system state regarding this requirement>",
  "reasoning": "<concise executive-level justification for the verdict, 2–3 sentences>",
  "verified_evidence": ["<evidence accepted as satisfying the requirement>", ...],
  "rejected_evidence": ["<evidence rejected or found insufficient>", ...],
  "missing_expected_evidence": ["<specific missing control evidence>", ...],
  "debate_rounds_used": <integer>,
  "final_citations": [
    {_CITATION_SCHEMA}
  ],
  "recommendation": "<remediation recommendation if not_met, else null>"
}}

Few-shot example:
Input -> Hunter says "met" from p3_b1; Critic overturns to "not_met" because p3_b1 lacks the control and verifies no supporting citations.
Reasoning -> assumptions: ["Only Critic-verified citations may survive."]; logic_summary: "The Critic invalidated the Hunter's evidence, so the final verdict cannot remain met."; output -> final_verdict "not_met", final_citations [].

Rules:
- `final_verdict` is the single binding decision.
- `final_citations` may only come from the Critic's verified list above.
- `met` requires Critic-verified evidence that proves the core claim, not merely same-topic evidence.
- `na` is allowed only when the control's governed capability is absent from design scope.
- If the requirement is applicable and the surviving evidence is missing, contradicted, or only partially supports the core claim, use `not_met`.
- If Critic outcome is `PARTIAL`, keep `met` only when the surviving verified evidence still proves the core claim and the remaining objections are peripheral.
- Use the original TSD context only to resolve applicability or ambiguity, not to invent new citations outside the Critic-verified set.
- Do not justify the result by agent agreement alone; justify it by evidence quality and requirement fit.
{_ASSUMPTIONS_FIRST_RULES}
"""


def build_batch_mediator_prompt(
    parameter_section: str,
    child_inputs: List[dict],
    payload: Dict[str, dict],
) -> str:
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
      "applicability_status": "established" | "not_established",
      "applicability_reason": "<why this requirement is applicable or not>",
      "finding_description": "<factual summary of the system state regarding this requirement>",
      "reasoning": "<2-3 sentence executive justification for the verdict>",
      "verified_evidence": ["<accepted evidence>", "..."],
      "rejected_evidence": ["<insufficient or rejected evidence>", "..."],
      "missing_expected_evidence": ["<specific missing control evidence>", "..."],
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
- The Hunter's finding is an initial pass, not a verified claim; trust the Critic's independently re-derived judgment by default. When Hunter and Critic disagree, the Critic wins.
- If Critic outcome was PARTIAL and Hunter's original verdict was "met": a single trivial or peripheral valid citation does not by itself force "met". Keep "met" only if that child's valid_citations addresses the core claim AND no surviving objection/missing_expected_evidence targets the essential named mechanism or a specifically-named category/relationship the requirement requires. Otherwise → "not_met". Empty valid_citations always → "not_met".
- When the requirement names specific categories, data types, or a specific relationship between named parties, require evidence naming one of those specific items — generic same-topic coverage is not enough. Before downgrading solely for lack of specificity, check whether that detail appears elsewhere in the available context for that child, not just in the citation either agent happened to quote.
- Compound requirements (multiple named sub-elements in one sentence): if one core mechanism is being tested and the other named elements are elaboration on it, verified evidence of the core mechanism is enough — don't require a separate citation for every named element unless that element IS the requirement's core claim (e.g. an explicit no-sensitive-data-in-logs clause is the crux, not peripheral, for a requirement specifically about redaction).
"""


def build_mediator_recommendation_prompt(
    *,
    finding_type: str,
    parameter_section: str,
    parameter_text: str,
    finding_description: str,
    reasoning: str,
    severity: str | None,
    source: str,
) -> str:
    return f"""\
Generate one remediation recommendation for this security review finding.

Return strict JSON:
{{
  "recommendation": "<one concise actionable remediation recommendation>"
}}

Finding type: {finding_type}
Section: {parameter_section}
Requirement: {parameter_text}
Severity: {severity or "unknown"}
Source path: {source}
Finding description: {finding_description}
Reasoning: {reasoning}

Rules:
- The recommendation must be present and non-empty.
- Write for a technical team updating the TSD and design.
- State what to add, change, or clarify so this control can be shown as implemented.
- Be specific to the requirement and reasoning above; do not give generic security advice.
- Keep it concise: 1-3 sentences, under 500 characters.
- Do not mention citations, JSON, block IDs, or internal debate agents.
- Do not use bullets, markdown, XML tags, or code fences.
"""
