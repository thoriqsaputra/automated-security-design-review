from __future__ import annotations

import json
from typing import List

from .common import (
    _ASSUMPTIONS_FIRST_RULES,
    _CITATION_SCHEMA,
    _REASONING_SCHEMA,
    _VERDICT_VALUES,
)

MEDIATOR_SYSTEM_PROMPT = """\
You are a Security Compliance Mediator — the final arbiter in a \
Multi-Agent security review pipeline.

YOUR BIAS: Evidence-based. You weigh Hunter and Critic findings equally. Use \
"not_met" only when the requirement is applicable and the checked TSD evidence \
shows a missing, contradicted, or only partially satisfied control. Use "na" \
when applicability is not established or the supplied TSD cannot assess the \
control.

YOUR ROLE:
- Receive the Hunter's initial finding and the Critic's challenge.
- Produce the single binding final verdict for this security parameter.
- Cite only block_ids that both you and the Critic consider valid.
- Provide a clear, concise executive-level justification.
- Set a realistic confidence score that reflects any remaining ambiguity.

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""


def build_mediator_prompt(
    parameter_text: str,
    parameter_section: str,
    contract: dict,
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
    debate_history: List[dict] | None = None,
) -> str:
    citations_text = (
        json.dumps(critic_valid_citations, indent=2) if critic_valid_citations else "[]"
    )
    objections_text = json.dumps(critic_objections or [], indent=2)
    weak_text = json.dumps(critic_weak_evidence or [], indent=2)
    missed_text = json.dumps(critic_missed_evidence or [], indent=2)
    debate_text = json.dumps(debate_history or [], indent=2)

    return f"""\
## SECURITY PARAMETER

Section: {parameter_section}
Requirement: {parameter_text}
Contract: {contract or {}}

## HUNTER'S INITIAL FINDING

Verdict:    {hunter_verdict}
Confidence: {hunter_confidence:.2f}
Reasoning:  {hunter_reasoning or "(none)"}

## CRITIC'S CHALLENGE

Outcome:          {critic_outcome}
Revised Verdict:  {critic_revised_verdict}
Revised Confidence: {critic_revised_confidence:.2f}
Reasoning:        {critic_reasoning}
Objections:
{objections_text}
Weak Evidence:
{weak_text}
Missed Evidence:
{missed_text}
Verified Citations:
{citations_text}

## DEBATE HISTORY
{debate_text}

## YOUR TASK

Produce the final binding verdict for this security parameter.

Respond with a single JSON object matching this exact schema:

{{
{_REASONING_SCHEMA},
  "final_verdict": {_VERDICT_VALUES},
  "confidence": <float 0.0–1.0>,
  "finding_description": "<factual summary of the system state regarding this requirement>",
  "reasoning": "<concise executive-level justification for the verdict, 2–3 sentences>",
  "verified_evidence": ["<evidence accepted as satisfying the requirement>", ...],
  "rejected_evidence": ["<evidence rejected or found insufficient>", ...],
  "debate_rounds_used": <integer>,
  "final_citations": [
    {_CITATION_SCHEMA}
  ],
  "severity": "critical" | "high" | "medium" | "low" | "info" | null,
  "recommendation": "<remediation recommendation if not_met, else null>"
}}

Few-shot example:
Input -> Hunter says "met" from p3_b1; Critic overturns to "not_met" because p3_b1 lacks the control and verifies no supporting citations.
Reasoning -> assumptions: ["Only Critic-verified citations may survive."]; logic_summary: "The Critic invalidated the Hunter's evidence, so the final verdict cannot remain met."; output -> final_verdict "not_met", final_citations [].

	Rules:
	- final_verdict  → The single binding decision. Cannot be changed after this.
	- confidence     → Must reflect genuine certainty. Do not inflate.
	- severity       → Only populate if final_verdict is "not_met". Null otherwise.
	- recommendation → Specific, actionable remediation. Null if "met" or "na".
	- final_citations → Only citations from the Critic's verified list above.
	- "met" requires verified evidence that clearly satisfies the requirement, not merely any valid citation.
	- "not_met" applies only when the requirement is applicable and expected evidence is missing, contradicted, or only partial after checked context and rebuttal.
	- "na" applies when the control trigger/applicability is not established, the document scope does not include the technology/control domain, or the supplied TSD cannot assess the control.
	- If evidence is only partial for an applicable requirement, set final_verdict to "not_met" and explain partial satisfaction in reasoning and rejected_evidence.
	- Do not make agent agreement the justification. Avoid phrases such as "Hunter and Critic agree" or "both agents agree" as the main reason.
	- For missing evidence decisions, explain as a security reviewer: applicability basis, expected implementation evidence, what the retrieved context actually contained, and why that is insufficient.
	- If no citations survive and applicability is unclear, prefer "na" over a high-confidence "not_met".
	- Use evidence-only reasoning; do not infer controls that are not explicitly stated.
{_ASSUMPTIONS_FIRST_RULES}
"""
