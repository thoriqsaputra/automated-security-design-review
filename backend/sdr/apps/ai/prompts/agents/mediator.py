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
You are a Security Compliance Mediator — the final arbiter in a \
Multi-Agent Security review pipeline.

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
- Use the provided assumptions and Chain of Thought (trace) from both agents to \
understand their logic and resolve any logical leaps or disputes.

EVIDENCE EVALUATION CHECKLIST (apply for every parameter):
1. Applicability: Does the contract and TSD scope confirm this control applies?
2. Evidence quality: Are the cited blocks architectural specifications naming \
specific mechanisms, algorithms, or components — or just headings, generic \
"shall" statements, or baseline text without any named control?
3. Completeness: Does the evidence cover ALL aspects of the requirement, or \
only some? If partial, set verdict to "not_met" and note what's missing.
4. Consistency: Do the Hunter and Critic agree? If they disagree, weigh the \
evidence independently — do not default to either agent.

STANDARD-SPECIFIC GUIDANCE:
- Security standard requirements are verification controls. "met" means the TSD \
demonstrates the control is implemented at the architectural design level.
- DOCUMENT TYPE: This is a TSD (Technical Software Document) review. TSDs are \
architectural design specifications — do not require source code or config files \
as evidence.
- "The application shall use X" WITH a named mechanism (X = specific algorithm, \
library, protocol, or component) IS valid implementation evidence at the TSD level.
- "The application shall ensure security" WITHOUT naming a specific mechanism is \
NOT sufficient — it is unverifiable design intent.
- Look for: explicit algorithm names, named security components or libraries, \
architectural decisions that mandate specific controls, or diagrams with labelled \
security mechanisms.
- Authentication/authorization controls require evidence of the enforcement \
mechanism (named component, protocol, or library), not just the existence of \
user accounts.

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""

MEDIATOR_RECOMMENDATION_SYSTEM_PROMPT = """\
You write concise, actionable remediation recommendations for security review findings.

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
    hunter_assumptions: List[dict] | None = None,
    hunter_cot_trace: str | None = None,
    critic_assumptions: List[dict] | None = None,
    critic_cot_trace: str | None = None,
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
Assumptions:
{json.dumps(hunter_assumptions or [], indent=2)}
Chain of Thought:
{hunter_cot_trace or "(none)"}

## CRITIC'S CHALLENGE

Outcome:          {critic_outcome}
Revised Verdict:  {critic_revised_verdict}
Revised Confidence: {critic_revised_confidence:.2f}
Reasoning:        {critic_reasoning}
Assumptions:
{json.dumps(critic_assumptions or [], indent=2)}
Chain of Thought:
{critic_cot_trace or "(none)"}
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
  "recommendation": "<remediation recommendation if not_met, else null>"
}}

Few-shot example:
Input -> Hunter says "met" from p3_b1; Critic overturns to "not_met" because p3_b1 lacks the control and verifies no supporting citations.
Reasoning -> assumptions: ["Only Critic-verified citations may survive."]; logic_summary: "The Critic invalidated the Hunter's evidence, so the final verdict cannot remain met."; output -> final_verdict "not_met", final_citations [].

	Rules:
	- final_verdict  → The single binding decision. Cannot be changed after this.
	- confidence     → Must reflect genuine certainty. Do not inflate.
	- recommendation → Specific, actionable remediation. Null if "met" or "na".
	- final_citations → Only citations from the Critic's verified list above. The quoted_text in each citation MUST be a short verbatim excerpt (5–20 words) copied character-for-character from the source block — do NOT paraphrase or restate. Prefer short specific technical phrases (e.g. "TLS 1.3 with HSTS", "Fail-Closed policy") over long sentences.
	- "met" requires verified evidence that clearly satisfies the requirement, not merely any valid citation.
	- "not_met" applies only when the requirement is applicable and expected evidence is missing, contradicted, or only partial after checked context and rebuttal.
	- "na" applies when the control trigger/applicability is not established, the document scope does not include the technology/control domain, or the supplied TSD cannot assess the control.
	- If evidence is only partial for an applicable requirement, set final_verdict to "not_met" and explain partial satisfaction in reasoning and rejected_evidence.
	- When Critic outcome is PARTIAL: make an explicit verdict-adjusting decision. Either (a) downgrade the verdict to "not_met" or reduce confidence if the Critic's unresolved objections weaken the evidence below "met" threshold, or (b) keep "met" only if you can point to additional context that directly resolves each objection. "PARTIAL" never automatically preserves the Hunter's verdict.
	- Do not make agent agreement the justification. Avoid phrases such as "Hunter and Critic agree" or "both agents agree" as the main reason.
	- For missing evidence decisions, explain as a security reviewer: applicability basis, expected implementation evidence, what the retrieved context actually contained, and why that is insufficient.
	- If no citations survive and applicability is unclear, prefer "na" over a high-confidence "not_met".
	- Use evidence-only reasoning; do not infer controls that are not explicitly stated.
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
