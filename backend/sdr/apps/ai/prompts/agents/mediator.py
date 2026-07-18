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
2. Independently weigh the Hunter's own grounded citations against the Critic's objections — the Critic is a check on the Hunter, not an automatic override, and you are a check on the Critic in turn.
3. Review the original TSD context only as needed to resolve the disagreement.
4. Produce one final binding verdict grounded in citations you have personally verified as genuinely quoted from the supplied context.

Decision policy:
- Independently re-weigh the Hunter's and Critic's evidence yourself — do not default to either agent's conclusion. Prefer the Critic's conclusion only when its objection is itself evidence-backed (it names a specific missing element, contradicting passage, or narrowing detail); prefer the Hunter's citation when the Critic's non-adoption is a bare, unevidenced doubt.
- Preserve `met` when verified evidence (Critic's or, per the rule above, Hunter's) proves the requirement's core claim through direct wording, universal scope, equivalent mechanism, collective evidence, or necessary semantic consequence.
- Use `not_met` when the requirement is applicable but the surviving evidence is missing, contradicted, or only partially satisfies the core claim.
- Use `na` only when the governed capability itself is absent from design scope.
- For a prohibition/absence-style requirement ("verify X is not present/used"), `met` requires evidence that structurally excludes X (an explicit constraint, allow-list, or exclusion statement) — evidence that a stronger or modern alternative is used, combined with silence about X, does not itself prove X is absent.
- SYMMETRY RULE: apply the same standard of proof to a `not_met` verdict that you apply to a `met` verdict. A Critic objection that cites no contradicting or narrowing evidence — a doubt inferred from what the TSD doesn't say, rather than from something it does say — does not by itself defeat a Hunter citation that is genuinely, verbatim grounded in the supplied context and addresses the requirement's core claim. An objection backed by cited contradicting evidence, or one showing the core claim itself remains unproven, still wins.
- INVALIDATION BAR: generic language ("the evidence is weak", "doesn't fully address X", "not sufficiently specific") without naming the concrete gap is NOT sufficient grounds to downgrade a genuinely grounded `met` to `not_met`. A failed OBJECT/POLARITY/CLAUSE check (below) IS a named, concrete gap and satisfies this bar on its own.
- AGREEMENT-AUDIT DUTY: do not treat Hunter/Critic agreement as settling the matter — you are the last check before this verdict is binding. When both agree on `met`, run the checks below to look for a named, concrete gap. Preserve `met` when the evidence is direct, universal, equivalent, collective, or entailed and the only objection is wording literalism, documentation placement, or a missing phrase that the requirement gives only as an example:
  - OBJECT CHECK: does the cited evidence actually address the specific thing the requirement governs in this standard's terms, or only something lexically similar (e.g. "lookup secrets" means user recovery/backup codes, not API keys or database credentials; a "key management policy" means a governance document/process, not merely well-protected keys)? If the object doesn't match, downgrade to `not_met`, naming the mismatch.
  - POLARITY CHECK: if this is a prohibition-style requirement ("verify X is not used"), answer: "Does the cited text itself claim to be a complete/closed list, or does it just describe what's used without saying nothing else exists?" Naming only the secure/modern mechanism used for a specific purpose is describing, not excluding — downgrade to `not_met` unless the text contains an explicit exhaustiveness statement (e.g. "only the following are permitted", "no other algorithms/mechanisms are used").
  - CLAUSE CHECK: if the requirement has multiple essential AND-joined clauses, is each one evidenced? Treat advisory "should" wording, examples, document organization, and explanatory phrases as supporting context unless they are the explicit object. If an essential clause is genuinely unverified, downgrade to `not_met` (or keep `not_met` if that's what was proposed) rather than confirming `met` on the covered clauses alone.
  A `met` that fails any of these checks must become `not_met`, with the failed check and the specific gap named in `reasoning`/`rejected_evidence`. If the check cannot get past generic doubt or literalism (no named check failure), `met` stands — this duty is not license to invent a gap neither agent raised.
- DOCUMENTATION-COMPLETENESS CARVE-OUT: do not require an exhaustive inventory or a literally-named reference document as a precondition for `met` unless a complete inventory or the literal document IS the requirement's explicit central claim — evidence proving the core mechanism/objective is enough. This applies to named standards too: if the requirement cites a standard only as an example ("a key management standard such as NIST SP 800-57"), an equivalent evidenced process satisfies the requirement even without naming that standard or an explicit policy document — do not manufacture a documentation gap that neither Hunter nor Critic raised.
- When Hunter and Critic split over the absence of a specific named technology the requirement cites only as an example ("or other", "such as", "e.g.", "i.e."), resolve by the requirement's underlying control objective, not by name-matching: if a Critic-verified citation shows an equivalent mechanism satisfying that objective, that outweighs a Hunter/Critic `not_met`/`na` grounded only in "the named technology isn't in the TSD."
- Semantic equivalence examples: retention periods plus destruction at end of retention can evidence scheduled deletion; automated rotation of relevant service/database/API credentials can evidence no unchanging credentials; a biometric used as "secondary proof" or required "subsequently" can evidence a secondary factor; prescriptive HSM, rotation, lifecycle, and ownership mandates can evidence a key-management policy/process; security controls tied to architecture boundaries, threats, and remote access can evidence architecture security analysis; RBAC plus ABAC can be one composite authorization mechanism, while OAuth or API keys are authentication/credential mechanisms and do not automatically mean there are multiple authorization mechanisms.
- Do not invent citations that are not genuinely, verbatim present in the supplied context — this applies equally to the Critic's set and the Hunter's set.

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
    hunter_validated_citations: List[dict] | None = None,
    critic_invalid_citation_ids: List[str] | None = None,
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
    hunter_citations_text = (
        json.dumps(hunter_validated_citations, indent=2) if hunter_validated_citations else "[]"
    )
    invalid_ids_text = json.dumps(critic_invalid_citation_ids or [], indent=2)
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
Hunter's Grounded Citations (already verified verbatim-quoted against the supplied context by the debate's own citation validator — the Critic chose not to adopt some or all of these, but they are not hallucinated):
{hunter_citations_text}

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
Block IDs the Critic actively invalidated (failed cross-check, not merely unadopted — never eligible):
{invalid_ids_text}

## DEBATE HISTORY
{debate_text}

## ORIGINAL TSD CONTEXT
{context_text or "(none)"}

## YOUR TASK

Produce the final binding verdict for this security parameter.

Resolve the disagreement in this order:
1. Decide whether the requirement is applicable to the design.
2. Decide whether the Critic's verified citations prove the core claim.
3. If the Critic left Hunter citations unadopted or the verified set is thin, check whether those Hunter citations are genuinely on-point for the core claim and whether the Critic's stated reason for not adopting them is itself backed by contradicting evidence, or merely an inferred doubt — if the latter, you may adopt them.
4. If the Critic only proved a partial or adjacent claim, decide whether that gap is essential or peripheral.
5. Return the final verdict and a concise executive justification, naming which pool (Critic's or Hunter's) your final citations came from and why.

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
Input -> Hunter says "met" from p3_b1; Critic overturns to "not_met" because p3_b1 lacks the control and instead cites p3_b4, which documents only a partial adjacent mechanism in the inspected scope.
Reasoning -> assumptions: ["Grounded met/not_met verdicts must keep at least one citation."]; logic_summary: "The Critic invalidated the Hunter's evidence and cited the inspected scope showing only partial adjacent coverage, so the final verdict is not_met with that scope citation."; output -> final_verdict "not_met", final_citations include p3_b4.

Rules:
- `final_verdict` is the single binding decision.
- `final_citations` may come from the Critic's verified list, or from the Hunter's grounded citations when the Critic's non-adoption was an inferred doubt rather than an evidence-backed rejection — never from a block ID in the Critic's invalidated list, and never a citation you cannot personally verify is genuinely quoted in the original TSD context supplied above.
- `met` requires citations you have verified prove the core claim, not merely same-topic evidence.
- `not_met` also requires at least one citation anchoring the inspected scope, partial implementation, contradiction, or closest surrounding evidence that justifies the failure finding. Do not return citationless `not_met`.
- `na` is allowed only when the control's governed capability is absent from design scope.
- If the requirement is applicable and the surviving evidence is missing, contradicted, or only partially supports the core claim, use `not_met`.
- If Critic outcome is `PARTIAL`, keep `met` only when the surviving verified evidence still proves the core claim and the remaining objections are peripheral. For a compound requirement (multiple named sub-elements or clauses in one sentence), an objection is peripheral — and `met` should hold — when it targets elaboration/context on the core mechanism rather than the mechanism itself; downgrade to `not_met` only when the objection targets the requirement's central claim.
- Accept direct, universal, equivalent, collective, or entailed evidence as satisfying. Do not downgrade because the TSD uses different wording, spreads evidence across sections, or lacks a separately titled artifact unless the requirement's central claim is that exact artifact or complete inventory.
- Use the original TSD context to resolve applicability, ambiguity, and to verify whether a Hunter citation the Critic didn't adopt is genuinely grounded. If no direct satisfying citation survives for a grounded verdict, select the closest inspected-scope citation from the evidence pools rather than returning an empty list.
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
        {{"block_id": "<verified block_id>", "page_number": <integer>, "quoted_text": "<short quote>", "bbox": {{"x0": null, "y0": null, "x1": null, "y1": null}}}}
      ],
      "severity": "critical" | "high" | "medium" | "low" | "info" | null,
      "recommendation": "<remediation if not_met, else null>"
    }}
  ]
}}

Rules:
- Use child_id exactly as supplied.
- final_citations should be selected from that child's Critic valid_citations first. If the Critic did not adopt a genuinely grounded Hunter citation and did not mark it invalid, you may use that Hunter citation.
- A "met" or "not_met" final verdict requires at least one citation for that same child. For "not_met", cite the inspected scope, partial implementation, contradiction, or closest surrounding evidence that anchors the failure finding.
- The Hunter's finding is an initial pass and the Critic is an independent re-derivation, but neither is automatically binding: independently re-weigh both against the requirement. When Hunter and Critic disagree, prefer the Critic's conclusion only when its objection names a specific missing element, contradicting passage, or narrowing detail — not when it is a bare, unevidenced doubt about a genuinely grounded Hunter citation.
- SYMMETRY RULE: apply the same standard of proof to "not_met" that you apply to "met". INVALIDATION BAR: generic language ("evidence is weak", "doesn't fully address X", "not specific enough") with no named evidence and no failed OBJECT/POLARITY/CLAUSE check (below) is NOT sufficient grounds to downgrade a genuinely grounded "met" to "not_met" — a failed check IS a named gap and satisfies this bar on its own.
- AGREEMENT-AUDIT DUTY: do not treat Hunter/Critic agreement on "met" as settling the matter — for every child, run the checks below for a named, concrete gap. Preserve "met" when the evidence is direct, universal, equivalent, collective, or entailed and the only objection is wording literalism, documentation placement, or a missing phrase that the requirement gives only as an example:
  - OBJECT CHECK: does the cited evidence address the specific thing the requirement governs in this standard's terms, or only something lexically similar (e.g. "lookup secrets" = user recovery/backup codes, not API keys or database credentials)? Mismatch → downgrade to "not_met", naming it.
  - POLARITY CHECK: for a prohibition requirement ("X is not used"), answer: "Does the cited text itself claim to be a complete/closed list, or does it just describe what's used without saying nothing else exists?" Naming only the secure/modern mechanism used for a specific purpose is describing, not excluding — downgrade to "not_met" unless the text contains an explicit exhaustiveness statement.
  - CLAUSE CHECK: for a requirement with multiple essential AND-joined clauses, is each evidenced? Treat advisory "should" wording, examples, document organization, and explanatory phrases as supporting context unless they are the explicit object. If an essential clause is genuinely unverified -> "not_met", not "met" on the covered clauses alone.
  If the check can't get past generic doubt or literalism (no named check failure), "met" stands — this duty is not license to invent a gap neither agent raised.
- DOCUMENTATION-COMPLETENESS CARVE-OUT: do not require an exhaustive inventory or a literally-named reference document/standard as a precondition for "met" unless that inventory/document IS the requirement's explicit central claim — evidence proving the core mechanism is enough, and a standard named only as an example ("such as", "e.g.") does not need to be cited by name.
- If Critic outcome was PARTIAL and Hunter's original verdict was "met": a single trivial or peripheral valid citation does not by itself force "met". Keep "met" only if that child's valid_citations addresses the core claim AND no surviving objection/missing_expected_evidence targets the essential named mechanism or a specifically-named category/relationship the requirement requires. Otherwise → "not_met", provided the objection names that specific gap (per the INVALIDATION BAR above) and final_citations anchors the inspected scope; do not default to "not_met" on silence alone.
- When the requirement names specific categories, data types, or a specific relationship between named parties, require evidence naming one of those specific items — generic same-topic coverage is not enough. Before downgrading solely for lack of specificity, check whether that detail appears elsewhere in the available context for that child, not just in the citation either agent happened to quote.
- Compound requirements (multiple named sub-elements in one sentence): if one core mechanism is being tested and the other named elements are elaboration on it, verified evidence of the core mechanism is enough — don't require a separate citation for every named element unless that element IS the requirement's core claim (e.g. an explicit no-sensitive-data-in-logs clause is the crux, not peripheral, for a requirement specifically about redaction).
- Semantic entailment examples: retention periods plus destruction at end of retention can evidence scheduled deletion; automated relevant credential rotation can evidence no unchanging credentials; "secondary proof" or a subsequently required biometric can evidence a secondary factor; prescriptive HSM/lifecycle/rotation mandates can evidence a key-management policy/process; security controls tied to boundaries and threats can evidence architecture security analysis; a uniform RBAC+ABAC authorization path can evidence a single composite authorization mechanism.
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
