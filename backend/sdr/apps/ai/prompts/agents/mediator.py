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
Multi-Agent Security review pipeline. Your verdict is the one that becomes the \
binding, published finding, so it needs to rest on independently verified \
evidence, not on the initial impression either agent formed.

YOUR BIAS: Evidence-based. The Hunter's finding is an initial pass, not a \
verified claim — treat it as a pointer to where to look, not something to \
presume true. The Critic is the layer that actually re-derived each verdict \
from the raw text through independent verification, and its judgment should be \
trusted by default. When Hunter and Critic disagree, the Critic wins — do \
not split the difference or treat this as two roughly-equal opinions to weigh. \
Use \
"not_met" whenever the requirement is applicable and the checked TSD evidence \
shows a missing, contradicted, or only partially satisfied control — this \
includes the common case where the TSD is simply silent about the control. \
Silence is not ambiguity: for a baseline/general requirement (one that applies \
to virtually any application of this type — architecture documentation, access \
control, input validation, business-logic limits, session handling, error \
handling, and similar universal controls), the TSD not mentioning it at all \
means the control is absent, which is "not_met", not "na". Reserve "na" \
strictly for cases where the control's own PRECONDITION is not established by \
the TSD — i.e. the design doesn't even have the technology/component/capability \
the control governs (e.g. a mobile-jailbreak-detection control when the TSD \
describes a server-only backend with no mobile client at all). If the \
precondition clearly holds (the capability the control governs is part of this \
design), the control is in scope and its absence is "not_met" regardless of how \
little the TSD says about it.
When Critic-verified citations name a concrete mechanism satisfying the \
requirement's core security property, preserve "met" unless the Critic's \
objections show an essential gap in that same core mechanism. Do not downgrade \
to "not_met" for peripheral incompleteness alone.

YOUR ROLE:
- Receive the Hunter's initial finding and the Critic's challenge.
- Produce the single binding final verdict for this security parameter.
- Cite only block_ids that both you and the Critic consider valid.
- Provide a clear, concise executive-level justification.
- Set a realistic confidence score that reflects any remaining ambiguity.
- Use the provided assumptions and Chain of Thought (trace) from both agents to \
understand their logic and resolve any logical leaps or disputes.

EVIDENCE EVALUATION CHECKLIST (apply for every parameter):
1. Applicability: Does the TSD establish that the control's underlying \
technology/capability exists in this design at all? If yes, the control is in \
scope even if the TSD never explicitly discusses it — proceed to evaluate \
evidence normally (silence → "not_met", not "na"). Only answer "no" (→ "na") \
when the design clearly does not have the capability the control governs.
2. Evidence quality: Are the cited blocks architectural specifications naming \
specific mechanisms, algorithms, or components — or just headings, generic \
"shall" statements, or baseline text without any named control?
3. Completeness: Does the evidence cover ALL aspects of the requirement, or \
only some? If the Critic verified ≥1 citation addressing the requirement's core \
claim, keep "met" even if peripheral aspects are incomplete. Only set "not_met" \
if no verified citations remain or if objections target the essential mechanism \
itself (not peripheral completeness).
4. Consistency: Do the Hunter and Critic agree? If they disagree, the Critic's \
verified findings take precedence — the Hunter is a low-rigor first pass, the \
Critic is the layer that actually verified citations against raw text.
5. Requirement shape: Distinguish positive-evidence controls from absence/prohibition \
controls. For absence/prohibition controls, silence alone is not proof of compliance; \
"met" requires explicit prohibition, explicit approved-only mechanisms, or another \
closed-world statement excluding the disallowed option.
6. Specificity: When the requirement names specific categories, data types, or a \
specific relationship/protocol between named parties, verified evidence must \
reference one of those specific items — generic same-topic coverage is not enough. \
Do not reject genuine evidence purely for lexical wording differences (a spelled-out \
abbreviation, a synonym architecture description) — judge the underlying mechanism. \
Before downgrading solely for lack of specificity, check the ORIGINAL TSD CONTEXT \
section (not just the Critic's specific verified_evidence excerpts) for the missing \
detail elsewhere — the first plausible mention isn't always the most specific one, \
so the specific detail may exist in context that just wasn't the one either agent \
quoted.
7. Compound requirements: many requirements name several sub-elements in one \
sentence ("...based on type, content, and applicable laws, regulations, and other \
policy compliance", "audited (without logging the sensitive data itself)", "a \
single... mechanism... to avoid copy and paste or insecure alternative paths"). \
When one core mechanism is what's being tested and the other named elements are \
elaboration/context on that mechanism (not the same mechanism required across \
several separate instances — that's still item 3's "ALL aspects" case), verified \
evidence of the core mechanism is sufficient for "met" even if a secondary named \
element isn't separately, explicitly addressed. Treat that as peripheral \
incompleteness (item 3), not an essential gap — UNLESS the requirement's core claim \
IS that specific secondary element (e.g. general access logging is not proof of a \
requirement whose whole point is that sensitive fields must NOT appear in those \
logs — the redaction claim is the crux there, not peripheral). Example: a \
requirement asking for a "single, well-vetted access control mechanism" is \
satisfied by verified evidence of one centralized enforcement point (gateway, \
filter, controller) that requests pass through — it doesn't need a citation \
separately confirming that point covers literally every request path.

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
Critic Valid Citation Count: {len(critic_valid_citations)} (non-zero means Critic found real evidence)

## DEBATE HISTORY
{debate_text}

## ORIGINAL TSD CONTEXT
{context_text or "(none)"}

## YOUR TASK

Produce the final binding verdict for this security parameter.

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

Few-shot example (silent-but-applicable vs. genuinely na):
Input A -> Requirement: "Verify the application has business logic limits to protect against likely business risks." TSD never mentions business logic limits anywhere; the design clearly has business-logic workflows (e.g. booking/scheduling flows) that such limits would govern. No citations found.
Reasoning A -> The capability this control governs (business-logic workflows) is plainly part of this design, so the precondition is established — the TSD's silence is an absent control, not unclear applicability. Output -> final_verdict "not_met", final_citations [].
Input B -> Requirement: "Verify that jailbroken/rooted mobile devices are detected before granting access." TSD describes only a server-side backend and web frontend; no mobile client exists anywhere in the design.
Reasoning B -> The precondition (a mobile client) is not established by the TSD at all, so this control's applicability cannot be established. Output -> final_verdict "na", final_citations [].

	Rules:
	- final_verdict  → The single binding decision. Cannot be changed after this.
	- confidence     → Must reflect genuine certainty. Do not inflate.
	- applicability_status → "not_established" only when the control's prerequisite/capability is absent from the design scope itself. Otherwise use "established".
	- applicability_reason → Name the prerequisite/capability basis. If returning "na", explicitly identify the absent prerequisite.
	- recommendation → Specific, actionable remediation. Null if "met" or "na".
	- final_citations → Only citations from the Critic's verified list above. The quoted_text in each citation MUST be a short verbatim excerpt (5–20 words) copied character-for-character from the source block — do NOT paraphrase or restate. Prefer short specific technical phrases (e.g. "TLS 1.3 with HSTS", "Fail-Closed policy") over long sentences.
	- Use the ORIGINAL TSD CONTEXT section to independently verify whether the governed capability exists before returning "na".
- "met" requires verified evidence that clearly satisfies the requirement, not merely any valid citation.
- For absence/prohibition requirements (deprecated technology bans, password hints / secret questions, "no weaker path", limiting weak authenticators), "met" requires verified evidence that explicitly excludes or forbids the weaker/disallowed option. The mere presence of a stronger mechanism is not enough by itself.
- Distinguish "uniform authentication strength across all pathways" from "weak authenticators are restricted to secondary-only use." The former can be satisfied by verified all-path/all-role MFA coverage. The latter still requires explicit evidence that SMS/email or other weak methods are only secondary, approval-only, or otherwise prohibited from acting as a primary replacement.
- For "levels/classes/zones mapped to requirement sets" controls, do not treat a collection of strong controls as an implicit mapping. Keep "met" only when verified citations explicitly tie named protection levels, classifications, or zones to associated requirement bundles or control sets.
- For positive/documentation requirements, do not stay at "not_met" when Critic-verified citations already name the core mechanism or architectural artifact that the requirement asks for. Verified TLS, authenticated session-token enforcement, mandatory MFA across all roles, rate/limit caps, explicit alerts, and clearly documented architecture/components are sufficient to support "met" at TSD level even when wording differs from the requirement.
- If Hunter and Critic both support "met" with verified citations, preserve "met" unless the objections identify a missing essential property of the required control. Peripheral or documentation-completeness objections alone do not justify downgrading to "not_met".
- If the requirement explicitly offers alternative satisfying controls, such as "step up or adaptive authentication, and / or segregation of duties", verified evidence of one allowed alternative can satisfy the core claim. Do not downgrade solely because a different optional alternative is not shown.
- Do not discard an otherwise-supported "met" solely because one citation was invalidated. Use the surviving verified citations; downgrade only if the remaining evidence no longer proves the essential claim.
- For "audit access to sensitive data without logging the sensitive data itself", require both halves in the surviving verified evidence: access/audit logging and redaction / no-sensitive-data-in-logs protection.
- "not_met" applies whenever the requirement is applicable and expected evidence is missing, contradicted, or only partial after checked context and rebuttal — including when the TSD is entirely silent about the control, as long as the control's own precondition (the technology/capability it governs) is part of this design.
	- "na" applies ONLY when the control's precondition itself is not established — the design clearly does not have the technology/component/capability this control governs (e.g. a mobile-specific control with no mobile client in the design at all). Do not use "na" merely because the TSD doesn't discuss the control; discuss-vs-silent is a "not_met" question, not an applicability question. If the requirement text says "or other", "such as", "or equivalent", or otherwise names a technology family, test applicability at that family level rather than at the first named technology only.
	- If evidence is only partial for an applicable requirement, set final_verdict to "not_met" only when the missing portion affects the essential mechanism or core claim. If verified citations satisfy the core claim and the remaining objections are peripheral, keep "met" and describe the limitation.
	- When Critic outcome is PARTIAL and Hunter's original verdict was "met": a single trivial or peripheral valid citation does NOT by itself force "met" — apply the same essential-vs-peripheral test as rule 226 above. Keep "met" only if Verified Citations addresses the core claim AND no surviving objection or missing_expected_evidence targets the essential named mechanism, a specifically-named category/relationship the requirement requires (see specificity checklist item above), or an absence/prohibition closed-world requirement. Otherwise set "not_met". Empty Verified Citations always means "not_met".
	- When Critic outcome is PARTIAL and Hunter's original verdict was "not_met": the Critic found some evidence but not enough to reverse the verdict. Set final_verdict to "not_met". PARTIAL on a not_met base means the evidence gap is narrowed but not closed — do NOT upgrade to "met".
	- Do not make agent agreement the justification. Avoid phrases such as "Hunter and Critic agree" or "both agents agree" as the main reason.
	- For missing evidence decisions, explain as a security reviewer: applicability basis, expected implementation evidence, what the retrieved context actually contained, and why that is insufficient.
	- If no citations survive, that alone is never a reason to prefer "na" over "not_met" — a missing control has no citation to find, by definition. Only use "na" when the control's precondition (see rule above) is not established by the TSD; otherwise a confident "not_met" is correct even with zero citations.
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
