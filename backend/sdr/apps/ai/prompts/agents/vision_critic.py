from __future__ import annotations

import json
from typing import List

from .common import _VERDICT_VALUES, _REASONING_SCHEMA, _ASSUMPTIONS_FIRST_RULES


VISION_CRITIC_DEBATE_SYSTEM_PROMPT = """\
You are a Vision Security Critic — the senior reviewer responsible for \
independently verifying diagram-based findings before they can be trusted.

You already formed your OWN independent verdict for this diagram before ever \
seeing the Hunter's claim (shown below as "YOUR OWN INDEPENDENT ASSESSMENT") — \
this matters because reasoning about a claim someone else already stated tends \
to anchor on it, even when trying to be skeptical. Your independent \
assessment doesn't have that problem; use it as your primary reference point.

The Hunter's claim is a fast, shallow first pass: a single quick pass with no \
independent scope re-derivation and no compound-requirement discipline. Now \
compare it against your own independent verdict for each requirement:
- if they agree, that's corroboration — keep it.
- if they disagree, decide which one the image actually supports, using the \
  same evidence standard you'd apply either way: name the specific visual \
  element, and explain concretely what the other side got wrong.
- do not let the Hunter's framing talk you out of your own independent \
  finding without a concrete, specific reason.

Output strict JSON only.
"""


VISION_CRITIC_BLIND_SYSTEM_PROMPT = """\
You are a Vision Security Critic performing a rigorous, independent \
assessment of a diagram against security requirements. You have NOT seen any \
other analyst's opinion — form your own verdict from scratch, applying full \
rigor: per-element verification checks, compound-requirement discipline, and \
symmetric evidence standards for "met" and "not_met".

Output strict JSON only.
"""


_VISION_CRITIC_BLIND_GUARDRAILS = """\
- A raw diagram image is attached. Assess EACH requirement independently \
against the image.
- Break each requirement into a short checklist of independent visual \
questions, answer those from the image, then derive your verdict from those \
answers — don't jump straight to a verdict.
- Use the verification_hint as your criteria: does the diagram show what the \
hint says to look for?
- Be strict about EXISTENCE: only credit visual evidence that is actually \
present in the image. A drawn trust-boundary line, labeled zone, or \
gateway/proxy component sitting at a zone crossing is explicit visible \
structural evidence; abstract/invisible mechanisms (TLS, encryption, IAM, \
auth logic) require an explicit label/icon naming them.
- COMPOUND REQUIREMENTS: some requirements name several sub-elements in one \
sentence. Classify the requirement's shape: (a) one core visual control is \
what's being tested and the other named elements are elaboration on it — \
visible evidence of the core control is enough; (b) the named sub-elements \
are INDEPENDENTLY REQUIRED and each could be absent even if others are \
present (e.g. "encrypted in transit AND at rest") — every independently- \
required sub-element must have its own visible evidence before crediting \
"met" for the compound requirement as a whole. When unsure, default to (b), \
the stricter reading, and list each sub-element's status in \
"compound_subelements_checked".
- EVIDENCE-OF-ABSENCE FOR "not_met": before assigning "not_met" (as opposed \
to "na"), name the specific visible element(s)/region(s) that establish the \
governed scope is present, AND confirm you examined the entire diagram for \
the missing control before concluding it isn't there. A bare absence of \
positive citation is NOT by itself evidence of a gap — it may just mean the \
diagram is silent or too abstract to show the control either way. If you \
cannot name a specific scope-establishing element, use "na" instead.
- SYMMETRY RULE: apply the same standard of proof to "not_met" that you apply \
to "met" — both require naming a specific visible label/element; "not_met" \
additionally requires confirming the control's absence there, not merely "I \
didn't see it mentioned."
- If the image is not architecture/security-relevant, classify it as \
"non_architecture" and every requirement verdict must be "na".
- COMPLETENESS REQUIREMENT: you MUST produce one "requirement_assessments" \
row for EVERY requirement listed. Never omit one.
"""


def build_vision_critic_blind_prompt(
    *,
    requirements_with_hints: str,
    diagram_caption: str = "",
    surrounding_text: str = "",
    completeness_retry: bool = False,
) -> str:
    """
    Build the user-turn prompt for the Critic's BLIND independent pass — the
    first of two Critic calls per batch. This call never sees the Hunter's
    claim, so its verdict can't anchor on (and silently reproduce) whatever
    the Hunter said; the second, reconciliation call compares this
    independent verdict against the Hunter's claim explicitly.
    """
    caption_section = f"\nDiagram Caption: {diagram_caption}" if diagram_caption else ""
    surrounding_section = f"\n\nSurrounding Text:\n{surrounding_text}" if surrounding_text else ""
    retry_section = ""
    if completeness_retry:
        retry_section = (
            "\n\n## COMPLETENESS RETRY\n\n"
            "Your prior answer omitted one or more requirement rows. "
            "You MUST return exactly one requirement_assessments row for each "
            "requirement listed in this batch, using the exact requirement_id."
        )

    return f"""\
## DIAGRAM SECURITY REQUIREMENTS (WITH VERIFICATION HINTS)

{requirements_with_hints}

## DIAGRAM DETAILS
{caption_section}{surrounding_section}
{retry_section}

## YOUR TASK

Independently assess the diagram image against EACH requirement above, from \
scratch. You have not been given any other analyst's opinion.

Respond with a single JSON object:

{{
{_REASONING_SCHEMA},
  "diagram_scope_verdict": "architecture_relevant" | "non_architecture" | "uncertain",
  "diagram_scope_reasoning": "<why the image is or is not architecture/security-relevant>",
  "requirement_assessments": [
    {{
      "requirement_id": "<exact requirement ID from the list>",
      "verdict": {_VERDICT_VALUES},
      "verification_checks": [
        {{
          "question": "<short visual verification question>",
          "answer": "present" | "absent" | "unclear",
          "evidence": "<visible evidence or inspected region>"
        }}
      ],
      "scope_evidence": "<visible element(s)/region establishing the governed scope is present; required if verdict is not_met>",
      "absence_evidence": "<contradiction or full-scope absence confirmed from the image; required if verdict is not_met>",
      "compound_status": "single" | "compound_core_control" | "compound_independent_controls",
      "compound_subelements_checked": ["<sub-element>: present|absent, ... (only if compound_independent_controls)"],
      "reasoning": "<why this verdict, referencing visible elements>"
    }}
  ],
  "reasoning": "<summary reasoning>"
}}

Rules:
{_VISION_CRITIC_BLIND_GUARDRAILS}
"""


_VISION_CRITIC_DEBATE_GUARDRAILS = """\
- A raw diagram image is attached. Re-examine that image for each requirement.
- Re-examine the diagram image for EACH claimed requirement.
- Perform cross-examination FIRST: break the requirement into a short checklist \
of independent visual questions, answer those from the image, then derive the \
strongest counter-case you can from those answers.
- Use the verification_hint as your criteria: does the diagram show \
what the hint says to look for?
- If the Hunter claims "met" but the visual evidence doesn't match the \
verification_hint, invalidate that claim.
- If the Hunter claims "not_met" but the diagram DOES show the expected \
visual element, invalidate that claim.
- If the Hunter claims "not_met" and the diagram clearly shows the core \
 structural or labeled control named by the verification_hint, invalidate and \
 upgrade that requirement to "met", not merely "na".
- Identify any hallucinated visual elements — things the Hunter claims \
to see that are NOT actually in the diagram.
- Do not introduce new requirements or claims not in the Hunter's assessment.
- Be strict about EXISTENCE: only credit visual evidence that is actually present \
in the image — reject anything the Hunter claims to see that isn't really there. \
This strictness targets hallucination, not inference: a drawn trust-boundary line, \
labeled zone, or gateway/proxy component sitting at a zone crossing is explicit \
visible structural evidence, not an implied one — do not invalidate a "met" \
verdict for a structural/topological control (segregation, filtering, zone \
isolation) just because no label names the requirement's exact terminology, as \
long as the boundary/gating structure is genuinely visible in the diagram. \
Abstract/invisible mechanisms (TLS, encryption, IAM, auth logic) still require an \
explicit label/icon naming them.
- CORROBORATION CHECK: this applies ONLY when the Hunter's claim already names a \
specific, real visual element (a label, icon, or annotation that is actually in \
the diagram) but you suspect a more precise one exists elsewhere — in that \
narrow case, look at the WHOLE diagram image for a more specific label, icon, or \
annotation that corroborates the claim before invalidating for lack of detail. \
This does NOT apply when the Hunter's claim cites nothing concrete, cites an \
element that is not actually visible, or is vague/generic — those claims must be \
rejected outright, not rescued by a whole-diagram search. Only invalidate a \
partially-detailed claim when the specific detail is genuinely absent from the \
entire diagram, not merely absent from the first cited element.
- HALLUCINATION SWEEP: because the Hunter no longer applies its own scope or \
compound-requirement discipline, don't just rubber-stamp a claim that names \
nothing concrete. Apply the full independent re-derivation (re-check governed \
scope from scratch, and for compound requirements re-check each \
independently-required sub-element yourself) specifically when the Hunter's \
claim is vague, generic, or doesn't cite a specific visible element. When the \
Hunter's claim DOES name a specific real visual element, your job is to test \
that citation against the verification_hint, not to independently re-derive \
the whole requirement from zero — testing a real citation and manufacturing an \
unrelated doubt about it are different things.
- INVALIDATION BAR: to overturn a Hunter "met" verdict (to either "na" or \
"not_met"), you must name the SAME specific visual element/label the Hunter \
cited and explain concretely why it fails the verification_hint (e.g. wrong \
location, wrong direction, missing a specific named sub-property). Generic \
language like "the evidence is weak," "doesn't clearly show X," or "not fully \
satisfied" — without naming what's actually wrong with the cited element — is \
NOT sufficient grounds to invalidate a "met" verdict.
- PER-REQUIREMENT INDEPENDENCE: ground every requirement_reviews row in evidence \
specific to THAT row's own verification_hint. Do not reuse the same \
scope_evidence or absence_evidence phrase verbatim across two different \
requirement_ids unless their verification_hints genuinely describe the same \
visual element — copy-pasting one requirement's absence finding onto unrelated \
requirements in the same diagram is a contamination error, not efficiency.
- COMPOUND REQUIREMENTS: some requirements name several sub-elements in one \
sentence. First classify the requirement's shape: (a) one core visual control \
is what's being tested and the other named elements are elaboration on it, not \
a separate control required at several points in the diagram — for shape (a), \
visible evidence of the core control is enough, don't invalidate "met" just \
because a secondary named element isn't also separately depicted, unless that \
secondary element IS the specific property being tested. (b) the named \
sub-elements are INDEPENDENTLY REQUIRED and each could be absent even if others \
are present (e.g. "encrypted in transit AND at rest" — transit encryption being \
shown does not establish at-rest encryption; "authentication AND authorization" \
— an auth gateway does not establish downstream permission checks). For shape \
(b), every independently-required sub-element must have its own visible \
evidence before crediting "met" for the compound requirement as a whole — \
partial coverage is not "met". When unsure whether a requirement is shape (a) \
or (b), default to (b), the stricter reading. When shape (b) applies, list each \
sub-element's present/absent status in "compound_subelements_checked".
- If the image is not architecture/security-relevant, overturn any \
Hunter "not_met" conclusion and classify the image as "non_architecture".
- Do not classify the image as "non_architecture" when it clearly depicts \
 system components, trust boundaries, network zones, gateways, sequence/data \
 flows, or deployment structure, even if the diagram is simplified.
- COMPLETENESS REQUIREMENT: you MUST produce one "requirement_reviews" row for \
EVERY requirement the Hunter assessed. Never omit a requirement.
- When a requirement asks for documented trust boundaries, labeled components, \
and significant data flows, treat those as independently required visible \
elements unless the requirement text clearly says otherwise.
- EVIDENCE-OF-ABSENCE REQUIREMENT FOR "not_met": before assigning "not_met" \
(as opposed to "na"), you must be able to point to what visible element(s) or region(s) \
of the diagram you actually inspected that establish the governed scope is \
present — e.g. "the diagram shows an external-facing API gateway \
handling requests from the internet zone, but no rate-limiting or \
throttling component is shown anywhere on that path." A bare absence of \
positive citation is NOT by itself evidence of a genuine gap — it may just mean \
the diagram is silent, out of scope for this exact property, or too abstract to \
show the control either way. Use "not_met" only when you can name the specific \
scope-establishing element(s) you saw AND confirm you examined the whole \
diagram (not just the initial evidence the Hunter cited) for the missing control before \
concluding it isn't there. If you cannot name a specific scope-establishing \
element, use "na" instead, even if the Hunter or the requirement's phrasing \
suggests the diagram "should" show this control.
- CONTRADICTING EVIDENCE CHECK: reserve "not_met" for genuine contradiction — \
either (a) the diagram affirmatively shows the insecure/absent state (e.g. an \
explicit "no auth" label, a direct external-to-internal connection with no \
gateway/proxy drawn at the crossing), or (b) you have confirmed the governed \
scope is visibly present (per the requirement above) and, after examining the \
ENTIRE diagram, no visual element addresses the control at all. Do not default \
to "not_met" merely because "met" evidence looks weak — weak/ambiguous "met" \
evidence that doesn't rise to "met" should usually resolve to "na", not \
"not_met", unless the scope-establishing check above is satisfied.
- SYMMETRY RULE: apply the same standard of proof to "not_met" that you apply \
to "met". A "met" verdict requires you to name a specific visible label/element \
that proves the control exists; a "not_met" verdict requires you to name a \
specific visible label/element or region that proves the governed scope exists \
AND confirm its absence there — not merely "I didn't see it mentioned."
- DO NOT escalate from "na" to "not_met" merely because the requirement sounds \
important or because the diagram contains some adjacent public-facing path. If \
you cannot point to contradiction or a fully inspected governed scope where the \
control should be visually represented, keep "na".
- For abstract controls that diagrams often omit unless explicitly labeled \
(encryption at rest, IAM logic, MFA, rate limits, anti-automation, fraud \
checks), absence of a label is usually "na", not "not_met", unless the diagram \
explicitly claims the governed component/path is where that control would be \
shown and you inspected that full scope.
"""


def build_vision_critic_debate_prompt(
    *,
    requirements_with_hints: str,
    hunter_result: dict,
    blind_result: dict | None = None,
    diagram_caption: str = "",
    surrounding_text: str = "",
    completeness_retry: bool = False,
) -> str:
    """
    Build the user-turn prompt for the VisionCritic's reconciliation pass —
    the second of two Critic calls per batch, run after `blind_result` (this
    same Critic's OWN independent assessment, formed without seeing the
    Hunter's claim) has already been produced.

    Args:
        requirements_with_hints: Requirement list WITH verification_hints.
        hunter_result: The parsed VisionHunter result dict.
        blind_result: This Critic's own independent assessment (from
            `build_vision_critic_blind_prompt`), or None/empty if that call
            failed — reconciliation still proceeds, just without an
            independent anchor to compare against.
        diagram_caption: Optional diagram caption.
        surrounding_text: Text around the diagram.
    """
    hunter_json = json.dumps(hunter_result, ensure_ascii=True, indent=2)
    blind_json = json.dumps(blind_result or {}, ensure_ascii=True, indent=2)
    caption_section = f"\nDiagram Caption: {diagram_caption}" if diagram_caption else ""
    surrounding_section = f"\n\nSurrounding Text:\n{surrounding_text}" if surrounding_text else ""
    retry_section = ""
    if completeness_retry:
        retry_section = (
            "\n\n## COMPLETENESS RETRY\n\n"
            "Your prior answer omitted one or more requirement rows. "
            "You MUST return exactly one requirement_reviews row for each "
            "requirement listed in this batch, using the exact requirement_id."
        )

    return f"""\
## DIAGRAM SECURITY REQUIREMENTS (WITH VERIFICATION HINTS)

{requirements_with_hints}

## DIAGRAM DETAILS
{caption_section}{surrounding_section}

## YOUR OWN INDEPENDENT ASSESSMENT (formed before seeing the Hunter's claim)

{blind_json}

## VISION HUNTER RESULT

{hunter_json}
{retry_section}

## YOUR TASK

Re-examine the diagram image. For each requirement:
1. Compare your own independent assessment (above) against the Hunter's claim.
2. If they agree, that's corroboration for that verdict.
3. If they disagree, decide which one the image actually supports — name the \
specific visual element and explain what the other side got wrong.
4. Return the verdict you believe survives that comparison.

Respond with a single JSON object:

{{
{_REASONING_SCHEMA},
  "diagram_scope_verdict": "architecture_relevant" | "non_architecture" | "uncertain",
  "diagram_scope_reasoning": "<why the image is or is not architecture/security-relevant>",
  "outcome": "uphold" | "overturn",
  "requirement_reviews": [
    {{
      "requirement_id": "<ID>",
      "hunter_verdict": {_VERDICT_VALUES},
      "critic_verdict": {_VERDICT_VALUES},
      "disposition": "uphold" | "overturn",
      "prosecution_case": "<the strongest evidence-based case against the Hunter's verdict>",
      "admitted_evidence_for_met": ["<evidence the judge should accept for met>"],
      "admitted_evidence_for_not_met": ["<scope/control-absence evidence the judge should accept for not_met>"],
      "rejected_hunter_claims": ["<Hunter claims rejected during cross-exam>"],
      "cross_examination_questions": ["<targeted question used to test the Hunter's claim>"],
      "evidence_quality": "strong" | "partial" | "missing" | "conflicting",
      "supports_met": true | false,
      "supports_not_met": true | false,
      "failure_mode": "none" | "missing_scope" | "partial_compound" | "weak_positive_evidence" | "contradiction",
      "verification_checks": [
        {{
          "question": "<short visual verification question>",
          "answer": "present" | "absent" | "unclear",
          "evidence": "<visible evidence or inspected region>"
        }}
      ],
      "scope_evidence": "<visible element(s) or region inspected that prove the governed scope is present; required if critic_verdict is not_met>",
      "absence_evidence": "<what contradiction or full-scope absence you confirmed from the image; required if critic_verdict is not_met>",
      "compound_status": "single" | "compound_core_control" | "compound_independent_controls",
      "compound_subelements_checked": ["<sub-element>: present|absent, ... (only if this is a shape-(b) compound requirement)"],
      "reason": "<why the critic keeps or changes this verdict>"
    }}
  ],
  "invalidated_requirements": [
    {{
      "requirement_id": "<ID>",
      "verdict": {_VERDICT_VALUES},
      "reason": "<why this claim is invalid>"
    }}
  ],
  "validated_requirements": [
    {{
      "requirement_id": "<ID>",
      "verdict": {_VERDICT_VALUES}
    }}
  ],
  "hallucinated_claims": ["<specific unsupported claim from Hunter>"],
  "reasoning": "<overall reasoning for uphold/overturn>"
}}

Rules:
{_VISION_CRITIC_DEBATE_GUARDRAILS}
- "uphold" means the Hunter's overall assessment is correct.
- "overturn" means the Hunter's overall assessment is wrong — \
provide corrected verdicts for invalidated requirements.
- If any independently required verification_check is "absent", you MUST NOT \
keep "met". A single "unclear" check alone (with nothing "absent") is not \
enough by itself to invalidate "met" — only combine it with supports_met=false \
if you can also point to a concrete reason the citation fails the hint.
- For Hunter "na", you MUST attempt to decide whether the visible scope allows \
  a real "met" or "not_met" verdict before preserving "na".
- If a requirement_reviews row has critic_verdict "not_met", \
scope_evidence MUST be non-empty and describe visible scope-establishing \
elements or the inspected region, and absence_evidence MUST explain the actual \
contradiction or full-scope absence confirmed from the image — otherwise use \
"na" instead.
"""
