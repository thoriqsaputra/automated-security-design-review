from __future__ import annotations

import json
from typing import List

from .common import _VERDICT_VALUES, _REASONING_SCHEMA, _ASSUMPTIONS_FIRST_RULES


VISION_MEDIATOR_DEBATE_SYSTEM_PROMPT = """\
You are a Security Diagram Mediator — the final arbiter whose verdict becomes \
the binding, published finding for this diagram.

Your job is to act like a judge in a courtroom-style debate. You do not simply \
average the Hunter and Critic. You decide which evidence is admitted, which \
claims are rejected, and what verdict survives the debate.

You DO see the diagram image. Use the image as an independent tie-breaker while \
also weighing the VisionHunter's assessment against the VisionCritic's challenge. \
The Hunter's assessment is an initial pass, not a verified claim. The Critic \
re-examined the image specifically to verify the Hunter's claims through \
independent inspection, and the Hunter may respond with a rebuttal. Your role is \
to judge that exchange, not to restate it.

Output strict JSON only.
"""


_VISION_MEDIATOR_EVIDENCE_CHECKLIST = """\
EVIDENCE EVALUATION (apply before finalizing verdict):
0. Diagram scope: Is the image architecture/security-relevant at all?
1. Visual grounding: Does the Hunter cite specific visible elements \
or vague/implied ones? Remember the Hunter is a fast, shallow first pass with no \
independent scope or compound-requirement discipline — its claim is a lead to \
verify, not evidence in itself.
1a. Critic independence: did the Critic re-derive scope and compound-completeness \
itself for this requirement, or did it merely echo/rubber-stamp the Hunter's \
claim without independent re-examination? Weight an independently re-derived \
Critic verdict much more heavily than one that just restates the Hunter.
2. Critic validation: Did the Critic confirm or invalidate the key claims?
3. Completeness: Are ALL assessed requirements addressed in the final verdict?
4. Consistency: Does the overall_verdict match the finding-level rollup of \
individual assessments? (any not_met => not_met; else any met => met; else na)
5. Ambiguity: If the Critic found hallucinated claims, prefer "na" over \
"not_met" for those requirements.
6. Compound requirements: when a requirement names several sub-elements, first \
decide whether they're elaboration on one core control (verified evidence of \
that core control is enough — don't downgrade just because a secondary named \
element isn't separately depicted) or independently-required sub-controls \
(each needs its own visible evidence; partial coverage should not be "met"). \
Default to the stricter (independently-required) reading when unsure.
7. Symmetric evidence bar: do not accept a "not_met" verdict merely because \
Hunter/Critic found no positive evidence for "met" — that default is "na", not \
"not_met". Only keep "not_met" when the Critic named specific \
scope-establishing visible elements or regions for this requirement and you can verify them \
yourself against the image.
7a. Absence evidence: a valid "not_met" case must also explain the actual \
contradiction or inspected full-scope absence. "The control is not mentioned" \
is not enough by itself.
8. Verifier precedence: if the Critic's verification_checks do not support a \
"met" verdict, do not keep "met". If the Critic marked a partial compound \
requirement, downgrade "met".
9. Courtroom rule: admit only specific, requirement-matching visible evidence. \
Do not let generic architecture structure substitute for a missing required \
visual subclaim.
"""


def build_vision_mediator_debate_prompt(
    *,
    requirements_with_hints: str,
    hunter_result: dict,
    critic_result: dict,
    hunter_rebuttal_result: dict | None = None,
    completeness_retry: bool = False,
) -> str:
    """
    Build the user-turn prompt for the VisionMediator.

    The Mediator also receives the diagram image and should resolve any
    Hunter/Critic disagreement requirement-by-requirement against that image.

    Args:
        hunter_result: The parsed VisionHunter result dict.
        critic_result: The parsed VisionCritic result dict.
    """
    hunter_json = json.dumps(hunter_result, ensure_ascii=True, indent=2)
    critic_json = json.dumps(critic_result, ensure_ascii=True, indent=2)
    rebuttal_json = json.dumps(hunter_rebuttal_result or {}, ensure_ascii=True, indent=2)
    retry_section = ""
    if completeness_retry:
        retry_section = (
            "\n\n## COMPLETENESS RETRY\n\n"
            "Your prior answer omitted one or more requirements or failed to "
            "give an explicit final verdict per requirement. Return exactly one "
            "assessed_requirements row for every requirement in this batch."
        )

    return f"""\
## DIAGRAM SECURITY REQUIREMENTS (WITH VERIFICATION HINTS)

{requirements_with_hints}

## VISION HUNTER ASSESSMENT

{hunter_json}

## VISION CRITIC CHALLENGE

{critic_json}

## HUNTER REBUTTAL

{rebuttal_json}
{retry_section}

## YOUR TASK

Inspect the attached diagram image, weigh the Hunter opening, the Critic
cross-examination, and the Hunter rebuttal, then produce the final binding
judgment for this diagram.

{_VISION_MEDIATOR_EVIDENCE_CHECKLIST}

Respond with a single JSON object:

{{
{_REASONING_SCHEMA},
  "diagram_scope_verdict": "architecture_relevant" | "non_architecture" | "uncertain",
  "diagram_scope_reasoning": "<why the image is or is not architecture/security-relevant>",
  "final_verdict": {_VERDICT_VALUES},
  "confidence": <float 0.0-1.0>,
  "finding_description": "<clear description of what was found, suitable for a security report>",
  "recommendation": "<actionable recommendation if not_met, otherwise null>",
  "assessed_requirements": [
    {{
      "requirement_id": "<ID>",
      "verdict": {_VERDICT_VALUES},
      "resolution_basis": "hunter_upheld" | "critic_upheld" | "mediator_tiebreak" | "same_verdict_after_cross_exam",
      "winning_side": "hunter" | "critic" | "split",
      "verifier_alignment": "supports_met" | "supports_not_met" | "ambiguous",
      "admitted_evidence": ["<evidence accepted by the judge>"],
      "rejected_evidence": ["<evidence rejected by the judge>"],
      "judge_reason": "<why this side won>",
      "summary": "<1-2 sentence summary of why this verdict>"
    }}
  ]
}}

Rules:
- If the image is judged "non_architecture", the final_verdict MUST be "na" and \
ALL assessed_requirements MUST have verdict "na".
- If all applicable requirements are "na", the final_verdict MUST be "na".
- The final_verdict must follow this rollup: any "not_met" => "not_met"; \
otherwise if at least one assessed requirement is "met" => "met"; otherwise "na".
- If the Critic overturned key claims and you agree, adjust those requirements \
accordingly.
- If the Critic upheld the Hunter and you agree, preserve those verdicts.
- Do not omit any requirement_id that already appears in the Hunter result. If \
you only disagree on some requirements, keep the Hunter's verdicts for the rest.
- If the Critic row is marked as fallback or incomplete, you must independently \
  inspect the image and emit a final verdict anyway; do not silently default to \
  Hunter without a stated evidentiary reason.
- Never keep "met" for a requirement whose Critic requirement_review has \
supports_met=false, any failed/unclear independently required verification_check, \
or failure_mode="partial_compound".
- Treat the Hunter rebuttal as narrow: it can rescue a disputed claim only if it
  points to specific visible evidence that answers the Critic's dispute.
- If the Critic and Hunter disagree, your verdict must explicitly choose a
  winning side or declare the evidence too ambiguous and fall back to "na".
- If the final requirement verdict stays the same as the Hunter's original
  verdict, do not claim the Critic "won". Use `resolution_basis` =
  `hunter_upheld` or `same_verdict_after_cross_exam`, with `winning_side` =
  `hunter` or `split`, and explain whether the Critic only refined rationale
  without changing the outcome.
- Prefer "not_met" over "na" ONLY when the Critic's requirement_reviews entry \
for this requirement names specific scope_evidence establishing the \
governed scope is visibly present AND you can independently confirm, by \
inspecting the attached image yourself, that no visible element addresses the \
control on that scope. If the Critic could not name such scope evidence, or \
you cannot independently confirm it against the image, use "na" instead — a \
"not_met" verdict must rest on the same standard of positive, named visual \
evidence that a "met" verdict requires, just evidence of the control's \
governed context rather than the control itself.
- If the Critic only argues from expectation or silence, and does not provide \
concrete absence_evidence, do not uphold "not_met"; prefer "na".
- Never finalize "not_met" for a requirement whose Critic requirement_review \
row has an empty or missing scope_evidence field, or an empty/missing \
absence_evidence field — downgrade to \
"na" in that case, and note the downgrade reason in that requirement's \
summary.
- recommendation is only required when final_verdict is "not_met".
- finding_description should read like a professional security finding.
{_ASSUMPTIONS_FIRST_RULES}
"""
