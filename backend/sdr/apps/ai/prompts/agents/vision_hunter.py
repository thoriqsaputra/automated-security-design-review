from __future__ import annotations

import json
from typing import List

from .common import _VERDICT_VALUES, _REASONING_SCHEMA, _ASSUMPTIONS_FIRST_RULES


VISION_HUNTER_SYSTEM_PROMPT = """\
You are a Security Diagram Hunter doing a quick first pass over diagram images \
to produce an opening claim about where a security requirement might be satisfied.

Look at the diagram once, check it against the given requirements, and give your \
best call for each one: "met", "not_met", or "na" if you genuinely can't tell.
Move quickly — this is a first pass, not a final verdict; a later reviewer will \
double-check your work against the full verification criteria.

Output strict JSON only.
"""


VISION_HUNTER_REBUTTAL_SYSTEM_PROMPT = """\
You are the Security Diagram Hunter responding to the Critic's cross-examination.

Do NOT re-analyze the whole diagram from scratch. Your job is only to defend or
withdraw disputed claims using visible evidence from the image.

For each disputed requirement:
- defend one specific claim the Critic rejected, if you can ground it clearly
- otherwise concede the point
- do not invent new controls, components, or flows

Output strict JSON only.
"""


_VISION_HUNTER_GUARDRAILS = """\
- A raw diagram image is attached. Base your claims on what you see in that image.
- Give your best-effort verdict per requirement using the verification hint as a guide.
- If a requirement clearly doesn't apply to what the diagram depicts, use "na".
- Each assessment MUST reference the exact requirement_id from the checklist.
- Claims without a matching requirement_id are invalid and will be discarded.
- Classify the image scope:
  - "architecture_relevant" if the image depicts system structure, deployment, \
network, trust boundaries, sequence/data flow, or security control scope.
  - "non_architecture" for screenshots, UI mockups, photos, logos/icons, charts, \
graphs, or decorative illustrations.
  - "uncertain" only when you genuinely cannot tell from the visible image.
- If the image scope is "non_architecture", ALL requirement verdicts MUST be "na" \
and the overall_verdict MUST be "na".
"""


def build_vision_hunter_prompt(
    *,
    requirements_text: str,
    diagram_caption: str = "",
    surrounding_text: str = "",
    tsd_context: str = "",
) -> str:
    """
    Build the user-turn prompt for the VisionHunter.

    Args:
        requirements_text: Compact one-line-per-item requirement list.
        diagram_caption: Optional diagram caption or title.
        surrounding_text: Text immediately around the diagram in the TSD.
        tsd_context: Broader TSD context (section heading, document scope).
    """
    caption_section = f"\nDiagram Caption: {diagram_caption}" if diagram_caption else ""
    surrounding_section = f"\n\nSurrounding Text:\n{surrounding_text}" if surrounding_text else ""
    context_section = f"\n\nTSD Context:\n{tsd_context}" if tsd_context else ""

    return f"""\
## DIAGRAM SECURITY REQUIREMENTS

{requirements_text}

## DIAGRAM DETAILS
{caption_section}{surrounding_section}{context_section}

## YOUR TASK

Evaluate the diagram image against EACH requirement listed above.
Before scoring requirements, decide whether the image is architecture/security-relevant.
For every requirement, state your assessment with visual evidence.

Respond with a single JSON object:

{{
{_REASONING_SCHEMA},
  "diagram_scope_verdict": "architecture_relevant" | "non_architecture" | "uncertain",
  "diagram_scope_reasoning": "<why the image is or is not architecture/security-relevant>",
  "requirement_assessments": [
    {{
      "requirement_id": "<exact requirement ID from the list>",
      "verdict": {_VERDICT_VALUES},
      "scope_present": "yes" | "no" | "unclear",
      "verification_subclaims": ["<visible subclaim the Hunter believes is satisfied or required>"],
      "required_visible_checks": [
        {{
          "check": "<specific visual condition from the verification hint>",
          "status": "present" | "absent" | "unclear"
        }}
      ],
      "strongest_met_evidence": "<best visible evidence for met, if any>",
      "strongest_scope_evidence": "<best visible evidence that the governed scope exists, if any>",
      "visual_evidence": "<what you see in the diagram that supports this verdict>",
      "uncertainty_reason": "<why you used na or why some checks remain unclear>",
      "reasoning": "<why this verdict, referencing visible elements>"
    }}
  ],
  "overall_verdict": {_VERDICT_VALUES},
  "confidence": <float 0.0-1.0>,
  "visual_elements_cited": ["<specific visible element names>"],
  "missing_controls": ["<explicit missing control if overall_verdict=not_met>"],
  "reasoning": "<summary reasoning for the overall verdict>"
}}

Rules:
{_VISION_HUNTER_GUARDRAILS}
- The overall_verdict must follow this rollup: \
any not_met => overall_verdict "not_met"; otherwise if at least one \
requirement is "met" => overall_verdict "met"; otherwise overall_verdict "na".
- You MUST assess every requirement. If a requirement is not relevant \
to this diagram, mark it "na" with reasoning.
"""


def build_vision_hunter_rebuttal_prompt(
    *,
    hunter_result: dict,
    critic_result: dict,
) -> str:
    hunter_json = json.dumps(hunter_result, ensure_ascii=True, indent=2)
    critic_json = json.dumps(critic_result, ensure_ascii=True, indent=2)

    return f"""\
## ORIGINAL HUNTER ASSESSMENT

{hunter_json}

## CRITIC CROSS-EXAMINATION

{critic_json}

## YOUR TASK

Respond ONLY for requirements the Critic disputed or weakened.
For each disputed requirement:
1. defend one visible claim the Critic rejected, OR
2. concede that the claim was too weak.

Respond with a single JSON object:

{{
{_REASONING_SCHEMA},
  "rebuttal_requirements": [
    {{
      "requirement_id": "<ID>",
      "stance": "defend" | "concede",
      "defended_claim": "<specific claim or subclaim being defended>",
      "rebuttal_evidence": "<visible evidence supporting the defense, or why you concede>",
      "reasoning": "<why the Critic is wrong, or why you concede>"
    }}
  ],
  "reasoning": "<short summary of the rebuttal>"
}}

Rules:
- Only address disputed requirements.
- If the Critic showed a required visual subclaim is absent or unclear and you
  cannot point to clear visible evidence, you MUST concede.
- Do not repeat the whole original assessment.
"""
