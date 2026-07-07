from __future__ import annotations

import json
from typing import List

from .common import _VERDICT_VALUES, _REASONING_SCHEMA, _ASSUMPTIONS_FIRST_RULES


VISION_CRITIC_DEBATE_SYSTEM_PROMPT = """\
You are a Vision Security Critic.

Your job is to re-examine the same diagram image and challenge the \
VisionHunter's claims. You verify whether the cited visual evidence \
actually exists in the diagram and whether it truly addresses the \
claimed requirement. You must also challenge whether the image is even \
an architecture/security-relevant diagram in the first place.

Use the verification_hint for each requirement as your criteria for \
what to look for.

Output strict JSON only.
"""


_VISION_CRITIC_DEBATE_GUARDRAILS = """\
- Re-examine the diagram image for EACH claimed requirement.
- The diagram has been overlaid with numbered markers (e.g., [1], [2]). Red markers label text blocks; blue markers label non-text visual elements. Ensure the Hunter explicitly cited correct marker numbers for its evidence. For each marker ID in the Hunter's "marker_ids_cited", verify that marker actually exists visibly in the diagram and that its content matches the claimed evidence. List any invalid citations in "invalid_marker_citations".
- Use the verification_hint as your criteria: does the diagram show \
what the hint says to look for?
- If the Hunter claims "met" but the visual evidence doesn't match the \
verification_hint, invalidate that claim.
- If the Hunter claims "not_met" but the diagram DOES show the expected \
visual element, invalidate that claim.
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
- If the image is not architecture/security-relevant, overturn any \
Hunter "not_met" conclusion and classify the image as "non_architecture".
- COMPLETENESS REQUIREMENT: for EVERY requirement the Hunter marked "met", \
you MUST add an entry for it to either "validated_requirements" (if you \
confirm the evidence) or "invalidated_requirements" (if you reject it). \
Never leave a Hunter "met" requirement unclassified in both lists.
"""


def build_vision_critic_debate_prompt(
    *,
    requirements_with_hints: str,
    hunter_result: dict,
    diagram_caption: str = "",
    surrounding_text: str = "",
) -> str:
    """
    Build the user-turn prompt for the VisionCritic in the debate pipeline.

    Args:
        requirements_with_hints: Requirement list WITH verification_hints.
        hunter_result: The parsed VisionHunter result dict.
        diagram_caption: Optional diagram caption.
        surrounding_text: Text around the diagram.
    """
    hunter_json = json.dumps(hunter_result, ensure_ascii=True, indent=2)
    caption_section = f"\nDiagram Caption: {diagram_caption}" if diagram_caption else ""
    surrounding_section = f"\n\nSurrounding Text:\n{surrounding_text}" if surrounding_text else ""

    return f"""\
## DIAGRAM SECURITY REQUIREMENTS (WITH VERIFICATION HINTS)

{requirements_with_hints}

## DIAGRAM DETAILS
{caption_section}{surrounding_section}

## VISION HUNTER RESULT

{hunter_json}

## YOUR TASK

Re-examine the diagram image. For each requirement the Hunter assessed:
1. Check whether the Hunter's cited visual evidence actually exists in the diagram.
2. Compare against the verification_hint — does the diagram show what the hint says to look for?
3. Identify any hallucinated or unsupported claims.

Respond with a single JSON object:

{{
{_REASONING_SCHEMA},
  "diagram_scope_verdict": "architecture_relevant" | "non_architecture" | "uncertain",
  "diagram_scope_reasoning": "<why the image is or is not architecture/security-relevant>",
  "outcome": "uphold" | "overturn",
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
  "invalid_marker_citations": ["<marker_id>: <why the cited marker is wrong or absent>"],
  "reasoning": "<overall reasoning for uphold/overturn>"
}}

Rules:
{_VISION_CRITIC_DEBATE_GUARDRAILS}
- "uphold" means the Hunter's overall assessment is correct.
- "overturn" means the Hunter's overall assessment is wrong — \
provide corrected verdicts for invalidated requirements.
"""
