from __future__ import annotations

from typing import List

from .common import _VERDICT_VALUES, _REASONING_SCHEMA, _ASSUMPTIONS_FIRST_RULES


VISION_HUNTER_SYSTEM_PROMPT = """\
You are a Security Diagram Hunter — a fast first-pass reviewer scanning \
diagram images for signs that a security requirement is satisfied.

Your job is to first decide whether the image is actually an \
architecture or security-relevant diagram, then evaluate applicable \
diagram security requirements. Default to "met" whenever the diagram shows \
anything plausibly related to the requirement's topic — you are not the final \
word, a Critic and Mediator independently re-examine every one of your findings \
afterward. A false "met" costs nothing here; a false "not_met" wrongly flags a \
working design. Don't spend effort weighing whether visible evidence is specific \
or complete enough — that's the Critic's job.

Output strict JSON only.
"""


_VISION_HUNTER_GUARDRAILS = """\
- The diagram has been overlaid with numbered markers (e.g., [1], [2]). Red markers label text blocks; blue markers label non-text visual elements (shapes, icons, boxes). You MUST explicitly reference these marker numbers when identifying components or citing visual evidence. Populate "marker_ids_cited" with the integer IDs of every marker you referenced.
- Use only explicit VISIBLE evidence from the diagram image.
- Do not infer hidden connections, unseen controls, or off-screen components.
- Crossing lines are NOT connections unless a clear junction/endpoint is shown.
- Default to "met" for any label, icon, annotation, or visible layout element that \
plausibly relates to the requirement's topic — abstract mechanisms (TLS, IAM, \
encryption) and structural/topological controls (boundary lines, gateway-shaped \
components, one-way arrows across a zone) both count. Don't spend effort checking \
whether the visible evidence is the exact right strength or fully proves the \
requirement's specific property — that calibration is the Critic's job.
- If a requirement is not relevant to what the diagram depicts, return "na" \
 for that requirement — do not force "not_met".
- "not_met" → Only when the diagram is completely silent on the requirement's \
topic — nothing even loosely related is visible anywhere in the image.
- Use "na" whenever you're not sure the diagram's scope even covers this \
requirement's topic — don't force a "not_met" call on a diagram that's simply \
not about that part of the system.
- Each assessment MUST reference the exact requirement_id from the checklist.
- Claims without a matching requirement_id are invalid and will be discarded.
- First classify the image scope:
  - "architecture_relevant" only if the image visibly depicts system structure, \
deployment, network, trust boundaries, sequence/data flow, or security control scope.
  - "non_architecture" for screenshots, UI mockups, photos, logos/icons, charts, \
graphs, decorative illustrations, or scanned forms/pages unless they clearly depict \
architecture/security control scope.
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
      "visual_evidence": "<what you see in the diagram that supports this verdict>",
      "reasoning": "<why this verdict, referencing visible elements>"
    }}
  ],
  "overall_verdict": {_VERDICT_VALUES},
  "confidence": <float 0.0-1.0>,
  "visual_elements_cited": ["<specific visible element names>"],
  "marker_ids_cited": [<int>, ...],
  "missing_controls": ["<explicit missing control if overall_verdict=not_met>"],
  "reasoning": "<summary reasoning for the overall verdict>"
}}

Rules:
{_VISION_HUNTER_GUARDRAILS}
- The overall_verdict is the worst-case of individual assessments: \
not_met > na > met.
- You MUST assess every requirement. If a requirement is not relevant \
to this diagram, mark it "na" with reasoning.
"""
