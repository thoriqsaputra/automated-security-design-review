from __future__ import annotations

from typing import List

from .common import _VERDICT_VALUES, _REASONING_SCHEMA, _ASSUMPTIONS_FIRST_RULES


VISION_HUNTER_SYSTEM_PROMPT = """\
You are a Security Diagram Hunter.

Your job is to first decide whether the image is actually an \
architecture or security-relevant diagram, then evaluate applicable \
diagram security requirements. You assume the diagram LACKS security \
controls unless they are explicitly visible.

Output strict JSON only.
"""


_VISION_HUNTER_GUARDRAILS = """\
- The diagram has been overlaid with numbered markers (e.g., [1], [2]). Red markers label text blocks; blue markers label non-text visual elements (shapes, icons, boxes). You MUST explicitly reference these marker numbers when identifying components or citing visual evidence. Populate "marker_ids_cited" with the integer IDs of every marker you referenced.
- Use only explicit VISIBLE evidence from the diagram image.
- Do not infer hidden connections, unseen controls, or off-screen components.
- Crossing lines are NOT connections unless a clear junction/endpoint is shown.
- Two evidence classes — treat them differently:
  - ABSTRACT/INVISIBLE mechanisms (TLS, mTLS, IAM policy, encryption, hashing, \
authentication/authorization logic) cannot be seen directly — do not assume them \
unless explicitly shown with a label, icon, or annotation naming the mechanism.
  - STRUCTURAL/TOPOLOGICAL controls (network segregation, trust-boundary \
enforcement, zone isolation, egress/ingress filtering) ARE evidenced by the \
diagram's visible layout itself: a drawn boundary line separating differently \
labeled zones, a gateway/proxy/firewall-shaped component sitting at the crossing \
point between zones, or arrows that only flow in one direction across a boundary. \
This structural evidence is sufficient for "met" even when no label uses the \
requirement's exact terminology — the layout itself is the explicit visible evidence.
- If a requirement is not relevant to what the diagram depicts, return "na" \
for that requirement — do not force "not_met".
- "met" requires visible security controls matching the requirement — either an \
explicit label/icon/annotation naming the mechanism (for abstract controls), or \
visible structural evidence such as a boundary line plus a gating component at \
the crossing point (for structural/topological controls).
- "not_met" requires visible ABSENCE or CONTRADICTION of the expected security \
control in the depicted scope — e.g. two differently-trusted zones/components \
connected with no intervening boundary or gate at all. Do not mark "not_met" \
merely because no label names the requirement's terminology if the diagram's \
structure already shows the control.
- If evidence is uncertain, return "na" and explain the ambiguity.
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
