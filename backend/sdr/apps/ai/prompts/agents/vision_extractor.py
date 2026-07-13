from __future__ import annotations

from typing import List


VISION_EXTRACTOR_SYSTEM_PROMPT = """\
You are a Diagram Structure Extractor. Your ONLY job is to describe, as literally \
and completely as possible, what is drawn in the attached diagram image — components, \
trust boundaries/zones, connections between them, and any visible text/labels/annotations.

Do NOT judge whether any security requirement is met. Do NOT reason about \
compliance, gaps, missing controls, or security implications. Only report what is \
visibly present in the image. If something is not visible or not legible, do not \
guess, infer, or fill it in from typical/expected architecture patterns.

Output strict JSON only.
"""


_VISION_EXTRACTOR_GUARDRAILS = """\
- A raw diagram image is attached. Base your extraction only on what you see in it.
- Assign each component, trust boundary, and flow a short local id (c1, c2, ... for \
components; tb1, tb2, ... for trust boundaries; f1, f2, ... for flows). These ids only \
need to be unique within this response.
- For each flow, set source_component_id/target_component_id to the ids of the \
components it visibly connects. If an endpoint isn't a component you extracted \
(e.g. an external actor outside any box), still create a component entry for it.
- For each trust boundary, list the component ids it visibly encloses in \
encloses_component_ids. If nothing is visibly enclosed, use an empty list.
- ONLY report a trust boundary when a genuinely visible security/network boundary \
marker is drawn AROUND a set of elements — a dashed-line perimeter, a solid box used \
specifically as a zone/segment delimiter, or an explicitly shaded/colored region. Do \
NOT report a trust boundary for a sequence-diagram phase grouping, a swimlane, a \
numbered step label, or any other informal grouping that isn't a drawn security \
boundary — even if it visually resembles a box or its label uses words like "flow" \
or "process". If diagram_style is "sequence_or_flow", trust_boundaries should \
normally be empty unless an explicit security-zone marker is still visibly drawn.
- Record only text/annotations that are actually visible (e.g. "TLS", "HTTPS", a lock \
icon, "auth required") in notes/security_annotations — do not add annotations that \
are not drawn or labeled in the image.
- When a component's visible name itself is security-relevant (for example WAF, API \
Gateway, Identity Provider, Auth Service, Vault, KMS, HSM, Secrets Manager), preserve \
that wording exactly in the component name/labels instead of abstracting it away.
- When a flow visibly indicates protected transport or authentication (for example \
HTTPS, TLS, mTLS, "authenticated API", lock icons, certificates), capture that in \
protocol and/or security_annotations as literally as possible.
- Classify the image scope:
  - "architecture_relevant" if the image depicts system structure, deployment, \
network, trust boundaries, sequence/data flow, or security control scope.
  - "non_architecture" for screenshots, UI mockups, photos, logos/icons, charts, \
graphs, or decorative illustrations.
  - "uncertain" only when you genuinely cannot tell from the visible image.
- If the image scope is "non_architecture", components/trust_boundaries/flows MUST \
all be empty lists.
- Classify diagram_style:
  - "architecture_or_dfd" for network/deployment/component diagrams and data-flow \
diagrams with boxes, zones, and directional data flow arrows between them.
  - "sequence_or_flow" for sequence diagrams, swimlanes, or step-by-step process/flow \
diagrams organized around interaction order rather than network/trust topology.
  - "other" if it fits neither.
"""


def build_vision_extractor_prompt(
    *,
    diagram_caption: str = "",
    surrounding_text: str = "",
) -> str:
    """
    Build the user-turn prompt for the VisionExtractor (Stage 1 of the
    extract-then-reason pipeline). This prompt asks only for a literal,
    structured description of the diagram — no requirement judgment.

    Args:
        diagram_caption: Optional diagram caption or title.
        surrounding_text: Text immediately around the diagram in the TSD.
    """
    caption_section = f"\nDiagram Caption: {diagram_caption}" if diagram_caption else ""
    surrounding_section = f"\n\nSurrounding Text:\n{surrounding_text}" if surrounding_text else ""

    return f"""\
## DIAGRAM DETAILS
{caption_section}{surrounding_section}

## YOUR TASK

Extract a complete, literal, structured description of the attached diagram image.
Decide first whether the image is architecture/security-relevant, then list every
visible component, trust boundary, and flow/connection you can identify.

Respond with a single JSON object:

{{
  "diagram_scope_verdict": "architecture_relevant" | "non_architecture" | "uncertain",
  "diagram_scope_reasoning": "<why the image is or is not architecture/security-relevant>",
  "diagram_style": "architecture_or_dfd" | "sequence_or_flow" | "other",
  "diagram_style_reasoning": "<one line: why this style classification>",
  "components": [
    {{
      "id": "c1",
      "name": "<component name as labeled, e.g. 'API Gateway'>",
      "type": "service" | "database" | "external_actor" | "process" | "data_store" | "network_zone" | "other",
      "labels": ["<visible text on or immediately next to this element>"],
      "notes": "<visible security-relevant annotation on/near this element, if any, else empty string>"
    }}
  ],
  "trust_boundaries": [
    {{
      "id": "tb1",
      "label": "<visible label, or 'unlabeled boundary' if the boundary is drawn but unlabeled>",
      "encloses_component_ids": ["c1", "c2"],
      "boundary_style": "dashed_line" | "solid_box" | "labeled_zone" | "other"
    }}
  ],
  "flows": [
    {{
      "id": "f1",
      "source_component_id": "c1",
      "target_component_id": "c2",
      "direction": "one_way" | "bidirectional" | "unclear",
      "label": "<visible edge label, verbatim, or empty string if unlabeled>",
      "protocol": "<visibly labeled protocol, e.g. 'HTTPS', or null if not labeled>",
      "security_annotations": ["<visible annotation on this flow, e.g. 'TLS lock icon', described>"]
    }}
  ],
  "other_visible_text": ["<any other legible text not captured above>"],
  "extraction_confidence": <float 0.0-1.0>
}}

Rules:
{_VISION_EXTRACTOR_GUARDRAILS}
- You must extract every component, trust boundary, and flow that is visibly present — \
do not skip elements for brevity.
"""


__all__: List[str] = [
    "VISION_EXTRACTOR_SYSTEM_PROMPT",
    "build_vision_extractor_prompt",
]
