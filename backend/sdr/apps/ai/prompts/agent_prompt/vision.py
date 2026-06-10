from __future__ import annotations

import json
from typing import Optional

from .common import _ASSUMPTIONS_FIRST_RULES, _REASONING_SCHEMA, _VERDICT_VALUES

VISION_ARCHITECT_SYSTEM_PROMPT = """\
You are an Architecture Diagram Extractor.

You MUST describe only what is explicitly visible in the diagram image.
You MUST NOT make security judgments in this pass.

Output strict JSON only.
"""

VISION_AUDITOR_SYSTEM_PROMPT = """\
You are a Security Diagram Auditor.

Evaluate security compliance ONLY from:
1) explicit visible evidence in the diagram image, and
2) the Architect pass narration provided by the user.

Output strict JSON only.
"""

VISION_CRITIC_SYSTEM_PROMPT = """\
You are a Vision Security Critic.

Re-check the same diagram and previous auditor result.
Identify unsupported claims and hallucinated evidence.
Decide whether to uphold or overturn the auditor result.

Output strict JSON only.
"""

# Backward compatibility alias for legacy imports.
VISION_SYSTEM_PROMPT = VISION_AUDITOR_SYSTEM_PROMPT

_VISION_GUARDRAILS = """\
- Use only explicit visible diagram evidence.
- Do not infer hidden connections or unseen controls.
- Crossing lines are NOT connections unless a clear junction/endpoint is shown.
- Do not assume TLS, mTLS, IAM, encryption, firewall, WAF, authentication, or authorization unless explicitly shown.
- If details are missing in a high-level/abstract diagram, return "na" instead of forcing "not_met".
- If evidence is uncertain, say so explicitly in ambiguity fields.
"""


def _diagram_details_section(
    diagram_caption: Optional[str],
    surrounding_text: Optional[str],
) -> str:
    caption_section = f"\nDiagram Caption: {diagram_caption}" if diagram_caption else ""
    surrounding_section = (
        f"\n\nSurrounding Text:\n{surrounding_text}" if surrounding_text else ""
    )
    return f"## DIAGRAM DETAILS{caption_section}{surrounding_section}"


def build_vision_architect_prompt(
    parameter_text: str,
    parameter_section: str,
    diagram_caption: Optional[str],
    surrounding_text: Optional[str],
) -> str:
    return f"""\
## SECURITY PARAMETER CONTEXT (DO NOT EVALUATE YET)

Section: {parameter_section}
Requirement: {parameter_text}

{_diagram_details_section(diagram_caption, surrounding_text)}

## YOUR TASK

Extract visible architecture information only.
Do not determine met/not_met/na in this pass.

Respond with a single JSON object matching this exact schema:

{{
  "diagram_title": "<concise 3-5 word descriptive title based on visible architecture>",
  "components": ["<visible component/service/node>"],
  "connections": ["<visible connection/path between components>"],
  "data_flows": ["<visible arrow-labelled data flow>"],
  "trust_boundaries": ["<visible boundary/zone marker>"],
  "visible_labels": ["<exact visible labels/text>"],
  "visible_security_controls": ["<explicitly shown TLS/mTLS/auth/encryption/etc labels>"],
  "unclear_regions": ["<ambiguous area where meaning is unclear>"],
  "visual_evidence": [
    {{
      "element": "<specific visible element>",
      "bbox": [<x1>, <y1>, <x2>, <y2>], 
      "why_relevant": "<why this element matters>"
    }}
  ],
  "notes": "<short neutral narration>"
}}

Rules:
- bbox coordinates should be normalized in [0,1] when possible; otherwise null.
{_VISION_GUARDRAILS}
- Do not include security verdicts in this pass.
"""


def build_vision_auditor_prompt(
    parameter_text: str,
    parameter_section: str,
    diagram_caption: Optional[str],
    surrounding_text: Optional[str],
    architect_result: dict,
) -> str:
    architect_json = json.dumps(architect_result, ensure_ascii=True)
    return f"""\
## SECURITY PARAMETER TO EVALUATE

Section: {parameter_section}
Requirement: {parameter_text}

{_diagram_details_section(diagram_caption, surrounding_text)}

## ARCHITECT PASS OUTPUT (TRUSTED CONTEXT)
{architect_json}

## YOUR TASK
Evaluate security compliance from explicit evidence only.

Respond with a single JSON object:
{{
{_REASONING_SCHEMA},
  "verdict": {_VERDICT_VALUES},
  "confidence": <float 0.0-1.0>,
  "reasoning": "<grounded security evaluation>",
  "visual_elements_cited": ["<short element names>"],
  "visual_evidence": [
    {{
      "element": "<element>",
      "bbox": [<x1>, <y1>, <x2>, <y2>],
      "why_relevant": "<evidence relevance>"
    }}
  ],
  "missing_controls": ["<explicit missing control if verdict=not_met>"],
  "ambiguous_elements": ["<ambiguous visual elements>"],
  "missing_information": ["<required details not visible>"],
  "ambiguity_reason": "<why this should be na if evidence is insufficient>"
}}

Rules:
- Return "na" when security-relevant detail is not explicitly visible.
- Return "not_met" only when the diagram explicitly shows absence/contradiction in depicted scope.
{_VISION_GUARDRAILS}
"""


def build_vision_critic_prompt(
    parameter_text: str,
    parameter_section: str,
    architect_result: dict,
    auditor_result: dict,
) -> str:
    architect_json = json.dumps(architect_result, ensure_ascii=True)
    auditor_json = json.dumps(auditor_result, ensure_ascii=True)
    return f"""\
## SECURITY PARAMETER
Section: {parameter_section}
Requirement: {parameter_text}

## ARCHITECT RESULT
{architect_json}

## AUDITOR RESULT
{auditor_json}

## YOUR TASK
Re-check for unsupported claims and hallucinated evidence.

Respond with strict JSON:
{{
{_REASONING_SCHEMA},
  "outcome": "uphold" | "overturn",
  "reasoning": "<why uphold/overturn>",
  "hallucinated_or_unsupported_claims": ["<claim lacking visible support>"],
  "revised_verdict": {_VERDICT_VALUES},
  "revised_confidence": <float 0.0-1.0>
}}

Rules:
{_VISION_GUARDRAILS}
"""


def build_vision_prompt(
    parameter_text: str,
    parameter_section: str,
    diagram_caption: Optional[str],
    surrounding_text: Optional[str],
) -> str:
    """
    Backward-compatible alias for older call sites.
    """
    return build_vision_auditor_prompt(
        parameter_text=parameter_text,
        parameter_section=parameter_section,
        diagram_caption=diagram_caption,
        surrounding_text=surrounding_text,
        architect_result={},
    )
