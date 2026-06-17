from __future__ import annotations

import json
from typing import Dict, List, Optional

from .common import (
    _ASSUMPTIONS_FIRST_RULES,
    _CITATION_SCHEMA,
    _REASONING_SCHEMA,
    _VERDICT_VALUES,
)

HUNTER_SYSTEM_PROMPT = """\
You are a Security Compliance Hunter — a specialist in identifying whether \
Technical Software Documents (TSDs) satisfy specific security requirements \
from the OWASP Application Security Verification Standard (ASVS).

YOUR BIAS: Assume NON-COMPLIANCE unless the TSD contains explicit, \
unambiguous evidence that the requirement is satisfied. Implicit, assumed, \
or aspirational compliance does NOT count.

YOUR ROLE:
- Analyse the provided TSD context chunks against the given security parameter.
- Search for direct evidence: explicit statements, configuration details, \
architectural decisions, or diagram elements that satisfy the requirement.
- Produce an initial verdict and cite the exact source block IDs as evidence.
- If the parameter is clearly out of scope for this type of TSD, use "na".

WHAT COUNTS AS EVIDENCE (in decreasing order of strength):
- STRONG: Code snippets, configuration files, library/framework usage with \
specific settings, middleware declarations, security filter chains.
- MODERATE: Architecture diagrams showing security layers, sequence diagrams \
with authentication/authorization steps, explicit "the system enforces..." \
statements with named components.
- WEAK (not sufficient alone for "met"): Section headings, requirement titles, \
baseline control text, "the application shall..." statements without \
implementation detail, generic mentions of security without specifics.

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""


def build_hunter_prompt(
    parameter_text: str,
    parameter_section: str,
    contract: Optional[dict],
    context_chunks: List[str],
    persona_focus: Optional[str] = None,
    killed_assumptions: Optional[List[dict]] = None,
    available_block_ids: Optional[List[str]] = None,
) -> str:
    chunks_text = "\n\n---\n\n".join(context_chunks)
    persona_block = f"\nPersona focus: {persona_focus}" if persona_focus else ""
    killed_block = ""
    if killed_assumptions:
        lines = []
        for item in killed_assumptions[:5]:
            if not isinstance(item, dict):
                continue
            text = (item.get("assumption") or item.get("reason") or "").strip()
            if text:
                lines.append(f"- {text}")
        if lines:
            killed_block = "\nDo not repeat these invalidated assumptions:\n" + "\n".join(lines)
    block_ids_block = ""
    if available_block_ids:
        block_ids_block = (
            "\nVALID CITATION BLOCK IDS (only cite block_ids from this list — "
            "do not invent or guess IDs):\n"
            + ", ".join(sorted(available_block_ids))
        )

    return f"""\
## SECURITY PARAMETER TO EVALUATE

Section: {parameter_section}
Requirement: {parameter_text}
Contract: {contract or {}}{persona_block}

## TSD CONTEXT
{chunks_text}

## YOUR TASK

Analyse the TSD context above and determine whether it satisfies the \
security parameter.

Respond with a single JSON object matching this exact schema:

{{
{_REASONING_SCHEMA},
  "verdict": {_VERDICT_VALUES},
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one paragraph explaining your verdict>",
  "checked_context": "<what context you checked and why it was sufficient or insufficient>",
  "evidence_quotes": ["<short verbatim snippets from context, empty if none>"],
  "evidence_assessment": "<why the evidence satisfies, partially satisfies, or fails the requirement>",
  "evidence_found": <true | false>,
  "citations": [
    {_CITATION_SCHEMA}
  ]
}}

Few-shot example:
Input -> Requirement requires TLS on internal service traffic. Context says "Service A calls Service B over mTLS" in block p4_b2.
Reasoning -> assumptions: ["Only retrieved context may be used."]; logic_summary: "The context explicitly states mTLS between the named services, so the control is evidenced."; output -> verdict "met" with citation p4_b2.

	Rules:
	- "met"     → TSD contains explicit evidence satisfying the requirement.
	- "not_met" → The requirement is applicable, and the TSD lacks implementation evidence or explicitly contradicts the requirement.
	- "na"      → The retrieved context does not establish the technology, data flow, control trigger, or document scope needed to assess this requirement.
	- citations → Only use block_ids that appear literally in the CONTEXT_CHUNK headers above. Do not invent or guess block_ids.
	- A "met" verdict must include at least one valid citation and an evidence quote.
	- A "not_met" verdict where evidence_found=true must include citations to the block_ids you examined and found insufficient. If you cannot identify a relevant block_id, set evidence_found=false instead.
	- First decide applicability, then implementation. A heading, graph node, requirement title, or baseline control text is not implementation evidence.
	- For "not_met", checked_context and evidence_assessment must identify the applicability basis and the specific missing control evidence.
	- Do not use generic phrases like "lacks explicit evidence" alone. Name the expected control, enforcement point, validation behavior, configuration, or component that is missing.
	- confidence → Your certainty in the verdict (1.0 = certain, 0.0 = guessing).
	- Use evidence-only reasoning; do not infer controls that are not explicitly stated.
{killed_block}{block_ids_block}
{_ASSUMPTIONS_FIRST_RULES}
"""


def build_batch_hunter_prompt(
    child_inputs: List[dict],
    parameter_section: str,
    context_chunks: List[str],
    killed_assumptions: Optional[List[dict]],
) -> str:
    chunks_text = "\n\n---\n\n".join(context_chunks)
    killed_text = json.dumps(killed_assumptions or [], indent=2)
    children_text = json.dumps(child_inputs, indent=2)
    return f"""\
## PARENT SECURITY SECTION
Section: {parameter_section}

## CHILD PARAMETERS
{children_text}

## SHARED TSD CONTEXT
{chunks_text}

## INVALIDATED ASSUMPTIONS TO AVOID
{killed_text}

Analyse each child parameter independently. Do not merge child requirements.
Return strict JSON with exactly one result object per child id:

{{
  "results": [
    {{
      "child_id": "<id from CHILD PARAMETERS>",
      "assumptions": ["<assumption>", "..."],
      "logic_summary": "<concise evidence-only reasoning>",
      "verdict": "met" | "not_met" | "na",
      "confidence": <float 0.0-1.0>,
      "reasoning": "<one paragraph explaining this child's verdict>",
      "checked_context": "<what context you checked and why it was sufficient or insufficient>",
      "evidence_quotes": ["<short verbatim snippets from context, empty if none>"],
      "evidence_assessment": "<why evidence satisfies or fails this child>",
      "evidence_found": <true | false>,
      "citations": [
        {{"block_id": "<CONTEXT_CHUNK id only>", "page_number": <integer>, "quoted_text": "<short quote>", "bbox": {{"x0": null, "y0": null, "x1": null, "y1": null}}}}
      ]
    }}
  ]
}}

Rules:
- Use child_id exactly as supplied.
- A "met" verdict must include at least one valid citation and evidence quote.
- A "not_met" verdict where evidence_found=true must include citations to the block_ids examined and found insufficient. If no relevant block_id exists, set evidence_found=false.
- For "not_met", explain what explicit evidence is missing for that child.
- Only cite block_ids that appear literally in the CONTEXT_CHUNK headers in SHARED TSD CONTEXT. Do not invent or guess block_ids.
"""

