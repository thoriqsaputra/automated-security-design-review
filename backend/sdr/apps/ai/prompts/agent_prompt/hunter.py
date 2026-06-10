from __future__ import annotations

from typing import List, Optional

from .common import (
    _ASSUMPTIONS_FIRST_RULES,
    _CITATION_SCHEMA,
    _REASONING_SCHEMA,
    _VERDICT_VALUES,
)

HUNTER_SYSTEM_PROMPT = """\
You are a Security Compliance Hunter — a specialist in identifying whether \
Technical Software Documents (TSDs) satisfy specific security requirements.

YOUR BIAS: Assume NON-COMPLIANCE unless the TSD contains explicit, \
unambiguous evidence that the requirement is satisfied. Implicit, assumed, \
or aspirational compliance does NOT count.

YOUR ROLE:
- Analyse the provided TSD context chunks against the given security parameter.
- Search for direct evidence: explicit statements, configuration details, \
architectural decisions, or diagram elements that satisfy the requirement.
- Produce an initial verdict and cite the exact source block IDs as evidence.
- If the parameter is clearly out of scope for this type of TSD, use "na".

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""


def build_hunter_prompt(
    parameter_text: str,
    parameter_section: str,
    contract: Optional[dict],
    context_chunks: List[str],
    diagram_captions: Optional[List[str]] = None,
    persona_focus: Optional[str] = None,
    killed_assumptions: Optional[List[dict]] = None,
) -> str:
    diagrams_section = ""
    if diagram_captions:
        formatted = "\n".join(
            f"  - Diagram {i + 1}: {cap}" for i, cap in enumerate(diagram_captions)
        )
        diagrams_section = f"\n\n### DIAGRAMS IN CONTEXT\n{formatted}"

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

    return f"""\
## SECURITY PARAMETER TO EVALUATE

Section: {parameter_section}
Requirement: {parameter_text}
Contract: {contract or {}}{persona_block}

## TSD CONTEXT
{chunks_text}{diagrams_section}

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
	- citations → List only block_ids from CONTEXT_CHUNK ids in the context above. Empty list if none.
	- A "met" verdict must include at least one valid citation and an evidence quote.
	- First decide applicability, then implementation. A heading, graph node, requirement title, or baseline control text is not implementation evidence.
	- For "not_met", checked_context and evidence_assessment must identify the applicability basis and the specific missing control evidence.
	- Do not use generic phrases like "lacks explicit evidence" alone. Name the expected control, enforcement point, validation behavior, configuration, or component that is missing.
	- confidence → Your certainty in the verdict (1.0 = certain, 0.0 = guessing).
	- Use evidence-only reasoning; do not infer controls that are not explicitly stated.
{killed_block}
{_ASSUMPTIONS_FIRST_RULES}
"""
