from __future__ import annotations

from typing import Dict, List, Optional

from .common import (
    _ASSUMPTIONS_FIRST_RULES,
    _build_block_ids_block,
    _CITATION_SCHEMA,
    _REASONING_SCHEMA,
    _VERDICT_VALUES,
)

HUNTER_SYSTEM_PROMPT = """\
You are a Security Compliance Hunter doing a quick first pass over Technical \
Software Documents (TSDs) to flag where a security requirement looks satisfied.

Skim the TSD context for anything on-topic for the requirement, cite it, and \
call it "met". Use "not_met" only when the context has nothing related at all. \
Use "na" when the requirement's topic doesn't apply to this kind of document.

Output strict JSON only.
"""


def build_hunter_prompt(
    parameter_text: str,
    parameter_section: str,
    context_chunks: List[str],
    killed_assumptions: Optional[List[dict]] = None,
    available_block_ids: Optional[List[str]] = None,
) -> str:
    chunks_text = "\n\n---\n\n".join(context_chunks)
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
    block_ids_block = _build_block_ids_block(available_block_ids)

    return f"""\
## SECURITY PARAMETER TO EVALUATE

Section: {parameter_section}
Requirement: {parameter_text}

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
  "applicability_status": "established" | "not_established",
  "applicability_reason": "<what prerequisite/capability makes this requirement applicable or not>",
  "reasoning": "<one paragraph explaining your verdict>",
  "checked_context": "<what context you checked and why it was sufficient or insufficient>",
  "evidence_quotes": ["<short verbatim snippets from context, empty if none>"],
  "evidence_assessment": "<why the evidence satisfies, partially satisfies, or fails the requirement>",
  "missing_expected_evidence": ["<specific missing control evidence>", ...],
  "evidence_found": <true | false>,
  "citations": [
    {_CITATION_SCHEMA}
  ]
}}

Few-shot example:
Input -> Requirement requires TLS on internal service traffic. Context says "Service A calls Service B over mTLS" in block p4_b2.
Reasoning -> assumptions: ["Only retrieved context may be used."]; logic_summary: "The context explicitly states mTLS between the named services, so the control is evidenced."; output -> verdict "met" with citation p4_b2.

	Rules:
	- "met"     → Default here. Any on-topic mention in the TSD context is enough — a quick triage pass, not a detailed audit of every term and mechanism.
	- "not_met" → Only when the context is completely silent on the requirement's topic — nothing even loosely related appears anywhere.
	- "na"      → The retrieved context does not establish that the underlying governed capability is present at all (e.g. a mobile-only control when the TSD describes a server-only backend).
	- applicability_status → "established" for "met" and "not_met". Use "not_established" only when the control's own prerequisite/capability is absent from the design scope.
	- applicability_reason → Briefly say what makes this applicable or not.
	- missing_expected_evidence → For "not_met", briefly name what's missing.
	- citations → You MUST copy the id, page_number, and bbox coordinates EXACTLY from the CONTEXT_CHUNK XML attributes into the JSON, and the CONTEXT_CHUNK must have citable="true". Do not invent or guess them. If attributes are missing, use null.
	- The quoted_text field MUST be a short verbatim excerpt (5–20 words) copied character-for-character from the CONTEXT_CHUNK text. Do NOT paraphrase, summarize, or construct your own sentence. Never write text that is not character-for-character present in the source chunk.
	- A "met" verdict must include at least one valid citation and an evidence quote.
	- confidence → Your certainty in the verdict (1.0 = certain, 0.0 = guessing).
{killed_block}{block_ids_block}
{_ASSUMPTIONS_FIRST_RULES}
"""
