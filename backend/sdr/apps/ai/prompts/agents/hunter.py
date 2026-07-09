from __future__ import annotations

import json
from typing import Dict, List, Optional

from .common import (
    _ASSUMPTIONS_FIRST_RULES,
    _build_block_ids_block,
    _CITATION_SCHEMA,
    _REASONING_SCHEMA,
    _VERDICT_VALUES,
)

HUNTER_SYSTEM_PROMPT = """\
You are a Security Compliance Hunter — a specialist in identifying whether \
Technical Software Documents (TSDs) satisfy specific security requirements \
from the selected security standard.

YOUR BIAS: Be evidence-first, not wording-first. Start skeptical of unsupported \
claims, but do NOT default to non-compliance when the TSD names a concrete \
mechanism, algorithm, control, or architectural decision that clearly satisfies \
the requirement's core security property. Generic, aspirational, or unnamed \
compliance still does NOT count.

DOCUMENT TYPE: You are reviewing a Technical Software Document (TSD) — an \
architectural design specification, not source code. TSDs describe what the \
system is designed to implement. Do not require code-level proof.

YOUR ROLE:
- Analyse the provided TSD context chunks against the given security parameter.
- Search for direct evidence: explicit statements, configuration details, \
architectural decisions, or diagram elements that satisfy the requirement.
- Produce an initial verdict and cite the exact source block IDs as evidence.
- If the parameter is clearly out of scope for this type of TSD, use "na".

WHAT COUNTS AS EVIDENCE (in decreasing order of strength):
- STRONG: Explicit statements naming specific security mechanisms, algorithms, \
libraries, or frameworks with implementation details (e.g. "AES-256 is used \
for data at rest", "OAuth 2.0 with PKCE enforced by the API gateway", \
"mTLS between services A and B").
- MODERATE (sufficient for "met"): Architectural mandates that name specific \
controls (e.g. "Section X mandates use of Y", "all Z components use W"), \
explicit design decisions referencing named security components, or \
architecture diagrams with labelled security layers. This level IS SUFFICIENT \
for a "met" verdict in a TSD review.
- WEAK (not sufficient alone): Section headings, requirement titles, baseline \
control text, "the application shall..." with no named mechanism, technology, \
or enforcement point, generic mentions of security without specifics.

"Explicit evidence naming a specific mechanism" is about the underlying \
security PROPERTY the requirement demands, not about the TSD repeating the \
requirement's own wording. Do not mark "not_met" just because the TSD uses \
different terminology than the requirement text — check whether a named, \
concrete mechanism achieves the same property. Example: requirement "audit \
access to sensitive data" is satisfied by "every attempt to access a \
restricted resource is logged and generates an alert," even though the TSD \
never says "audit" or "sensitive data" verbatim — the named logging+alerting \
mechanism covers the same property. This is still distinct from genuinely \
generic evidence ("the system is secure," a bare topic mention) which \
remains WEAK regardless of wording.

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
    block_ids_block = _build_block_ids_block(available_block_ids)

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
	- "met"     → TSD contains explicit evidence (STRONG or MODERATE) satisfying the requirement. Architectural design mandates naming specific mechanisms count as MODERATE and are sufficient for "met".
	- If a cited block names a concrete mechanism that clearly satisfies the requirement's core security property, prefer "met" even when the TSD uses different terminology than the requirement text.
	- For requirements framed as forbidding or limiting something (for example unsupported/deprecated technology, password hints, secret questions, weak authenticators, or "no weaker path" style controls), do NOT return "met" from silence alone. "Met" requires explicit prohibition, explicit approved-only mechanism inventory, or other closed-world evidence showing the weaker/disallowed option is not used.
	- Distinguish "uniform authentication strength across all pathways" from "weak authenticators are limited to secondary-only use." Uniform MFA or stronger methods across all roles can satisfy the former. The latter still requires explicit evidence restricting SMS/email or other weak methods to secondary/approval-only use, or explicitly forbidding them as primary authentication.
	- "not_met" → The requirement is applicable, and the TSD lacks implementation evidence or explicitly contradicts the requirement.
	- "na"      → The retrieved context does not establish that the underlying governed capability (not just the exact named technology or standard) — e.g. any data flow, control trigger, or document scope this requirement's security objective depends on — is present at all. A different mechanism serving the same capability (e.g. REST instead of GraphQL, OOB SMS/email instead of TOTP, a single service instead of many) does NOT make the requirement inapplicable; missing evidence for it is "not_met", not "na". If the requirement text says "or other", "such as", "or equivalent", or otherwise names a technology family, treat that family-level capability as the applicability test.
	- applicability_status → "established" for "met" and "not_met". Use "not_established" only when the control's own prerequisite/capability is absent from the design scope, not merely undocumented.
	- applicability_reason → Name the prerequisite/capability basis. For "not_met", state why the control applies. For "na", state which prerequisite is absent.
	- missing_expected_evidence → Required for "not_met". Name the specific implementation evidence that should have appeared in the TSD (mechanism, enforcement point, configuration, component, flow, or validation behavior).
	- citations → You MUST copy the id, page_number, and bbox coordinates EXACTLY from the CONTEXT_CHUNK XML attributes into the JSON, and the CONTEXT_CHUNK must have citable="true". Do not invent or guess them. If attributes are missing, use null.
	- The quoted_text field MUST be a short verbatim excerpt (5–20 words) copied character-for-character from the CONTEXT_CHUNK text. Do NOT paraphrase, summarize, or construct your own sentence. Pick the most distinctive technical phrase or named mechanism directly from the chunk text. Shorter, specific quotes (e.g. "TLS 1.3 with HSTS enforcement", "parameterized queries and prepared statements") are better than long sentences — they are easier to verify and almost never contain paraphrasing errors. Never write text that is not character-for-character present in the source chunk.
	- A "met" verdict must include at least one valid citation and an evidence quote.
	- A "not_met" verdict MUST include citations to the most relevant context chunks examined when such chunks exist — cite chunks that were relevant but insufficient to demonstrate why they fall short. If the requirement is clearly applicable but no citable chunk supports the control, use "not_met" with evidence_found=false. Use "na" only when the governed capability itself is absent from the design.
	- First decide applicability, then implementation. A heading, graph node, requirement title, or baseline control text is not implementation evidence.
	- Do not downgrade to "not_met" merely because evidence is architecture-level rather than code-level. This is a TSD review; named architectural mandates and named security components are valid implementation evidence.
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
    available_block_ids: Optional[List[str]] = None,
) -> str:
    chunks_text = "\n\n---\n\n".join(context_chunks)
    killed_text = json.dumps(killed_assumptions or [], indent=2)
    children_text = json.dumps(child_inputs, indent=2)
    block_ids_block = _build_block_ids_block(available_block_ids)
    return f"""\
## PARENT SECURITY SECTION
Section: {parameter_section}

## CHILD PARAMETERS
{children_text}

## SHARED TSD CONTEXT
{chunks_text}

## INVALIDATED ASSUMPTIONS TO AVOID
{killed_text}
{block_ids_block}

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
      "applicability_status": "established" | "not_established",
      "applicability_reason": "<what prerequisite/capability makes this requirement applicable or not>",
      "reasoning": "<one paragraph explaining this child's verdict>",
      "checked_context": "<what context you checked and why it was sufficient or insufficient>",
      "evidence_quotes": ["<short verbatim snippets from context, empty if none>"],
      "evidence_assessment": "<why evidence satisfies or fails this child>",
      "missing_expected_evidence": ["<specific missing control evidence>", "..."],
      "evidence_found": <true | false>,
      "citations": [
        {{"block_id": "<CONTEXT_CHUNK id>", "page_number": <CONTEXT_CHUNK page_number or null>, "quoted_text": "<verbatim quote>", "bbox": {{"x0": <bbox_x0 or null>, "y0": <bbox_y0 or null>, "x1": <bbox_x1 or null>, "y1": <bbox_y1 or null>}}}}
      ]
    }}
  ]
}}

Rules:
- Use child_id exactly as supplied.
- A "met" verdict must include at least one valid citation and evidence quote.
- A "not_met" verdict where evidence_found=true must include citations to the block_ids examined and found insufficient. If no relevant block_id exists, set evidence_found=false.
- Use applicability_status="established" for "met" and "not_met". Only use "not_established" when the control's prerequisite/capability is absent from the design scope itself.
- For "not_met", explain what explicit evidence is missing for that child.
- You MUST copy the id, page_number, and bbox coordinates EXACTLY from the CONTEXT_CHUNK XML attributes into the JSON, and the CONTEXT_CHUNK must have citable="true". Do not invent or guess them. If missing, use null.
"""
