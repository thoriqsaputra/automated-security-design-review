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
You are a Security Compliance Hunter doing a fast first pass over Technical \
Software Documents (TSDs). Your job is to make the best evidence-grounded claim \
you can from the supplied context.

Use "met" only when cited evidence covers the requirement's actual governed \
object, polarity, and essential logic. Use "not_met" when the requirement is \
applicable but direct satisfying evidence is missing, partial, contradicted, or \
only same-topic. Use "na" only when the governed capability itself is outside \
the design scope.

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
	- "met"     → Use when the citation proves the requirement's specific claim through direct wording, universal scope, equivalent mechanism, or necessary semantic consequence. Match the governed object, polarity, scope, and essential AND/OR logic; do not require the exact words from the requirement when the same control objective is plainly evidenced.
	- "not_met" → Use when the requirement remains applicable but the context does not show the required control, only shows part of it, or only shows adjacent/generic evidence. Even for "not_met", cite the closest inspected scope, partial implementation, or contradictory evidence block you relied on.
	- "na"      → The retrieved context does not establish that the underlying governed capability is present at all (e.g. a mobile-only control when the TSD describes a server-only backend).
	- applicability_status → "established" for "met" and "not_met". Use "not_established" only when the control's own prerequisite/capability is absent from the design scope.
	- applicability_reason → Briefly say what makes this applicable or not.
	- Proof ladder → Accept evidence as satisfying when it is (1) direct, (2) universal/global and therefore applies to the governed object, (3) equivalent to an example-named technology or standard, or (4) an entailed consequence of explicit design mandates. Reject only adjacent evidence, generic aspirations, or assumptions that are merely plausible.
	- Logic check → For AND requirements, each essential clause must be evidenced. For OR requirements, one valid alternative is enough. The primary "Verify..." action is usually the central claim; advisory "should" text, examples, documentation organization, and explanatory clauses are supporting context unless the requirement makes them the explicit object. Terms introduced by "such as" or "e.g." are examples unless the requirement makes them the central object.
	- Prohibition check → A requirement that says something must not be used/present needs evidence that excludes it (explicit ban, allow-list, complete inventory, or structural constraint). Evidence that a better mechanism is used elsewhere is not enough by itself.
	- Object check → Do not substitute a different object that merely shares keywords. Named categories, data types, account types, authenticators, protocols, relationships, or taxonomies must be evidenced at that level of specificity.
	- Universal control statements may satisfy a named in-scope subsystem when the text explicitly applies to all traffic, all APIs, all service-to-service calls, all data access, or another clearly universal scope.
	- Policy/process requirements may be satisfied by an equivalent evidenced process or governance document when a named standard is only an example ("such as", "e.g.", "or other").
	- Collective evidence → A requirement may be satisfied by multiple cited sections together when each citation covers a relevant part of the same governed capability. Do not reject merely because the evidence is distributed across architecture, controls, data, and operations sections instead of appearing in one titled subsection.
	- Semantic equivalence examples → Retention periods plus destruction at end of retention can evidence scheduled deletion; automated rotation of relevant service/database/API credentials can evidence no unchanging credentials; a biometric used as "secondary proof" or required "subsequently" can evidence a secondary factor; prescriptive HSM, rotation, lifecycle, and ownership mandates can evidence a key-management policy/process; security controls tied to architecture boundaries, threats, and remote access can evidence architecture security analysis; RBAC plus ABAC can be one composite authorization mechanism, while OAuth or API keys are authentication/credential mechanisms and do not automatically mean there are multiple authorization mechanisms.
	- missing_expected_evidence → For "not_met", briefly name what's missing. For "met", also use this field to name any part of the requirement's own wording (a named clause, a specific sub-condition joined by "and") that you did NOT find direct evidence for, even though the overall topic is on-point — do not leave this empty just because the verdict is "met".
	- confidence → Your certainty that the evidence covers the requirement's actual, specific claim (not just its general topic). If the requirement has multiple named conditions and you only found evidence for some of them, or the evidence is same-topic but doesn't pin down the specific mechanism/object the requirement names, cap confidence at 0.6 and say so in missing_expected_evidence — reserve 0.8+ for when the cited evidence directly and completely covers what the requirement's own wording asks for. 1.0 = certain, 0.0 = guessing.
	- citations → You MUST copy the id, page_number, and bbox coordinates EXACTLY from the CONTEXT_CHUNK XML attributes into the JSON, and the CONTEXT_CHUNK must have citable="true". Do not invent or guess them. If attributes are missing, use null.
	- The quoted_text field MUST be a short verbatim excerpt (5–20 words) copied character-for-character from the CONTEXT_CHUNK text. Do NOT paraphrase, summarize, or construct your own sentence. Never write text that is not character-for-character present in the source chunk.
	- If the block that satisfies the requirement spans a chunk boundary or contains OCR noise so no single clean 5–20 word span is copyable, shorten the quote to the longest contiguous run you CAN copy character-for-character (even 2–4 words) rather than omitting the citation — a short exact quote is always preferable to no citation for a genuinely met requirement. Only omit the citation entirely when no contiguous exact substring of any length exists in the block.
	- A "met" or "not_met" verdict must include at least one valid citation and an evidence quote.
 
{killed_block}{block_ids_block}
{_ASSUMPTIONS_FIRST_RULES}
"""
