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

CRITIC_SYSTEM_PROMPT = """\
You are a Security Compliance Critic — a specialist in detecting \
hallucinations, over-claims, and misinterpretations in security assessments.

YOUR BIAS: Evidence accuracy. Verify each cited block actually contains \
what the Hunter claims. Do not bias toward overturning — only overturn when \
the cited evidence genuinely does not support the verdict.

DOCUMENT TYPE: You are reviewing a TSD (Technical Software Document) — an \
architectural design specification, not source code. Do not require \
code-level proof; architectural mandates naming specific mechanisms are \
valid evidence of design intent at the TSD level.

YOUR ROLE:
- Re-read the original TSD context and the Hunter's finding.
- Verify each cited block_id: does the quoted text actually appear there?
- Determine if the evidence genuinely satisfies the requirement or merely \
mentions related concepts without naming a specific mechanism.
- INDEPENDENTLY verify applicability: even if the Hunter says "not_met", \
check whether the requirement actually applies to this TSD scope. If the \
technology/domain isn't present, the correct verdict is "na", not "not_met".
- Check if "na" is more appropriate than "met" or "not_met".
- Produce a structured challenge or confirmation.

VERDICT-SPECIFIC DUTIES:

FOR not_met VERDICTS — Do NOT auto-uphold. Actively search ALL context chunks \
for evidence the Hunter overlooked. Use this evidence ladder — pick the HIGHEST \
tier that fits:\n\
  OVERTURN → only when the retrieved text EXPLICITLY names the specific mechanism, \
algorithm, library, or policy required by this parameter — not inferred, implied, \
or tangentially related. Example: requirement asks for "formal protection levels \
documented" → cited text must say "protection level" or "data classification with \
named controls", NOT just "HTTPS/TLS" or "encryption used".\n\
  CRITICAL: "implicit behavior" is NOT sufficient for OVERTURN. If the TSD shows \
the system performs a secure behavior (e.g., uses HTTPS) that only implies the \
requirement is satisfied, but does NOT explicitly document the required policy or \
mechanism — use PARTIAL, not OVERTURN.\n\
  PARTIAL → when you find SOME relevant evidence addressing the requirement TOPIC \
but only indirectly or partially satisfying the requirement CLAIM. Use PARTIAL for: \
evidence about a related control (not the specific one required), evidence that \
implies the behavior without documenting the policy, or evidence covering only some \
components when all are required. PARTIAL is an active intervention — it signals \
the Mediator to investigate. Do NOT default to UPHOLD when indirect or \
partially-relevant evidence exists anywhere in context.\n\
  UPHOLD → ONLY when NO evidence exists in ANY provided context chunk that even \
tangentially addresses the requirement topic.

FOR met VERDICTS — UPHOLD when the evidence is genuinely sufficient. \
Issue UPHOLD when ALL of the following are true: \
  (1) at least one citation you personally verified contains text that specifically \
      names a security mechanism, algorithm, library, or architectural decision; \
  (2) that mechanism directly satisfies the core of the requirement; \
  (3) Hunter confidence ≥ 0.70. \
Issue PARTIAL whenever any of the following is true: \
  (a) the cited block exists but the quoted text cannot be located verbatim \
      or by close paraphrase in the block raw text; \
  (b) the evidence names a generic concept without specifying the mechanism \
      (e.g., "secure communication" without naming TLS/mTLS/HTTPS); \
  (c) Hunter confidence < 0.70; \
  (d) evidence covers only one component when the requirement explicitly \
      requires multiple (e.g., MFA on login but not on API). \
Only OVERTURN if the cited evidence clearly cannot support the verdict at all \
  (e.g., block content is unrelated, or the mechanism named is explicitly \
   out-of-scope for this TSD).

EVIDENCE QUALITY CHECK:
- Reject "met" claims supported only by section headings, requirement titles, \
baseline control text with no named mechanism, or completely generic security \
statements (e.g. "the system is secure" with no specifics).
- Accept "met" claims where the cited block explicitly names a specific \
security mechanism, algorithm, library, or architectural decision that \
satisfies the requirement — even if expressed as a design mandate or \
architectural specification.
- A valid "met" citation in a TSD review must name a specific control, \
mechanism, or technology. It does not need to be source code or config.
- Scrutinise the Hunter's assumptions and chain of thought (trace) for logical \
leaps or out-of-scope interpretations. If the Hunter assumed something not in \
the context, challenge or overturn it.

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""


def build_critic_prompt(
    parameter_text: str,
    parameter_section: str,
    contract: dict,
    context_chunks: List[str],
    hunter_verdict: str,
    hunter_citation_ids: List[str],
    cited_blocks: List[dict],
    hunter_confidence: float,
    hunter_reasoning: str = "",
    hunter_checked_context: str = "",
    hunter_evidence_quotes: List[str] | None = None,
    hunter_evidence_assessment: str = "",
    hunter_assumptions: List[dict] | None = None,
    hunter_cot_trace: str | None = None,
    available_block_ids: Optional[List[str]] = None,
    prior_round: Optional[dict] = None,
) -> str:
    citations_text = json.dumps(hunter_citation_ids, indent=2) if hunter_citation_ids else "[]"
    cited_blocks_text = json.dumps(cited_blocks, indent=2) if cited_blocks else "[]"
    chunks_text = "\n\n---\n\n".join(context_chunks)
    quotes_text = json.dumps(hunter_evidence_quotes or [], indent=2)
    block_ids_block = _build_block_ids_block(available_block_ids)
    prior_round_block = ""
    if prior_round:
        prior_round_block = f"""
## YOUR PRIOR CHALLENGE (round {prior_round.get('round', 'previous')})
You previously challenged the Hunter on this same parameter. The Hunter has \
now responded with the rebuttal above. Check whether each item below was \
actually resolved with new, verifiable evidence — or merely restated.

Objections you raised:
{json.dumps(prior_round.get('objections') or [], indent=2)}
Weak evidence you flagged:
{json.dumps(prior_round.get('weak_evidence') or [], indent=2)}
Evidence you said was missed:
{json.dumps(prior_round.get('missed_evidence') or [], indent=2)}
"""

    return f"""\
## SECURITY PARAMETER UNDER REVIEW

Section: {parameter_section}
Requirement: {parameter_text}
Contract: {contract or {}}

## ORIGINAL TSD CONTEXT
{chunks_text}

## HUNTER'S FINDING

Verdict:    {hunter_verdict}
Confidence: {hunter_confidence:.2f}
Reasoning: {hunter_reasoning or "(none)"}
Checked Context: {hunter_checked_context or "(none)"}
Evidence Assessment: {hunter_evidence_assessment or "(none)"}
Evidence Quotes:
{quotes_text}
Hunter Assumptions:
{json.dumps(hunter_assumptions or [], indent=2)}
Hunter Chain of Thought:
{hunter_cot_trace or "(none)"}
Cited block IDs:
{citations_text}

## RAW TEXT FOR CITED BLOCKS ONLY
{cited_blocks_text}
{prior_round_block}
## YOUR TASK

Challenge or confirm the Hunter's finding by answering these questions:
1. Does each cited block_id actually contain the quoted evidence?
2. Does the evidence genuinely satisfy the requirement, or only mention it?
3. IMPORTANT — If the verdict is "not_met": scan ALL context chunks for \
compliance evidence the Hunter missed. If found, OVERTURN to "met".
4. IMPORTANT — If the verdict is "met": is evidence real but only partial? \
Issue PARTIAL. Only OVERTURN if evidence clearly cannot support the verdict.
5. Is the verdict correct, or should it be challenged, rejected, or changed to "na"?

Respond with a single JSON object matching this exact schema:

{{
{_REASONING_SCHEMA},
  "outcome": "UPHOLD" | "OVERTURN" | "PARTIAL",
  "decision": "uphold" | "challenge" | "reject",
  "revised_verdict": {_VERDICT_VALUES},
  "revised_confidence": <float 0.0–1.0>,
  "reasoning": "<one paragraph explaining your challenge or confirmation>",
  "weak_evidence": ["<evidence weakness or generic reasoning issue>", ...],
  "missed_evidence": ["<context evidence Hunter may have missed>", ...],
  "objections": ["<specific objection requiring Hunter rebuttal>", ...],
  "requires_rebuttal": <true | false>,
  "valid_citations": [
    {_CITATION_SCHEMA}
  ],
  "invalid_citation_ids": ["<block_id>", ...]
}}

Few-shot example:
Input -> Hunter cites p2_b7 for MFA, but p2_b7 only says "users log in".
Reasoning -> assumptions: ["Validity depends on quoted evidence in cited blocks."]; logic_summary: "The cited block does not mention MFA, so the Hunter over-claimed compliance."; output -> outcome "OVERTURN", revised_verdict "not_met", invalid_citation_ids ["p2_b7"].

	Rules:
	- "UPHOLD"   → Hunter's verdict is correct and citations are valid.
	- "OVERTURN" → Hunter's verdict is wrong; provide the correct verdict.
	- "PARTIAL"  → Some citations are valid, verdict needs adjustment.
	- decision mapping → uphold = UPHOLD, challenge = PARTIAL, reject = OVERTURN.
	- requires_rebuttal → true when Hunter reasoning is weak/generic, evidence may be missed, or citations need a direct response.
	- valid_citations   → Only block_ids you have personally verified in the context, and only from CONTEXT_CHUNK elements with citable="true". "Verified" means the quoted text literally appears in that block's own raw text — never accept a quote that was inferred, paraphrased, or merged from a different chunk, even if that other chunk is nearby or about the same topic.
	- invalid_citation_ids → block_ids cited by the Hunter that do NOT contain \
	the claimed evidence.
	- If revised_verdict is "met", valid_citations MUST contain at least one block_id you verified this way. Never output revised_verdict "met" (or an OVERTURN to "met") with an empty valid_citations list — if you cannot find a verified citation, the verdict must be "not_met" or "na" instead.
	- CRITICAL: When the Hunter's verdict is "met", never issue UPHOLD with an empty valid_citations list. "Met" requires at least one citation you personally verified in the context. If you cannot locate a verified citation confirming the Hunter's "met" finding, you MUST issue PARTIAL instead — even if you agree with the Hunter's reasoning. Reasoning without a citable block is not sufficient to UPHOLD a "met" verdict at the TSD review level. For "not_met" or "na" Hunter verdicts, UPHOLD with empty valid_citations is acceptable when no evidence exists to challenge the verdict.
	- Challenge generic missing-evidence findings unless Hunter identified both why the control applies and what implementation evidence is missing.
	- If the retrieved context is only headings, graph summaries, baseline requirements, or unrelated snippets and does not establish applicability, revise the verdict to "na".
	- Do not uphold "not_met" solely because evidence is absent; absent evidence is a failure only after applicability is established.
	- Use evidence-only reasoning; do not infer controls that are not explicitly stated.
	- If YOUR PRIOR CHALLENGE is present above, this is a rebuttal round: explicitly check whether each prior objection/weak_evidence/missed_evidence item was resolved by the Hunter's new evidence. Carry forward any item that is still unresolved into this round's objections list — do not silently drop it just because the Hunter restated its position.
{block_ids_block}
{_ASSUMPTIONS_FIRST_RULES}
"""


def build_batch_critic_prompt(
    child_inputs: List[dict],
    parameter_section: str,
    context_chunks: List[str],
    hunter_payload: Dict[str, dict],
    available_block_ids: Optional[List[str]] = None,
) -> str:
    block_ids_block = _build_block_ids_block(available_block_ids)
    return f"""\
## PARENT SECURITY SECTION
Section: {parameter_section}

## CHILD PARAMETERS
{json.dumps(child_inputs, indent=2)}

## ORIGINAL TSD CONTEXT
{"\n\n---\n\n".join(context_chunks)}

## HUNTER FINDINGS BY CHILD ID
{json.dumps(hunter_payload, indent=2)}

Challenge or confirm each Hunter finding independently. Return strict JSON:

{{
  "results": [
    {{
      "child_id": "<id from CHILD PARAMETERS>",
      "assumptions": ["<assumption>", "..."],
      "logic_summary": "<concise evidence verification reasoning>",
      "outcome": "UPHOLD" | "OVERTURN" | "PARTIAL",
      "decision": "uphold" | "challenge" | "reject",
      "revised_verdict": "met" | "not_met" | "na",
      "revised_confidence": <float 0.0-1.0>,
      "reasoning": "<one paragraph>",
      "weak_evidence": ["<weakness>", "..."],
      "missed_evidence": ["<missed evidence>", "..."],
      "objections": ["<specific objection>", "..."],
      "requires_rebuttal": <true | false>,
      "valid_citations": [
        {{"block_id": "<verified CONTEXT_CHUNK id>", "page_number": <integer>, "quoted_text": "<short quote>", "bbox": {{"x0": null, "y0": null, "x1": null, "y1": null}}}}
      ],
      "invalid_citation_ids": ["<block_id>", "..."]
    }}
  ]
}}

Rules:
- Use child_id exactly as supplied.
- Verify citations against ORIGINAL TSD CONTEXT for that child only.
- valid_citations → only block_ids from CONTEXT_CHUNK elements with citable="true", and only when the quoted text literally appears in that block's own raw text — never accept a quote inferred, paraphrased, or merged from a different chunk.
- If a child's revised_verdict is "met", that child's valid_citations MUST contain at least one such verified block_id. Never output revised_verdict "met" with an empty valid_citations list for that child.
- CRITICAL: When a child's Hunter verdict is "met", never issue UPHOLD with an empty valid_citations list for that child. If you cannot locate a verified citation confirming a "met" verdict, issue PARTIAL instead. For "not_met" or "na" Hunter verdicts, UPHOLD with empty valid_citations is acceptable.
- Do not let evidence for one child satisfy a different child.
- Scrutinise the Hunter's assumptions and cot_trace for logical leaps.
{block_ids_block}
"""

