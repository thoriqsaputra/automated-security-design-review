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

FOR not_met VERDICTS — Do NOT auto-uphold. Actively search all context chunks \
for evidence the Hunter overlooked. If you find unambiguous compliance evidence \
(e.g., an explicit architectural mandate naming the required mechanism), issue \
OVERTURN to "met" with a verified citation from a citable chunk. Only uphold \
not_met when no supporting evidence exists anywhere in the context.

FOR met VERDICTS — UPHOLD is the EXCEPTION, not the default. \
Issue UPHOLD ONLY when ALL of the following are true: \
  (1) at least two independent citations each naming a specific mechanism; \
  (2) the evidence covers ALL scoped aspects of the requirement (not just one layer); \
  (3) Hunter confidence ≥ 0.90. \
Issue PARTIAL (your DEFAULT response) whenever any of the following is true: \
  (a) only one citation, or citations only from generic NFR/baseline sections; \
  (b) the mechanism is named but HOW it is applied is not described; \
  (c) Hunter confidence < 0.90; \
  (d) evidence covers only one component or data-flow path when the requirement \
      applies to multiple (e.g., MFA on login but not on API access). \
Only OVERTURN if the cited evidence clearly cannot support the verdict at all.

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
- Do not let evidence for one child satisfy a different child.
- Scrutinise the Hunter's assumptions and cot_trace for logical leaps.
{block_ids_block}
"""

