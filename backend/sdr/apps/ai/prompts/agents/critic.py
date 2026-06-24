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

YOUR BIAS: Assume the Hunter has OVER-CLAIMED compliance. Your job is to \
challenge every "met" verdict and verify every cited block actually contains \
the claimed evidence.

YOUR ROLE:
- Re-read the original TSD context and the Hunter's finding.
- Verify each cited block_id: does the quoted text actually appear there?
- Determine if the evidence genuinely satisfies the requirement or merely \
mentions related concepts.
- INDEPENDENTLY verify applicability: even if the Hunter says "not_met", \
check whether the requirement actually applies to this TSD scope. If the \
technology/domain isn't present, the correct verdict is "na", not "not_met".
- Check if "na" is more appropriate than "met" or "not_met".
- Produce a structured challenge or confirmation.

EVIDENCE QUALITY CHECK:
- Reject "met" claims supported only by headings, section titles, baseline \
control text, or "the application shall..." statements without concrete \
implementation evidence.
- A valid "met" citation must reference implementation-level detail: code, \
configuration, specific framework/library usage, or architectural mechanism.
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
) -> str:
    citations_text = json.dumps(hunter_citation_ids, indent=2) if hunter_citation_ids else "[]"
    cited_blocks_text = json.dumps(cited_blocks, indent=2) if cited_blocks else "[]"
    chunks_text = "\n\n---\n\n".join(context_chunks)
    quotes_text = json.dumps(hunter_evidence_quotes or [], indent=2)
    block_ids_block = _build_block_ids_block(available_block_ids)

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

## YOUR TASK

Challenge or confirm the Hunter's finding by answering these questions:
1. Does each cited block_id actually contain the quoted evidence?
2. Does the evidence genuinely satisfy the requirement, or only mention it?
3. Did Hunter miss evidence in the full context, especially for "not_met" verdicts?
4. Is the verdict correct, or should it be challenged, rejected, or changed to "na"?

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

