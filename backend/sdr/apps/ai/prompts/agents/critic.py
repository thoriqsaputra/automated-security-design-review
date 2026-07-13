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
You are a Security Compliance Critic. Your job is to independently verify the Hunter's claim against the TSD context and return a structured challenge or confirmation.

Priority order:
1. Verify whether the cited evidence is real and correctly quoted.
2. Decide whether the evidence proves the requirement's core claim.
3. Search the rest of the provided context for evidence the Hunter missed.
4. Decide applicability at the governed-capability level, not only by named technology wording.

Use this review ladder:
- UPHOLD: the evidence is real and sufficiently proves the core claim.
- PARTIAL: the evidence is real but incomplete, indirect, too generic, contradicted elsewhere, or only covers part of the claim.
- OVERTURN: the Hunter's verdict is materially wrong.

Evidence policy:
- A TSD is architectural evidence, not source code. Named mechanisms, protocols, components, and explicit design mandates count as evidence.
- Reject generic claims, headings, or same-topic mentions that do not name a concrete mechanism.
- Do not use silence alone to prove a prohibition or absence-style requirement.
- When the Hunter says not_met, actively scan the full provided context for missed evidence before upholding.

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""


def build_critic_prompt(
    parameter_text: str,
    parameter_section: str,
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
Cited block IDs:
{citations_text}

## RAW TEXT FOR CITED BLOCKS ONLY
{cited_blocks_text}
{prior_round_block}
## YOUR TASK

Challenge or confirm the Hunter's finding.

Work in this order:
1. Verify each cited block and quoted evidence.
2. Judge whether the evidence proves the core claim, only part of it, or none of it.
3. If Hunter said not_met, scan the full context for missed evidence before upholding.
4. Decide whether the requirement is applicable to this design at the governed-capability level.

Respond with a single JSON object matching this exact schema:

{{
{_REASONING_SCHEMA},
  "outcome": "UPHOLD" | "OVERTURN" | "PARTIAL",
  "decision": "uphold" | "challenge" | "reject",
  "revised_verdict": {_VERDICT_VALUES},
  "revised_confidence": <float 0.0–1.0>,
  "applicability_status": "established" | "not_established",
  "applicability_reason": "<why this requirement is applicable or not>",
  "reasoning": "<one paragraph explaining your challenge or confirmation>",
  "weak_evidence": ["<evidence weakness or generic reasoning issue>", ...],
  "missed_evidence": ["<context evidence Hunter may have missed>", ...],
  "objections": ["<specific objection requiring Hunter rebuttal>", ...],
  "requires_rebuttal": <true | false>,
  "missing_expected_evidence": ["<specific missing control evidence>", ...],
  "valid_citations": [
    {_CITATION_SCHEMA}
  ],
  "invalid_citation_ids": ["<block_id>", ...]
}}

Few-shot example:
Input -> Hunter cites p2_b7 for MFA, but p2_b7 only says "users log in".
Reasoning -> assumptions: ["Validity depends on quoted evidence in cited blocks."]; logic_summary: "The cited block does not mention MFA, so the Hunter over-claimed compliance."; output -> outcome "OVERTURN", revised_verdict "not_met", invalid_citation_ids ["p2_b7"].
Rules:
- `UPHOLD` means the Hunter's verdict is materially correct after your verification.
- `PARTIAL` means some evidence is real but the claim remains incomplete, too generic, or only partially supported.
- `OVERTURN` means the Hunter's verdict is materially wrong and you are replacing it.
- `decision` must map cleanly to the outcome: `uphold` -> `UPHOLD`, `challenge` -> `PARTIAL`, `reject` -> `OVERTURN`.
- `valid_citations` may only contain personally verified citable block_ids from the provided context.
- If `revised_verdict` is `met`, `valid_citations` must be non-empty.
- When the Hunter's verdict is `met`, do not issue `UPHOLD` with empty `valid_citations`.
- Use `na` only when the governed capability itself is absent from the design scope, not merely because the exact named technology differs.
- If `revised_verdict` is `not_met`, fill `missing_expected_evidence` with the specific missing implementation evidence.
- In rebuttal rounds, explicitly check whether prior objections were resolved; keep unresolved objections active.
- Use evidence-only reasoning. Do not infer undocumented controls.
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
      "applicability_status": "established" | "not_established",
      "applicability_reason": "<why this requirement is applicable or not>",
      "reasoning": "<one paragraph>",
      "weak_evidence": ["<weakness>", "..."],
      "missed_evidence": ["<missed evidence>", "..."],
      "objections": ["<specific objection>", "..."],
      "requires_rebuttal": <true | false>,
      "missing_expected_evidence": ["<specific missing control evidence>", "..."],
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
- Use applicability_status="not_established" only when the child's prerequisite/capability is absent from the design scope. A silent TSD on an otherwise relevant control remains "established" and should usually be "not_met".
- Do not revise "not_met" to "na" merely because the exact named standard/mechanism (e.g. GraphQL, WS-Security, TOTP) is absent from the stack. If the design still has the governed capability via a different mechanism (a business logic layer, REST APIs, another second-factor, internal API endpoints as a service boundary, etc.), applicability remains established and silence is a "not_met" problem, not "na".
- CRITICAL: When a child's Hunter verdict is "met", never issue UPHOLD with an empty valid_citations list for that child. If you cannot locate a verified citation confirming a "met" verdict, issue PARTIAL instead. For "not_met" or "na" Hunter verdicts, UPHOLD with empty valid_citations is acceptable.
- Do not let evidence for one child satisfy a different child.
- Scrutinise the Hunter's assumptions and reasoning summary for logical leaps.
- Treat the Hunter's finding as a lead to independently verify, not a conclusion — re-derive the correct verdict yourself from the raw context.
- Do not reject genuine evidence merely because the Hunter's wording differs lexically from the requirement text (e.g. a spelled-out abbreviation, a synonym architecture description) — judge the underlying mechanism.
- When the requirement names specific categories, data types, or a specific relationship/protocol between named parties, verified evidence must reference one of those specific items, not just generic same-topic coverage. But before downgrading for lack of specificity, check the OTHER context chunks (not just the one the Hunter cited) for the missing detail — the first plausible block cited isn't always the most specific one available.
- Before UPHOLD on "met", also scan the other provided context chunks for anything that contradicts or narrows the cited evidence (e.g. marks it optional, deprecated, or scoped to a different component); if found, issue PARTIAL or OVERTURN instead.
- Compound requirements (multiple named sub-elements in one sentence): if the requirement tests one core mechanism and the other named elements are elaboration/context on it, verified evidence of the core mechanism is sufficient — don't demand a separate citation for every named element unless the requirement's core claim IS that specific element (e.g. an explicit no-sensitive-data-in-logs clause is the crux, not peripheral, if the requirement is about redaction specifically).
{block_ids_block}
"""
