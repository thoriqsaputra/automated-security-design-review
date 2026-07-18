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
- UPHOLD: the evidence is real and sufficiently proves the core claim — a genuine, on-topic mechanism addressing the requirement's central assertion. Do not withhold UPHOLD merely because the evidence doesn't exhaustively enumerate every component, instance, or sub-detail the wording could be read to imply; that is a documentation-completeness bar, not proof of the core claim, and only applies when a complete inventory or catalog IS the requirement's explicit central claim.
- PARTIAL: the evidence is real but incomplete, indirect, too generic, contradicted elsewhere, or only covers part of the claim.
- OVERTURN: the Hunter's verdict is materially wrong.

Evidence policy:
- A TSD is architectural evidence, not source code. Named mechanisms, protocols, components, and explicit design mandates count as evidence.
- Reject generic claims, headings, or same-topic mentions that do not name a concrete mechanism.
- Proof ladder: accept evidence as satisfying when it is direct, universal/global and therefore applies to the governed object, equivalent to an example-named mechanism/standard, or a necessary semantic consequence of explicit design mandates. Do not require the exact phrase from the requirement when the cited mechanism proves the same control objective. Reject evidence only when it is adjacent, merely plausible, contradictory, or about a different governed object.
- Collective evidence may satisfy one requirement when multiple cited sections together cover the same governed capability. Do not downgrade merely because evidence is distributed across architecture, threat/security controls, data, and operations sections instead of appearing under one heading or in one inventory.
- Do not use silence alone to prove a prohibition or absence-style requirement. Evidence that a strong or modern mechanism is used does not by itself prove a weaker or legacy alternative is absent — architectures often retain undocumented legacy paths. A prohibition-style requirement is only "met" when there is direct evidence that structurally excludes the prohibited option (an explicit constraint, allow-list, or exclusion statement), not merely evidence that a better option happens to be used elsewhere.
- When the Hunter says not_met, actively scan the full provided context for missed evidence before upholding. Even when you agree with not_met, fill `missed_evidence` with short, specific descriptions of the exact evidence that would satisfy this requirement (named mechanisms, sections, or document areas to check) — this is what triggers another retrieval pass, so a bare "no missed evidence" forecloses that pass even when the TSD may cover it elsewhere.
- If the Hunter's not_met/na reasoning rests on the absence of a specific named technology or product, check whether the requirement's own wording marks that technology as an example rather than the requirement itself ("or other", "such as", "e.g.", "i.e." — e.g. "GraphQL or other data layer authorization logic"). If so, restate the requirement's underlying control objective in technology-neutral terms (e.g. "authorization enforced at the business logic layer, not the query-interface layer") and re-scan the context for an equivalent mechanism that satisfies that objective, even if it uses different tooling than the example named. If you find one with citable evidence, OVERTURN to `met`. Only uphold `not_met`/`na` if the entire control class — not just the named example — is genuinely absent or inapplicable.
- SYMMETRY RULE: apply the same standard of proof to overturning "met" that you apply to upholding it. Upholding "met" requires a real, on-topic citation; overturning to "not_met"/PARTIAL requires the same rigor in the other direction — you must name the specific missing element, contradicting passage, or narrowing detail that disqualifies the Hunter's evidence. "The context doesn't fully establish this" or a bare absence of an exhaustive citation is not, by itself, proof the core claim is false — it is proof only that you must keep scanning before you overturn.
- INVALIDATION BAR: to overturn or downgrade a Hunter "met" verdict, you must name a concrete, specific gap — either (a) the SAME specific evidence the Hunter cited, explaining concretely why it fails, or (b) a failed OBJECT/POLARITY/CLAUSE check below, naming which one and why. A failed check IS a named gap and satisfies this bar on its own. Generic language like "the evidence is weak," "doesn't fully address X," or "not sufficiently specific" — with no named evidence and no failed check — is NOT sufficient grounds to invalidate a "met" verdict.

Before choosing an outcome, run these checks and report them in the JSON (`requirement_object`, `requirement_polarity`, `clause_coverage`, `evidence_relation`, `risk_flags`):
- OBJECT CHECK: restate in one line the specific thing the requirement actually governs, using the term's meaning in this security standard's context — not just the closest matching keyword. E.g. "lookup secrets" means pre-generated user recovery/backup codes, NOT API keys, database credentials, or other infrastructure secrets; "cryptographic key management policy" means a governance document/process, not merely evidence that keys happen to be well-protected. If the Hunter's/your cited evidence is about a different, merely lexically-similar object, it does NOT satisfy this requirement — OVERTURN or PARTIAL, naming the object mismatch.
- POLARITY CHECK: classify the requirement as one of `positive` (a mechanism must exist), `prohibition` (something must NOT be used/present), or `policy` (an explicit document/process must exist). For `prohibition`, before UPHOLD/`met`, answer this exact verification question and let the answer decide the outcome: "Does the cited text itself claim to be a complete/closed list (e.g. 'only the following are permitted', 'no other algorithms/mechanisms are used', 'this is the complete list of...'), or does it just describe what's used without saying nothing else exists?" A section that merely names the secure/modern mechanisms currently in use is describing, not excluding — it is NOT a complete inventory unless the text contains that kind of explicit exhaustiveness statement, and "we use X instead" / silence about the prohibited item is never enough on its own; architectures often retain undocumented legacy paths alongside modern ones. For `policy`: a named standard in the requirement text introduced by "such as", "e.g.", "or other" is an example, not a mandatory literal citation — an equivalent evidenced process/document satisfies the requirement even without naming that standard.
- CLAUSE CHECK: decompose the requirement into its essential AND-joined clauses (ignore clauses that are elaboration/context on the same core mechanism, per the compound-requirement rule elsewhere in this prompt). The primary "Verify..." action is usually the central claim; advisory "should" wording, examples, document-placement instructions, and explanatory clauses are supporting context unless they are the explicit object being verified. For each essential clause, record whether it is evidenced by direct, universal, equivalent, or entailed evidence and by which citation. `met`/UPHOLD requires every essential clause evidenced; if the Hunter's own reasoning or evidence_assessment admits an essential clause is unverified, that alone is grounds for PARTIAL — do not UPHOLD `met` over the Hunter's own stated gap.
- LOGIC CHECK: classify each clause as `required`, `alternative`, `example`, or `context`. For OR/AND-OR groups, mark alternatives with the same `group_id` and `group_operator="any"`; one evidenced alternative satisfies that group. For pure AND requirements, use `group_operator="all"`.
- EVIDENCE RELATION CHECK: classify the strongest evidence as `direct`, `universal`, `equivalent`, `partial`, or `none`. `direct` names the object/control itself. `universal` applies to the object by an explicit all/global scope. `equivalent` satisfies an example-named technology through the same control objective. `partial` is adjacent or incomplete. `none` means no satisfying evidence.
- RISK FLAGS: include every applicable flag from this closed set: `absence_inference`, `object_substitution`, `generic_scope`, `compound_logic`, `policy_literalism`, `example_literalism`, `universal_scope_ignored`, `citation_thin`, `none`. Use `none` only when no other flag applies. A unanimous `met` with any real risk flag will receive another debate round, so be precise.
- Semantic equivalence examples: retention periods plus destruction at end of retention can evidence scheduled deletion; automated rotation of relevant service/database/API credentials can evidence no unchanging credentials; a biometric used as "secondary proof" or required "subsequently" can evidence a secondary factor; prescriptive HSM, rotation, lifecycle, and ownership mandates can evidence a key-management policy/process; security controls tied to architecture boundaries, threats, and remote access can evidence architecture security analysis; RBAC plus ABAC can be one composite authorization mechanism, while OAuth or API keys are authentication/credential mechanisms and do not automatically mean there are multiple authorization mechanisms.

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
3. If Hunter said not_met, scan the full context for missed evidence before upholding. This scan is not optional even when the Hunter's stated reason is "no mention of [named technology]" — a requirement naming one technology as an example ("or other", "such as", "e.g.") is not proven not_met just because that exact technology is absent; you must still check whether the context describes an equivalent mechanism doing the same job under different tooling.
4. Decide whether the requirement is applicable to this design at the governed-capability level, not by whether the named technology in the requirement's own wording appears verbatim in the TSD.

Respond with a single JSON object matching this exact schema:

{{
{_REASONING_SCHEMA},
  "requirement_object": "<one line: what specific thing this requirement governs, in the standard's terms>",
  "requirement_polarity": "positive" | "prohibition" | "policy",
  "clause_coverage": [
    {{"clause": "<clause text>", "role": "required" | "alternative" | "example" | "context", "group_id": "<short id or null>", "group_operator": "all" | "any" | null, "evidenced": <true | false>, "evidence_relation": "direct" | "universal" | "equivalent" | "partial" | "none", "citation_id": "<block_id or null>"}}
  ],
  "evidence_relation": "direct" | "universal" | "equivalent" | "partial" | "none",
  "risk_flags": ["absence_inference" | "object_substitution" | "generic_scope" | "compound_logic" | "policy_literalism" | "example_literalism" | "universal_scope_ignored" | "citation_thin" | "none", ...],
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

Few-shot examples:
Input -> Hunter cites p2_b7 for MFA, but p2_b7 only says "users log in".
Reasoning -> assumptions: ["Validity depends on quoted evidence in cited blocks."]; logic_summary: "The cited block does not mention MFA, so the Hunter over-claimed compliance."; output -> outcome "OVERTURN", revised_verdict "not_met", invalid_citation_ids ["p2_b7"].

Input -> Requirement: "Verify GraphQL or other data layer authorization logic is implemented at the business logic layer." Hunter verdict not_met, checked_context: "No mention of GraphQL found." Context includes p3_b12: "All access decisions are evaluated by the Attribute Based Access Control (ABAC) engine before the request reaches any data resource, based on business-context attributes."
Reasoning -> assumptions: ["'GraphQL' is one named example of a data-layer-authorization risk, not the requirement itself — the core claim is that authorization logic lives at the business layer, not the transport/query layer."]; logic_summary: "The TSD never mentions GraphQL, but p3_b12 shows an equivalent mechanism (ABAC) enforcing authorization at the business-context layer before data access — this satisfies the requirement's underlying control objective even though the named example technology is absent."; output -> outcome "OVERTURN", revised_verdict "met", valid_citations include p3_b12.

Input -> Requirement: "Verify the application does not use unsupported, insecure, or deprecated client-side technologies such as Flash, Shockwave, ActiveX, Silverlight, or client-side Java applets." Hunter verdict met, confidence 1.0, citing p9_b2: "The presentation layer uses React and modern responsive web design; supported browsers are Chrome, Firefox, Safari, Edge."
Reasoning -> assumptions: ["This is a prohibition requirement: 'met' requires evidence the banned technologies are structurally excluded, not merely that a modern alternative is also used."]; logic_summary: "requirement_polarity=prohibition. p9_b2 only shows React/modern browsers are used; it never states Flash/ActiveX/Silverlight/applets are banned, blocked, or absent from the build — architectures can retain legacy plugin support alongside a modern frontend. No structural exclusion is cited."; output -> outcome "OVERTURN", revised_verdict "not_met", weak_evidence: ["Evidence shows a modern frontend stack is used but never states the named deprecated technologies are excluded — presence of a good alternative does not prove absence of the prohibited one."].

Input -> Requirement: "Verify that lookup secrets are resistant to offline attacks, such as predictable values." Hunter verdict met, confidence 1.0, citing p24_b7: "API keys and database credentials are injected at runtime via an isolated Enterprise Secrets Vault with automated rotation; no hardcoded credentials are permitted."
Reasoning -> assumptions: ["'Lookup secrets' in this security standard means pre-generated one-time user recovery/backup codes (a second-factor mechanism), not infrastructure secrets like API keys or database credentials."]; logic_summary: "requirement_object=user recovery/backup codes used for authentication. p24_b7 is about infrastructure secret management (API keys, DB credentials) via a vault — a different, only lexically-similar object. It says nothing about how user-facing lookup secrets/recovery codes are generated or whether they resist offline guessing."; output -> outcome "OVERTURN", revised_verdict "not_met", weak_evidence: ["Cited evidence addresses infrastructure secret storage, not user-facing lookup/recovery secrets — object mismatch, requirement not actually addressed."].

Input -> Requirement: "Verify that known insecure block modes (e.g. ECB), padding modes (e.g. PKCS#1 v1.5), ciphers with small block sizes (e.g. Triple-DES, Blowfish), and weak hashing algorithms (e.g. MD5, SHA1) are not used unless required for backwards compatibility." Hunter verdict met, confidence 1.0, citing p21_b0: "all data at rest is secured using AES-256-GCM"; p33_b0: "passwords are hashed using Argon2id".
Reasoning -> assumptions: ["requirement_polarity=prohibition. The verification question is whether p21_b0/p33_b0 themselves claim to be a complete/closed list of every algorithm the system uses, or just describe what's used for these specific purposes."]; logic_summary: "The cited blocks name AES-256-GCM and Argon2id for data-at-rest encryption and password hashing respectively — they are on-topic and each secure, but neither states 'these are the only cryptographic algorithms used anywhere' or excludes ECB/PKCS#1v1.5/Triple-DES/MD5/SHA1 from other parts of the system (e.g. legacy integrations, third-party libraries, other hashing uses). Describing what's used for two specific purposes is not the same as an exhaustiveness statement, so the prohibition is not proven met from this evidence alone — do not treat 'the section doesn't mention anything insecure' as if it said 'nothing else is used'."; output -> outcome "OVERTURN", revised_verdict "not_met", weak_evidence: ["Evidence confirms secure algorithms for the two named purposes but contains no closed-list/exhaustiveness statement ruling out insecure algorithms elsewhere in the system — silence is not exclusion."].
Rules:
- `UPHOLD` means the Hunter's verdict is materially correct after your verification.
- `PARTIAL` means some evidence is real but the claim remains incomplete, too generic, or only partially supported.
- `OVERTURN` means the Hunter's verdict is materially wrong and you are replacing it.
- `decision` must map cleanly to the outcome: `uphold` -> `UPHOLD`, `challenge` -> `PARTIAL`, `reject` -> `OVERTURN`.
- `valid_citations` may only contain personally verified citable block_ids from the provided context.
- If `revised_verdict` is `met` or `not_met`, `valid_citations` must be non-empty.
- When the Hunter's verdict is `met`, do not issue `UPHOLD` with empty `valid_citations`.
- Use `na` only when the governed capability itself is absent from the design scope, not merely because the exact named technology differs.
- If `revised_verdict` is `not_met`, cite the closest inspected scope, partial implementation, contradiction, or surrounding control evidence block that proves what you checked. Do not return citationless `not_met`.
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
      "requirement_object": "<one line: what specific thing this requirement governs, in the standard's terms>",
      "requirement_polarity": "positive" | "prohibition" | "policy",
      "clause_coverage": [
        {{"clause": "<clause text>", "role": "required" | "alternative" | "example" | "context", "group_id": "<short id or null>", "group_operator": "all" | "any" | null, "evidenced": <true | false>, "evidence_relation": "direct" | "universal" | "equivalent" | "partial" | "none", "citation_id": "<block_id or null>"}}
      ],
      "evidence_relation": "direct" | "universal" | "equivalent" | "partial" | "none",
      "risk_flags": ["absence_inference" | "object_substitution" | "generic_scope" | "compound_logic" | "policy_literalism" | "example_literalism" | "universal_scope_ignored" | "citation_thin" | "none", "..."],
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
- If a child's revised_verdict is "met" or "not_met", that child's valid_citations MUST contain at least one such verified block_id. Never output revised_verdict "met" or "not_met" with an empty valid_citations list for that child.
- Use applicability_status="not_established" only when the child's prerequisite/capability is absent from the design scope. A silent TSD on an otherwise relevant control remains "established" and should usually be "not_met".
- Do not revise "not_met" to "na" merely because the exact named standard/mechanism (e.g. GraphQL, WS-Security, TOTP) is absent from the stack. If the design still has the governed capability via a different mechanism (a business logic layer, REST APIs, another second-factor, internal API endpoints as a service boundary, etc.), applicability remains established and silence is a "not_met" problem, not "na".
- CRITICAL: Never issue UPHOLD/PARTIAL/OVERTURN to a child's "met" or "not_met" revised_verdict with an empty valid_citations list. If the requirement is applicable but not satisfied, cite the closest inspected scope, partial implementation, contradiction, or surrounding control evidence. Empty valid_citations is acceptable only for "na".
- Do not let evidence for one child satisfy a different child.
- Scrutinise the Hunter's assumptions and reasoning summary for logical leaps.
- Treat the Hunter's finding as a lead to independently verify, not a conclusion — re-derive the correct verdict yourself from the raw context.
- Do not reject genuine evidence merely because the Hunter's wording differs lexically from the requirement text (e.g. a spelled-out abbreviation, a synonym architecture description) — judge the underlying mechanism.
- When the requirement names specific categories, data types, or a specific relationship/protocol between named parties, verified evidence must reference one of those specific items, not just generic same-topic coverage. But before downgrading for lack of specificity, check the OTHER context chunks (not just the one the Hunter cited) for the missing detail — the first plausible block cited isn't always the most specific one available.
- Before UPHOLD on "met", also scan the other provided context chunks for anything that contradicts or narrows the cited evidence (e.g. marks it optional, deprecated, or scoped to a different component); if found, issue PARTIAL or OVERTURN instead.
- SYMMETRY RULE: apply the same standard of proof to overturning "met" as you do to upholding it. An OVERTURN or PARTIAL against a Hunter "met" is only valid when you name the same specific evidence the Hunter cited and explain concretely why it fails, name the specific missing/contradicting evidence in `missed_evidence`/`weak_evidence`/`objections`, OR name a failed OBJECT/POLARITY/CLAUSE check (below) — a failed check IS a named gap. Generic language ("evidence is weak", "doesn't fully address X", "not specific enough") with no named evidence and no failed check is NOT sufficient grounds to overturn a "met". For a "not_met" revised verdict, cite the inspected scope or partial evidence that anchors the gap; do not return citationless "not_met".
- OBJECT CHECK (fill `requirement_object` for every child): restate in one line the specific thing the requirement actually governs, using the term's meaning in this security standard's context, not just the nearest keyword match. E.g. "lookup secrets" = pre-generated user recovery/backup codes, NOT API keys or database credentials; "key management policy" = a governance document/process, not merely evidence that keys happen to be well-protected. Evidence about a different, only lexically-similar object does NOT satisfy the requirement — OVERTURN or PARTIAL, naming the mismatch in `weak_evidence`.
- POLARITY CHECK (fill `requirement_polarity` for every child): classify as `positive` (a mechanism must exist), `prohibition` (something must NOT be used), or `policy` (a document/process must exist). For `prohibition`, before UPHOLD/`met`, answer: "Does the cited text itself claim to be a complete/closed list, or does it just describe what's used without saying nothing else exists?" Naming only the secure/modern mechanisms used for a specific purpose is describing, not excluding — `met` requires the text to contain an explicit exhaustiveness statement (e.g. "only the following are permitted", "no other algorithms are used"), not just "we use X instead" or silence about the prohibited item. For `policy`: a standard named via "such as"/"e.g."/"or other" is an example, not a mandatory literal citation.
- CLAUSE CHECK (fill `clause_coverage` for every child): decompose the requirement into clauses and classify each as `required`, `alternative`, `example`, or `context`. For AND requirements, every required clause must be evidenced. For OR/AND-OR requirements, one evidenced alternative in an `any` group is sufficient. If the Hunter's own reasoning admits an essential required clause is unverified, that alone is grounds for PARTIAL.
- EVIDENCE RELATION CHECK (fill `evidence_relation` for every child and each clause): classify the strongest evidence as `direct`, `universal`, `equivalent`, `partial`, or `none`. `direct` names the object/control itself. `universal` applies through explicit all/global scope. `equivalent` satisfies an example-named technology through the same control objective. `partial` is adjacent or incomplete.
- RISK FLAGS (fill `risk_flags` for every child): include all applicable flags from `absence_inference`, `object_substitution`, `generic_scope`, `compound_logic`, `policy_literalism`, `example_literalism`, `universal_scope_ignored`, `citation_thin`; use `none` only when no other flag applies.
- Compound requirements (multiple named sub-elements in one sentence): if the requirement tests one core mechanism and the other named elements are elaboration/context on it, verified evidence of the core mechanism is sufficient — don't demand a separate citation for every named element unless the requirement's core claim IS that specific element (e.g. an explicit no-sensitive-data-in-logs clause is the crux, not peripheral, if the requirement is about redaction specifically).
- Semantic entailment examples: retention periods plus destruction at end of retention can evidence scheduled deletion; automated relevant credential rotation can evidence no unchanging credentials; "secondary proof" or a subsequently required biometric can evidence a secondary factor; prescriptive HSM/lifecycle/rotation mandates can evidence a key-management policy/process; security controls tied to boundaries and threats can evidence architecture security analysis; a uniform RBAC+ABAC authorization path can evidence a single composite authorization mechanism.
{block_ids_block}
"""
