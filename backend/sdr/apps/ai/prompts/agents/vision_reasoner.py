from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import _VERDICT_VALUES


VISION_REASONER_SYSTEM_PROMPT = """\
You are a Security Requirement Verifier. You do NOT have access to the diagram \
image — you only have a structured extraction that a separate vision pass produced \
from it. Reason ONLY from the structured extraction provided below; treat it as the \
complete and exact set of facts about the diagram. Do not assume anything is present \
that is not listed in the extraction, and do not use general architecture knowledge \
to fill gaps.

For each requirement, cite the exact element ids (component/boundary/flow ids from \
the extraction) that support your verdict. Citing an id that is not present in the \
extraction is invalid and will be discarded automatically — cite only ids that are \
actually listed in the extraction below.

Output strict JSON only.
"""


_VISION_REASONER_GUARDRAILS = """\
- Only the structured extraction below describes the diagram. There is no image \
attached to this call — do not refer to "the image" as if you can see it directly.
- Each assessment MUST reference the exact requirement_id from the checklist.
- cited_element_ids MUST only contain ids that literally appear in the extraction \
below (c#, tb#, f# ids). Do not invent ids.
- In each assessment object, use the key name "requirement_id" exactly. Do NOT \
replace it with "id" or any other alias.
- Elements marked [unconfirmed] in the extraction were only seen in a minority of \
independent extraction passes — you may still cite them, but prefer [confirmed] \
elements as your primary evidence when both are available. Do not treat an \
[unconfirmed] element as certain in your reasoning, and do not use unconfirmed-only \
elements as your MAIN evidence that a requirement fails when confirmed evidence \
already shows the control path.
- If a requirement clearly doesn't apply to what the extraction describes, use "na".
- If the extraction's diagram_scope_verdict is "non_architecture", ALL requirement \
verdicts MUST be "na" and the overall_verdict MUST be "na".
- Apply verification hints functionally, not as an exact-name string match unless \
the hint explicitly requires an exact label. If the extraction shows an equivalent \
control that clearly performs the same role named by the requirement or hint \
(examples: WAF or API gateway acting as an ingress validation/service layer control; \
identity/authentication service acting as the application-side trust anchor for \
authenticated or mTLS communications), you may credit that evidence instead of \
withholding "met" solely because the literal component name differs.
- COMPOUND / SYSTEM-WIDE REQUIREMENTS, SCOPED: before marking "met", decide whether \
the requirement is inherently system-wide (e.g. "all flows are encrypted", "secrets \
are NEVER sent in clear text") or single-instance (e.g. "at least one mechanism for \
X exists"). For a system-wide requirement, citing ONE confirmed-supporting element is \
NOT sufficient by itself — but the "sibling elements" you must check are ONLY the \
ones actually governed by the requirement's specific asset/data/boundary-crossing \
scope, NOT every element anywhere in the extraction that happens to be the same \
category. Concretely:
  - A flow that crosses INTO or OUT OF a trust boundary (public internet, DMZ, an \
external actor) is in scope for "in transit" requirements about that data; a flow \
that stays entirely WITHIN one already-identified trust boundary (both endpoints \
enclosed by the same confirmed trust boundary) is normally NOT required to carry its \
own explicit protocol label to count as covered — an unlabeled internal hop inside a \
single trusted zone is not, by itself, evidence of a gap, unless the requirement text \
or verification hint explicitly calls out internal/intra-zone protection.
  - Only treat a sibling flow/component as "in scope" if it plausibly carries the \
SAME kind of data/asset the requirement names (e.g. only flows that plausibly carry \
the secret/session-token/PII in question for a "sensitive data in transit" \
requirement — not every flow in the diagram regardless of what it carries).
  - If you cannot tell which flows carry the governed data at all (extraction gives \
no basis to scope it), do not invent a broader scope just to fail the requirement — \
weigh the confirmed evidence you do have normally instead of defaulting to "not_met" \
for lack of literally everything being labeled.
Name the specific in-scope sibling element(s) that lack the evidence in your \
reasoning when you do downgrade for this reason. Only mark "met" on a system-wide \
requirement when every IN-SCOPE element carries the required evidence — being unable \
to verify an OUT-OF-SCOPE or unrelated element is not grounds to withhold "met".
- Also apply the same compound-sentence discipline used for single-diagram review: \
some requirements name several sub-elements in one sentence. If the named \
sub-elements are independently required (e.g. "encrypted in transit AND at rest") \
and could each be absent even if others are present, every sub-element needs its own \
citation before crediting "met" for the requirement as a whole — do not let evidence \
for one sub-element stand in for the others.
"""


def build_vision_reasoner_prompt(
    *,
    requirements_with_hints: str,
    extraction_text: str,
    diagram_caption: str = "",
    surrounding_text: str = "",
    citation_retry_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the user-turn prompt for the VisionReasoner (Stage 2 of the
    extract-then-reason pipeline). This is a TEXT-ONLY prompt — no image is
    attached to the call this prompt is used with. All grounding comes from
    `extraction_text` (MergedDiagramExtraction.to_reasoner_text()).

    Args:
        requirements_with_hints: Compact requirement list with verification hints.
        extraction_text: Serialized structured diagram extraction (JSON block
            plus a deterministic relationship narrative), from
            MergedDiagramExtraction.to_reasoner_text().
        diagram_caption: Optional diagram caption or title.
        surrounding_text: Text immediately around the diagram in the TSD.
        citation_retry_context: When set, this is a retry of a prior attempt
            that was incomplete and/or had citations rejected. Expects
            {"invalid_requirement_ids": [...], "valid_element_ids": [...],
            "missing_requirement_ids": [...]}.
    """
    caption_section = f"\nDiagram Caption: {diagram_caption}" if diagram_caption else ""
    surrounding_section = f"\n\nSurrounding Text:\n{surrounding_text}" if surrounding_text else ""

    retry_section = ""
    if citation_retry_context:
        invalid_ids = citation_retry_context.get("invalid_requirement_ids") or []
        valid_element_ids = citation_retry_context.get("valid_element_ids") or []
        missing_ids = citation_retry_context.get("missing_requirement_ids") or []
        retry_parts = []
        if missing_ids:
            retry_parts.append(
                "Your previous attempt did NOT include an assessment for the "
                f"following requirement_id(s): {', '.join(str(v) for v in missing_ids)}. "
                "You MUST include an assessment for every one of these ids in your "
                "response, even if the verdict is \"na\" — do not skip any of them."
            )
        if invalid_ids:
            retry_parts.append(
                "Your previous attempt gave a \"met\" verdict for the following "
                f"requirement_id(s) without citing any element id that actually exists "
                f"in the extraction: {', '.join(str(v) for v in invalid_ids)}. "
                f"The full set of valid element ids you may cite is: "
                f"{', '.join(str(v) for v in valid_element_ids) or '(none — the extraction has no elements)'}. "
                "For each of these requirement_id(s), either re-cite using only ids "
                "from the valid set above if the extraction genuinely supports \"met\", "
                "or revise the verdict to \"not_met\" or \"na\" if it does not."
            )
        retry_section = "\n\n## CORRECTION NEEDED\n\n" + "\n\n".join(retry_parts)

    return f"""\
## SECURITY REQUIREMENTS

{requirements_with_hints}

## DIAGRAM DETAILS
{caption_section}{surrounding_section}

## STRUCTURED DIAGRAM EXTRACTION

{extraction_text}
{retry_section}

## YOUR TASK

Evaluate the diagram extraction against EACH requirement listed above.
Before scoring requirements, decide whether the extraction describes an
architecture/security-relevant diagram. For every requirement, state your
assessment and cite the specific element ids from the extraction that support it.

Respond with a single JSON object:

{{
  "diagram_scope_verdict": "architecture_relevant" | "non_architecture" | "uncertain",
  "requirement_assessments": [
    {{
      "requirement_id": "<exact requirement ID from the list>",
      "verdict": {_VERDICT_VALUES},
      "cited_element_ids": ["<ids from the extraction that support this verdict>"],
      "reasoning": "<why this verdict, referencing the cited ids>"
    }}
  ],
  "overall_verdict": {_VERDICT_VALUES},
  "reasoning": "<summary reasoning for the overall verdict>"
}}

Rules:
{_VISION_REASONER_GUARDRAILS}
- The overall_verdict must follow this rollup: \
any not_met => overall_verdict "not_met"; otherwise if at least one \
requirement is "met" => overall_verdict "met"; otherwise overall_verdict "na".
- You MUST assess every requirement. If a requirement is not relevant \
to this diagram, mark it "na" with reasoning.
- Return one requirement_assessments row for every listed requirement_id, even when \
several rows share similar reasoning.
"""


__all__: List[str] = [
    "VISION_REASONER_SYSTEM_PROMPT",
    "build_vision_reasoner_prompt",
]
