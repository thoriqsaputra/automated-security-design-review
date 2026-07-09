from __future__ import annotations

from typing import List


DIAGRAM_GATEKEEPER_SYSTEM_PROMPT = """\
You are a Diagram Requirement Gatekeeper.

Your only job is to decide, for each candidate security requirement in a \
batch, whether it is genuinely CHECKABLE from the attached diagram — i.e. \
could a well-annotated version of this diagram plausibly show evidence that \
would let a reviewer determine whether the requirement is met or not met?

This is a scope/relevance judgment, NOT a met/not_met verdict. Do not judge \
whether the diagram actually shows the control correctly — only whether this \
diagram's type and content is capable of depicting this kind of control at \
all. A requirement about network segmentation is relevant to an architecture \
diagram even if segmentation isn't drawn (that would be a "not_met" \
judgment, made elsewhere, by a different agent) — it is NOT relevant if the \
requirement is about something a diagram fundamentally cannot show regardless \
of quality (e.g. process/documentation checks like "verify a security \
analysis was performed").

Output strict JSON only.
"""


def build_diagram_gatekeeper_prompt(*, requirements_batch: List) -> str:
    """
    Build the user-turn prompt for one gatekeeper batch call.

    Args:
        requirements_batch: a slice of requirement objects (ordinal, stable_key,
            requirement_text, verification_hint) — a chunk of the full candidate
            pool, not necessarily the whole thing.
    """
    lines = []
    for req in requirements_batch:
        req_id = getattr(req, "stable_key", f"D-{getattr(req, 'ordinal', 0)}")
        text = getattr(req, "requirement_text", "")
        hint = getattr(req, "verification_hint", "")
        lines.append(f"- [{req_id}] {text}")
        if hint:
            lines.append(f"  VERIFY: {hint}")
    requirements_text = "\n".join(lines)

    return f"""\
## CANDIDATE REQUIREMENTS

{requirements_text}

## YOUR TASK

For EACH candidate requirement above, decide if it is checkable from the \
attached diagram image.

Respond with a single JSON object:

{{
  "assessments": [
    {{
      "requirement_id": "<exact requirement ID from the list, in brackets above>",
      "relevant": true | false,
      "reasoning": "<one short sentence>"
    }}
  ]
}}

Rules:
- You MUST include an entry for every requirement listed above — do not skip any.
- Each assessment MUST reference the exact requirement_id from the list.
- Claims without a matching requirement_id are invalid and will be discarded.
"""
