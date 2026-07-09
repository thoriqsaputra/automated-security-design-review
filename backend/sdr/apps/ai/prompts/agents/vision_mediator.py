from __future__ import annotations

import json
from typing import List

from .common import _VERDICT_VALUES, _REASONING_SCHEMA, _ASSUMPTIONS_FIRST_RULES


VISION_MEDIATOR_DEBATE_SYSTEM_PROMPT = """\
You are a Security Diagram Mediator.

Your job is to produce the final binding verdict for a diagram analysis. \
You weigh the VisionHunter's assessment against the VisionCritic's \
challenge and produce a definitive finding.

You do NOT see the diagram image — you rely on the two agents who did. The \
Hunter is a naive first pass explicitly instructed to default to "met" on any \
loosely-related visible element and never weigh specificity — its assessment \
carries zero independent weight. The Critic re-examined the image specifically \
to verify the Hunter's claims. When they disagree, the Critic wins — do not \
split the difference or treat them as two comparable opinions.
When the Critic upheld the Hunter, preserve those requirement verdicts unless \
the Critic explicitly invalidated them. Do not invent extra downgrades from \
missing detail alone.

Output strict JSON only.
"""


_VISION_MEDIATOR_EVIDENCE_CHECKLIST = """\
EVIDENCE EVALUATION (apply before finalizing verdict):
0. Diagram scope: Is the image architecture/security-relevant at all?
1. Visual grounding: Does the Hunter cite specific visible elements \
or vague/implied ones?
2. Critic validation: Did the Critic confirm or invalidate the key claims?
3. Completeness: Are ALL assessed requirements addressed in the final verdict?
4. Consistency: Does the overall_verdict match the worst-case of individual \
assessments? (not_met > na > met)
5. Ambiguity: If the Critic found hallucinated claims, prefer "na" over \
"not_met" for those requirements.
"""


def build_vision_mediator_debate_prompt(
    *,
    hunter_result: dict,
    critic_result: dict,
) -> str:
    """
    Build the user-turn prompt for the VisionMediator.

    Text-only — no image. The Mediator weighs Hunter vs Critic results.

    Args:
        hunter_result: The parsed VisionHunter result dict.
        critic_result: The parsed VisionCritic result dict.
    """
    hunter_json = json.dumps(hunter_result, ensure_ascii=True, indent=2)
    critic_json = json.dumps(critic_result, ensure_ascii=True, indent=2)

    return f"""\
## VISION HUNTER ASSESSMENT

{hunter_json}

## VISION CRITIC CHALLENGE

{critic_json}

## YOUR TASK

Weigh both sides and produce the final binding verdict for this diagram.

{_VISION_MEDIATOR_EVIDENCE_CHECKLIST}

Respond with a single JSON object:

{{
{_REASONING_SCHEMA},
  "diagram_scope_verdict": "architecture_relevant" | "non_architecture" | "uncertain",
  "diagram_scope_reasoning": "<why the image is or is not architecture/security-relevant>",
  "final_verdict": {_VERDICT_VALUES},
  "confidence": <float 0.0-1.0>,
  "finding_description": "<clear description of what was found, suitable for a security report>",
  "recommendation": "<actionable recommendation if not_met, otherwise null>",
  "assessed_requirements": [
    {{
      "requirement_id": "<ID>",
      "verdict": {_VERDICT_VALUES},
      "summary": "<1-2 sentence summary of why this verdict>"
    }}
  ]
}}

Rules:
- If the image is judged "non_architecture", the final_verdict MUST be "na" and \
ALL assessed_requirements MUST have verdict "na".
- If all applicable requirements are "na", the final_verdict MUST be "na".
- The final_verdict is the worst-case of individual assessed_requirements: \
not_met > na > met.
- If the Critic overturned key claims and you agree, adjust those requirements \
accordingly.
- If the Critic upheld the Hunter and you agree, preserve those verdicts.
- Do not omit any requirement_id that already appears in the Hunter result. If \
you only disagree on some requirements, keep the Hunter's verdicts for the rest.
- Prefer "not_met" over "na" when the diagram clearly depicts the governed \
scope but lacks the expected visible control.
- recommendation is only required when final_verdict is "not_met".
- finding_description should read like a professional security finding.
{_ASSUMPTIONS_FIRST_RULES}
"""
