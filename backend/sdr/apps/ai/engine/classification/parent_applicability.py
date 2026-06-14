from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.prompts.analysis import (
    PARENT_APPLICABILITY_SYSTEM_PROMPT,
    build_parent_applicability_prompt,
)
from sdr.apps.ai.engine.classification.json_utils import parse_json_with_repair

logger = logging.getLogger(__name__)


@dataclass
class ParentApplicabilityResult:
    applicable: bool
    confidence: float
    reasoning: str
    evidence: List[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applicable": self.applicable,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "error": self.error,
        }


def classify_parent_applicability(
    *,
    category_code: str,
    version_label: str,
    parent_title: str,
    parent_description: str,
    child_requirements: List[str],
    retrieved_context: str,
) -> ParentApplicabilityResult:
    context_text = (retrieved_context or "").strip()
    if not child_requirements:
        return ParentApplicabilityResult(
            applicable=True,
            confidence=0.0,
            reasoning="No child requirements were supplied; treated as applicable.",
            evidence=[],
            error="missing_child_requirements",
        )
    if not context_text:
        return ParentApplicabilityResult(
            applicable=True,
            confidence=0.0,
            reasoning="No retrieved TSD context was available; treated as applicable.",
            evidence=[],
            error="missing_retrieved_context",
        )

    child_block = "\n".join(f"- {item}" for item in child_requirements[:8])
    prompt = build_parent_applicability_prompt(
        category_code=category_code,
        version_label=version_label,
        parent_title=parent_title,
        parent_description=parent_description,
        child_block=child_block,
        context_text=context_text,
    )
    try:
        response = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": PARENT_APPLICABILITY_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            component="parent_applicability",
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        if response.error or not response.content:
            return ParentApplicabilityResult(
                applicable=True,
                confidence=0.0,
                reasoning="Parent applicability classification failed; treated as applicable.",
                evidence=[],
                error=str(response.error or "empty_response"),
            )

        parsed, parse_error = parse_json_with_repair(
            response.content,
            component="parent_applicability",
            max_tokens=500,
            chat_completion_fn=chat_completion,
        )
        if not isinstance(parsed, dict):
            return ParentApplicabilityResult(
                applicable=True,
                confidence=0.0,
                reasoning="Parent applicability classification could not be parsed; treated as applicable.",
                evidence=[],
                error=parse_error or "invalid_response",
            )
        applicable = bool(parsed.get("applicable", True))
        confidence = max(0.0, min(float(parsed.get("confidence") or 0.0), 1.0))
        evidence = parsed.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        return ParentApplicabilityResult(
            applicable=applicable,
            confidence=confidence,
            reasoning=str(parsed.get("reasoning") or "").strip(),
            evidence=[str(item) for item in evidence[:5]],
        )
    except Exception as exc:
        logger.exception("classify_parent_applicability: failed")
        return ParentApplicabilityResult(
            applicable=True,
            confidence=0.0,
            reasoning="Parent applicability classification raised an exception; treated as applicable.",
            evidence=[],
            error=str(exc),
        )
