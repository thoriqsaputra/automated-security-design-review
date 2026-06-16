from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

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
    decision_mode: str = "unknown"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applicable": self.applicable,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "decision_mode": self.decision_mode,
            "error": self.error,
        }


_STOPWORDS = {
    "a",
    "an",
    "and",
    "application",
    "applications",
    "architectural",
    "architecture",
    "control",
    "controls",
    "design",
    "document",
    "for",
    "from",
    "in",
    "is",
    "of",
    "or",
    "require",
    "required",
    "requirement",
    "requirements",
    "security",
    "shall",
    "should",
    "standard",
    "system",
    "that",
    "the",
    "this",
    "to",
    "use",
    "using",
    "with",
}


def _normalize_terms(values: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = (value or "").strip().lower()
        if len(text) < 3 or text in _STOPWORDS or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _extract_scope_terms(
    *,
    parent_title: str,
    parent_description: str,
    child_requirements: List[str],
    query_details: Optional[Dict[str, Any]] = None,
) -> List[str]:
    raw_terms: List[str] = []
    query_details = query_details or {}
    for item in query_details.get("family_scope_terms") or []:
        raw_terms.append(str(item))
    raw_terms.extend(
        re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", " ".join([parent_title, parent_description]))
    )
    for requirement in child_requirements[:4]:
        raw_terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", requirement))
    return _normalize_terms(raw_terms)


def _match_scope_terms(scope_terms: Sequence[str], context_text: str) -> List[str]:
    lowered = (context_text or "").lower()
    matches = []
    for term in scope_terms:
        if term and term in lowered:
            matches.append(term)
    return matches[:8]


def classify_parent_applicability(
    *,
    category_code: str,
    version_label: str,
    parent_title: str,
    parent_description: str,
    child_requirements: List[str],
    retrieved_context: str,
    query_details: Optional[Dict[str, Any]] = None,
) -> ParentApplicabilityResult:
    context_text = (retrieved_context or "").strip()
    if not child_requirements:
        return ParentApplicabilityResult(
            applicable=False,
            confidence=0.0,
            reasoning="No child requirements were supplied, so applicability could not be established.",
            evidence=[],
            decision_mode="missing_child_requirements",
            error="missing_child_requirements",
        )
    if not context_text:
        return ParentApplicabilityResult(
            applicable=False,
            confidence=0.0,
            reasoning="No retrieved TSD context was available, so applicability could not be established.",
            evidence=[],
            decision_mode="missing_context",
            error="missing_retrieved_context",
        )
    scope_terms = _extract_scope_terms(
        parent_title=parent_title,
        parent_description=parent_description,
        child_requirements=child_requirements,
        query_details=query_details,
    )
    matched_scope_terms = _match_scope_terms(scope_terms, context_text)
    if scope_terms and not matched_scope_terms:
        return ParentApplicabilityResult(
            applicable=False,
            confidence=0.9,
            reasoning="The retrieved TSD context does not contain direct scope signals for this control family.",
            evidence=[],
            decision_mode="no_scope_match",
            error=None,
        )

    child_block = "\n".join(f"- {item}" for item in child_requirements[:8])
    prompt = build_parent_applicability_prompt(
        category_code=category_code,
        version_label=version_label,
        parent_title=parent_title,
        parent_description=parent_description,
        child_block=child_block,
        context_text=context_text,
        scope_terms=matched_scope_terms or scope_terms[:8],
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
                applicable=False,
                confidence=0.0,
                reasoning="Parent applicability classification failed, so applicability could not be established.",
                evidence=matched_scope_terms[:5],
                decision_mode="error",
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
                applicable=False,
                confidence=0.0,
                reasoning="Parent applicability classification could not be parsed, so applicability could not be established.",
                evidence=matched_scope_terms[:5],
                decision_mode="parse_error",
                error=parse_error or "invalid_response",
            )
        applicable = bool(parsed.get("applicable", True))
        confidence = max(0.0, min(float(parsed.get("confidence") or 0.0), 1.0))
        evidence = parsed.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        decision_mode = str(parsed.get("decision_mode") or "").strip().lower()
        if not decision_mode:
            decision_mode = "positive_match" if applicable else "negative_match"
        if applicable and not matched_scope_terms:
            applicable = False
            decision_mode = "no_scope_match"
            confidence = max(confidence, 0.8)
        return ParentApplicabilityResult(
            applicable=applicable,
            confidence=confidence,
            reasoning=str(parsed.get("reasoning") or "").strip(),
            evidence=[str(item) for item in (evidence[:5] or matched_scope_terms[:5])],
            decision_mode=decision_mode,
        )
    except Exception as exc:
        logger.exception("classify_parent_applicability: failed")
        return ParentApplicabilityResult(
            applicable=False,
            confidence=0.0,
            reasoning="Parent applicability classification raised an exception, so applicability could not be established.",
            evidence=matched_scope_terms[:5],
            decision_mode="exception",
            error=str(exc),
        )
