from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.utils.parsing import strip_markdown_code_blocks, strip_thinking_block
logger = logging.getLogger(__name__)


@dataclass
class ASVSLevelClassification:
    level: Optional[int]
    confidence: float
    reasoning: str
    evidence: List[str]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "error": self.error,
        }


def _extract_json_payload(text: str) -> str:
    cleaned = strip_markdown_code_blocks(strip_thinking_block(text or "{}")).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()
    return cleaned


def _coerce_level(value: Any) -> Optional[int]:
    if isinstance(value, int) and value in (1, 2, 3):
        return value
    text = str(value or "").strip().upper()
    if text.startswith("L"):
        text = text[1:].strip()
    if text in {"1", "2", "3"}:
        return int(text)
    return None


def classify_tsd_asvs_level(tsd_text: str, levels: Iterable[Any]) -> ASVSLevelClassification:
    level_rows = sorted(list(levels), key=lambda item: item.level)
    if not level_rows:
        return ASVSLevelClassification(
            level=1,
            confidence=0.0,
            reasoning="ASVS level definitions were unavailable; defaulted to L1.",
            evidence=[],
            error="missing_level_definitions",
        )

    sample_text = (tsd_text or "")[:12000]
    if not sample_text.strip():
        return ASVSLevelClassification(
            level=1,
            confidence=0.0,
            reasoning="The TSD had no text available for ASVS level classification; defaulted to L1.",
            evidence=[],
            error="empty_tsd_text",
        )

    definitions = "\n".join(
        f"L{row.level} - {row.name}: {row.classification_guidance}" for row in level_rows
    )
    prompt = f"""
Classify this Technical Software Document into exactly one OWASP ASVS verification level.

Use these level definitions:
{definitions}

Return only valid JSON:
{{
  "level": 1 | 2 | 3,
  "confidence": 0.0,
  "reasoning": "short explanation",
  "evidence": ["short quote or signal", "..."]
}}

TSD SAMPLE:
{sample_text}
"""
    try:
        response = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify application technical design documents against OWASP ASVS "
                        "levels. Prefer the lowest level that fits the documented risk and assurance "
                        "needs. Return strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            component="tsd_asvs_level_classification",
            temperature=0.0,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        if response.error or not response.content:
            return ASVSLevelClassification(
                level=1,
                confidence=0.0,
                reasoning="ASVS level classification failed; defaulted to L1.",
                evidence=[],
                error=str(response.error or "empty_response"),
            )

        parsed = json.loads(_extract_json_payload(response.content))
        level = _coerce_level(parsed.get("level"))
        confidence = float(parsed.get("confidence") or 0.0)
        evidence = parsed.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        if level is None:
            return ASVSLevelClassification(
                level=1,
                confidence=0.0,
                reasoning="ASVS level classifier returned an invalid level; defaulted to L1.",
                evidence=[str(item) for item in evidence[:5]],
                error="invalid_level",
            )
        return ASVSLevelClassification(
            level=level,
            confidence=max(0.0, min(confidence, 1.0)),
            reasoning=str(parsed.get("reasoning") or "").strip(),
            evidence=[str(item) for item in evidence[:5]],
        )
    except Exception as exc:
        logger.exception("classify_tsd_asvs_level: failed")
        return ASVSLevelClassification(
            level=1,
            confidence=0.0,
            reasoning="ASVS level classification raised an exception; defaulted to L1.",
            evidence=[],
            error=str(exc),
        )


def filter_parameters_for_asvs_level(parameters: List[Any], effective_level: Optional[int]) -> tuple[List[Any], Dict[str, int]]:
    if effective_level not in (1, 2, 3):
        return list(parameters), {
            "before_count": len(parameters),
            "after_count": len(parameters),
            "excluded_by_level_count": 0,
            "unknown_level_included_count": sum(1 for item in parameters if getattr(item, "asvs_level", None) is None),
        }

    included: List[Any] = []
    excluded = 0
    unknown = 0
    for parameter in parameters:
        level = getattr(parameter, "asvs_level", None)
        if level is None:
            unknown += 1
            included.append(parameter)
        elif int(level) <= effective_level:
            included.append(parameter)
        else:
            excluded += 1

    return included, {
        "before_count": len(parameters),
        "after_count": len(included),
        "excluded_by_level_count": excluded,
        "unknown_level_included_count": unknown,
    }
