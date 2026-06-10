from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sdr.apps.reviews.models.choices import FindingType

RUBRIC_VERSION = "v1"

_BASE_SCORES = {
    "business_logic_concurrency": 8.0,
    "transaction_integrity": 8.0,
    "iam_access_control": 7.0,
    "architecture_network": 7.0,
    "data_crypto_privacy": 7.0,
    "general": 5.0,
}


@dataclass(frozen=True)
class DeterministicSeverity:
    severity: Optional[str]
    score: Optional[float]
    analysis: Optional[Dict[str, Any]]


def calculate_deterministic_severity(
    *,
    met_status: Optional[str],
    confidence_score: Optional[float],
    domain: Optional[str],
    finding_type: str,
    raw_final_verdict: Optional[str] = None,
    ambiguous_elements: Optional[List[str]] = None,
    missing_information: Optional[List[str]] = None,
) -> DeterministicSeverity:
    if met_status != "not_met":
        return DeterministicSeverity(severity=None, score=None, analysis=None)

    normalized_domain = (domain or "general").strip().lower() or "general"
    base_score = _BASE_SCORES.get(normalized_domain, _BASE_SCORES["general"])
    confidence = _clamp_confidence(confidence_score)
    confidence_modifier = _confidence_modifier(confidence)
    modifiers: List[Dict[str, Any]] = [
        {
            "kind": "confidence",
            "confidence_score": confidence,
            "delta": confidence_modifier,
        }
    ]

    final_score = base_score + confidence_modifier
    has_ambiguity = bool(ambiguous_elements)
    has_missing_information = bool(missing_information)

    if finding_type == FindingType.REQUIREMENT and raw_final_verdict == "partial":
        if final_score > 6.5:
            modifiers.append(
                {
                    "kind": "partial_verdict_cap",
                    "cap": 6.5,
                    "applied": True,
                }
            )
        final_score = min(final_score, 6.5)

    if finding_type == FindingType.DIAGRAM and (has_ambiguity or has_missing_information):
        if final_score > 6.5:
            modifiers.append(
                {
                    "kind": "diagram_ambiguity_cap",
                    "cap": 6.5,
                    "applied": True,
                }
            )
        final_score = min(final_score, 6.5)

    final_score = round(max(0.0, min(10.0, final_score)), 1)
    final_severity = _severity_bucket(final_score)
    analysis = {
        "source": "rubric_v1",
        "version": RUBRIC_VERSION,
        "inputs": {
            "met_status": met_status,
            "confidence_score": confidence,
            "domain": normalized_domain,
            "finding_type": finding_type,
            "raw_final_verdict": raw_final_verdict,
            "ambiguous_elements_present": has_ambiguity,
            "missing_information_present": has_missing_information,
        },
        "modifiers": modifiers,
        "final_score": final_score,
        "final_severity": final_severity,
    }
    return DeterministicSeverity(
        severity=final_severity,
        score=final_score,
        analysis=analysis,
    )


def _clamp_confidence(value: Optional[float]) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def _confidence_modifier(confidence: float) -> float:
    if confidence >= 0.9:
        return 1.0
    if confidence >= 0.8:
        return 0.5
    if confidence >= 0.7:
        return 0.0
    return -0.5


def _severity_bucket(score: float) -> str:
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    return "low"
