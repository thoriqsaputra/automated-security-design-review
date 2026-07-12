from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from sdr.apps.reviews.models.choices import FindingType

RUBRIC_VERSION = "v2"

_BASE_SCORES = {
    "business_logic_concurrency": 7.5,
    "transaction_integrity": 7.5,
    "iam_access_control": 7.0,
    "architecture_network": 6.5,
    "data_crypto_privacy": 7.0,
    "general": 5.0,
}

_DOMAIN_ALIASES = (
    (("auth", "iam", "access"), 7.0, "iam_access_control"),
    (("crypto", "privacy", "data"), 7.0, "data_crypto_privacy"),
    (("transaction", "payment", "concurrency"), 7.5, "transaction_integrity"),
    (("network", "architecture", "infrastructure", "api"), 6.5, "architecture_network"),
)

_HIGH_IMPACT_CAPABILITIES = {
    "admin",
    "auth",
    "crypto",
    "transaction",
    "database",
    "secrets",
}

_EXPOSURE_CAPABILITIES = {
    "api",
    "browser",
    "mobile",
    "file_upload",
    "external_integration",
    "infrastructure",
}

_KEYWORD_RISK_TERMS = {
    "access control",
    "authentication",
    "authorization",
    "credential",
    "encryption",
    "injection",
    "mfa",
    "payment",
    "pii",
    "secret",
    "tenant",
    "token",
    "upload",
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
    requirement_text: Optional[str] = None,
    requirement_metadata: Optional[Dict[str, Any]] = None,
    analysis_trace: Optional[Dict[str, Any]] = None,
    citation_count: int = 0,
    evidence_metadata: Optional[Dict[str, Any]] = None,
) -> DeterministicSeverity:
    if met_status != "not_met":
        return DeterministicSeverity(severity=None, score=None, analysis=None)

    normalized_domain = (domain or "general").strip().lower() or "general"
    base_score, domain_source = _resolve_base_score(normalized_domain)
    requirement_metadata = requirement_metadata or {}
    analysis_trace = analysis_trace or {}
    evidence_metadata = evidence_metadata or _extract_evidence_metadata(analysis_trace)
    evidence_quality = evidence_metadata.get("evidence_quality") or {}
    implementation_evidence_count = _safe_int(
        evidence_quality.get("implementation_evidence_count"),
        default=0,
    )
    selected_evidence_count = _safe_int(evidence_quality.get("selected_count"), default=0)
    capabilities = _collect_capabilities(requirement_metadata, analysis_trace)
    impact_capabilities = sorted(capabilities & _HIGH_IMPACT_CAPABILITIES)
    exposure_capabilities = sorted(capabilities & _EXPOSURE_CAPABILITIES)
    keyword_signals = _collect_keyword_signals(requirement_text, analysis_trace)
    confidence = _clamp_confidence(confidence_score)
    confidence_delta = _confidence_modifier(confidence)
    impact_delta = min(len(impact_capabilities) * 0.5, 1.5)
    exposure_delta = min(len(exposure_capabilities) * 0.4, 0.8)
    keyword_delta = min(len(keyword_signals) * 0.2, 0.6)

    dimensions = {
        "domain_base": {
            "domain": normalized_domain,
            "source": domain_source,
            "score": base_score,
        },
        "impact_capabilities": {
            "matched": impact_capabilities,
            "delta": round(impact_delta, 1),
            "max_delta": 1.5,
        },
        "exposure_capabilities": {
            "matched": exposure_capabilities,
            "delta": round(exposure_delta, 1),
            "max_delta": 0.8,
        },
        "keyword_signals": {
            "matched": keyword_signals,
            "delta": round(keyword_delta, 1),
            "max_delta": 0.6,
        },
        "confidence_adjustment": {
            "confidence_score": confidence,
            "delta": confidence_delta,
        },
    }

    final_score = base_score + impact_delta + exposure_delta + keyword_delta + confidence_delta
    has_ambiguity = bool(ambiguous_elements)
    has_missing_information = bool(missing_information)
    caps: List[Dict[str, Any]] = []

    if _is_requirement_finding(finding_type) and raw_final_verdict == "partial":
        applied = final_score > 6.0
        caps.append({"kind": "partial_verdict_cap", "cap": 6.0, "applied": applied})
        final_score = min(final_score, 6.0)

    if _is_requirement_finding(finding_type) and citation_count <= 0:
        applied = final_score > 6.5
        caps.append({"kind": "zero_verified_citations_cap", "cap": 6.5, "applied": applied})
        final_score = min(final_score, 6.5)

    if _is_diagram_finding(finding_type) and (has_ambiguity or has_missing_information):
        applied = final_score > 6.5
        caps.append({"kind": "diagram_ambiguity_cap", "cap": 6.5, "applied": applied})
        final_score = min(final_score, 6.5)

    final_score = round(max(0.0, min(10.0, final_score)), 1)
    final_severity = _severity_bucket(final_score)
    analysis = {
        "source": "deterministic_risk_rubric",
        "version": RUBRIC_VERSION,
        "inputs": {
            "met_status": met_status,
            "confidence_score": confidence,
            "domain": normalized_domain,
            "finding_type": str(finding_type),
            "raw_final_verdict": raw_final_verdict,
            "citation_count": max(0, int(citation_count or 0)),
            "implementation_evidence_count": implementation_evidence_count,
            "selected_evidence_count": selected_evidence_count,
            "ambiguous_elements_present": has_ambiguity,
            "missing_information_present": has_missing_information,
        },
        "dimensions": dimensions,
        "caps": caps,
        "final_score": final_score,
        "final_severity": final_severity,
        "explanation": _build_explanation(
            final_severity=final_severity,
            final_score=final_score,
            domain=normalized_domain,
            impact_capabilities=impact_capabilities,
            exposure_capabilities=exposure_capabilities,
            keyword_signals=keyword_signals,
            caps=caps,
        ),
    }
    return DeterministicSeverity(
        severity=final_severity,
        score=final_score,
        analysis=analysis,
    )


def _resolve_base_score(domain: str) -> tuple[float, str]:
    if domain in _BASE_SCORES:
        return _BASE_SCORES[domain], domain
    for terms, score, source in _DOMAIN_ALIASES:
        if any(term in domain for term in terms):
            return score, source
    return _BASE_SCORES["general"], "general"


def _extract_evidence_metadata(analysis_trace: Dict[str, Any]) -> Dict[str, Any]:
    query_details = analysis_trace.get("retrieval_query_details") or {}
    metadata = query_details.get("retrieval_evidence_metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _collect_capabilities(
    requirement_metadata: Dict[str, Any],
    analysis_trace: Dict[str, Any],
) -> set[str]:
    capabilities: set[str] = set()
    for source in (
        requirement_metadata,
        requirement_metadata.get("analysis_trace") if isinstance(requirement_metadata.get("analysis_trace"), dict) else {},
        analysis_trace,
        analysis_trace.get("contract") if isinstance(analysis_trace.get("contract"), dict) else {},
        analysis_trace.get("retrieval_query_details") if isinstance(analysis_trace.get("retrieval_query_details"), dict) else {},
    ):
        if not isinstance(source, dict):
            continue
        for key in (
            "required_capabilities",
            "optional_capabilities",
            "matched_capabilities",
            "domain_keywords",
        ):
            capabilities.update(_normalized_strings(source.get(key)))
    return capabilities


def _collect_keyword_signals(
    requirement_text: Optional[str],
    analysis_trace: Dict[str, Any],
) -> List[str]:
    parts = [requirement_text or ""]
    for key in ("mediator_decision_basis", "hunter_claim", "critic_verification"):
        value = analysis_trace.get(key)
        if isinstance(value, dict):
            parts.append(" ".join(str(item) for item in value.values()))
    text = " ".join(parts).lower()
    return sorted(term for term in _KEYWORD_RISK_TERMS if re.search(rf"\b{re.escape(term)}\b", text))


def _normalized_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    for item in values:
        normalized = re.sub(r"\s+", "_", str(item or "").strip().lower())
        if normalized:
            yield normalized


def _is_requirement_finding(finding_type: str) -> bool:
    return finding_type == FindingType.REQUIREMENT or str(finding_type) == FindingType.REQUIREMENT.value


def _is_diagram_finding(finding_type: str) -> bool:
    return finding_type == FindingType.DIAGRAM or str(finding_type) == FindingType.DIAGRAM.value


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_explanation(
    *,
    final_severity: str,
    final_score: float,
    domain: str,
    impact_capabilities: List[str],
    exposure_capabilities: List[str],
    keyword_signals: List[str],
    caps: List[Dict[str, Any]],
) -> str:
    signal_parts = []
    if impact_capabilities:
        signal_parts.append(f"impact capabilities: {', '.join(impact_capabilities)}")
    if exposure_capabilities:
        signal_parts.append(f"exposure capabilities: {', '.join(exposure_capabilities)}")
    if keyword_signals:
        signal_parts.append(f"risk keywords: {', '.join(keyword_signals)}")
    if not signal_parts:
        signal_parts.append("no elevated impact or exposure signals")
    applied_caps = [str(cap["kind"]) for cap in caps if cap.get("applied")]
    cap_text = f" Applied caps: {', '.join(applied_caps)}." if applied_caps else ""
    return (
        f"Severity is {final_severity} ({final_score:.1f}) from domain '{domain}' with "
        f"{'; '.join(signal_parts)}.{cap_text}"
    )


def _clamp_confidence(value: Optional[float]) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def _confidence_modifier(confidence: float) -> float:
    if confidence >= 0.9:
        return 0.4
    if confidence >= 0.8:
        return 0.2
    if confidence >= 0.65:
        return 0.0
    if confidence >= 0.5:
        return -0.4
    return -0.8


def _severity_bucket(score: float) -> str:
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    return "low"
