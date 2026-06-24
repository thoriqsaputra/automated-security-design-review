from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Set, Tuple

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate


_IMPLEMENTATION_TERMS = {
    "use",
    "uses",
    "using",
    "implemented",
    "configured",
    "enabled",
    "enforced",
    "validated",
    "verified",
    "required",
    "requires",
    "authenticate",
    "authenticated",
    "authorize",
    "authorized",
    "encrypt",
    "encrypted",
    "token",
    "oauth",
    "oidc",
    "jwt",
    "pkce",
    "jwks",
    "mfa",
    "rbac",
    # Logging / monitoring
    "log",
    "logging",
    "logged",
    "audit",
    "audited",
    "monitor",
    "monitored",
    "monitoring",
    "alert",
    "alerting",
    # Input validation
    "sanitiz",
    "validat",
    "escape",
    "escaped",
    "whitelist",
    "allowlist",
    "denylist",
    "blacklist",
    "schema",
    # Rate limiting / abuse prevention
    "rate limit",
    "rate-limit",
    "throttle",
    "throttled",
    "throttling",
    "quota",
    "backoff",
    "lockout",
    # Session management
    "session",
    "cookie",
    "timeout",
    "expir",
    # Error handling
    "exception",
    "fail-safe",
    "fail-secure",
    "error handling",
    # Access control (non-AuthN)
    "access control",
    "least privilege",
    "deny by default",
    "policy",
    # Data protection
    "mask",
    "masked",
    "redact",
    "redacted",
    "anonymiz",
    "pseudonymiz",
    "pii",
    # Network / infra isolation
    "firewall",
    "segmentation",
    "isolat",
    "sandbox",
}

_WEAK_CHUNK_PREFIXES = (
    "--- VECTOR RESULT",
    "--- GRAPH RESULT",
    "--- GRAPH PATH",
    "GRAPH NODE:",
)


class EvidenceGrader:
    def __init__(self, max_context_chunks: int) -> None:
        self.max_context_chunks = max_context_chunks

    def classify_candidate_evidence(
        self,
        candidate: RetrievalCandidate,
        *,
        keywords: List[str],
    ) -> Tuple[str, str]:
        text = (candidate.text or "").strip()
        lowered = text.lower()
        if not text:
            return "empty", "empty chunk"
        if candidate.metadata.get("non_tsd_evidence"):
            return "baseline_requirement", "standard baseline text is not TSD evidence"
        if text.startswith(_WEAK_CHUNK_PREFIXES) or lowered.startswith("graph node:"):
            return "graph_summary", "graph summary is structural context, not implementation evidence"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(text) < 120 and len(lines) <= 2:
            return "heading_only", "short heading-like chunk has no implementation detail"

        keyword_hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
        implementation_hits = sum(1 for term in _IMPLEMENTATION_TERMS if term in lowered)
        if candidate.block_ids and implementation_hits > 0 and len(text) >= 80:
            return "implementation_evidence", "TSD block contains implementation/security terms"
        if candidate.block_ids and keyword_hits > 0:
            return "weak_context", "TSD block is relevant but lacks clear implementation language"
        return "weak_context", "retrieved text lacks clear implementation evidence"

    def grade_and_filter_candidates(
        self,
        candidates: List[RetrievalCandidate],
        *,
        query_text: str,
        keywords: List[str],
    ) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
        graded: List[RetrievalCandidate] = []
        rejected: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        applicability_terms: Set[str] = set()
        query_terms = {term.lower() for term in keywords if len(term) >= 3}

        for candidate in candidates:
            kind, reason = self.classify_candidate_evidence(candidate, keywords=keywords)
            counts[kind] = counts.get(kind, 0) + 1
            metadata = dict(candidate.metadata or {})
            metadata["evidence_kind"] = kind
            metadata["evidence_reason"] = reason
            candidate.metadata = metadata

            lowered = (candidate.text or "").lower()
            for term in query_terms:
                if term in lowered:
                    applicability_terms.add(term)

            if kind in {"baseline_requirement", "empty"}:
                rejected.append({"id": candidate.id, "kind": kind, "reason": reason})
                continue
            graded.append(candidate)

        implementation = [c for c in graded if c.metadata.get("evidence_kind") == "implementation_evidence"]
        fallback = [c for c in graded if c.metadata.get("evidence_kind") != "implementation_evidence"]
        implementation.sort(key=lambda c: c.score, reverse=True)
        fallback.sort(key=lambda c: c.score, reverse=True)
        selected = implementation + fallback

        if not selected and candidates:
            selected = sorted(
                (c for c in candidates if c.metadata.get("evidence_kind") not in {"baseline_requirement", "empty"}),
                key=lambda c: c.score,
                reverse=True,
            )

        metadata = {
            "evidence_quality": {
                "counts": counts,
                "implementation_evidence_count": len(implementation),
                "selected_count": len(selected[: self.max_context_chunks]),
                "rejected": rejected[:20],
                "applicability_terms": sorted(applicability_terms),
                "applicability_signal": bool(applicability_terms),
                "query_hash": hashlib.sha256((query_text or "").encode("utf-8")).hexdigest()[:12],
            }
        }
        return selected[: self.max_context_chunks], metadata

    def apply_keyword_coverage_boost(
        self,
        candidates: List[RetrievalCandidate],
        keywords: List[str],
    ) -> List[RetrievalCandidate]:
        if not keywords:
            return candidates
        keyword_set = {k.lower() for k in keywords}
        boosted: List[RetrievalCandidate] = []
        for candidate in candidates:
            text_lower = (candidate.text or "").lower()
            coverage = sum(1 for kw in keyword_set if kw in text_lower)
            candidate.metadata["keyword_coverage"] = coverage
            candidate.score = float(candidate.score) + (0.05 * coverage)
            boosted.append(candidate)
        return boosted

    def generate_followup_query(self, original_query: str, candidates: List[RetrievalCandidate], keyword_extractor) -> str:
        words = keyword_extractor(original_query)
        seen = set(words)
        for candidate in candidates[:5]:
            for token in keyword_extractor(candidate.text)[:8]:
                if token not in seen:
                    words.append(token)
                    seen.add(token)
                if len(words) >= 16:
                    break
            if len(words) >= 16:
                break
        return " ".join(words[:16]) or original_query


__all__ = ["EvidenceGrader"]
