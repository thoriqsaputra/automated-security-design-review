from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class DomainClassification:
    primary_domain: str
    secondary_domains: List[str]
    reason: str
    matched_terms: Dict[str, List[str]]


DOMAIN_TERMS: Dict[str, List[str]] = {
    "business_logic_concurrency": [
        "concurrency",
        "locking",
        "optimistic locking",
        "pessimistic locking",
        "distributed lock",
        "race condition",
        "double-spending",
        "state transition",
        "retry safety",
    ],
    "transaction_integrity": [
        "atomic transaction",
        "transaction",
        "isolation level",
        "select for update",
        "idempotency",
        "idempotency key",
        "rollback",
    ],
    "iam_access_control": [
        "iam",
        "role",
        "permission",
        "rbac",
        "mfa",
        "auth",
        "authentication",
        "authorization",
        "least privilege",
    ],
    "architecture_network": [
        "network",
        "subnet",
        "firewall",
        "tls",
        "mtls",
        "ingress",
        "egress",
        "service-to-service",
    ],
    "data_crypto_privacy": [
        "encrypt",
        "encryption",
        "crypto",
        "key",
        "kms",
        "pii",
        "privacy",
        "tokenization",
        "at rest",
        "in transit",
    ],
}

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "business_logic_concurrency": [
        "concurrency",
        "atomic transaction",
        "transaction",
        "locking",
        "optimistic locking",
        "pessimistic locking",
        "distributed lock",
        "isolation level",
        "SELECT FOR UPDATE",
        "idempotency",
        "idempotency key",
        "race condition",
        "double-spending",
        "state transition",
        "retry safety",
        "rollback",
    ],
    "transaction_integrity": [
        "atomic transaction",
        "transaction",
        "idempotency",
        "idempotency key",
        "rollback",
        "isolation level",
        "select for update",
        "saga",
        "compensating transaction",
        "two-phase commit",
        "eventual consistency",
        "outbox pattern",
    ],
    "iam_access_control": ["iam", "auth", "rbac", "mfa", "authorization", "permission"],
    "architecture_network": ["network", "subnet", "tls", "firewall", "ingress", "egress"],
    "data_crypto_privacy": ["encryption", "privacy", "kms", "key", "tokenization"],
    "general": ["security", "control"],
}


def classify_requirement_domain(
    child_requirement: str,
    parent_title: str = "",
    parent_description: str = "",
    extra_parts: Sequence[str] | None = None,
) -> DomainClassification:
    child = (child_requirement or "").lower()
    parent = " ".join([parent_title or "", parent_description or ""]).lower()
    extra = " ".join([(p or "") for p in (extra_parts or [])]).lower()

    scores: Dict[str, float] = {domain: 0.0 for domain in DOMAIN_TERMS}
    matched_terms: Dict[str, List[str]] = {domain: [] for domain in DOMAIN_TERMS}

    for domain, terms in DOMAIN_TERMS.items():
        for term in terms:
            c = term in child
            p = term in parent
            e = term in extra
            if c:
                scores[domain] += 5.0
                matched_terms[domain].append(term)
            elif p:
                scores[domain] += 1.5
                matched_terms[domain].append(term)
            elif e:
                scores[domain] += 1.0
                matched_terms[domain].append(term)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_domain, top_score = ranked[0]
    if top_score <= 0:
        return DomainClassification(
            primary_domain="general",
            secondary_domains=[],
            reason="No domain-specific terms matched.",
            matched_terms={},
        )

    secondaries = [domain for domain, score in ranked[1:] if score > 0]
    reason = (
        f"Selected '{top_domain}' from weighted term matches "
        f"(child text prioritized over parent context)."
    )
    return DomainClassification(
        primary_domain=top_domain,
        secondary_domains=secondaries,
        reason=reason,
        matched_terms={k: v for k, v in matched_terms.items() if v},
    )
