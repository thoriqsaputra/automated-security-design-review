from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .candidate import RetrievalCandidate


_CLEARANCE_RANK = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}


@dataclass
class UserContext:
    user_id: str | None = None
    clearance: str = "internal"
    tenant_id: str | None = None


def filter_candidates_by_policy(
    candidates: List[RetrievalCandidate],
    user_context: UserContext,
) -> List[RetrievalCandidate]:
    allowed: List[RetrievalCandidate] = []
    user_rank = _CLEARANCE_RANK.get(user_context.clearance, _CLEARANCE_RANK["internal"])

    for candidate in candidates:
        metadata = candidate.metadata or {}
        candidate_tenant = metadata.get("tenant_id")
        sensitivity = metadata.get("sensitivity", "internal")
        candidate_rank = _CLEARANCE_RANK.get(sensitivity, _CLEARANCE_RANK["internal"])

        if user_context.tenant_id and candidate_tenant and user_context.tenant_id != candidate_tenant:
            continue
        if candidate_rank > user_rank:
            continue
        allowed.append(candidate)

    return allowed
