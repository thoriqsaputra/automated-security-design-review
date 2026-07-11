from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate, _key_for_candidate, _merge_dupe

# Cormack et al. 2009 "Reciprocal Rank Fusion outperforms Condorcet and
# individual rank learning methods" — k=60 is the conventional default.
RRF_K_DEFAULT = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Iterable[RetrievalCandidate]],
    *,
    k: int = RRF_K_DEFAULT,
) -> List[RetrievalCandidate]:
    """Fuses N pre-ranked candidate lists via Reciprocal Rank Fusion.

    Each list is assumed to already be ordered best-first by its own searcher
    (dense RAPTOR leaf/multi-level, BM25, each query-expansion variant, etc).
    score(d) = sum over lists L containing d of 1/(k + rank_L(d) + 1), where
    rank_L(d) is d's 0-based position in L. A candidate found near the top of
    two lists outranks one found at the top of only one list, which is the
    point of RRF over a single-signal top-k cutoff.

    Sets each surviving candidate's .score to the fused value — this becomes
    the score used everywhere downstream (tier ordering, keyword-coverage
    boost, cross-encoder tie-breaking), same as the boosted score produced by
    dedupe_candidates today.
    """
    fused: Dict[str, RetrievalCandidate] = {}
    scores: Dict[str, float] = {}
    order: List[str] = []

    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list):
            key = _key_for_candidate(candidate)
            contribution = 1.0 / (k + rank + 1)

            if key not in fused:
                fused[key] = candidate
                scores[key] = contribution
                order.append(key)
                continue

            fused[key] = _merge_dupe(fused[key], candidate)
            scores[key] += contribution

    for key, candidate in fused.items():
        candidate.score = scores[key]
        candidate.metadata["rrf_score"] = scores[key]

    return sorted((fused[key] for key in order), key=lambda c: c.score, reverse=True)


__all__ = ["reciprocal_rank_fusion", "RRF_K_DEFAULT"]
