from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate, _key_for_candidate


def _candidate_snapshot(candidate: RetrievalCandidate) -> Dict[str, Any]:
    return {
        "key": _key_for_candidate(candidate),
        "id": candidate.id,
        "source_type": candidate.source_type,
        "score": float(candidate.score),
        "block_ids": list(candidate.block_ids or []),
        "level": int(candidate.metadata.get("level", 0) or 0),
        "evidence_kind": candidate.metadata.get("evidence_kind"),
        "token_count": int(candidate.token_count or 0),
    }


@dataclass
class HybridRetrievalTrace:
    """Diagnostic capture of every stage of execute_hybrid.

    Populated only when a trace object is explicitly passed to
    HybridRetrievalRouter.retrieve(trace=...) — the production path passes
    None and pays no cost. Used by the retrieval diagnosis eval script to
    classify, per expected gold block, exactly which pipeline stage lost it
    (search miss vs fusion vs evidence grading vs rerank vs truncation).
    """

    strategy: Optional[str] = None
    queries: List[str] = field(default_factory=list)
    query_embedding: List[float] = field(default_factory=list)
    fusion_method: Optional[str] = None
    max_context_chunks: int = 0
    # list_name ("bm25[0]", "dense[1]", "raptor_multi") -> ranked snapshots
    per_list_candidates: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    fused: List[Dict[str, Any]] = field(default_factory=list)
    graded: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    secondary_search_triggered: bool = False
    # Candidate keys in tier order, score-sorted within tiers (what the final
    # ranking would be with the cross-encoder disabled), before truncation.
    pre_rerank_order: List[str] = field(default_factory=list)
    # Candidate keys after within-tier cross-encoder rerank, before truncation.
    post_rerank_order: List[str] = field(default_factory=list)
    final: List[Dict[str, Any]] = field(default_factory=list)

    def record_list(self, name: str, candidates: Iterable[RetrievalCandidate]) -> None:
        self.per_list_candidates[name] = [_candidate_snapshot(c) for c in candidates]

    def record_fused(self, candidates: Iterable[RetrievalCandidate]) -> None:
        self.fused = [_candidate_snapshot(c) for c in candidates]

    def record_graded(self, candidates: Iterable[RetrievalCandidate]) -> None:
        self.graded = [_candidate_snapshot(c) for c in candidates]

    def record_final(self, candidates: Iterable[RetrievalCandidate]) -> None:
        self.final = [_candidate_snapshot(c) for c in candidates]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "queries": list(self.queries),
            "fusion_method": self.fusion_method,
            "max_context_chunks": self.max_context_chunks,
            "per_list_candidates": self.per_list_candidates,
            "fused": self.fused,
            "graded": self.graded,
            "rejected": self.rejected,
            "secondary_search_triggered": self.secondary_search_triggered,
            "pre_rerank_order": list(self.pre_rerank_order),
            "post_rerank_order": list(self.post_rerank_order),
            "final": self.final,
        }


__all__ = ["HybridRetrievalTrace"]
