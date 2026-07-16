from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from sdr.core.config import settings

from sdr.apps.ai.retrieval.searchers.raptor import RAPTORSearchResponse


class RetrievalStrategy(Enum):
    RAPTOR_LOW = "raptor_low"
    RAPTOR_HIGH = "raptor_high"
    HYBRID = "hybrid"
    # Eval-only baseline (via force_strategy) — vanilla top-k cosine search
    # over leaf nodes, no threshold gate, no hierarchy, no BM25/expansion.
    FLAT_TOPK = "flat_topk"


class QueryType(Enum):
    FACT_BASED = "fact_based"
    REASONING_BASED = "reasoning_based"
    GLOBAL_ARCHITECTURAL = "global_architectural"
    MULTI_HOP_SECURITY = "multi_hop_security"


@dataclass
class AdvancedRetrievalConfig:
    enable_cross_encoder_rerank: bool = True
    hybrid_max_workers: int = 3
    retrieve_many_max_concurrency: int = 2
    # "agreement_boost" (default) keeps today's dedupe + flat per-source score
    # bump. "rrf" fuses per-searcher ranked lists via Reciprocal Rank Fusion
    # instead, as the primary merge step in execute_hybrid.
    fusion_method: Literal["agreement_boost", "rrf"] = "agreement_boost"
    rrf_k: int = 60
    # Recall-safety floor: guaranteed final-context slots for each constituent
    # signal's primary-query top-N (see AI_RETRIEVAL_PROTECTED_* settings).
    # dense floor = raptor_low's top-k; raptor floor rescues raptor_high-only
    # wins found (but under-protected) in hybrid's own multi-level branch,
    # sized to stay well under max_context_chunks alongside the other floors.
    protected_dense_top_n: int = 7
    protected_bm25_top_n: int = 2
    protected_raptor_top_n: int = 7
    # Query-expansion variant branches (bm25[1..n], dense[1..n]) are
    # supplementary in the main protected floors above, so a vocabulary-gap
    # chunk they alone surface can still be lost to truncation/reranking
    # against the abstract primary query. This gives each variant branch its
    # own (smaller) protected floor.
    protected_variant_top_n: int = 5
    rerank_with_variants: bool = True
    summary_leaves_per_grounding: int = 1
    hybrid_dense_top_k: int = 20
    hybrid_bm25_top_k: int = 20
    rerank_score_weight: float = 0.72

    @classmethod
    def from_settings(cls) -> "AdvancedRetrievalConfig":
        return cls(
            enable_cross_encoder_rerank=bool(
                getattr(settings, "AI_RETRIEVAL_ENABLE_CROSS_ENCODER_RERANK", True)
            ),
            hybrid_max_workers=max(1, int(getattr(settings, "AI_RETRIEVAL_HYBRID_MAX_WORKERS", 3))),
            retrieve_many_max_concurrency=max(
                1,
                int(getattr(settings, "AI_RETRIEVAL_MANY_MAX_CONCURRENCY", 2)),
            ),
            fusion_method=getattr(settings, "AI_RETRIEVAL_FUSION_METHOD", "agreement_boost"),
            rrf_k=int(getattr(settings, "AI_RETRIEVAL_RRF_K", 60)),
            protected_dense_top_n=max(0, int(getattr(settings, "AI_RETRIEVAL_PROTECTED_DENSE_TOP_N", 3))),
            protected_bm25_top_n=max(0, int(getattr(settings, "AI_RETRIEVAL_PROTECTED_BM25_TOP_N", 2))),
            protected_raptor_top_n=max(0, int(getattr(settings, "AI_RETRIEVAL_PROTECTED_RAPTOR_TOP_N", 7))),
            protected_variant_top_n=max(0, int(getattr(settings, "AI_RETRIEVAL_PROTECTED_VARIANT_TOP_N", 5))),
            rerank_with_variants=bool(getattr(settings, "AI_RETRIEVAL_RERANK_WITH_VARIANTS", True)),
            summary_leaves_per_grounding=max(
                1, int(getattr(settings, "AI_RETRIEVAL_SUMMARY_LEAVES_PER_GROUNDING", 1))
            ),
            hybrid_dense_top_k=max(1, int(getattr(settings, "AI_RETRIEVAL_HYBRID_DENSE_TOP_K", 20))),
            hybrid_bm25_top_k=max(1, int(getattr(settings, "AI_RETRIEVAL_HYBRID_BM25_TOP_K", 20))),
            rerank_score_weight=min(
                1.0,
                max(0.0, float(getattr(settings, "AI_RETRIEVAL_RERANK_SCORE_WEIGHT", 0.72))),
            ),
        )


@dataclass
class RetrievalResult:
    context_chunks: List[str] = field(default_factory=list)
    context_chunk_block_ids: List[List[str]] = field(default_factory=list)
    context_chunk_levels: List[int] = field(default_factory=list)
    source_block_ids: List[str] = field(default_factory=list)
    block_source_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    diagram_block_ids: List[str] = field(default_factory=list)
    strategy_used: RetrievalStrategy = RetrievalStrategy.HYBRID
    query_embedding: List[float] = field(default_factory=list)
    raptor_response: Optional["RAPTORSearchResponse"] = None
    evidence_metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return len(self.context_chunks) == 0

    @property
    def total_chunks(self) -> int:
        return len(self.context_chunks)

    def get_diagram_block_ids(self) -> List[str]:
        if self.diagram_block_ids:
            return list(self.diagram_block_ids)
        return [bid for bid in self.source_block_ids if "_d" in bid]


__all__ = [
    "AdvancedRetrievalConfig",
    "QueryType",
    "RetrievalResult",
    "RetrievalStrategy",
]
