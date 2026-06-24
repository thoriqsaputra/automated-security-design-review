from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from sdr.core.config import settings

from sdr.apps.ai.retrieval.searchers.graph import GraphSearchResponse
from sdr.apps.ai.retrieval.searchers.raptor import RAPTORSearchResponse
from sdr.apps.ai.retrieval.searchers.vector import VectorSearchResponse


class RetrievalStrategy(Enum):
    VECTOR_ONLY = "vector_only"
    RAPTOR_LOW = "raptor_low"
    RAPTOR_HIGH = "raptor_high"
    GRAPH_TRAVERSE = "graph_traverse"
    GRAPH_LOCAL = "graph_local"
    HYBRID = "hybrid"


class QueryType(Enum):
    FACT_BASED = "fact_based"
    REASONING_BASED = "reasoning_based"
    GLOBAL_ARCHITECTURAL = "global_architectural"
    MULTI_HOP_SECURITY = "multi_hop_security"


@dataclass
class AdvancedRetrievalConfig:
    enable_cross_encoder_rerank: bool = False
    hybrid_max_workers: int = 3
    graph_local_max_workers: int = 3
    retrieve_many_max_concurrency: int = 2

    @classmethod
    def from_settings(cls) -> "AdvancedRetrievalConfig":
        return cls(
            enable_cross_encoder_rerank=bool(
                getattr(settings, "AI_RETRIEVAL_ENABLE_CROSS_ENCODER_RERANK", False)
            ),
            hybrid_max_workers=max(1, int(getattr(settings, "AI_RETRIEVAL_HYBRID_MAX_WORKERS", 3))),
            graph_local_max_workers=max(1, int(getattr(settings, "AI_RETRIEVAL_GRAPH_LOCAL_MAX_WORKERS", 3))),
            retrieve_many_max_concurrency=max(
                1,
                int(getattr(settings, "AI_RETRIEVAL_MANY_MAX_CONCURRENCY", 2)),
            ),
        )


@dataclass
class RetrievalResult:
    context_chunks: List[str] = field(default_factory=list)
    # Index-aligned with context_chunks: context_chunk_block_ids[i] is the list of
    # real document block ids backing context_chunks[i]. Empty list means that chunk
    # has no traceable source block (e.g. the VECTOR_ONLY full-text fallback) and must
    # stay non-citable.
    context_chunk_block_ids: List[List[str]] = field(default_factory=list)
    source_block_ids: List[str] = field(default_factory=list)
    block_source_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    diagram_block_ids: List[str] = field(default_factory=list)
    strategy_used: RetrievalStrategy = RetrievalStrategy.VECTOR_ONLY
    query_embedding: List[float] = field(default_factory=list)
    vector_response: Optional["VectorSearchResponse"] = None
    raptor_response: Optional["RAPTORSearchResponse"] = None
    graph_response: Optional["GraphSearchResponse"] = None
    graph_node_ids: List[str] = field(default_factory=list)
    graph_edge_ids: List[str] = field(default_factory=list)
    grounded_texts: List[Dict[str, Any]] = field(default_factory=list)
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
