from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from sdr.apps.ai.retrieval.searchers.graph import GraphSearchResponse
from sdr.apps.ai.retrieval.searchers.raptor import RAPTORSearchResponse
from sdr.apps.ai.retrieval.searchers.vector import VectorSearchResponse


class RetrievalStrategy(Enum):
    VECTOR_ONLY = "vector_only"
    RAPTOR_LOW = "raptor_low"
    RAPTOR_HIGH = "raptor_high"
    GRAPH_TRAVERSE = "graph_traverse"
    GRAPH_LOCAL = "graph_local"
    GRAPH_GLOBAL = "graph_global"
    IR_COT_GRAPH = "ir_cot_graph"
    HYBRID = "hybrid"


class QueryType(Enum):
    FACT_BASED = "fact_based"
    REASONING_BASED = "reasoning_based"
    GLOBAL_ARCHITECTURAL = "global_architectural"
    MULTI_HOP_SECURITY = "multi_hop_security"


@dataclass
class AdvancedRetrievalConfig:
    enable_graph_global: bool = False
    enable_ir_cot: bool = False
    ir_cot_max_iterations: int = 2
    enable_community_llm_summary: bool = False
    enable_cross_encoder_rerank: bool = False
    graph_global_support_blocks_per_community: int = 3


@dataclass
class RetrievalResult:
    context_chunks: List[str] = field(default_factory=list)
    source_block_ids: List[str] = field(default_factory=list)
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
