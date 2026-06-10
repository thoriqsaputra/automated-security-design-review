"""
Hybrid Retrieval Router — orchestrates all three retrieval strategies
(Vector, RAPTOR, Graph) into a single unified context retrieval pipeline.

Responsibility:
    For each security parameter (CategoryParameterChild [3]), the router
    decides which retrieval strategy or combination of strategies to use,
    executes them, merges the results, and returns a unified RetrievalResult
    containing the context chunks and source block_ids needed by the
    Hunter agent (agents/hunter.py).

Strategy selection:
    VECTOR_ONLY     — simple factual parameters ("Is TLS 1.2+ enforced?")
                      Fast path — single pgvector cosine distance query.
    RAPTOR_LOW      — parameters needing precise section-level evidence
                      Uses RAPTOR leaf nodes (level 0).
    RAPTOR_HIGH     — cross-cutting parameters spanning multiple sections
                      Uses RAPTOR multi-level search (levels 0, 1, 2).
    GRAPH_TRAVERSE  — relationship parameters ("Is auth enforced on all paths?")
                      Uses GraphSearcher with auto-selected sub-strategy.
    HYBRID          — complex parameters needing all three strategies
                      Merges and re-ranks results from all three searchers.

Embedding reuse:
    The router generates ONE embedding per parameter query and passes it
    as precomputed_embedding to all three searchers — this avoids N
    redundant Bedrock API calls [4] for the same text.

Dependency chain:
    retrieval/vector_search.py   (VectorSearcher, VectorSearchResponse)
    retrieval/raptor_search.py   (RAPTORSearcher, RAPTORSearchResponse)
    retrieval/graph_search.py    (GraphSearcher, GraphSearchResponse)
         ↓
    retrieval/router.py          ← YOU ARE HERE
         ↓
    analysis_service.py
"""

from __future__ import annotations

import logging
import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

def _safe_execute(func, *args, **kwargs):
    return func(*args, **kwargs)


from sdr.apps.ai.client import get_embedding
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree
from sdr.apps.ai.tsd_processing.graph_builder import TSDGraph, _normalise_entity_id
from sdr.apps.standards.models import (
    CategoryParameterChild,
    StandardCategory,
    StandardIngestionJob,
)
from sdr.apps.standards.utils import build_parameter_analysis_text
from .graph_search import (
    GraphSearcher,
    GraphSearchResponse,
    GraphSearchStrategy,
    GraphTraversalConfig,
    _extract_keywords,
    _infer_relation_types_from_parameter,
)
from .raptor_search import (
    RAPTORSearcher,
    RAPTORSearchResponse,
    RAPTOR_LEVEL_LOW,
    RAPTOR_LEVEL_MID,
    RAPTOR_LEVEL_HIGH,
)
from .vector_search import (
    VectorSearcher,
    VectorSearchResponse,
)
from .candidate import RetrievalCandidate, dedupe_candidates, merge_candidates
from .keyword_search import KeywordSearcher
from .policy import UserContext, filter_candidates_by_policy
from .reranker import SafeOptionalReranker
from .graph_communities import GraphCommunityService, CommunitySummary

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except Exception:
    nx = None


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class RetrievalStrategy(Enum):
    """
    The retrieval strategy selected by the router for a given parameter.
    Stored in RetrievalResult for audit and analysis_service.py logging.
    """
    VECTOR_ONLY    = "vector_only"
    RAPTOR_LOW     = "raptor_low"
    RAPTOR_HIGH    = "raptor_high"
    GRAPH_TRAVERSE = "graph_traverse"
    GRAPH_LOCAL    = "graph_local"
    GRAPH_GLOBAL   = "graph_global"
    IR_COT_GRAPH   = "ir_cot_graph"
    HYBRID         = "hybrid"


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Top-K values per strategy — balanced for context window budget
_VECTOR_TOP_K    = 8
_RAPTOR_TOP_K    = 5
_RAPTOR_PER_LEVEL = 3
_GRAPH_TOP_K     = 6

# Minimum number of graph relations inferred before triggering GRAPH or HYBRID
_GRAPH_TRIGGER_RELATION_COUNT = 2

# Minimum number of keywords before triggering RAPTOR_HIGH or HYBRID
_HYBRID_TRIGGER_KEYWORD_COUNT = 4

# Confidence threshold for HYBRID re-ranking —
# vector results below this similarity are deprioritised in the merged set
_VECTOR_DERANK_THRESHOLD = 0.65

# Maximum total context chunks returned to the Hunter agent —
# prevents context window overflow when HYBRID merges many results
_MAX_CONTEXT_CHUNKS = 12

# Embedding dimensions — consistent across the entire pipeline [5]
_EMBEDDING_DIMENSIONS = 1024

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
}

_WEAK_CHUNK_PREFIXES = (
    "--- VECTOR RESULT",
    "--- GRAPH RESULT",
    "--- GRAPH PATH",
    "GRAPH NODE:",
)


# ---------------------------------------------------------------------------
# Retrieval result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    The unified output of the HybridRetrievalRouter for a single parameter.

    Consumed directly by analysis_service.py which passes:
        - context_chunks → HunterAgent.run(context_chunks=...)
        - diagram_blocks  → VisionAgent.run(diagram=...)
        - source_block_ids → TSD-backed citation resolution only

    strategy_used and per-strategy responses are stored for audit logging
    and for the Mediator's citation reconciliation step.
    """
    # The merged, de-duplicated context chunks for the Hunter agent
    context_chunks: List[str] = field(default_factory=list)
    # All source block_ids covered by the retrieved context
    source_block_ids: List[str] = field(default_factory=list)
    # Explicit diagram block IDs selected by retrieval/inference.
    diagram_block_ids: List[str] = field(default_factory=list)
    # The strategy the router selected
    strategy_used: RetrievalStrategy = RetrievalStrategy.VECTOR_ONLY
    # The shared query embedding — reused by analysis_service.py
    query_embedding: List[float] = field(default_factory=list)
    # Per-strategy responses for audit — None if strategy was not used
    vector_response: Optional[VectorSearchResponse] = None
    raptor_response: Optional[RAPTORSearchResponse] = None
    graph_response: Optional[GraphSearchResponse] = None
    graph_node_ids: List[str] = field(default_factory=list)
    graph_edge_ids: List[str] = field(default_factory=list)
    grounded_texts: List[Dict[str, Any]] = field(default_factory=list)
    evidence_metadata: Dict[str, Any] = field(default_factory=dict)
    # Error message if retrieval failed — context_chunks may be empty
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return len(self.context_chunks) == 0

    @property
    def total_chunks(self) -> int:
        return len(self.context_chunks)

    def get_diagram_block_ids(self) -> List[str]:
        """
        Returns block_ids that correspond to diagram blocks —
        identified by the "p{page}_d{idx}" format used by the TSD ingestor.
        Used by analysis_service.py to select diagrams for VisionAgent.
        """
        if self.diagram_block_ids:
            return list(self.diagram_block_ids)
        return [bid for bid in self.source_block_ids if "_d" in bid]


# ---------------------------------------------------------------------------
# Hybrid Retrieval Router
# ---------------------------------------------------------------------------

class HybridRetrievalRouter:
    """
    Orchestrates Vector, RAPTOR, and Graph retrieval strategies into a
    single unified context retrieval pipeline for the Multi-Agent debate.

    The router is the single entry point for context retrieval in
    analysis_service.py. It is called once per security parameter per
    TSD analysis run.

    Key design principles:
        1. ONE embedding per parameter — generated once and passed to all
           three searchers as precomputed_embedding to avoid redundant
           Bedrock API calls [4].
        2. Strategy is auto-selected from parameter keyword analysis —
           no manual configuration required per parameter.
        3. HYBRID merges all three result sets and re-ranks by a composite
           score — vector similarity weighted higher for factual params,
           graph relevance weighted higher for relationship params.
        4. Context chunks are always capped at _MAX_CONTEXT_CHUNKS to
           prevent the Hunter agent's context window from overflowing.

    Usage:
        router = HybridRetrievalRouter()
        result = router.retrieve(
            parameter=child_parameter,
            category=category,
            raptor_tree=raptor_tree,
            graph=tsd_graph,
        )
        # Pass result.context_chunks to HunterAgent.run()
    """

    def __init__(
        self,
        vector_top_k: int = _VECTOR_TOP_K,
        raptor_top_k: int = _RAPTOR_TOP_K,
        graph_top_k: int = _GRAPH_TOP_K,
        max_context_chunks: int = _MAX_CONTEXT_CHUNKS,
        advanced_config: Optional[AdvancedRetrievalConfig] = None,
    ) -> None:
        self.vector_top_k = vector_top_k
        self.raptor_top_k = raptor_top_k
        self.graph_top_k = graph_top_k
        self.max_context_chunks = max_context_chunks
        self.advanced_config = advanced_config or AdvancedRetrievalConfig()

        self._vector_searcher = VectorSearcher()
        self._raptor_searcher = RAPTORSearcher()
        self._graph_searcher = GraphSearcher()
        self._keyword_searcher = KeywordSearcher()
        self._reranker = SafeOptionalReranker(
            enable_cross_encoder=self.advanced_config.enable_cross_encoder_rerank,
        )
        self._community_service = GraphCommunityService(
            enable_llm_summary=self.advanced_config.enable_community_llm_summary,
        )

        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        parameter: CategoryParameterChild,
        category: StandardCategory,
        raptor_tree: Optional[RAPTORTree] = None,
        graph: Optional[TSDGraph] = None,
        ingestion_job: Optional[StandardIngestionJob] = None,
        force_strategy: Optional[RetrievalStrategy] = None,
        override_query_text: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Primary entry point — retrieves unified context for a single
        security parameter using the most appropriate strategy.

        Pipeline:
            1. Validate inputs.
            2. Generate one shared query embedding.
            3. Select retrieval strategy from parameter analysis.
            4. Execute the selected strategy.
            5. Cap context chunks at _MAX_CONTEXT_CHUNKS.
            6. Return populated RetrievalResult.

        Args:
            parameter:      The CategoryParameterChild to retrieve context for [3].
            category:       The StandardCategory to scope vector search to [3].
            raptor_tree:    The RAPTORTree built from the TSD document.
                            Required for RAPTOR_LOW, RAPTOR_HIGH, HYBRID.
            graph:          The TSDGraph built from the TSD document.
                            Required for GRAPH_TRAVERSE, HYBRID.
            ingestion_job:  Optional specific job to scope vector search to.
            force_strategy: Optional — override automatic strategy selection.
                            Used by analysis_service.py for specific parameter
                            types (e.g. always use GRAPH_TRAVERSE for
                            inter-service parameters).

        Returns:
            RetrievalResult — never raises. Check .error for failures.
        """
        query_text = (override_query_text or "").strip() or build_parameter_analysis_text(parameter).strip()

        if not query_text:
            msg = (
                f"Parameter id={parameter.id} has empty requirement text — "
                "cannot retrieve context."
            )
            self.logger.warning("HybridRetrievalRouter.retrieve: %s", msg)
            return RetrievalResult(error=msg)

        # ------------------------------------------------------------------
        # Step 1: Generate shared query embedding
        # ------------------------------------------------------------------
        query_embedding = self._generate_query_embedding(query_text)

        if not query_embedding:
            msg = (
                f"Failed to generate embedding for parameter "
                f"id={parameter.id} — falling back to text-only retrieval."
            )
            self.logger.warning("HybridRetrievalRouter.retrieve: %s", msg)
            # Continue without embedding — RAPTOR text search still works
            # but vector search and graph embedding scoring will be degraded

        # ------------------------------------------------------------------
        # Step 2: Analyse parameter to select strategy
        # ------------------------------------------------------------------
        keywords = _extract_keywords(query_text)
        inferred_relations = _infer_relation_types_from_parameter(keywords)
        query_entities = self._extract_query_entities(query_text)
        query_type = self._classify_query_type(query_text, keywords, inferred_relations)

        strategy = force_strategy or self._select_strategy(
            query_text=query_text,
            keywords=keywords,
            inferred_relations=inferred_relations,
            query_entities=query_entities,
            query_type=query_type,
            raptor_tree=raptor_tree,
            graph=graph,
        )

        self.logger.info(
            "HybridRetrievalRouter.retrieve: strategy=%s for parameter "
            "id=%s '%s...'",
            strategy.value,
            parameter.id,
            query_text[:60],
        )

        # ------------------------------------------------------------------
        # Step 3: Execute strategy
        # ------------------------------------------------------------------
        try:
            if strategy == RetrievalStrategy.VECTOR_ONLY:
                return self._execute_vector_only(
                    query_text=query_text,
                    category=category,
                    ingestion_job=ingestion_job,
                    query_embedding=query_embedding,
                )

            elif strategy == RetrievalStrategy.RAPTOR_LOW:
                return self._execute_raptor_low(
                    query_text=query_text,
                    raptor_tree=raptor_tree,
                    query_embedding=query_embedding,
                )

            elif strategy == RetrievalStrategy.RAPTOR_HIGH:
                return self._execute_raptor_high(
                    query_text=query_text,
                    raptor_tree=raptor_tree,
                    query_embedding=query_embedding,
                )

            elif strategy == RetrievalStrategy.GRAPH_TRAVERSE:
                return self._execute_graph_traverse(
                    query_text=query_text,
                    graph=graph,
                    raptor_tree=raptor_tree,
                    query_embedding=query_embedding,
                )

            elif strategy == RetrievalStrategy.GRAPH_LOCAL:
                return self._execute_graph_local(
                    query_text=query_text,
                    category=category,
                    ingestion_job=ingestion_job,
                    raptor_tree=raptor_tree,
                    graph=graph,
                    query_embedding=query_embedding,
                    keywords=keywords,
                    query_entities=query_entities,
                )

            elif strategy == RetrievalStrategy.GRAPH_GLOBAL:
                return self._execute_graph_global(
                    query_text=query_text,
                    category=category,
                    ingestion_job=ingestion_job,
                    raptor_tree=raptor_tree,
                    graph=graph,
                    query_embedding=query_embedding,
                    keywords=keywords,
                    query_entities=query_entities,
                )

            elif strategy == RetrievalStrategy.IR_COT_GRAPH:
                return self._execute_ir_cot_graph(
                    query_text=query_text,
                    category=category,
                    ingestion_job=ingestion_job,
                    raptor_tree=raptor_tree,
                    graph=graph,
                    query_embedding=query_embedding,
                    keywords=keywords,
                    query_entities=query_entities,
                )

            elif strategy == RetrievalStrategy.HYBRID:
                return self._execute_hybrid(
                    query_text=query_text,
                    category=category,
                    ingestion_job=ingestion_job,
                    raptor_tree=raptor_tree,
                    graph=graph,
                    query_embedding=query_embedding,
                    keywords=keywords,
                    inferred_relations=inferred_relations,
                )

            else:
                # Unknown strategy — safe fallback to VECTOR_ONLY
                self.logger.error(
                    "HybridRetrievalRouter.retrieve: unknown strategy '%s' "
                    "— falling back to VECTOR_ONLY.",
                    strategy,
                )
                return self._execute_vector_only(
                    query_text=query_text,
                    category=category,
                    ingestion_job=ingestion_job,
                    query_embedding=query_embedding,
                )

        except Exception as exc:
            msg = f"Strategy execution failed for strategy={strategy.value}: {exc}"
            self.logger.exception(
                "HybridRetrievalRouter.retrieve: %s for parameter id=%s",
                msg,
                parameter.id,
            )
            return RetrievalResult(
                error=msg,
                strategy_used=strategy,
                query_embedding=query_embedding,
            )
        finally:
            pass

    # ------------------------------------------------------------------
    # Strategy execution methods
    # ------------------------------------------------------------------

    def _execute_vector_only(
        self,
        query_text: str,
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        query_embedding: List[float],
    ) -> RetrievalResult:
        """
        VECTOR_ONLY — single pgvector cosine distance query.
        Fastest path — used for simple factual parameters.
        """
        vector_response = self._vector_searcher.search(
            query_text=query_text,
            category=category,
            top_k=self.vector_top_k,
            ingestion_job=ingestion_job,
            precomputed_embedding=query_embedding or None,
        )

        if vector_response.error:
            self.logger.warning(
                "HybridRetrievalRouter._execute_vector_only: "
                "vector search error: %s",
                vector_response.error,
            )

        context_chunks = self._build_chunks_from_vector(vector_response)
        source_block_ids = self._collect_block_ids_from_vector(vector_response)

        return RetrievalResult(
            context_chunks=context_chunks[:self.max_context_chunks],
            source_block_ids=source_block_ids,
            strategy_used=RetrievalStrategy.VECTOR_ONLY,
            query_embedding=query_embedding,
            vector_response=vector_response,
            raptor_response=None,
            graph_response=None,
            error=vector_response.error,
        )

    def _normalize_embedding_diagnostics(
        self,
        result: RetrievalResult,
        graph: Optional[TSDGraph],
    ) -> Dict[str, Any]:
        metadata = dict(getattr(result, "evidence_metadata", {}) or {})
        graph_stats = metadata.get("graph_embedding_stats")
        if not isinstance(graph_stats, dict):
            graph_stats = getattr(graph, "embedding_stats", {}) or {}

        normalized = {
            "graph_embedding_rerank_applied": bool(metadata.get("graph_embedding_rerank_applied", False)),
            "graph_result_count": int(metadata.get("graph_result_count", len(getattr(getattr(result, "graph_response", None), "results", []) or []))),
            "graph_embedding_stats": {
                "entity_attempted": int(graph_stats.get("entity_attempted", 0)),
                "entity_succeeded": int(graph_stats.get("entity_succeeded", 0)),
                "entity_failed": int(graph_stats.get("entity_failed", 0)),
                "relation_attempted": int(graph_stats.get("relation_attempted", 0)),
                "relation_succeeded": int(graph_stats.get("relation_succeeded", 0)),
                "relation_failed": int(graph_stats.get("relation_failed", 0)),
            },
        }
        metadata.update(normalized)
        result.evidence_metadata = metadata
        return normalized

    def _execute_raptor_low(
        self,
        query_text: str,
        raptor_tree: Optional[RAPTORTree],
        query_embedding: List[float],
    ) -> RetrievalResult:
        """
        RAPTOR_LOW — searches Level-0 leaf nodes of the RAPTOR tree.
        Used when the parameter needs precise section-level evidence
        with exact citation block_ids.
        """
        if raptor_tree is None or raptor_tree.is_empty():
            self.logger.warning(
                "HybridRetrievalRouter._execute_raptor_low: "
                "RAPTORTree is None or empty — returning empty result."
            )
            return RetrievalResult(
                strategy_used=RetrievalStrategy.RAPTOR_LOW,
                query_embedding=query_embedding,
                error="RAPTORTree is not available.",
            )

        raptor_response = self._raptor_searcher.search(
            query_text=query_text,
            tree=raptor_tree,
            level=RAPTOR_LEVEL_LOW,
            top_k=self.raptor_top_k,
            precomputed_embedding=query_embedding or None,
        )

        if raptor_response.error:
            self.logger.warning(
                "HybridRetrievalRouter._execute_raptor_low: "
                "RAPTOR search error: %s",
                raptor_response.error,
            )

        context_chunks = raptor_response.get_context_chunks()
        source_block_ids = raptor_response.all_source_block_ids

        return RetrievalResult(
            context_chunks=context_chunks[:self.max_context_chunks],
            source_block_ids=source_block_ids,
            strategy_used=RetrievalStrategy.RAPTOR_LOW,
            query_embedding=query_embedding,
            vector_response=None,
            raptor_response=raptor_response,
            graph_response=None,
            error=raptor_response.error,
        )

    def _execute_raptor_high(
        self,
        query_text: str,
        raptor_tree: Optional[RAPTORTree],
        query_embedding: List[float],
    ) -> RetrievalResult:
        """
        RAPTOR_HIGH — searches across all levels (0, 1, 2) of the RAPTOR tree.
        Used for cross-cutting parameters that need context from multiple
        abstraction levels simultaneously.
        """
        if raptor_tree is None or raptor_tree.is_empty():
            self.logger.warning(
                "HybridRetrievalRouter._execute_raptor_high: "
                "RAPTORTree is None or empty — returning empty result."
            )
            return RetrievalResult(
                strategy_used=RetrievalStrategy.RAPTOR_HIGH,
                query_embedding=query_embedding,
                error="RAPTORTree is not available.",
            )

        raptor_response = self._raptor_searcher.search_multi_level(
            query_text=query_text,
            tree=raptor_tree,
            levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
            top_k_per_level=_RAPTOR_PER_LEVEL,
            precomputed_embedding=query_embedding or None,
        )

        if raptor_response.error:
            self.logger.warning(
                "HybridRetrievalRouter._execute_raptor_high: "
                "RAPTOR multi-level search error: %s",
                raptor_response.error,
            )

        context_chunks = raptor_response.get_context_chunks()
        source_block_ids = raptor_response.all_source_block_ids

        return RetrievalResult(
            context_chunks=context_chunks[:self.max_context_chunks],
            source_block_ids=source_block_ids,
            strategy_used=RetrievalStrategy.RAPTOR_HIGH,
            query_embedding=query_embedding,
            vector_response=None,
            raptor_response=raptor_response,
            graph_response=None,
            error=raptor_response.error,
        )

    def _execute_graph_traverse(
        self,
        query_text: str,
        graph: Optional[TSDGraph],
        raptor_tree: Optional[RAPTORTree],
        query_embedding: List[float],
    ) -> RetrievalResult:
        """
        GRAPH_TRAVERSE — traverses the TSDGraph to find relationship evidence.
        Enriches with RAPTOR_LOW context for the matched entities'
        source_block_ids so the Hunter has both structural evidence
        (graph) and textual evidence (RAPTOR) in its context window.
        """
        if graph is None or graph.is_empty():
            self.logger.warning(
                "HybridRetrievalRouter._execute_graph_traverse: "
                "TSDGraph is None or empty — falling back to RAPTOR_LOW."
            )
            return self._execute_raptor_low(
                query_text=query_text,
                raptor_tree=raptor_tree,
                query_embedding=query_embedding,
            )

        graph_response = self._graph_searcher.search(
            parameter_text=query_text,
            graph=graph,
            query_embedding=query_embedding or None,
            top_k=self.graph_top_k,
        )

        if graph_response.error:
            self.logger.warning(
                "HybridRetrievalRouter._execute_graph_traverse: "
                "graph search error: %s — falling back to RAPTOR_LOW.",
                graph_response.error,
            )
            return self._execute_raptor_low(
                query_text=query_text,
                raptor_tree=raptor_tree,
                query_embedding=query_embedding,
            )

        # Build context chunks from graph results — entity descriptions
        # and relation summaries form the textual context
        context_chunks = self._build_chunks_from_graph(graph_response)
        source_block_ids = graph_response.all_source_block_ids

        # Enrich with RAPTOR_LOW chunks for the same block_ids
        # so the Hunter has raw TSD text alongside graph structure
        if raptor_tree and not raptor_tree.is_empty() and source_block_ids:
            raptor_response = self._raptor_searcher.search(
                query_text=query_text,
                tree=raptor_tree,
                level=RAPTOR_LEVEL_LOW,
                top_k=3,   # conservative — graph is the primary source
                precomputed_embedding=query_embedding or None,
            )
            if not raptor_response.is_empty:
                # Append RAPTOR chunks after graph chunks
                context_chunks.extend(raptor_response.get_context_chunks())
                for bid in raptor_response.all_source_block_ids:
                    if bid not in source_block_ids:
                        source_block_ids.append(bid)
        else:
            raptor_response = None

        return RetrievalResult(
            context_chunks=context_chunks[:self.max_context_chunks],
            source_block_ids=source_block_ids,
            strategy_used=RetrievalStrategy.GRAPH_TRAVERSE,
            query_embedding=query_embedding,
            vector_response=None,
            raptor_response=raptor_response if raptor_tree else None,
            graph_response=graph_response,
            evidence_metadata={
                "graph_embedding_rerank_applied": bool(query_embedding),
                "graph_embedding_stats": getattr(graph, "embedding_stats", {}),
                "graph_result_count": len(graph_response.results),
            },
            error=None,
        )

    def _execute_hybrid(
        self,
        query_text: str,
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        raptor_tree: Optional[RAPTORTree],
        graph: Optional[TSDGraph],
        query_embedding: List[float],
        keywords: List[str],
        inferred_relations: set,
    ) -> RetrievalResult:
        """
        HYBRID — executes all three strategies and merges their results.

        Merge order (highest priority first):
            1. Graph results — relationship evidence (most unique signal)
            2. RAPTOR results — multi-level textual evidence
            3. Vector results — semantic similarity (broadest coverage)

        Vector results below _VECTOR_DERANK_THRESHOLD are appended last
        so high-confidence graph and RAPTOR evidence always leads the
        context window. This directly combats "lost in the middle" syndrome
        by placing the strongest evidence at the top of the context.
        """
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ThreadPoolExecutor-3") as executor:
            fut_vec = executor.submit(
                _safe_execute, self._vector_searcher.search,
                query_text=query_text,
                category=category,
                top_k=self.vector_top_k,
                ingestion_job=ingestion_job,
                precomputed_embedding=query_embedding or None
            )

            if raptor_tree and not raptor_tree.is_empty():
                fut_rap = executor.submit(
                    _safe_execute, self._raptor_searcher.search_collapsed_raptor,
                    query_text=query_text,
                    tree=raptor_tree,
                    top_k=max(self.raptor_top_k, _RAPTOR_PER_LEVEL * 2),
                    max_tokens=4000,
                    allowed_levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
                    precomputed_embedding=query_embedding or None
                )
            else:
                fut_rap = None

            fut_bm25 = executor.submit(
                _safe_execute, self._keyword_searcher.search,
                query_text=query_text,
                tree=raptor_tree,
                top_k=max(self.vector_top_k, self.raptor_top_k, 20),
                allowed_levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH]
            )

            vector_response = fut_vec.result()
            raptor_response = fut_rap.result() if fut_rap else None
            bm25_candidates = fut_bm25.result()

        dense_candidates: List[RetrievalCandidate] = []
        for idx, result in enumerate(vector_response.results):
            text = result.child.requirement_text or ""
            if not text:
                continue
            dense_candidates.append(
                RetrievalCandidate(
                    id=f"dense:{getattr(result.child, 'stable_key', idx)}",
                    source_type="dense",
                    text=text,
                    score=float(result.cosine_similarity),
                    block_ids=[],
                    metadata={
                        "sensitivity": "internal",
                        "evidence_kind": "baseline_requirement",
                        "non_tsd_evidence": True,
                    },
                    token_count=max(1, len(text) // 4),
                )
            )

        raptor_candidates: List[RetrievalCandidate] = []
        if raptor_response and not raptor_response.is_empty:
            for result in raptor_response.results:
                raptor_candidates.append(
                    RetrievalCandidate(
                        id=result.node.node_id,
                        source_type="raptor",
                        text=result.node.text,
                        score=float(result.cosine_similarity),
                        block_ids=list(result.source_block_ids),
                        metadata={
                            "level": result.node.level,
                            "page_numbers": list(result.node.page_numbers),
                            "section_heading": result.node.section_heading,
                            "sensitivity": "internal",
                        },
                        token_count=result.node.token_estimate,
                    )
                )

        merged = merge_candidates(bm25_candidates, dense_candidates, raptor_candidates)
        deduped = dedupe_candidates(merged)
        policy_filtered = filter_candidates_by_policy(deduped, UserContext())
        scored_for_coverage = self._apply_keyword_coverage_boost(policy_filtered, keywords)
        evidence_filtered, evidence_metadata = self._grade_and_filter_candidates(
            scored_for_coverage,
            query_text=query_text,
            keywords=keywords,
        )
        reranked = self._reranker.rerank(
            query=query_text,
            candidates=evidence_filtered,
            top_k=self.max_context_chunks,
        )
        merged_chunks = [c.text for c in reranked if c.text]
        all_source_block_ids: List[str] = []
        seen_block_ids: Set[str] = set()
        for candidate in reranked:
            for block_id in candidate.block_ids:
                if block_id not in seen_block_ids:
                    seen_block_ids.add(block_id)
                    all_source_block_ids.append(block_id)

        return RetrievalResult(
            context_chunks=merged_chunks[:self.max_context_chunks],
            source_block_ids=all_source_block_ids,
            strategy_used=RetrievalStrategy.HYBRID,
            query_embedding=query_embedding,
            vector_response=vector_response,
            raptor_response=raptor_response,
            graph_response=None,
            evidence_metadata=evidence_metadata,
            error=None,
        )

    def _execute_graph_local(
        self,
        query_text: str,
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        raptor_tree: Optional[RAPTORTree],
        graph: Optional[TSDGraph],
        query_embedding: List[float],
        keywords: List[str],
        query_entities: List[str],
    ) -> RetrievalResult:
        if graph is None or graph.is_empty():
            return self._execute_hybrid(
                query_text=query_text,
                category=category,
                ingestion_job=ingestion_job,
                raptor_tree=raptor_tree,
                graph=graph,
                query_embedding=query_embedding,
                keywords=keywords,
                inferred_relations=set(),
            )

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ThreadPoolExecutor-4") as executor:
            fut_graph = executor.submit(
                _safe_execute, self._graph_searcher.search_local,
                query_entities=query_entities,
                graph=graph,
                query_embedding=query_embedding or None,
                user_context=UserContext(),
                traversal_config=GraphTraversalConfig()
            )

            fut_vec = executor.submit(
                _safe_execute, self._vector_searcher.search,
                query_text=query_text,
                category=category,
                top_k=self.vector_top_k,
                ingestion_job=ingestion_job,
                precomputed_embedding=query_embedding or None
            )

            fut_bm25 = executor.submit(
                _safe_execute, self._keyword_searcher.search,
                query_text=query_text,
                tree=raptor_tree,
                top_k=max(self.vector_top_k, self.raptor_top_k, 20),
                allowed_levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH]
            )

            graph_response = fut_graph.result()
            if graph_response.error:
                return self._execute_hybrid(
                    query_text=query_text,
                    category=category,
                    ingestion_job=ingestion_job,
                    raptor_tree=raptor_tree,
                    graph=graph,
                    query_embedding=query_embedding,
                    keywords=keywords,
                    inferred_relations=set(),
                )

            vector_response = fut_vec.result()
            bm25_candidates = fut_bm25.result()
        dense_candidates: List[RetrievalCandidate] = []
        for idx, result in enumerate(vector_response.results):
            text = result.child.requirement_text or ""
            if not text:
                continue
            dense_candidates.append(
                RetrievalCandidate(
                    id=f"dense:{getattr(result.child, 'stable_key', idx)}",
                    source_type="dense",
                    text=text,
                    score=float(result.cosine_similarity),
                    block_ids=[],
                    metadata={
                        "sensitivity": "internal",
                        "evidence_kind": "baseline_requirement",
                        "non_tsd_evidence": True,
                    },
                    token_count=max(1, len(text) // 4),
                )
            )
        graph_candidates = self._graph_response_to_candidates(graph_response)
        merged = merge_candidates(bm25_candidates, dense_candidates, graph_candidates)
        deduped = dedupe_candidates(merged)
        policy_filtered = filter_candidates_by_policy(deduped, UserContext())
        scored_for_coverage = self._apply_keyword_coverage_boost(policy_filtered, keywords)
        evidence_filtered, evidence_metadata = self._grade_and_filter_candidates(
            scored_for_coverage,
            query_text=query_text,
            keywords=keywords,
        )
        reranked = self._reranker.rerank(query=query_text, candidates=evidence_filtered, top_k=self.max_context_chunks)

        chunks = [c.text for c in reranked if c.text]
        block_ids: List[str] = []
        seen: Set[str] = set()
        for candidate in reranked:
            for block_id in candidate.block_ids:
                if block_id not in seen:
                    seen.add(block_id)
                    block_ids.append(block_id)
        return RetrievalResult(
            context_chunks=chunks[:self.max_context_chunks],
            source_block_ids=block_ids,
            strategy_used=RetrievalStrategy.GRAPH_LOCAL,
            query_embedding=query_embedding,
            vector_response=vector_response,
            raptor_response=None,
            graph_response=graph_response,
            graph_node_ids=list(graph_response.graph_node_ids),
            graph_edge_ids=list(graph_response.graph_edge_ids),
            grounded_texts=list(graph_response.grounded_texts),
            evidence_metadata={
                **evidence_metadata,
                "query_entities": query_entities,
                "graph_node_ids": graph_response.graph_node_ids,
                "graph_edge_ids": graph_response.graph_edge_ids,
                "graph_embedding_rerank_applied": bool(query_embedding),
                "graph_embedding_stats": getattr(graph, "embedding_stats", {}),
            },
            error=None,
        )

    def _graph_response_to_candidates(self, graph_response: GraphSearchResponse) -> List[RetrievalCandidate]:
        candidates: List[RetrievalCandidate] = []
        for idx, result in enumerate(graph_response.results):
            entity = result.entity
            text_lines = [
                f"GRAPH NODE: {entity.name} ({entity.entity_type})",
            ]
            for rel in result.relevant_relations:
                text_lines.append(f"{rel.source_entity_id} --{rel.relation_type}--> {rel.target_entity_id}")
            text = "\n".join(text_lines)
            candidates.append(
                RetrievalCandidate(
                    id=f"graph:{entity.entity_id}:{idx}",
                    source_type="raptor",
                    text=text,
                    score=float(result.relevance_score),
                    block_ids=list(result.source_block_ids),
                    metadata={
                        "graph_node_id": entity.entity_id,
                        "graph_edge_ids": [f"{r.source_entity_id}->{r.target_entity_id}" for r in result.relevant_relations],
                        "grounded_texts": (entity.grounded_texts or []),
                        "entity_similarity": float(getattr(result, "entity_similarity", 0.0)),
                        "relation_similarity": float(getattr(result, "relation_similarity", 0.0)),
                        "blended_score": float(getattr(result, "blended_score", result.relevance_score)),
                        "sensitivity": entity.sensitivity,
                        "tenant_id": entity.tenant_id,
                    },
                    token_count=max(1, len(text) // 4),
                )
            )
        return candidates

    def _execute_graph_global(
        self,
        query_text: str,
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        raptor_tree: Optional[RAPTORTree],
        graph: Optional[TSDGraph],
        query_embedding: List[float],
        keywords: List[str],
        query_entities: List[str],
    ) -> RetrievalResult:
        if not self.advanced_config.enable_graph_global or graph is None or graph.is_empty():
            return self._execute_hybrid(
                query_text=query_text,
                category=category,
                ingestion_job=ingestion_job,
                raptor_tree=raptor_tree,
                graph=graph,
                query_embedding=query_embedding,
                keywords=keywords,
                inferred_relations=set(),
            )

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ThreadPoolExecutor-5") as executor:
            def _get_community_cands():
                communities = self._community_service.detect_communities(graph)
                if not communities:
                    return None
                summaries = self._community_service.summarize_communities(graph, communities)
                cands = self._community_summaries_to_candidates(
                    graph=graph,
                    summaries=summaries,
                    keywords=keywords,
                    query_embedding=query_embedding or None,
                )
                return filter_candidates_by_policy(cands, UserContext())

            fut_comm = executor.submit(_safe_execute, _get_community_cands)

            if query_entities:
                fut_graph = executor.submit(
                    _safe_execute, self._graph_searcher.search_local,
                    query_entities=query_entities,
                    graph=graph,
                    query_embedding=query_embedding or None,
                    user_context=UserContext(),
                    traversal_config=GraphTraversalConfig()
                )
            else:
                fut_graph = None

            fut_vec = executor.submit(
                _safe_execute, self._vector_searcher.search,
                query_text=query_text,
                category=category,
                top_k=self.vector_top_k,
                ingestion_job=ingestion_job,
                precomputed_embedding=query_embedding or None
            )

            community_candidates = fut_comm.result()
            if not community_candidates:
                return self._execute_hybrid(
                    query_text=query_text,
                    category=category,
                    ingestion_job=ingestion_job,
                    raptor_tree=raptor_tree,
                    graph=graph,
                    query_embedding=query_embedding,
                    keywords=keywords,
                    inferred_relations=set(),
                )

            graph_candidates: List[RetrievalCandidate] = []
            if fut_graph:
                graph_local_response = fut_graph.result()
                if not graph_local_response.error:
                    graph_candidates = self._graph_response_to_candidates(graph_local_response)
            
            vector_response = fut_vec.result()
        dense_candidates: List[RetrievalCandidate] = []
        for idx, result in enumerate(vector_response.results):
            text = result.child.requirement_text or ""
            if not text:
                continue
            dense_candidates.append(
                RetrievalCandidate(
                    id=f"dense:{getattr(result.child, 'stable_key', idx)}",
                    source_type="dense",
                    text=text,
                    score=float(result.cosine_similarity),
                    block_ids=[],
                    metadata={
                        "sensitivity": "internal",
                        "evidence_kind": "baseline_requirement",
                        "non_tsd_evidence": True,
                    },
                    token_count=max(1, len(text) // 4),
                )
            )

        merged = merge_candidates(community_candidates, graph_candidates, dense_candidates)
        deduped = dedupe_candidates(merged)
        policy_filtered = filter_candidates_by_policy(deduped, UserContext())
        scored_for_coverage = self._apply_keyword_coverage_boost(policy_filtered, keywords)
        evidence_filtered, evidence_metadata = self._grade_and_filter_candidates(
            scored_for_coverage,
            query_text=query_text,
            keywords=keywords,
        )
        reranked = self._reranker.rerank(query=query_text, candidates=evidence_filtered, top_k=self.max_context_chunks)

        chunks = [c.text for c in reranked if c.text]
        block_ids: List[str] = []
        seen: Set[str] = set()
        for candidate in reranked:
            for block_id in candidate.block_ids:
                if block_id not in seen:
                    seen.add(block_id)
                    block_ids.append(block_id)

        selected_community_ids = [
            c.metadata.get("community_id")
            for c in reranked
            if c.metadata.get("community_id")
        ]
        return RetrievalResult(
            context_chunks=chunks[:self.max_context_chunks],
            source_block_ids=block_ids,
            strategy_used=RetrievalStrategy.GRAPH_GLOBAL,
            query_embedding=query_embedding,
            vector_response=vector_response,
            raptor_response=None,
            graph_response=None,
            evidence_metadata={
                **evidence_metadata,
                "query_entities": query_entities,
                "selected_communities": selected_community_ids,
                "mode": "graph_global",
                "graph_embedding_rerank_applied": bool(query_embedding),
                "graph_embedding_stats": getattr(graph, "embedding_stats", {}),
            },
            error=None,
        )

    def _execute_ir_cot_graph(
        self,
        query_text: str,
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        raptor_tree: Optional[RAPTORTree],
        graph: Optional[TSDGraph],
        query_embedding: List[float],
        keywords: List[str],
        query_entities: List[str],
    ) -> RetrievalResult:
        max_iterations = min(max(1, self.advanced_config.ir_cot_max_iterations), 3)
        current_query = query_text
        accumulated_candidates: List[RetrievalCandidate] = []
        seen_block_ids: Set[str] = set()
        evidence_metadata: Dict[str, Any] = {"ir_cot_iterations": 0, "queries": []}

        for iteration in range(max_iterations):
            evidence_metadata["queries"].append(current_query)
            local_keywords = _extract_keywords(current_query)
            local_entities = self._extract_query_entities(current_query)

            if self.advanced_config.enable_graph_global:
                result = self._execute_graph_global(
                    query_text=current_query,
                    category=category,
                    ingestion_job=ingestion_job,
                    raptor_tree=raptor_tree,
                    graph=graph,
                    query_embedding=query_embedding,
                    keywords=local_keywords,
                    query_entities=local_entities,
                )
            else:
                result = self._execute_graph_local(
                    query_text=current_query,
                    category=category,
                    ingestion_job=ingestion_job,
                    raptor_tree=raptor_tree,
                    graph=graph,
                    query_embedding=query_embedding,
                    keywords=local_keywords,
                    query_entities=local_entities,
                )

            new_candidates = [
                RetrievalCandidate(
                    id=f"ircot:{iteration}:{idx}",
                    source_type="raptor",
                    text=text,
                    score=1.0 - (0.1 * iteration),
                    block_ids=list(result.source_block_ids),
                    metadata={"sensitivity": "internal"},
                    token_count=max(1, len(text) // 4),
                )
                for idx, text in enumerate(result.context_chunks)
                if text
            ]
            accumulated_candidates.extend(new_candidates)
            new_blocks = set(result.source_block_ids) - seen_block_ids
            seen_block_ids.update(result.source_block_ids)
            evidence_metadata["ir_cot_iterations"] = iteration + 1
            if not new_blocks:
                break
            if len(seen_block_ids) >= self.max_context_chunks:
                break
            current_query = self._generate_followup_query(query_text, accumulated_candidates)

        deduped = dedupe_candidates(accumulated_candidates)
        policy_filtered = filter_candidates_by_policy(deduped, UserContext())
        evidence_filtered, quality_metadata = self._grade_and_filter_candidates(
            policy_filtered,
            query_text=query_text,
            keywords=keywords,
        )
        evidence_metadata.update(quality_metadata)
        reranked = self._reranker.rerank(query=query_text, candidates=evidence_filtered, top_k=self.max_context_chunks)
        chunks = [c.text for c in reranked if c.text]
        block_ids: List[str] = []
        seen: Set[str] = set()
        for candidate in reranked:
            for block_id in candidate.block_ids:
                if block_id not in seen:
                    seen.add(block_id)
                    block_ids.append(block_id)
        return RetrievalResult(
            context_chunks=chunks[:self.max_context_chunks],
            source_block_ids=block_ids,
            strategy_used=RetrievalStrategy.IR_COT_GRAPH,
            query_embedding=query_embedding,
            evidence_metadata=evidence_metadata,
            error=None,
        )

    def _community_summaries_to_candidates(
        self,
        graph: TSDGraph,
        summaries: List[CommunitySummary],
        keywords: List[str],
        query_embedding: Optional[List[float]] = None,
    ) -> List[RetrievalCandidate]:
        candidates: List[RetrievalCandidate] = []
        keyword_set = {k.lower() for k in keywords}
        for summary in summaries:
            combined_text = f"{summary.title}\n{summary.summary}"
            coverage = sum(1 for k in keyword_set if k in combined_text.lower())
            score = 0.3 + (0.1 * min(coverage, 5))
            community_sim = self._community_similarity(
                graph=graph,
                community_id=summary.community_id,
                text=combined_text,
                query_embedding=query_embedding,
            )
            if community_sim > 0.0:
                score = (0.65 * score) + (0.35 * community_sim)
            candidates.append(
                RetrievalCandidate(
                    id=f"community:{summary.community_id}",
                    source_type="raptor",
                    text=combined_text,
                    score=score,
                    block_ids=list(summary.block_ids),
                    metadata={
                        "community_id": summary.community_id,
                        "key_entities": summary.key_entities,
                        "key_relationships": summary.key_relationships,
                        "community_similarity": community_sim,
                        "sensitivity": summary.sensitivity,
                        "tenant_id": summary.source_metadata.get("tenant_id"),
                    },
                    token_count=max(1, len(combined_text) // 4),
                )
            )
        return candidates

    def _classify_candidate_evidence(
        self,
        candidate: RetrievalCandidate,
        *,
        keywords: List[str],
    ) -> Tuple[str, str]:
        text = (candidate.text or "").strip()
        lowered = text.lower()
        if not text:
            return "empty", "empty chunk"
        if candidate.metadata.get("non_tsd_evidence") or candidate.source_type == "dense":
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

    def _grade_and_filter_candidates(
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
            kind, reason = self._classify_candidate_evidence(candidate, keywords=keywords)
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
        selected = implementation + fallback

        if not selected and candidates:
            selected = [
                c for c in candidates
                if c.metadata.get("evidence_kind") not in {"baseline_requirement", "empty"}
            ][: self.max_context_chunks]

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

    def _generate_followup_query(self, original_query: str, candidates: List[RetrievalCandidate]) -> str:
        words = _extract_keywords(original_query)
        seen = set(words)
        for candidate in candidates[:5]:
            for token in _extract_keywords(candidate.text)[:8]:
                if token not in seen:
                    words.append(token)
                    seen.add(token)
                if len(words) >= 16:
                    break
            if len(words) >= 16:
                break
        return " ".join(words[:16]) or original_query

    def _apply_keyword_coverage_boost(
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

    def _community_similarity(
        self,
        graph: Optional[TSDGraph],
        community_id: str,
        text: str,
        query_embedding: Optional[List[float]],
    ) -> float:
        if graph is None or not query_embedding or not text:
            return 0.0
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_key = f"{community_id}:{text_hash}"
        vector = graph.community_embedding_cache.get(cache_key)
        if not vector:
            vector = get_embedding(text=text, dimensions=_EMBEDDING_DIMENSIONS) or []
            if vector:
                graph.community_embedding_cache[cache_key] = vector
        return self._cosine_similarity(query_embedding, vector)

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_strategy(
        self,
        query_text: str,
        keywords: List[str],
        inferred_relations: set,
        query_entities: List[str],
        query_type: QueryType,
        raptor_tree: Optional[RAPTORTree],
        graph: Optional[TSDGraph],
    ) -> RetrievalStrategy:
        """
        Automatically selects the most appropriate retrieval strategy
        for a given parameter based on keyword and resource availability.
        """
        has_raptor = raptor_tree is not None and not raptor_tree.is_empty()
        has_graph = graph is not None and not graph.is_empty()
        has_many_keywords = len(keywords) >= _HYBRID_TRIGGER_KEYWORD_COUNT
        has_enough_relations = len(inferred_relations) >= _GRAPH_TRIGGER_RELATION_COUNT

        if query_type == QueryType.FACT_BASED:
            if has_raptor:
                return RetrievalStrategy.HYBRID
            return RetrievalStrategy.VECTOR_ONLY

        if query_type == QueryType.MULTI_HOP_SECURITY and has_graph:
            if self.advanced_config.enable_ir_cot:
                return RetrievalStrategy.IR_COT_GRAPH
            return RetrievalStrategy.GRAPH_LOCAL

        if query_type == QueryType.REASONING_BASED and has_graph:
            if self._graph_entities_are_close(graph, query_entities):
                return RetrievalStrategy.GRAPH_LOCAL
            if has_raptor:
                return RetrievalStrategy.HYBRID
            return RetrievalStrategy.GRAPH_TRAVERSE

        if query_type == QueryType.GLOBAL_ARCHITECTURAL:
            if has_graph and self.advanced_config.enable_graph_global and self._has_community_summaries(graph):
                if self.advanced_config.enable_ir_cot:
                    return RetrievalStrategy.IR_COT_GRAPH
                return RetrievalStrategy.GRAPH_GLOBAL
            if has_raptor:
                return RetrievalStrategy.HYBRID

        # RAPTOR_HIGH — cross-cutting parameter with many keywords
        if has_raptor and has_many_keywords:
            return RetrievalStrategy.RAPTOR_HIGH

        # RAPTOR_LOW — raptor available, moderate keyword count
        if has_raptor:
            return RetrievalStrategy.RAPTOR_LOW

        # VECTOR_ONLY — safe fallback when no tree or graph available
        return RetrievalStrategy.VECTOR_ONLY

    def _extract_query_entities(self, query_text: str) -> List[str]:
        keywords = _extract_keywords(query_text)
        entities: List[str] = []
        seen: Set[str] = set()
        for keyword in keywords:
            normalized = _normalise_entity_id(keyword)
            if normalized and normalized not in seen:
                seen.add(normalized)
                entities.append(normalized)
        return entities

    def _classify_query_type(self, query_text: str, keywords: List[str], inferred_relations: Set[str]) -> QueryType:
        text = query_text.lower()
        global_markers = (
            "across all", "overall", "end-to-end", "entire architecture", "global",
            "how does the architecture", "main security risks", "trust boundaries", "dependencies",
        )
        multi_hop_markers = (
            "bypass", "trace", "audit this request path", "leak", "reach the", "request path",
            "permissions", "tenant data",
        )
        reasoning_markers = ("between", "path", "flow", "across", "relationship", "all services")
        if any(marker in text for marker in global_markers):
            return QueryType.GLOBAL_ARCHITECTURAL
        if any(marker in text for marker in multi_hop_markers):
            return QueryType.MULTI_HOP_SECURITY
        if any(marker in text for marker in reasoning_markers) or len(inferred_relations) >= 1:
            return QueryType.REASONING_BASED
        if len(keywords) <= 4:
            return QueryType.FACT_BASED
        return QueryType.REASONING_BASED

    def _has_community_summaries(self, graph: Optional[TSDGraph]) -> bool:
        if graph is None or graph.is_empty():
            return False
        communities = self._community_service.detect_communities(graph)
        return len(communities) > 0

    def _graph_entities_are_close(self, graph: Optional[TSDGraph], query_entities: List[str]) -> bool:
        if graph is None or graph.is_empty() or len(query_entities) < 2 or graph.graph is None or nx is None:
            return False
        matching_ids: List[str] = []
        for entity_name in query_entities:
            normalized = _normalise_entity_id(entity_name)
            if normalized in graph.entities:
                matching_ids.append(normalized)
                continue
            matches = graph.find_entities_by_name_fragment(entity_name)
            if matches:
                matching_ids.append(matches[0].entity_id)
        if len(matching_ids) < 2:
            return False
        for i, source_id in enumerate(matching_ids):
            for target_id in matching_ids[i + 1:]:
                try:
                    # Undirected distance to capture local topology regardless of edge direction.
                    distance = nx.shortest_path_length(graph.graph.to_undirected(), source=source_id, target=target_id)
                    if distance <= 2:
                        return True
                except Exception:
                    continue
        return False

    # ------------------------------------------------------------------
    # Context chunk builders
    # ------------------------------------------------------------------

    def _build_chunks_from_vector(
        self,
        vector_response: VectorSearchResponse,
    ) -> List[str]:
        """
        Converts VectorSearchResponse results into context chunk strings
        for the Hunter agent.

        Each chunk includes the requirement text from the matched
        CategoryParameterChild [3] along with its section heading for
        positional context — mirroring the banner format used by
        chunk_text_with_context() [1].

        Args:
            vector_response: The VectorSearchResponse from VectorSearcher.

        Returns:
            List of formatted context chunk strings.
        """
        chunks: List[str] = []

        for idx, result in enumerate(vector_response.results, start=1):
            child = result.child
            section = (
                child.parent.title
                if hasattr(child, "parent") and child.parent
                else "Unknown Section"
            )
            similarity_pct = round(result.cosine_similarity * 100, 1)
            chunk = (
                f"--- VECTOR RESULT {idx} "
                f"[similarity={similarity_pct}%] "
                f"[section={section}] ---\n\n"
                f"{child.requirement_text}"
            )
            chunks.append(chunk)

        return chunks

    def _build_chunks_from_graph(
        self,
        graph_response: GraphSearchResponse,
    ) -> List[str]:
        """
        Converts GraphSearchResponse results into context chunk strings
        for the Hunter agent.

        Each chunk describes an entity and its security-relevant relations
        in plain text — the Hunter agent is prompted to look for
        compliance evidence, so structured prose is more useful than
        raw graph data structures.

        For PATH_ANALYSIS results, each path is described as a chain
        of entity → relation → entity hops with security metadata
        (is_encrypted, requires_auth, protocol) per edge.

        Args:
            graph_response: The GraphSearchResponse from GraphSearcher.

        Returns:
            List of formatted context chunk strings.
        """
        chunks: List[str] = []

        # Entity/relation chunks
        for idx, result in enumerate(graph_response.results, start=1):
            entity = result.entity
            lines = [
                f"--- GRAPH RESULT {idx} "
                f"[strategy={graph_response.strategy_used.value}] "
                f"[entity={entity.name}] "
                f"[type={entity.entity_type}] ---",
            ]

            if entity.description:
                lines.append(f"Description: {entity.description}")

            if result.relevant_relations:
                lines.append("Security-relevant relationships:")
                for rel in result.relevant_relations:
                    target_entity_id = rel.target_entity_id
                    rel_desc = rel.description or rel.relation_type
                    security_flags: List[str] = []

                    if rel.is_encrypted is True:
                        security_flags.append("encrypted=yes")
                    elif rel.is_encrypted is False:
                        security_flags.append("encrypted=NO")

                    if rel.requires_auth is True:
                        security_flags.append("auth=yes")
                    elif rel.requires_auth is False:
                        security_flags.append("auth=NO")

                    if rel.protocol:
                        security_flags.append(f"protocol={rel.protocol}")

                    flags_str = (
                        f" [{', '.join(security_flags)}]"
                        if security_flags
                        else ""
                    )
                    lines.append(
                        f"  → {rel.relation_type} → {target_entity_id}: "
                        f"{rel_desc}{flags_str}"
                    )

            chunks.append("\n".join(lines))

        # Path analysis chunks
        for idx, path_result in enumerate(graph_response.path_results, start=1):
            lines = [
                f"--- GRAPH PATH {idx} "
                f"[length={path_result.length}] "
                f"[secured={path_result.is_fully_secured}] ---",
                f"Path: {' → '.join(path_result.entity_names)}",
                f"Has authentication: {path_result.has_auth}",
                f"Has encryption: {path_result.has_encryption}",
                f"Fully secured: {path_result.is_fully_secured}",
            ]
            if path_result.path_relations:
                lines.append("Edge details:")
                for rel in path_result.path_relations:
                    flags: List[str] = []
                    if rel.is_encrypted is not None:
                        flags.append(f"encrypted={'yes' if rel.is_encrypted else 'NO'}")
                    if rel.requires_auth is not None:
                        flags.append(f"auth={'yes' if rel.requires_auth else 'NO'}")
                    if rel.protocol:
                        flags.append(f"protocol={rel.protocol}")
                    flags_str = f" [{', '.join(flags)}]" if flags else ""
                    lines.append(
                        f"  {rel.source_entity_id} "
                        f"--[{rel.relation_type}]--> "
                        f"{rel.target_entity_id}{flags_str}"
                    )
            chunks.append("\n".join(lines))

        return chunks

    def _collect_block_ids_from_vector(
        self,
        vector_response: VectorSearchResponse,
    ) -> List[str]:
        """
        Vector search is treated as a ranking hint only.

        Vector results come from the parameter knowledge base, not the
        current TSD document, so their identifiers are not suitable for
        click-to-source citation anchors in the review output.

        The router still returns the vector chunks as low-priority context,
        but this helper intentionally returns an empty list so the analysis
        pipeline only persists TSD-backed citation ids from RAPTOR or graph
        retrieval.

        Args:
            vector_response: The VectorSearchResponse from VectorSearcher.

        Returns:
            Empty list.
        """
        return []

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def _generate_query_embedding(self, query_text: str) -> List[float]:
        """
        Generates a single embedding vector for the query text using
        Amazon Titan via get_embedding() from client.py [4].

        This is called ONCE per parameter — the returned vector is
        passed as precomputed_embedding to all three searchers so
        no redundant Bedrock API calls are made.

        Returns an empty list on failure — callers handle gracefully
        by passing None to precomputed_embedding which causes each
        searcher to fall back to text-only or generate their own.

        Args:
            query_text: The normalised parameter text to embed.

        Returns:
            A list of floats (embedding vector), or [] on failure.
        """
        try:
            vector = get_embedding(
                text=query_text,
                dimensions=_EMBEDDING_DIMENSIONS,
            )   # [4]

            if not vector:
                self.logger.error(
                    "HybridRetrievalRouter._generate_query_embedding: "
                    "get_embedding() returned empty vector for query '%s...'",
                    query_text[:60],
                )
                return []

            return vector

        except Exception as exc:
            self.logger.error(
                "HybridRetrievalRouter._generate_query_embedding: "
                "unexpected error: %s",
                exc,
            )
            return []


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def retrieve_context_for_parameter(
    parameter: CategoryParameterChild,
    category: StandardCategory,
    raptor_tree: Optional[RAPTORTree] = None,
    graph: Optional[TSDGraph] = None,
    ingestion_job: Optional[StandardIngestionJob] = None,
    force_strategy: Optional[RetrievalStrategy] = None,
) -> RetrievalResult:
    """
    Module-level convenience wrapper around HybridRetrievalRouter.retrieve().

    Instantiates a HybridRetrievalRouter with default settings and
    executes a single parameter retrieval. Used directly by
    analysis_service.py as the primary context retrieval entry point.

    Args:
        parameter:      The CategoryParameterChild to retrieve context for [3].
        category:       The StandardCategory to scope vector search to.
        raptor_tree:    The RAPTORTree built from the TSD document.
        graph:          The TSDGraph built from the TSD document.
        ingestion_job:  Optional specific job to scope vector search to.
        force_strategy: Optional explicit strategy override.

    Returns:
        RetrievalResult — never raises.
    """
    router = HybridRetrievalRouter()
    return router.retrieve(
        parameter=parameter,
        category=category,
        raptor_tree=raptor_tree,
        graph=graph,
        ingestion_job=ingestion_job,
        force_strategy=force_strategy,
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Strategy enum — used by analysis_service.py to force strategies
    "RetrievalStrategy",
    # Result dataclass — consumed by analysis_service.py
    "RetrievalResult",
    # Main router class
    "HybridRetrievalRouter",
    # Convenience function — primary entry point for analysis_service.py
    "retrieve_context_for_parameter",
]
