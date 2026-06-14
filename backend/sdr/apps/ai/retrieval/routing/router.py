from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Dict, List, Optional, Set

from sdr.apps.ai.client import get_embedding
from sdr.apps.ai.retrieval.core import AdvancedRetrievalConfig, RetrievalCandidate, RetrievalResult, RetrievalStrategy
from sdr.apps.ai.retrieval.graph.communities import CommunitySummary, GraphCommunityService
from sdr.apps.ai.retrieval.postprocessing.chunk_builders import (
    build_chunks_from_graph,
    build_chunks_from_vector,
    collect_block_ids_from_vector,
    community_summaries_to_candidates,
    graph_response_to_candidates,
)
from sdr.apps.ai.retrieval.postprocessing.evidence_grader import EvidenceGrader
from sdr.apps.ai.retrieval.postprocessing.reranker import SafeOptionalReranker
from sdr.apps.ai.retrieval.routing.executors import RetrievalRouteExecutor
from sdr.apps.ai.retrieval.routing.strategy_selector import RetrievalStrategySelector
from sdr.apps.ai.retrieval.searchers.graph import (
    GraphSearchResponse,
    GraphSearcher,
    _extract_keywords,
    _infer_relation_types_from_parameter,
)
from sdr.apps.ai.retrieval.searchers.keyword import KeywordSearcher
from sdr.apps.ai.retrieval.searchers.raptor import RAPTORSearchResponse, RAPTORSearcher
from sdr.apps.ai.retrieval.searchers.vector import VectorSearchResponse, VectorSearcher
from sdr.apps.ai.tsd_processing.graph_builder import TSDGraph
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree
from sdr.apps.standards.models import CategoryParameterChild, StandardCategory, StandardIngestionJob
from sdr.apps.standards.utils import build_parameter_analysis_text

logger = logging.getLogger(__name__)

_VECTOR_TOP_K = 8
_RAPTOR_TOP_K = 5
_GRAPH_TOP_K = 6
_MAX_CONTEXT_CHUNKS = 12
_EMBEDDING_DIMENSIONS = 1024


class HybridRetrievalRouter:
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
        self._strategy_selector = RetrievalStrategySelector()
        self._route_executor = RetrievalRouteExecutor()
        self._evidence_grader = EvidenceGrader(max_context_chunks=self.max_context_chunks)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

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
        query_text = (override_query_text or "").strip() or build_parameter_analysis_text(parameter).strip()
        if not query_text:
            msg = f"Parameter id={parameter.id} has empty requirement text — cannot retrieve context."
            self.logger.warning("HybridRetrievalRouter.retrieve: %s", msg)
            return RetrievalResult(error=msg)

        query_embedding = self._generate_query_embedding(query_text)
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

        try:
            if strategy == RetrievalStrategy.VECTOR_ONLY:
                return self._execute_vector_only(
                    query_text=query_text,
                    category=category,
                    ingestion_job=ingestion_job,
                    query_embedding=query_embedding,
                )
            if strategy == RetrievalStrategy.RAPTOR_LOW:
                return self._execute_raptor_low(query_text=query_text, raptor_tree=raptor_tree, query_embedding=query_embedding)
            if strategy == RetrievalStrategy.RAPTOR_HIGH:
                return self._execute_raptor_high(query_text=query_text, raptor_tree=raptor_tree, query_embedding=query_embedding)
            if strategy == RetrievalStrategy.GRAPH_TRAVERSE:
                return self._execute_graph_traverse(
                    query_text=query_text,
                    graph=graph,
                    raptor_tree=raptor_tree,
                    query_embedding=query_embedding,
                )
            if strategy == RetrievalStrategy.GRAPH_LOCAL:
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
            if strategy == RetrievalStrategy.GRAPH_GLOBAL:
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
            if strategy == RetrievalStrategy.IR_COT_GRAPH:
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
        except Exception as exc:
            msg = f"Strategy execution failed for strategy={strategy.value}: {exc}"
            self.logger.exception("HybridRetrievalRouter.retrieve: %s for parameter id=%s", msg, parameter.id)
            return RetrievalResult(error=msg, strategy_used=strategy, query_embedding=query_embedding)

    def _select_strategy(self, **kwargs) -> RetrievalStrategy:
        kwargs.pop("query_text", None)
        return self._strategy_helper().select_strategy(
            advanced_config=self.advanced_config,
            has_community_summaries=self._has_community_summaries,
            **kwargs,
        )

    def _extract_query_entities(self, query_text: str) -> List[str]:
        return self._strategy_helper().extract_query_entities(query_text, _extract_keywords)

    def _classify_query_type(self, query_text: str, keywords: List[str], inferred_relations: Set[str]):
        return self._strategy_helper().classify_query_type(query_text, keywords, inferred_relations)

    def _has_community_summaries(self, graph: Optional[TSDGraph]) -> bool:
        if graph is None or graph.is_empty():
            return False
        return len(self._community_service.detect_communities(graph)) > 0

    def _execute_vector_only(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_vector_only(self, **kwargs)

    def _execute_raptor_low(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_raptor_low(self, **kwargs)

    def _execute_raptor_high(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_raptor_high(self, **kwargs)

    def _execute_graph_traverse(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_graph_traverse(self, **kwargs)

    def _execute_hybrid(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_hybrid(self, **kwargs)

    def _execute_graph_local(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_graph_local(self, **kwargs)

    def _execute_graph_global(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_graph_global(self, **kwargs)

    def _execute_ir_cot_graph(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_ir_cot_graph(self, **kwargs)

    def _graph_response_to_candidates(self, graph_response: GraphSearchResponse) -> List[RetrievalCandidate]:
        return graph_response_to_candidates(graph_response)

    def _community_summaries_to_candidates(
        self,
        graph: TSDGraph,
        summaries: List[CommunitySummary],
        keywords: List[str],
        query_embedding: Optional[List[float]] = None,
    ) -> List[RetrievalCandidate]:
        return community_summaries_to_candidates(
            graph=graph,
            summaries=summaries,
            keywords=keywords,
            similarity_resolver=self._community_similarity,
            query_embedding=query_embedding,
        )

    def _classify_candidate_evidence(self, candidate: RetrievalCandidate, *, keywords: List[str]):
        return self._grader().classify_candidate_evidence(candidate, keywords=keywords)

    def _grade_and_filter_candidates(self, candidates: List[RetrievalCandidate], *, query_text: str, keywords: List[str]):
        return self._grader().grade_and_filter_candidates(candidates, query_text=query_text, keywords=keywords)

    def _generate_followup_query(self, original_query: str, candidates: List[RetrievalCandidate]) -> str:
        return self._grader().generate_followup_query(original_query, candidates, _extract_keywords)

    def _apply_keyword_coverage_boost(self, candidates: List[RetrievalCandidate], keywords: List[str]) -> List[RetrievalCandidate]:
        return self._grader().apply_keyword_coverage_boost(candidates, keywords)

    def _build_chunks_from_vector(self, vector_response: VectorSearchResponse) -> List[str]:
        return build_chunks_from_vector(vector_response)

    def _build_chunks_from_graph(self, graph_response: GraphSearchResponse) -> List[str]:
        return build_chunks_from_graph(graph_response)

    def _collect_block_ids_from_vector(self, vector_response: VectorSearchResponse) -> List[str]:
        return collect_block_ids_from_vector(vector_response)

    def _vector_results_to_candidates(self, vector_response: VectorSearchResponse) -> List[RetrievalCandidate]:
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
        return dense_candidates

    def _raptor_results_to_candidates(self, raptor_response: Optional[RAPTORSearchResponse]) -> List[RetrievalCandidate]:
        candidates: List[RetrievalCandidate] = []
        if raptor_response and not raptor_response.is_empty:
            for result in raptor_response.results:
                candidates.append(
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
        return candidates

    def _collect_candidate_block_ids(self, candidates: List[RetrievalCandidate]) -> List[str]:
        block_ids: List[str] = []
        seen: Set[str] = set()
        for candidate in candidates:
            for block_id in candidate.block_ids:
                if block_id not in seen:
                    seen.add(block_id)
                    block_ids.append(block_id)
        return block_ids

    def _normalize_embedding_diagnostics(self, result: RetrievalResult, graph: Optional[TSDGraph]) -> Dict[str, Any]:
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

    def _generate_query_embedding(self, query_text: str) -> List[float]:
        try:
            return get_embedding(text=query_text, dimensions=_EMBEDDING_DIMENSIONS) or []
        except Exception as exc:
            self.logger.error("HybridRetrievalRouter._generate_query_embedding: unexpected error: %s", exc)
            return []

    def _strategy_helper(self) -> RetrievalStrategySelector:
        helper = getattr(self, "_strategy_selector", None)
        if helper is None:
            helper = RetrievalStrategySelector()
            self._strategy_selector = helper
        return helper

    def _executor(self) -> RetrievalRouteExecutor:
        helper = getattr(self, "_route_executor", None)
        if helper is None:
            helper = RetrievalRouteExecutor()
            self._route_executor = helper
        return helper

    def _grader(self) -> EvidenceGrader:
        helper = getattr(self, "_evidence_grader", None)
        if helper is None:
            helper = EvidenceGrader(max_context_chunks=getattr(self, "max_context_chunks", _MAX_CONTEXT_CHUNKS))
            self._evidence_grader = helper
        return helper


def retrieve_context_for_parameter(
    parameter: CategoryParameterChild,
    category: StandardCategory,
    raptor_tree: Optional[RAPTORTree] = None,
    graph: Optional[TSDGraph] = None,
    ingestion_job: Optional[StandardIngestionJob] = None,
    force_strategy: Optional[RetrievalStrategy] = None,
) -> RetrievalResult:
    router = HybridRetrievalRouter()
    return router.retrieve(
        parameter=parameter,
        category=category,
        raptor_tree=raptor_tree,
        graph=graph,
        ingestion_job=ingestion_job,
        force_strategy=force_strategy,
    )


__all__ = [
    "AdvancedRetrievalConfig",
    "HybridRetrievalRouter",
    "RetrievalResult",
    "RetrievalStrategy",
    "retrieve_context_for_parameter",
]
