from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

from sdr.apps.ai.retrieval.core import RetrievalCandidate, RetrievalResult, RetrievalStrategy, dedupe_candidates, merge_candidates
from sdr.apps.ai.retrieval.searchers.graph import GraphSearchResponse, GraphTraversalConfig, _extract_keywords
from sdr.apps.ai.retrieval.searchers.raptor import RAPTOR_LEVEL_HIGH, RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTORSearchResponse
from sdr.apps.ai.retrieval.searchers.vector import VectorSearchResponse
from sdr.apps.ai.tsd_processing.graph_builder import TSDGraph
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob

logger = logging.getLogger(__name__)


def _safe_execute(func, *args, on_error=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.exception("Retrieval branch failed in thread pool: %s", exc)
        if on_error is not None:
            return on_error(exc)
        raise


def _bounded_pool_size(configured_max: int, branch_count: int) -> int:
    return max(1, min(max(1, int(configured_max)), max(1, int(branch_count))))


def _canonical_source_type(source_type: Optional[str]) -> str:
    normalized = str(source_type or "").strip().lower()
    if normalized in {"raptor", "bm25"}:
        return "raptor"
    if normalized in {"graph"}:
        return "graph"
    if normalized in {"dense", "vector"}:
        return "vector"
    return normalized or "unknown"


def _source_descriptor_for_keys(source_keys: Set[str]) -> Dict[str, Any]:
    normalized = {key for key in (_canonical_source_type(item) for item in source_keys) if key}
    if normalized == {"graph"}:
        return {"key": "graph", "label": "Graph"}
    if normalized == {"raptor"}:
        return {"key": "raptor", "label": "RAPTOR"}
    if normalized == {"vector"}:
        return {"key": "vector", "label": "Vector"}
    if "graph" in normalized and "raptor" in normalized:
        return {"key": "hybrid_agreement", "label": "Hybrid Agreement"}
    if len(normalized) == 1:
        key = next(iter(normalized))
        return {"key": key, "label": key.replace("_", " ").title()}
    ordered = sorted(normalized)
    return {"key": "+".join(ordered), "label": " + ".join(item.replace("_", " ").title() for item in ordered)}


def _merge_block_source_entry(block_source_map: Dict[str, Dict[str, Any]], block_id: str, source_type: str) -> None:
    if not block_id:
        return
    canonical = _canonical_source_type(source_type)
    existing = dict(block_source_map.get(block_id) or {})
    source_keys = set(existing.get("source_keys") or [])
    if existing.get("retrieval_origin"):
        source_keys.add(str(existing["retrieval_origin"]))
    if canonical:
        source_keys.add(canonical)
    descriptor = _source_descriptor_for_keys(source_keys)
    block_source_map[block_id] = {
        "retrieval_origin": descriptor["key"],
        "retrieval_origin_label": descriptor["label"],
        "source_keys": sorted(source_keys),
    }


def _build_uniform_block_source_map(block_ids: List[str], source_type: str) -> Dict[str, Dict[str, Any]]:
    block_source_map: Dict[str, Dict[str, Any]] = {}
    for block_id in block_ids or []:
        _merge_block_source_entry(block_source_map, block_id, source_type)
    return block_source_map


def _build_block_source_map_from_candidates(candidates: List[RetrievalCandidate]) -> Dict[str, Dict[str, Any]]:
    block_source_map: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates or []:
        merged_sources = candidate.metadata.get("merged_sources", [candidate.source_type]) if isinstance(candidate.metadata, dict) else [candidate.source_type]
        for block_id in candidate.block_ids or []:
            for source_type in merged_sources:
                _merge_block_source_entry(block_source_map, block_id, str(source_type))
    return block_source_map


class RetrievalRouteExecutor:
    def execute_vector_only(
        self,
        router,
        *,
        query_text: str,
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        query_embedding: List[float],
    ) -> RetrievalResult:
        vector_response = router._vector_searcher.search(
            query_text=query_text,
            category=category,
            top_k=router.vector_top_k,
            ingestion_job=ingestion_job,
            precomputed_embedding=query_embedding or None,
        )
        context_chunks = router._build_chunks_from_vector(vector_response)
        source_block_ids = router._collect_block_ids_from_vector(vector_response)
        return RetrievalResult(
            context_chunks=context_chunks[: router.max_context_chunks],
            source_block_ids=source_block_ids,
            block_source_map={},
            strategy_used=RetrievalStrategy.VECTOR_ONLY,
            query_embedding=query_embedding,
            vector_response=vector_response,
            error=vector_response.error,
        )

    def execute_raptor_low(self, router, *, query_text: str, raptor_tree: Optional[RAPTORTree], query_embedding: List[float]) -> RetrievalResult:
        if raptor_tree is None or raptor_tree.is_empty():
            return RetrievalResult(
                strategy_used=RetrievalStrategy.RAPTOR_LOW,
                query_embedding=query_embedding,
                error="RAPTORTree is not available.",
            )
        raptor_response = router._raptor_searcher.search(
            query_text=query_text,
            tree=raptor_tree,
            level=RAPTOR_LEVEL_LOW,
            top_k=router.raptor_top_k,
            precomputed_embedding=query_embedding or None,
        )
        return RetrievalResult(
            context_chunks=raptor_response.get_context_chunks()[: router.max_context_chunks],
            source_block_ids=raptor_response.all_source_block_ids,
            block_source_map=_build_uniform_block_source_map(
                raptor_response.all_source_block_ids,
                "raptor",
            ),
            strategy_used=RetrievalStrategy.RAPTOR_LOW,
            query_embedding=query_embedding,
            raptor_response=raptor_response,
            error=raptor_response.error,
        )

    def execute_raptor_high(self, router, *, query_text: str, raptor_tree: Optional[RAPTORTree], query_embedding: List[float]) -> RetrievalResult:
        if raptor_tree is None or raptor_tree.is_empty():
            return RetrievalResult(
                strategy_used=RetrievalStrategy.RAPTOR_HIGH,
                query_embedding=query_embedding,
                error="RAPTORTree is not available.",
            )
        raptor_response = router._raptor_searcher.search_multi_level(
            query_text=query_text,
            tree=raptor_tree,
            levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
            top_k_per_level=3,
            precomputed_embedding=query_embedding or None,
        )
        return RetrievalResult(
            context_chunks=raptor_response.get_context_chunks()[: router.max_context_chunks],
            source_block_ids=raptor_response.all_source_block_ids,
            block_source_map=_build_uniform_block_source_map(
                raptor_response.all_source_block_ids,
                "raptor",
            ),
            strategy_used=RetrievalStrategy.RAPTOR_HIGH,
            query_embedding=query_embedding,
            raptor_response=raptor_response,
            error=raptor_response.error,
        )

    def execute_graph_traverse(
        self,
        router,
        *,
        query_text: str,
        graph: Optional[TSDGraph],
        raptor_tree: Optional[RAPTORTree],
        query_embedding: List[float],
    ) -> RetrievalResult:
        if graph is None or graph.is_empty():
            return router._execute_raptor_low(query_text=query_text, raptor_tree=raptor_tree, query_embedding=query_embedding)
        graph_response = router._graph_searcher.search(
            parameter_text=query_text,
            graph=graph,
            query_embedding=query_embedding or None,
            top_k=router.graph_top_k,
        )
        if graph_response.error:
            return router._execute_raptor_low(query_text=query_text, raptor_tree=raptor_tree, query_embedding=query_embedding)
        context_chunks = router._build_chunks_from_graph(graph_response)
        source_block_ids = graph_response.all_source_block_ids
        block_source_map = _build_uniform_block_source_map(source_block_ids, "graph")
        if raptor_tree and not raptor_tree.is_empty() and source_block_ids:
            raptor_response = router._raptor_searcher.search(
                query_text=query_text,
                tree=raptor_tree,
                level=RAPTOR_LEVEL_LOW,
                top_k=3,
                precomputed_embedding=query_embedding or None,
            )
            if not raptor_response.is_empty:
                context_chunks.extend(raptor_response.get_context_chunks())
                for bid in raptor_response.all_source_block_ids:
                    _merge_block_source_entry(block_source_map, bid, "raptor")
                    if bid not in source_block_ids:
                        source_block_ids.append(bid)
        else:
            raptor_response = None
        return RetrievalResult(
            context_chunks=context_chunks[: router.max_context_chunks],
            source_block_ids=source_block_ids,
            block_source_map=block_source_map,
            strategy_used=RetrievalStrategy.GRAPH_TRAVERSE,
            query_embedding=query_embedding,
            raptor_response=raptor_response if raptor_tree else None,
            graph_response=graph_response,
            evidence_metadata={
                "block_source_map": block_source_map,
                "graph_embedding_rerank_applied": bool(query_embedding),
                "graph_embedding_stats": getattr(graph, "embedding_stats", {}),
                "graph_result_count": len(graph_response.results),
            },
        )

    def execute_hybrid(
        self,
        router,
        *,
        query_text: str,
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        raptor_tree: Optional[RAPTORTree],
        graph: Optional[TSDGraph],
        query_embedding: List[float],
        keywords: List[str],
        inferred_relations: set,
    ) -> RetrievalResult:
        branch_count = 2 + int(bool(raptor_tree and not raptor_tree.is_empty()))
        max_workers = _bounded_pool_size(router.advanced_config.hybrid_max_workers, branch_count)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ThreadPoolExecutor-3") as executor:
            fut_vec = executor.submit(
                _safe_execute,
                router._vector_searcher.search,
                query_text=query_text,
                category=category,
                top_k=router.vector_top_k,
                ingestion_job=ingestion_job,
                precomputed_embedding=query_embedding or None,
                on_error=lambda exc: VectorSearchResponse(error=str(exc)),
            )
            if raptor_tree and not raptor_tree.is_empty():
                fut_rap = executor.submit(
                    _safe_execute,
                    router._raptor_searcher.search_collapsed_raptor,
                    query_text=query_text,
                    tree=raptor_tree,
                    top_k=max(router.raptor_top_k, 6),
                    max_tokens=4000,
                    allowed_levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
                    precomputed_embedding=query_embedding or None,
                    on_error=lambda exc: RAPTORSearchResponse(error=str(exc)),
                )
            else:
                fut_rap = None
            fut_bm25 = executor.submit(
                _safe_execute,
                router._keyword_searcher.search,
                query_text=query_text,
                tree=raptor_tree,
                top_k=max(router.vector_top_k, router.raptor_top_k, 20),
                allowed_levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
                on_error=lambda exc: [],
            )
            vector_response = fut_vec.result()
            raptor_response = fut_rap.result() if fut_rap else None
            bm25_candidates = fut_bm25.result()

        dense_candidates = router._vector_results_to_candidates(vector_response)
        raptor_candidates = router._raptor_results_to_candidates(raptor_response)
        merged = merge_candidates(bm25_candidates, dense_candidates, raptor_candidates)
        deduped = dedupe_candidates(merged)
        scored = router._apply_keyword_coverage_boost(deduped, keywords)
        evidence_filtered, evidence_metadata = router._grade_and_filter_candidates(
            scored,
            query_text=query_text,
            keywords=keywords,
        )
        reranked = router._reranker.rerank(query=query_text, candidates=evidence_filtered, top_k=router.max_context_chunks)
        merged_chunks = [c.text for c in reranked if c.text]
        all_source_block_ids = router._collect_candidate_block_ids(reranked)
        block_source_map = _build_block_source_map_from_candidates(reranked)
        return RetrievalResult(
            context_chunks=merged_chunks[: router.max_context_chunks],
            source_block_ids=all_source_block_ids,
            block_source_map=block_source_map,
            strategy_used=RetrievalStrategy.HYBRID,
            query_embedding=query_embedding,
            vector_response=vector_response,
            raptor_response=raptor_response,
            evidence_metadata={
                **evidence_metadata,
                "block_source_map": block_source_map,
            },
        )

    def execute_graph_local(
        self,
        router,
        *,
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
            return router._execute_hybrid(
                query_text=query_text,
                category=category,
                ingestion_job=ingestion_job,
                raptor_tree=raptor_tree,
                graph=graph,
                query_embedding=query_embedding,
                keywords=keywords,
                inferred_relations=set(),
            )
        max_workers = _bounded_pool_size(router.advanced_config.graph_local_max_workers, 3)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ThreadPoolExecutor-4") as executor:
            fut_graph = executor.submit(
                _safe_execute,
                router._graph_searcher.search_local,
                query_entities=query_entities,
                graph=graph,
                query_embedding=query_embedding,
                traversal_config=GraphTraversalConfig(),
                on_error=lambda exc: GraphSearchResponse(error=str(exc)),
            )
            fut_vec = executor.submit(
                _safe_execute,
                router._vector_searcher.search,
                query_text=query_text,
                category=category,
                top_k=router.vector_top_k,
                ingestion_job=ingestion_job,
                precomputed_embedding=query_embedding or None,
                on_error=lambda exc: VectorSearchResponse(error=str(exc)),
            )
            fut_bm25 = executor.submit(
                _safe_execute,
                router._keyword_searcher.search,
                query_text=query_text,
                tree=raptor_tree,
                top_k=max(router.vector_top_k, router.raptor_top_k, 20),
                allowed_levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
                on_error=lambda exc: [],
            )
            graph_response = fut_graph.result()
            if graph_response.error:
                return router._execute_hybrid(
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
        dense_candidates = router._vector_results_to_candidates(vector_response)
        graph_candidates = router._graph_response_to_candidates(graph_response)
        merged = merge_candidates(bm25_candidates, dense_candidates, graph_candidates)
        deduped = dedupe_candidates(merged)
        scored = router._apply_keyword_coverage_boost(deduped, keywords)
        evidence_filtered, evidence_metadata = router._grade_and_filter_candidates(
            scored,
            query_text=query_text,
            keywords=keywords,
        )
        reranked = router._reranker.rerank(query=query_text, candidates=evidence_filtered, top_k=router.max_context_chunks)
        chunks = [c.text for c in reranked if c.text]
        block_ids = router._collect_candidate_block_ids(reranked)
        block_source_map = _build_block_source_map_from_candidates(reranked)
        return RetrievalResult(
            context_chunks=chunks[: router.max_context_chunks],
            source_block_ids=block_ids,
            block_source_map=block_source_map,
            strategy_used=RetrievalStrategy.GRAPH_LOCAL,
            query_embedding=query_embedding,
            vector_response=vector_response,
            graph_response=graph_response,
            graph_node_ids=list(graph_response.graph_node_ids),
            graph_edge_ids=list(graph_response.graph_edge_ids),
            grounded_texts=list(graph_response.grounded_texts),
            evidence_metadata={
                **evidence_metadata,
                "block_source_map": block_source_map,
                "query_entities": query_entities,
                "graph_node_ids": graph_response.graph_node_ids,
                "graph_edge_ids": graph_response.graph_edge_ids,
                "graph_embedding_rerank_applied": bool(query_embedding),
                "graph_embedding_stats": getattr(graph, "embedding_stats", {}),
            },
        )

__all__ = ["RetrievalRouteExecutor"]
