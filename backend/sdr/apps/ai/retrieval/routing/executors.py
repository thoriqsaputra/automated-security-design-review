from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set

from sdr.apps.ai.retrieval.core import RetrievalCandidate, RetrievalResult, RetrievalStrategy, dedupe_candidates, merge_candidates
from sdr.apps.ai.retrieval.searchers.graph import GraphSearchResponse, GraphTraversalConfig, _extract_keywords
from sdr.apps.ai.retrieval.searchers.raptor import RAPTOR_LEVEL_HIGH, RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID
from sdr.apps.ai.tsd_processing.graph_builder import TSDGraph
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob


def _safe_execute(func, *args, **kwargs):
    return func(*args, **kwargs)


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
                    if bid not in source_block_ids:
                        source_block_ids.append(bid)
        else:
            raptor_response = None
        return RetrievalResult(
            context_chunks=context_chunks[: router.max_context_chunks],
            source_block_ids=source_block_ids,
            strategy_used=RetrievalStrategy.GRAPH_TRAVERSE,
            query_embedding=query_embedding,
            raptor_response=raptor_response if raptor_tree else None,
            graph_response=graph_response,
            evidence_metadata={
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
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ThreadPoolExecutor-3") as executor:
            fut_vec = executor.submit(
                _safe_execute,
                router._vector_searcher.search,
                query_text=query_text,
                category=category,
                top_k=router.vector_top_k,
                ingestion_job=ingestion_job,
                precomputed_embedding=query_embedding or None,
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
        return RetrievalResult(
            context_chunks=merged_chunks[: router.max_context_chunks],
            source_block_ids=all_source_block_ids,
            strategy_used=RetrievalStrategy.HYBRID,
            query_embedding=query_embedding,
            vector_response=vector_response,
            raptor_response=raptor_response,
            evidence_metadata=evidence_metadata,
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
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ThreadPoolExecutor-4") as executor:
            fut_graph = executor.submit(
                _safe_execute,
                router._graph_searcher.search_local,
                query_entities=query_entities,
                graph=graph,
                query_embedding=query_embedding,
                traversal_config=GraphTraversalConfig(),
            )
            fut_vec = executor.submit(
                _safe_execute,
                router._vector_searcher.search,
                query_text=query_text,
                category=category,
                top_k=router.vector_top_k,
                ingestion_job=ingestion_job,
                precomputed_embedding=query_embedding or None,
            )
            fut_bm25 = executor.submit(
                _safe_execute,
                router._keyword_searcher.search,
                query_text=query_text,
                tree=raptor_tree,
                top_k=max(router.vector_top_k, router.raptor_top_k, 20),
                allowed_levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
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
        return RetrievalResult(
            context_chunks=chunks[: router.max_context_chunks],
            source_block_ids=block_ids,
            strategy_used=RetrievalStrategy.GRAPH_LOCAL,
            query_embedding=query_embedding,
            vector_response=vector_response,
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
        )

    def execute_graph_global(
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
        if not router.advanced_config.enable_graph_global or graph is None or graph.is_empty():
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
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ThreadPoolExecutor-5") as executor:
            def _get_community_cands():
                communities = router._community_service.detect_communities(graph)
                if not communities:
                    return None
                summaries = router._community_service.summarize_communities(graph, communities)
                return router._community_summaries_to_candidates(
                    graph=graph,
                    summaries=summaries,
                    keywords=keywords,
                    query_embedding=query_embedding or None,
                )

            fut_comm = executor.submit(_safe_execute, _get_community_cands)
            fut_graph = None
            if query_entities:
                fut_graph = executor.submit(
                    _safe_execute,
                    router._graph_searcher.search_local,
                    query_entities=query_entities,
                    graph=graph,
                    query_embedding=query_embedding,
                    traversal_config=GraphTraversalConfig(),
                )
            fut_vec = executor.submit(
                _safe_execute,
                router._vector_searcher.search,
                query_text=query_text,
                category=category,
                top_k=router.vector_top_k,
                ingestion_job=ingestion_job,
                precomputed_embedding=query_embedding or None,
            )
            community_candidates = fut_comm.result()
            if not community_candidates:
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
            graph_candidates: List[RetrievalCandidate] = []
            if fut_graph:
                graph_local_response = fut_graph.result()
                if not graph_local_response.error:
                    graph_candidates = router._graph_response_to_candidates(graph_local_response)
            vector_response = fut_vec.result()
        dense_candidates = router._vector_results_to_candidates(vector_response)
        merged = merge_candidates(community_candidates, graph_candidates, dense_candidates)
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
        selected_community_ids = [c.metadata.get("community_id") for c in reranked if c.metadata.get("community_id")]
        return RetrievalResult(
            context_chunks=chunks[: router.max_context_chunks],
            source_block_ids=block_ids,
            strategy_used=RetrievalStrategy.GRAPH_GLOBAL,
            query_embedding=query_embedding,
            vector_response=vector_response,
            evidence_metadata={
                **evidence_metadata,
                "query_entities": query_entities,
                "selected_communities": selected_community_ids,
                "mode": "graph_global",
                "graph_embedding_rerank_applied": bool(query_embedding),
                "graph_embedding_stats": getattr(graph, "embedding_stats", {}),
            },
        )

    def execute_ir_cot_graph(
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
        max_iterations = min(max(1, router.advanced_config.ir_cot_max_iterations), 3)
        current_query = query_text
        accumulated_candidates: List[RetrievalCandidate] = []
        seen_block_ids: Set[str] = set()
        evidence_metadata: Dict[str, Any] = {"ir_cot_iterations": 0, "queries": []}

        for iteration in range(max_iterations):
            evidence_metadata["queries"].append(current_query)
            local_keywords = _extract_keywords(current_query)
            local_entities = router._extract_query_entities(current_query)
            if router.advanced_config.enable_graph_global:
                result = router._execute_graph_global(
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
                result = router._execute_graph_local(
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
            if not new_blocks or len(seen_block_ids) >= router.max_context_chunks:
                break
            current_query = router._generate_followup_query(query_text, accumulated_candidates)

        deduped = dedupe_candidates(accumulated_candidates)
        evidence_filtered, quality_metadata = router._grade_and_filter_candidates(
            deduped,
            query_text=query_text,
            keywords=keywords,
        )
        evidence_metadata.update(quality_metadata)
        reranked = router._reranker.rerank(query=query_text, candidates=evidence_filtered, top_k=router.max_context_chunks)
        chunks = [c.text for c in reranked if c.text]
        block_ids = router._collect_candidate_block_ids(reranked)
        return RetrievalResult(
            context_chunks=chunks[: router.max_context_chunks],
            source_block_ids=block_ids,
            strategy_used=RetrievalStrategy.IR_COT_GRAPH,
            query_embedding=query_embedding,
            evidence_metadata=evidence_metadata,
        )


__all__ = ["RetrievalRouteExecutor"]
