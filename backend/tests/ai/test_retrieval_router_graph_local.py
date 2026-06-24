from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate
from sdr.apps.ai.retrieval.searchers.graph import (
    GraphSearchResponse,
    GraphSearchResult,
)
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.tsd_processing.graph_builder import GraphEntity, GraphRelation, TSDGraph


def _router() -> HybridRetrievalRouter:
    router = HybridRetrievalRouter.__new__(HybridRetrievalRouter)
    router.vector_top_k = 8
    router.raptor_top_k = 5
    router.graph_top_k = 6
    router.max_context_chunks = 12
    router.advanced_config = SimpleNamespace(graph_local_max_workers=3, hybrid_max_workers=3)
    router._vector_searcher = SimpleNamespace()
    router._raptor_searcher = SimpleNamespace()
    router._graph_searcher = SimpleNamespace()
    router._keyword_searcher = SimpleNamespace()
    router._reranker = SimpleNamespace(rerank=lambda query, candidates, top_k: list(candidates))
    return router


def test_execute_graph_local_returns_grounded_texts(monkeypatch):
    router = _router()
    graph = TSDGraph(document_name="Example TSD", total_entities=1)
    entity = GraphEntity(
        entity_id="api_gateway",
        name="API Gateway",
        entity_type="component",
        source_block_ids=["p1_b1"],
    )
    relation = GraphRelation(
        source_entity_id="api_gateway",
        target_entity_id="auth_service",
        relation_type="authenticates_with",
        source_block_ids=["p1_b1"],
    )
    graph.entities[entity.entity_id] = entity
    graph.total_entities = 1
    graph.embedding_stats = {"entity_succeeded": 1}

    graph_response = GraphSearchResponse(
        results=[
            GraphSearchResult(
                entity=entity,
                relevant_relations=[relation],
                relevance_score=0.92,
                source_block_ids=["p1_b1"],
            )
        ],
        graph_node_ids=["api_gateway"],
        graph_edge_ids=["api_gateway->auth_service"],
        grounded_texts=[{"text": "API Gateway authenticates requests", "block_id": "p1_b1"}],
    )
    bm25_candidates = [
        RetrievalCandidate(
            id="bm25:1",
            source_type="keyword",
            text="API Gateway authenticates requests",
            score=0.8,
            block_ids=["p1_b1"],
        )
    ]

    router._graph_searcher.search_local = lambda **kwargs: graph_response
    router._keyword_searcher.search = lambda **kwargs: bm25_candidates

    result = router._execute_graph_local(
        query_text="Use MFA for admin access",
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        raptor_tree=None,
        graph=graph,
        query_embedding=[0.1, 0.2],
        keywords=["mfa", "admin"],
        query_entities=["api gateway"],
    )

    assert result.grounded_texts == graph_response.grounded_texts
    assert result.graph_node_ids == ["api_gateway"]
    assert result.graph_edge_ids == ["api_gateway->auth_service"]
    assert result.error is None
