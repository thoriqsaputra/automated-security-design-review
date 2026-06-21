from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate
from sdr.apps.ai.retrieval.core.types import AdvancedRetrievalConfig
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.retrieval.searchers.graph import GraphSearchResponse, GraphSearchResult
from sdr.apps.ai.retrieval.searchers.raptor import RAPTORSearchResponse, RAPTORSearchResult
from sdr.apps.ai.retrieval.searchers.vector import VectorSearchResponse, VectorSearchResult
from sdr.apps.ai.tsd_processing.graph_builder import GraphEntity, TSDGraph
from sdr.apps.ai.tsd_processing.raptor import RAPTORNode


def _router() -> HybridRetrievalRouter:
    router = HybridRetrievalRouter.__new__(HybridRetrievalRouter)
    router.vector_top_k = 8
    router.raptor_top_k = 5
    router.graph_top_k = 6
    router.max_context_chunks = 12
    router.advanced_config = AdvancedRetrievalConfig()
    router._vector_searcher = SimpleNamespace()
    router._raptor_searcher = SimpleNamespace()
    router._graph_searcher = SimpleNamespace()
    router._keyword_searcher = SimpleNamespace()
    router._reranker = SimpleNamespace(rerank=lambda query, candidates, top_k: list(candidates))
    return router


def _graph_with_entity() -> TSDGraph:
    graph = TSDGraph(document_name="Example TSD", total_entities=1)
    graph.entities["api_gateway"] = GraphEntity(
        entity_id="api_gateway",
        name="API Gateway",
        entity_type="component",
        source_block_ids=["p1_b1"],
    )
    return graph


def _graph_response() -> GraphSearchResponse:
    entity = GraphEntity(
        entity_id="api_gateway",
        name="API Gateway",
        entity_type="component",
        source_block_ids=["p1_b1"],
    )
    return GraphSearchResponse(
        results=[
            GraphSearchResult(
                entity=entity,
                relevant_relations=[],
                relevance_score=0.9,
                source_block_ids=["p1_b1"],
            )
        ],
        graph_node_ids=["api_gateway"],
        graph_edge_ids=[],
        grounded_texts=[{"text": "API Gateway authenticates requests", "block_id": "p1_b1"}],
    )


def test_graph_local_survives_vector_searcher_exception():
    router = _router()
    graph = _graph_with_entity()

    router._graph_searcher.search_local = lambda **kwargs: _graph_response()
    router._vector_searcher.search = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("vector backend down"))
    router._keyword_searcher.search = lambda **kwargs: [
        RetrievalCandidate(id="bm25:1", source_type="keyword", text="API Gateway authenticates requests", score=0.5, block_ids=["p1_b1"])
    ]

    result = router._execute_graph_local(
        query_text="API Gateway auth",
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        raptor_tree=None,
        graph=graph,
        query_embedding=[0.1, 0.2],
        keywords=["auth"],
        query_entities=["api gateway"],
    )

    assert result.error is None
    assert result.vector_response.error == "vector backend down"
    assert result.context_chunks


def test_hybrid_survives_keyword_searcher_exception():
    router = _router()
    raptor_node = RAPTORNode(
        node_id="level0_node0",
        level=0,
        text="The service enforces authentication using OAuth and requires MFA for admin access.",
        source_block_ids=["p1_b1"],
    )
    router._vector_searcher.search = lambda **kwargs: VectorSearchResponse(
        results=[VectorSearchResult(child=SimpleNamespace(requirement_text="Use MFA", stable_key="c1"), cosine_distance=0.1)]
    )
    router._raptor_searcher.search_collapsed_raptor = lambda **kwargs: RAPTORSearchResponse(
        results=[RAPTORSearchResult(node=raptor_node, cosine_similarity=0.8, source_block_ids=["p1_b1"])]
    )
    router._keyword_searcher.search = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bm25 index corrupt"))

    result = router._execute_hybrid(
        query_text="Use MFA for admin access",
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        raptor_tree=SimpleNamespace(is_empty=lambda: False),
        graph=None,
        query_embedding=[0.1, 0.2],
        keywords=["mfa"],
        inferred_relations=set(),
    )

    assert result.error is None
    assert result.context_chunks


def test_all_branches_failing_in_hybrid_returns_gracefully_not_via_router_catch_all():
    router = _router()
    router._vector_searcher.search = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("vector down"))
    router._raptor_searcher.search_collapsed_raptor = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("raptor down"))
    router._keyword_searcher.search = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bm25 down"))

    result = router._execute_hybrid(
        query_text="Use MFA for admin access",
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        raptor_tree=None,
        graph=None,
        query_embedding=[0.1, 0.2],
        keywords=["mfa"],
        inferred_relations=set(),
    )

    assert result.error is None
    assert result.vector_response.error == "vector down"
    assert result.context_chunks == []
