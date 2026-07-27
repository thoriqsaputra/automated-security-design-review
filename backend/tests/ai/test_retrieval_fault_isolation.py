from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate
from sdr.apps.ai.retrieval.core.types import AdvancedRetrievalConfig
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.retrieval.searchers.raptor import RAPTORSearchResponse, RAPTORSearchResult
from sdr.apps.ai.tsd_processing.raptor import RAPTORNode


def _router() -> HybridRetrievalRouter:
    router = HybridRetrievalRouter.__new__(HybridRetrievalRouter)
    router.vector_top_k = 8
    router.raptor_top_k = 5
    router.max_context_chunks = 12
    router.advanced_config = AdvancedRetrievalConfig()
    router._raptor_searcher = SimpleNamespace(
        search_multi_level=lambda **kwargs: RAPTORSearchResponse(results=[]),
    )
    router._keyword_searcher = SimpleNamespace()
    router._reranker = SimpleNamespace(
        rerank=lambda query, candidates, top_k, **kwargs: list(candidates)
    )
    return router


def test_hybrid_survives_keyword_searcher_exception():
    router = _router()
    raptor_node = RAPTORNode(
        node_id="level0_node0",
        level=0,
        text="The service enforces authentication using OAuth and requires MFA for admin access.",
        source_block_ids=["p1_b1"],
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
        query_embedding=[0.1, 0.2],
        keywords=["mfa"],
        query_variants=[],
    )

    assert result.error is None
    assert result.context_chunks


def test_hybrid_returns_empty_result_when_all_branches_fail():
    router = _router()
    router._raptor_searcher.search_collapsed_raptor = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("raptor down"))
    router._keyword_searcher.search = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bm25 down"))

    result = router._execute_hybrid(
        query_text="Use MFA for admin access",
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        raptor_tree=SimpleNamespace(is_empty=lambda: False),
        query_embedding=[0.1, 0.2],
        keywords=["mfa"],
        query_variants=[],
    )

    assert result.error is None
    assert result.context_chunks == []


def test_hybrid_keeps_keyword_evidence_when_dense_branch_fails():
    router = _router()
    router._raptor_searcher.search_collapsed_raptor = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("raptor dense down"))
    router._keyword_searcher.search = lambda **kwargs: [
        RetrievalCandidate(
            id="bm25:1",
            source_type="keyword",
            text="API Gateway authenticates requests and requires MFA for admin access.",
            score=0.6,
            block_ids=["p1_b1"],
        )
    ]

    result = router._execute_hybrid(
        query_text="Use MFA for admin access",
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        raptor_tree=SimpleNamespace(is_empty=lambda: False),
        query_embedding=[0.1, 0.2],
        keywords=["mfa"],
        query_variants=[],
    )

    assert result.error is None
    assert result.context_chunks
