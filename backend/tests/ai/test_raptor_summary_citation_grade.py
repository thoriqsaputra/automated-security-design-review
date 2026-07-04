from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.engine.debate.debate_input_factory import DebateInputFactory
from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate
from sdr.apps.ai.retrieval.core.types import AdvancedRetrievalConfig
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.retrieval.searchers.raptor import RAPTORSearchResponse, RAPTORSearchResult
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


def test_get_context_chunk_levels_is_index_aligned_with_mixed_levels():
    # A literal leaf (level 0) and a synthesized multi-page summary (level 2)
    # returned together — the levels list must line up 1:1 with the chunks
    # and block-id lists produced by the response's other accessors.
    leaf_node = RAPTORNode(
        node_id="level0_node5",
        level=0,
        text="DATABASE_PASSWORD is an AES 256 encrypted master password.",
        source_block_ids=["p21_b0"],
    )
    summary_node = RAPTORNode(
        node_id="level2_node1",
        level=2,
        text="Section 10 emphasizes that the DATABASE_PASSWORD is an AES 256 encrypted master password.",
        source_block_ids=["p8_b0", "p9_b1", "p21_b0"],
    )
    response = RAPTORSearchResponse(
        results=[
            RAPTORSearchResult(node=leaf_node, cosine_similarity=0.9, source_block_ids=["p21_b0"]),
            RAPTORSearchResult(node=summary_node, cosine_similarity=0.7, source_block_ids=["p8_b0", "p9_b1", "p21_b0"]),
        ]
    )

    chunks = response.get_context_chunks()
    block_ids = response.get_context_chunk_block_ids()
    levels = response.get_context_chunk_levels()

    assert len(chunks) == len(block_ids) == len(levels) == 2
    assert levels == [0, 2]
    assert block_ids[1] == ["p8_b0", "p9_b1", "p21_b0"]


def test_execute_raptor_high_populates_context_chunk_levels():
    router = _router()
    leaf_node = RAPTORNode(node_id="level0_node5", level=0, text="Literal page text.", source_block_ids=["p21_b0"])
    summary_node = RAPTORNode(node_id="level2_node1", level=2, text="Synthesized summary text.", source_block_ids=["p8_b0"])
    router._raptor_searcher.search_multi_level = lambda **kwargs: RAPTORSearchResponse(
        results=[
            RAPTORSearchResult(node=leaf_node, cosine_similarity=0.9, source_block_ids=["p21_b0"]),
            RAPTORSearchResult(node=summary_node, cosine_similarity=0.7, source_block_ids=["p8_b0"]),
        ]
    )

    result = router._execute_raptor_high(
        query_text="database password",
        raptor_tree=SimpleNamespace(is_empty=lambda: False),
        query_embedding=[0.1, 0.2],
    )

    assert result.context_chunk_levels == [0, 2]
    assert len(result.context_chunks) == len(result.context_chunk_levels)


def test_execute_hybrid_populates_context_chunk_levels_from_candidate_metadata():
    router = _router()
    raptor_node = RAPTORNode(node_id="level0_node0", level=0, text="Literal page text.", source_block_ids=["p1_b1"])
    router._raptor_searcher.search_collapsed_raptor = lambda **kwargs: RAPTORSearchResponse(
        results=[RAPTORSearchResult(node=raptor_node, cosine_similarity=0.8, source_block_ids=["p1_b1"])]
    )
    router._keyword_searcher.search = lambda **kwargs: [
        RetrievalCandidate(
            id="bm25:summary",
            source_type="keyword",
            text="Synthesized summary text mentioning MFA.",
            score=0.5,
            block_ids=["p8_b0"],
            metadata={"level": 2},
        )
    ]

    result = router._execute_hybrid(
        query_text="Use MFA for admin access",
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        raptor_tree=SimpleNamespace(is_empty=lambda: False),
        query_embedding=[0.1, 0.2],
        keywords=["mfa"],
    )

    assert len(result.context_chunks) == len(result.context_chunk_levels)
    assert 2 in result.context_chunk_levels
    assert 0 in result.context_chunk_levels


def test_build_context_chunk_map_marks_summary_node_as_non_citable():
    factory = DebateInputFactory()
    chunk_map = factory.build_context_chunk_map(
        ["Synthesized summary text mentioning the DATABASE_PASSWORD."],
        chunk_block_ids=[["p8_b0", "p9_b1"]],
        chunk_levels=[2],
    )

    entry = chunk_map["p8_b0"]
    assert entry["citation_grade"] is False
    assert entry["evidence_kind"] == "hierarchical_summary"


def test_build_context_chunk_map_keeps_literal_leaf_citable():
    factory = DebateInputFactory()
    chunk_map = factory.build_context_chunk_map(
        ["The DATABASE_PASSWORD is an AES 256 encrypted master password."],
        chunk_block_ids=[["p21_b0"]],
        chunk_levels=[0],
    )

    entry = chunk_map["p21_b0"]
    assert entry["citation_grade"] is True
    assert entry["evidence_kind"] != "hierarchical_summary"
