from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.retrieval.searchers.vector import VectorSearcher, VectorSearchResult


def _searcher_with_results(results):
    searcher = VectorSearcher.__new__(VectorSearcher)
    searcher.embedding_dimensions = 2
    searcher.max_cosine_distance = 0.5
    searcher.default_top_k = 10
    import logging
    searcher.logger = logging.getLogger("test.vector_searcher")
    searcher._get_active_job = lambda category: SimpleNamespace(id=1)
    searcher._execute_search = lambda **kwargs: results
    return searcher


def test_search_relaxes_threshold_when_nothing_clears_it():
    far_result = VectorSearchResult(
        child=SimpleNamespace(stable_key="child-1"),
        cosine_distance=0.9,  # well above the 0.5 floor
    )
    searcher = _searcher_with_results([far_result])

    response = searcher.search(
        query_text="logging",
        category=SimpleNamespace(id=1, code="web_application"),
        precomputed_embedding=[1.0, 0.0],
    )

    assert response.error is None
    assert response.threshold_relaxed is True
    assert len(response.results) == 1
    assert response.results[0].child.stable_key == "child-1"


def test_search_does_not_relax_when_threshold_is_met():
    close_result = VectorSearchResult(
        child=SimpleNamespace(stable_key="child-1"),
        cosine_distance=0.1,
    )
    searcher = _searcher_with_results([close_result])

    response = searcher.search(
        query_text="logging",
        category=SimpleNamespace(id=1, code="web_application"),
        precomputed_embedding=[1.0, 0.0],
    )

    assert response.threshold_relaxed is False
    assert len(response.results) == 1
