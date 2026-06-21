from __future__ import annotations

from sdr.apps.ai.retrieval.searchers.keyword import KeywordSearcher
from sdr.apps.ai.tsd_processing.raptor import RAPTORNode, RAPTORTree


def _tree(texts: list[str]) -> RAPTORTree:
    nodes = [
        RAPTORNode(node_id=f"level0_node{i}", level=0, text=text, source_block_ids=[f"b{i}"])
        for i, text in enumerate(texts)
    ]
    return RAPTORTree(document_name="test", levels=[nodes], total_nodes=len(nodes))


def test_scores_are_normalized_into_unit_range():
    tree = _tree(
        [
            "Authentication tokens are validated and authenticated on every request to the gateway.",
            "The weather today is unrelated to authentication or tokens whatsoever.",
            "Authentication authentication authentication tokens tokens tokens validated everywhere repeatedly.",
        ]
    )
    results = KeywordSearcher().search(query_text="authentication tokens validated", tree=tree, top_k=10)
    assert results
    for candidate in results:
        assert 0.0 <= candidate.score <= 1.0


def test_single_result_does_not_raise_and_yields_degenerate_score():
    tree = _tree(["Authentication tokens are validated on every request."])
    results = KeywordSearcher().search(query_text="authentication tokens", tree=tree, top_k=10)
    assert len(results) == 1
    assert results[0].score == 0.0


def test_all_equal_scores_do_not_divide_by_zero():
    tree = _tree(
        [
            "This text shares no overlapping vocabulary with the search terms at all.",
            "Neither does this completely unrelated passage about something else entirely.",
        ]
    )
    results = KeywordSearcher().search(query_text="nonexistent missing absent", tree=tree, top_k=10)
    assert len(results) == 2
    for candidate in results:
        assert candidate.score == 0.0
