from __future__ import annotations

from sdr.apps.ai.retrieval.searchers.raptor import RAPTORSearcher
from sdr.apps.ai.tsd_processing.raptor import RAPTORNode, RAPTORTree


def _tree_with_low_similarity_node() -> RAPTORTree:
    node = RAPTORNode(
        node_id="level0_node0",
        level=0,
        text="The gateway logs every request to a central audit trail.",
        embedding=[1.0, 0.0],
        has_embedding=True,
        source_block_ids=["p1_b1"],
    )
    return RAPTORTree(document_name="Example TSD", levels=[[node]], total_nodes=1, max_level=0)


def test_search_relaxes_threshold_when_nothing_clears_it():
    searcher = RAPTORSearcher(min_cosine_similarity=0.6)
    tree = _tree_with_low_similarity_node()
    # Orthogonal query vector -> cosine similarity 0.0, well below the 0.6 floor.
    response = searcher.search(
        query_text="logging",
        tree=tree,
        precomputed_embedding=[0.0, 1.0],
    )
    assert response.error is None
    assert response.threshold_relaxed is True
    assert len(response.results) == 1
    assert response.results[0].node.node_id == "level0_node0"


def test_search_does_not_relax_when_threshold_is_met():
    searcher = RAPTORSearcher(min_cosine_similarity=0.6)
    tree = _tree_with_low_similarity_node()
    response = searcher.search(
        query_text="logging",
        tree=tree,
        precomputed_embedding=[1.0, 0.0],
    )
    assert response.threshold_relaxed is False
    assert len(response.results) == 1
