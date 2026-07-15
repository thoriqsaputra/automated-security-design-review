"""Leaf-grounding: a hierarchical_summary candidate surviving to the final
context gets its top-N highest-cosine literal leaf descendants inserted too,
fixing the coverage=1/recall=0 failure mode (a judge never sees the summary's
paraphrase as evidence for the literal blocks it claims to cover) — a summary
can union many distinct blocks, so a single leaf isn't always enough."""
from unittest.mock import MagicMock

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate
from sdr.apps.ai.retrieval.routing.executors import _ground_summaries_with_leaves


def _leaf_node(node_id: str, text: str, embedding, block_ids):
    node = MagicMock()
    node.node_id = node_id
    node.level = 0
    node.text = text
    node.embedding = embedding
    node.has_embedding = True
    node.source_block_ids = block_ids
    node.page_numbers = [1]
    node.section_heading = None
    node.token_estimate = len(text) // 4
    node.children = []
    return node


def _summary_node(node_id: str, children):
    node = MagicMock()
    node.node_id = node_id
    node.level = 1
    node.children = children
    return node


def _summary_candidate(node_id: str, block_ids) -> RetrievalCandidate:
    return RetrievalCandidate(
        id=node_id,
        source_type="raptor",
        text="An LLM-synthesized summary spanning several sections.",
        score=0.9,
        block_ids=block_ids,
        metadata={"level": 1, "evidence_kind": "hierarchical_summary"},
    )


def _router_stub():
    router = MagicMock()

    def classify(candidate, *, keywords):
        text = candidate.text or ""
        if not text:
            return "empty", "empty"
        return "implementation_evidence", "TSD block contains implementation/security terms"

    router._classify_candidate_evidence.side_effect = classify
    return router


def test_summary_gets_grounded_with_its_best_leaf_by_default():
    leaf_a = _leaf_node("leaf_a", "The service enforces MFA for admins.", [1.0, 0.0], ["p1_b1"])
    leaf_b = _leaf_node("leaf_b", "Unrelated boilerplate text.", [0.0, 1.0], ["p1_b2"])
    summary = _summary_node("summary_1", [leaf_a, leaf_b])

    tree = MagicMock()
    tree.is_empty.return_value = False
    tree.get_all_nodes.return_value = [summary, leaf_a, leaf_b]

    candidates = [_summary_candidate("summary_1", ["p1_b1", "p1_b2"])]
    result = _ground_summaries_with_leaves(
        _router_stub(), candidates,
        raptor_tree=tree, query_embedding=[1.0, 0.0], keywords=["mfa"],
        max_context_chunks=16,
    )

    ids = [c.id for c in result]
    assert "summary_1" in ids
    assert "leaf_a" in ids  # higher cosine with query embedding
    assert "leaf_b" not in ids  # default leaves_per_summary=1: budget-conscious
    grounded = next(c for c in result if c.id == "leaf_a")
    assert grounded.metadata["leaf_grounded_for"] == "summary_1"
    assert grounded.metadata["level"] == 0


def test_leaves_per_summary_is_configurable():
    leaf_a = _leaf_node("leaf_a", "The service enforces MFA for admins.", [1.0, 0.0], ["p1_b1"])
    leaf_b = _leaf_node("leaf_b", "Related session timeout policy.", [0.9, 0.1], ["p1_b2"])
    summary = _summary_node("summary_1", [leaf_a, leaf_b])

    tree = MagicMock()
    tree.is_empty.return_value = False
    tree.get_all_nodes.return_value = [summary, leaf_a, leaf_b]

    candidates = [_summary_candidate("summary_1", ["p1_b1", "p1_b2"])]
    result = _ground_summaries_with_leaves(
        _router_stub(), candidates,
        raptor_tree=tree, query_embedding=[1.0, 0.0], keywords=["mfa"],
        max_context_chunks=16, leaves_per_summary=2,
    )

    ids = [c.id for c in result]
    assert "leaf_a" in ids
    assert "leaf_b" in ids


def test_already_present_leaf_is_not_duplicated():
    leaf_a = _leaf_node("leaf_a", "The service enforces MFA.", [1.0, 0.0], ["p1_b1"])
    summary = _summary_node("summary_1", [leaf_a])
    tree = MagicMock()
    tree.is_empty.return_value = False
    tree.get_all_nodes.return_value = [summary, leaf_a]

    existing_leaf = RetrievalCandidate(
        id="leaf_a", source_type="bm25", text="already here", score=0.5,
        block_ids=["p1_b1"], metadata={"level": 0, "evidence_kind": "implementation_evidence"},
    )
    candidates = [_summary_candidate("summary_1", ["p1_b1"]), existing_leaf]

    result = _ground_summaries_with_leaves(
        _router_stub(), candidates,
        raptor_tree=tree, query_embedding=[1.0, 0.0], keywords=[],
        max_context_chunks=16,
    )

    assert len(result) == 2  # no duplicate leaf_a added
    assert [c.id for c in result] == ["summary_1", "leaf_a"]


def test_no_summaries_returns_input_unchanged():
    tree = MagicMock()
    tree.is_empty.return_value = False
    candidates = [
        RetrievalCandidate(id="leaf", source_type="bm25", text="x", score=0.5,
                            metadata={"level": 0, "evidence_kind": "implementation_evidence"})
    ]
    result = _ground_summaries_with_leaves(
        _router_stub(), candidates, raptor_tree=tree, query_embedding=[1.0], keywords=[],
        max_context_chunks=16,
    )
    assert result is candidates


def test_grader_rejected_leaf_is_not_grounded():
    empty_leaf = _leaf_node("empty_leaf", "", [1.0, 0.0], ["p1_b1"])
    summary = _summary_node("summary_1", [empty_leaf])
    tree = MagicMock()
    tree.is_empty.return_value = False
    tree.get_all_nodes.return_value = [summary, empty_leaf]

    candidates = [_summary_candidate("summary_1", ["p1_b1"])]
    result = _ground_summaries_with_leaves(
        _router_stub(), candidates, raptor_tree=tree, query_embedding=[1.0, 0.0], keywords=[],
        max_context_chunks=16,
    )
    assert [c.id for c in result] == ["summary_1"]  # empty leaf rejected, not inserted


def test_budget_eviction_never_evicts_protected_or_grounded():
    leaf_a = _leaf_node("leaf_a", "The service enforces MFA.", [1.0, 0.0], ["p1_b1"])
    summary = _summary_node("summary_1", [leaf_a])
    tree = MagicMock()
    tree.is_empty.return_value = False
    tree.get_all_nodes.return_value = [summary, leaf_a]

    filler = [
        RetrievalCandidate(id=f"filler_{i}", source_type="bm25", text=f"filler {i}", score=1.0 - i * 0.01,
                            metadata={"level": 0, "evidence_kind": "implementation_evidence"})
        for i in range(16)
    ]
    protected_filler = filler[-1]
    protected_filler.metadata["protected_slot_source"] = "dense"
    candidates = [_summary_candidate("summary_1", ["p1_b1"])] + filler

    result = _ground_summaries_with_leaves(
        _router_stub(), candidates, raptor_tree=tree, query_embedding=[1.0, 0.0], keywords=[],
        max_context_chunks=16,
    )

    assert len(result) == 16
    ids = [c.id for c in result]
    assert "leaf_a" in ids
    assert "summary_1" in ids
    assert protected_filler.id in ids  # protected item survives eviction
