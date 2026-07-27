"""Recall-safety floor: protected single-signal slots in execute_hybrid.

A hybrid ensemble must never return a context strictly worse than any of its
constituent signals — the top-N of each signal's primary-query ranking is
guaranteed a final-context slot unless the evidence grader legitimately
rejected it (baseline_requirement/empty).
"""
from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate, _key_for_candidate
from sdr.apps.ai.retrieval.routing.executors import (
    _enforce_protected_slots,
    _protected_candidate_keys,
)


def _candidate(cid: str, score: float, kind: str = "implementation_evidence", text: str = "The service enforces MFA.") -> RetrievalCandidate:
    return RetrievalCandidate(
        id=cid,
        source_type="dense",
        text=text,
        score=score,
        block_ids=[f"p1_{cid}"],
        metadata={"evidence_kind": kind, "level": 0},
    )


def test_protected_keys_take_top_n_per_signal_in_priority_order():
    dense = [_candidate(f"d{i}", 1.0 - i * 0.1) for i in range(5)]
    bm25 = [_candidate(f"b{i}", 1.0 - i * 0.1) for i in range(5)]
    raptor = [_candidate(f"r{i}", 1.0 - i * 0.1) for i in range(5)]

    protected = _protected_candidate_keys(
        primary_dense=dense, primary_bm25=bm25, raptor_multi=raptor,
        dense_n=3, bm25_n=2, raptor_n=2,
    )

    assert set(protected.values()) == {"dense", "bm25", "raptor"}
    assert [k.removeprefix("id:") for k in protected] == ["d0", "d1", "d2", "b0", "b1", "r0", "r1"]


def test_zero_top_n_disables_the_floor_entirely():
    dense = [_candidate("d0", 1.0)]
    protected = _protected_candidate_keys(
        primary_dense=dense, primary_bm25=[], raptor_multi=[],
        dense_n=0, bm25_n=0, raptor_n=0,
    )
    assert protected == {}

    reranked = [_candidate("x", 0.9)]
    result = _enforce_protected_slots(reranked, [dense[0], reranked[0]], protected, max_context_chunks=16)
    assert result is reranked  # byte-identical current behavior


def test_shared_candidate_across_signals_keeps_first_signal_label():
    shared = _candidate("shared", 1.0)
    protected = _protected_candidate_keys(
        primary_dense=[shared], primary_bm25=[shared], raptor_multi=[],
        dense_n=1, bm25_n=1, raptor_n=0,
    )
    assert protected == {"id:shared": "dense"}


def test_missing_protected_candidate_is_appended_at_tail():
    gold = _candidate("gold", 0.2)
    reranked = [_candidate(f"c{i}", 1.0 - i * 0.01) for i in range(5)]
    graded_pool = reranked + [gold]
    protected = {"id:gold": "dense"}

    result = _enforce_protected_slots(reranked, graded_pool, protected, max_context_chunks=16)

    assert [c.id for c in result[:5]] == [f"c{i}" for i in range(5)]  # head untouched
    assert result[-1].id == "gold"
    assert result[-1].metadata["protected_slot_source"] == "dense"


def test_tail_eviction_respects_budget_and_never_evicts_protected():
    reranked = [_candidate(f"c{i}", 1.0 - i * 0.01) for i in range(16)]
    gold = _candidate("gold", 0.1)
    graded_pool = reranked + [gold]
    protected = {"id:gold": "bm25", _key_for_candidate(reranked[15]): "raptor"}

    result = _enforce_protected_slots(reranked, graded_pool, protected, max_context_chunks=16)

    assert len(result) == 16
    ids = [c.id for c in result]
    assert "gold" in ids
    assert "c15" in ids  # protected in-list survivor not evicted
    assert "c14" not in ids  # lowest-ranked NON-protected evicted instead


def test_grader_rejected_protected_candidate_stays_excluded():
    rejected = _candidate("rejected", 1.0, kind="baseline_requirement")
    reranked = [_candidate("kept", 0.9)]
    graded_pool = [reranked[0]]  # 'rejected' absent — grader filtered it
    protected = {"id:rejected": "dense"}

    result = _enforce_protected_slots(reranked, graded_pool, protected, max_context_chunks=16)

    assert [c.id for c in result] == ["kept"]


def test_already_present_protected_candidate_is_not_duplicated():
    gold = _candidate("gold", 0.9)
    reranked = [gold, _candidate("other", 0.8)]
    protected = {"id:gold": "dense"}

    result = _enforce_protected_slots(reranked, list(reranked), protected, max_context_chunks=16)

    assert [c.id for c in result] == ["gold", "other"]
    assert gold.metadata["protected_slot_source"] == "dense"
