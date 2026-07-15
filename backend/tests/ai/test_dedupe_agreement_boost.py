from __future__ import annotations

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate, dedupe_candidates


def _candidate(id_: str, source_type: str, score: float, block_id: str = "p1_b1") -> RetrievalCandidate:
    return RetrievalCandidate(
        id=id_,
        source_type=source_type,
        text="The gateway enforces TLS for all inter-service traffic.",
        score=score,
        block_ids=[block_id],
    )


def test_single_source_candidate_is_not_boosted():
    candidates = [_candidate("only", "bm25", 0.5)]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 1
    assert deduped[0].score == 0.5
    assert "agreement_count" not in deduped[0].metadata
    assert "agreement_boost" not in deduped[0].metadata


# NOTE: dedupe keys on the candidate's stable node id (see _key_for_candidate) —
# searchers that find the same RAPTOR node emit the SAME id with a different
# source_type, which is what these fixtures model. Distinct ids never merge,
# even with identical text/block_ids (a leaf and a summary can share a first
# block id; merging by block/text would leak the summary's breadth onto the leaf).


def test_two_independent_sources_boost_score_above_either_raw_score():
    candidates = [
        _candidate("node_1", "bm25", 0.5),
        _candidate("node_1", "dense", 0.6),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 1
    merged = deduped[0]
    assert merged.metadata["agreement_count"] == 2
    assert merged.score > 0.6
    assert merged.score == 0.6 + 0.08


def test_three_or_more_sources_hit_the_boost_cap():
    candidates = [
        _candidate("node_1", "bm25", 0.5),
        _candidate("node_1", "dense", 0.5),
        _candidate("node_1", "keyword", 0.5),
        _candidate("node_1", "raptor", 0.5),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 1
    merged = deduped[0]
    assert merged.metadata["agreement_count"] == 4
    # 0.08 * 3 = 0.24, exactly at the cap
    assert merged.metadata["agreement_boost"] == 0.24
    assert merged.score == 0.5 + 0.24


def test_boost_is_not_double_counted_across_repeated_merge_steps():
    # Same source_type repeated should not inflate agreement_count beyond
    # the number of *distinct* source types.
    candidates = [
        _candidate("node_1", "bm25", 0.4),
        _candidate("node_1", "bm25", 0.45),
        _candidate("node_1", "dense", 0.5),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 1
    merged = deduped[0]
    assert merged.metadata["agreement_count"] == 2
    assert merged.score == 0.5 + 0.08


def test_distinct_node_ids_never_merge_even_with_identical_text_and_blocks():
    candidates = [
        _candidate("leaf_node", "bm25", 0.5),
        _candidate("summary_node", "dense", 0.6),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 2
    assert all("agreement_count" not in c.metadata for c in deduped)
