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


def test_two_independent_sources_boost_score_above_either_raw_score():
    candidates = [
        _candidate("bm25_hit", "bm25", 0.5),
        _candidate("dense_hit", "dense", 0.6),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 1
    merged = deduped[0]
    assert merged.metadata["agreement_count"] == 2
    assert merged.score > 0.6
    assert merged.score == 0.6 + 0.08


def test_three_or_more_sources_hit_the_boost_cap():
    candidates = [
        _candidate("bm25_hit", "bm25", 0.5),
        _candidate("dense_hit", "dense", 0.5),
        _candidate("keyword_hit", "keyword", 0.5),
        _candidate("raptor_hit", "raptor", 0.5),
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
        _candidate("bm25_a", "bm25", 0.4),
        _candidate("bm25_b", "bm25", 0.45),
        _candidate("dense_a", "dense", 0.5),
    ]
    deduped = dedupe_candidates(candidates)
    assert len(deduped) == 1
    merged = deduped[0]
    assert merged.metadata["agreement_count"] == 2
    assert merged.score == 0.5 + 0.08
