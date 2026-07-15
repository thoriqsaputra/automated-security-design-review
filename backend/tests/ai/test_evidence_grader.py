from __future__ import annotations

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate
from sdr.apps.ai.retrieval.postprocessing.evidence_grader import EvidenceGrader


def _impl_candidate(id_: str, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        id=id_,
        source_type="bm25",
        text=(
            "The service enforces authentication using OAuth and requires MFA "
            "for all administrative endpoints before granting access."
        ),
        score=score,
        block_ids=["p1_b1"],
    )


def _fallback_candidate(id_: str, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        id=id_,
        source_type="raptor",
        text="This section briefly mentions the system without implementation detail words present here.",
        score=score,
        block_ids=["p1_b2"],
    )


def test_high_score_implementation_candidate_wins_despite_late_arrival():
    grader = EvidenceGrader(max_context_chunks=10)
    candidates = [
        _impl_candidate("low", 0.2),
        _impl_candidate("high", 0.95),
    ]
    selected, _ = grader.grade_and_filter_candidates(candidates, query_text="mfa", keywords=["mfa"])
    assert selected[0].id == "high"
    assert selected[1].id == "low"


def test_implementation_tier_outranks_higher_scoring_fallback():
    grader = EvidenceGrader(max_context_chunks=10)
    candidates = [
        _fallback_candidate("fallback_high", 0.99),
        _impl_candidate("impl_low", 0.3),
    ]
    selected, _ = grader.grade_and_filter_candidates(candidates, query_text="mfa", keywords=["mfa"])
    assert selected[0].id == "impl_low"
    assert selected[1].id == "fallback_high"


def test_grader_orders_tiers_but_never_truncates():
    # The grader deliberately returns the FULL tier-ordered pool — truncation
    # happens downstream in _rerank_within_tiers, after the cross-encoder has
    # seen every tier member, so a genuinely better candidate beyond an early
    # per-tier cutoff still gets a chance to surface.
    grader = EvidenceGrader(max_context_chunks=2)
    candidates = [
        _fallback_candidate("fallback_low_first", 0.1),
        _fallback_candidate("fallback_high_last", 0.9),
        _impl_candidate("impl_only", 0.5),
    ]
    selected, _ = grader.grade_and_filter_candidates(candidates, query_text="mfa", keywords=["mfa"])
    assert [c.id for c in selected] == ["impl_only", "fallback_high_last", "fallback_low_first"]
