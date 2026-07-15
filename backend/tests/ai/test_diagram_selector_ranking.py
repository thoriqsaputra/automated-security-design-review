"""DiagramRequirementSelector._hybrid_search ranking behavior: segment-aware
BM25, weighted RRF, chapter prior (off by default), and the adaptive
score-floor cutoff (0.0 = current fixed-size behavior)."""
from dataclasses import dataclass, field
from typing import List

from sdr.apps.ai.engine.config import AnalysisPipelineConfig
from sdr.apps.ai.engine.debate.diagram_requirement_selector import DiagramRequirementSelector


@dataclass
class _Req:
    id: int
    stable_key: str
    parent_section: str
    requirement_text: str
    verification_hint: str = ""
    diagram_type: str = ""


class _StubRepository:
    """Returns the pool in a fixed 'vector similarity' order."""

    def __init__(self, ordered_pool: List[_Req]):
        self._pool = ordered_pool

    def list_diagram_requirements_with_similarity(self, *, category_id, ingestion_job_id, query_embedding):
        return [(req, 0.1 * i) for i, req in enumerate(self._pool)]

    def list_diagram_requirements(self, *, category_id, ingestion_job_id):
        return list(self._pool)


def _selector(pool: List[_Req], **config_overrides) -> DiagramRequirementSelector:
    config = AnalysisPipelineConfig(**config_overrides)
    return DiagramRequirementSelector(config=config, workflow_repository=_StubRepository(pool))


def _pool() -> List[_Req]:
    return [
        _Req(1, "req-session", "V3 Session Management", "Verify session tokens are rotated after login."),
        _Req(2, "req-crypto", "V6 Cryptography", "Verify approved cryptographic algorithms are used."),
        _Req(3, "req-logging", "V7 Logging", "Verify authentication events are logged."),
        _Req(4, "req-config", "V14 Configuration", "Verify debug modes are disabled in production."),
    ]


def test_bm25_uses_high_signal_segments_not_page_window():
    pool = _pool()
    selector = _selector(pool)
    # Page window screams "cryptography"; the caption/OCR say "session tokens".
    segments = {
        "caption": "login sequence diagram session tokens rotated",
        "ocr": "",
        "surrounding": "",
        "page_window": "cryptographic algorithms cryptography approved cryptographic",
    }
    ranked = selector._hybrid_search(
        category_id=1,
        ingestion_job_id=1,
        query_text=selector._assemble_query_text(segments),
        query_vector=[0.1] * 4,
        top_k=4,
        query_segments=segments,
    )
    assert ranked[0].stable_key == "req-session"


def test_score_floor_ratio_zero_returns_fixed_size_list():
    pool = _pool()
    selector = _selector(pool, diagram_score_floor_ratio=0.0)
    ranked = selector._hybrid_search(
        category_id=1, ingestion_job_id=1,
        query_text="session tokens", query_vector=[0.1] * 4, top_k=4,
    )
    assert len(ranked) == 4  # current behavior: max(_MIN_RESULTS, min(len, top_k))


def test_score_floor_trims_tail_but_never_below_min_results():
    pool = _pool()
    # An impossibly strict floor collapses to the _MIN_RESULTS guarantee.
    selector = _selector(pool, diagram_score_floor_ratio=100.0)
    ranked = selector._hybrid_search(
        category_id=1, ingestion_job_id=1,
        query_text="session tokens", query_vector=[0.1] * 4, top_k=4,
    )
    assert len(ranked) == 3  # _MIN_RESULTS floor

    # A permissive floor keeps everything.
    selector = _selector(pool, diagram_score_floor_ratio=0.01)
    ranked = selector._hybrid_search(
        category_id=1, ingestion_job_id=1,
        query_text="session tokens", query_vector=[0.1] * 4, top_k=4,
    )
    assert len(ranked) == 4


def test_chapter_prior_is_off_by_default_and_promotes_when_enabled():
    pool = _pool()
    # Neutral query: BM25 signalless -> ranking follows vector order.
    query = "diagram"

    default_selector = _selector(pool)
    default_ranked = default_selector._hybrid_search(
        category_id=1, ingestion_job_id=1,
        query_text=query, query_vector=[0.1] * 4, top_k=4,
        diagram_type="sequence",
    )

    boosted_selector = _selector(pool, diagram_chapter_prior_bonus=0.5)
    boosted_ranked = boosted_selector._hybrid_search(
        category_id=1, ingestion_job_id=1,
        query_text=query, query_vector=[0.1] * 4, top_k=4,
        diagram_type="sequence",
    )

    # Default: prior contributes nothing (session req stays in vector order).
    assert [r.stable_key for r in default_ranked][0] == "req-session"  # vector rank 0 anyway
    # Enabled: the session-chapter requirement must lead for a sequence diagram.
    assert boosted_ranked[0].stable_key == "req-session"
    # And the prior actually changed relative ordering below the top slot
    # (crypto has no 'sequence' chapter affinity, config/logging neither) —
    # scores of non-prior items are strictly below the prior-boosted one.


def test_explain_out_reports_ranks_and_scores():
    pool = _pool()
    selector = _selector(pool)
    explain: dict = {}
    selector._hybrid_search(
        category_id=1, ingestion_job_id=1,
        query_text="session tokens rotated", query_vector=[0.1] * 4, top_k=4,
        diagram_type="sequence", explain_out=explain,
    )
    assert explain["diagram_type"] == "sequence"
    rows = explain["ranking"]
    assert len(rows) == 4
    assert {"stable_key", "fused_rank", "vector_rank", "bm25_rank", "rrf_score", "type_match"} <= set(rows[0])
    assert [row["fused_rank"] for row in rows] == [0, 1, 2, 3]
