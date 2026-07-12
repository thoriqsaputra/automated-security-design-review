from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate
from sdr.apps.ai.retrieval.core.types import AdvancedRetrievalConfig
from sdr.apps.ai.retrieval.routing.executors import _grade_with_secondary_search
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter


def _router() -> HybridRetrievalRouter:
    router = HybridRetrievalRouter.__new__(HybridRetrievalRouter)
    router.vector_top_k = 8
    router.raptor_top_k = 5
    router.max_context_chunks = 12
    router.advanced_config = AdvancedRetrievalConfig()
    router._generate_query_embedding = lambda text: [0.1, 0.2]
    return router


# A weak-context candidate: relevant (mentions "attribute"), has block_ids, but
# uses none of EvidenceGrader._IMPLEMENTATION_TERMS, so it grades "weak_context"
# with implementation_evidence_count == 0 on the first pass.
_WEAK_TEXT = (
    "The system defines role based and attribute based access models that "
    "describe how client attributes and permissions are organized for "
    "authorization decisions."
)

# An implementation-grade candidate (mentions session/validated/authenticated)
# that the secondary BM25 round should surface. Must be >= 120 chars to clear
# EvidenceGrader's heading_only length gate before the implementation-term check runs.
_STRONG_TEXT = (
    "Session tokens are validated server-side and authenticated for every "
    "incoming request to ensure attribute integrity and to prevent end-user "
    "tampering with stored values."
)


def test_secondary_search_triggers_when_no_implementation_evidence_found():
    router = _router()
    calls = {"search": 0}

    def fake_search(**kwargs):
        calls["search"] += 1
        return [
            RetrievalCandidate(
                id="bm25:extra",
                source_type="keyword",
                text=_STRONG_TEXT,
                score=0.6,
                block_ids=["b2"],
            )
        ]

    router._keyword_searcher = SimpleNamespace(search=fake_search)

    weak_candidate = RetrievalCandidate(
        id="dense:1",
        source_type="dense",
        text=_WEAK_TEXT,
        score=0.5,
        block_ids=["b1"],
    )

    evidence_filtered, evidence_metadata = _grade_with_secondary_search(
        router,
        [weak_candidate],
        query_text="Verify that policy attributes used in authorization decisions cannot be manipulated by end users.",
        keywords=["attribute"],
        raptor_tree=None,
        has_raptor=False,
    )

    assert calls["search"] == 1
    assert evidence_metadata["secondary_search_triggered"] is True
    assert evidence_metadata.get("secondary_search_query")
    assert any(c.id == "bm25:extra" for c in evidence_filtered)
    assert evidence_metadata["evidence_quality"]["implementation_evidence_count"] >= 1


def test_secondary_search_skipped_when_implementation_evidence_already_present():
    router = _router()
    calls = {"search": 0}

    def fake_search(**kwargs):
        calls["search"] += 1
        return []

    router._keyword_searcher = SimpleNamespace(search=fake_search)

    strong_candidate = RetrievalCandidate(
        id="dense:1",
        source_type="dense",
        text=_STRONG_TEXT,
        score=0.5,
        block_ids=["b1"],
    )

    evidence_filtered, evidence_metadata = _grade_with_secondary_search(
        router,
        [strong_candidate],
        query_text="Verify that session integrity is enforced.",
        keywords=["session"],
        raptor_tree=None,
        has_raptor=False,
    )

    assert calls["search"] == 0
    assert evidence_metadata["secondary_search_triggered"] is False
    assert any(c.id == "dense:1" for c in evidence_filtered)
