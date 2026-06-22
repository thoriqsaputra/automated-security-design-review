from types import SimpleNamespace

from sdr.apps.ai.engine.classification.parent_applicability import (
    ParentApplicabilityResult,
    _extract_scope_terms,
)
from sdr.apps.ai.engine.config import AnalysisPipelineConfig
from sdr.apps.ai.engine.debate.text_debate_coordinator import TextDebateCoordinator
from sdr.apps.ai.engine.dto import AnalysisSummary
from sdr.apps.ai.retrieval.core.types import RetrievalResult


def _coordinator(config: AnalysisPipelineConfig, retrieval_service) -> TextDebateCoordinator:
    return TextDebateCoordinator(
        config=config,
        retrieval_service=retrieval_service,
        debate_service=SimpleNamespace(),
        persistence_service=SimpleNamespace(),
        debate_input_factory=SimpleNamespace(),
        progress_service=SimpleNamespace(),
        run_state_service=SimpleNamespace(raise_if_cancelled=lambda *a, **k: None),
        mediator_agent_factory=SimpleNamespace(),
    )


def _parent(parent_id: int, title: str = "Session Controls", description: str = "Browser session requirements"):
    return SimpleNamespace(id=parent_id, title=title, description=description)


def _child(child_id: int, parent, text: str):
    return SimpleNamespace(
        id=child_id,
        parent=parent,
        requirement_text=text,
        requirement_text_normalized=text.lower(),
        details="",
    )


class _FakeRetrievalService:
    """Stubs the subset of RetrievalService used by the gate, with a configurable
    sequence of classify_parent_applicability outcomes driven via context_chunks."""

    def __init__(self, *, narrow_chunks, expanded_chunks, retry_max_chunks=14):
        self.narrow_chunks = narrow_chunks
        self.expanded_chunks = expanded_chunks
        self.retry_max_chunks = retry_max_chunks
        self.retrieve_for_parent_group_calls = []

    def get_retrieve_many_max_concurrency(self):
        return 1

    def get_parent_retrieval_retry_max_context_chunks(self):
        return self.retry_max_chunks

    def retrieve_for_parent_group(self, *, max_context_chunks_override=None, **kwargs):
        self.retrieve_for_parent_group_calls.append(max_context_chunks_override)
        if max_context_chunks_override == self.retry_max_chunks:
            return RetrievalResult(context_chunks=self.expanded_chunks, source_block_ids=["p9_b1"])
        return RetrievalResult(context_chunks=self.narrow_chunks, source_block_ids=["p1_b1"])


def test_mid_confidence_not_applicable_triggers_retry_then_excludes(monkeypatch):
    config = AnalysisPipelineConfig(
        parent_applicability_enabled=True,
        parent_applicability_confidence_threshold=0.7,
        parent_applicability_retry_confidence_floor=0.35,
        parent_applicability_fallback_mode="assume_applicable",
    )
    retrieval = _FakeRetrievalService(
        narrow_chunks=["sparse, ambiguous context"],
        expanded_chunks=["The TSD explicitly states there is no browser session handling anywhere."],
    )
    coordinator = _coordinator(config, retrieval)
    parent = _parent(1)
    children = [_child(1, parent, "Use secure cookie flags.")]
    summary = AnalysisSummary()

    call_log = []

    def fake_classify(**kwargs):
        chunks = kwargs["retrieved_context"]
        call_log.append(chunks)
        if "no browser session handling" in chunks:
            return ParentApplicabilityResult(
                applicable=False,
                confidence=0.88,
                reasoning="TSD explicitly rules out browser sessions.",
                evidence=["no browser session handling"],
                decision_mode="negative_match",
            )
        return ParentApplicabilityResult(
            applicable=False,
            confidence=0.45,
            reasoning="Ambiguous.",
            evidence=[],
            decision_mode="unclear",
        )

    monkeypatch.setattr(
        "sdr.apps.ai.engine.debate.text_debate_coordinator.classify_parent_applicability",
        fake_classify,
    )
    monkeypatch.setattr(
        coordinator,
        "persist_parent_not_applicable_children",
        lambda **kwargs: None,
    )

    applicable_parameters, _ = coordinator.apply_parent_applicability_gate(
        review=SimpleNamespace(id=1),
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1, version_no=1),
        parameters=children,
        indexes=SimpleNamespace(),
        tsd_document=SimpleNamespace(),
        summary=summary,
    )

    assert len(call_log) == 2, "expected a first pass and one retry classification call"
    assert applicable_parameters == []
    assert summary.applicability["parents_not_applicable"] == 1
    assert summary.applicability["parents"][0]["retry_attempted"] is True
    assert retrieval.retrieve_for_parent_group_calls == [None, 14]


def test_very_low_confidence_fails_open_without_retry(monkeypatch):
    config = AnalysisPipelineConfig(
        parent_applicability_enabled=True,
        parent_applicability_confidence_threshold=0.7,
        parent_applicability_retry_confidence_floor=0.35,
        parent_applicability_fallback_mode="assume_applicable",
    )
    retrieval = _FakeRetrievalService(narrow_chunks=[], expanded_chunks=["should not be reached"])
    coordinator = _coordinator(config, retrieval)
    parent = _parent(2)
    children = [_child(2, parent, "Use secure cookie flags.")]
    summary = AnalysisSummary()

    def fake_classify(**kwargs):
        return ParentApplicabilityResult(
            applicable=True,
            confidence=0.0,
            reasoning="No retrieved TSD context was available, so applicability could not be established. Failing open.",
            evidence=[],
            decision_mode="missing_context",
            error="missing_retrieved_context",
        )

    monkeypatch.setattr(
        "sdr.apps.ai.engine.debate.text_debate_coordinator.classify_parent_applicability",
        fake_classify,
    )

    applicable_parameters, _ = coordinator.apply_parent_applicability_gate(
        review=SimpleNamespace(id=2),
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1, version_no=1),
        parameters=children,
        indexes=SimpleNamespace(),
        tsd_document=SimpleNamespace(),
        summary=summary,
    )

    assert applicable_parameters == children
    assert retrieval.retrieve_for_parent_group_calls == [None]
    assert summary.applicability["parents"][0]["retry_attempted"] is False


def test_retry_skipped_when_retrieval_service_lacks_capability(monkeypatch):
    config = AnalysisPipelineConfig(
        parent_applicability_enabled=True,
        parent_applicability_confidence_threshold=0.7,
        parent_applicability_retry_confidence_floor=0.35,
        parent_applicability_fallback_mode="assume_applicable",
    )
    bare_retrieval = SimpleNamespace(get_retrieve_many_max_concurrency=lambda: 1)
    coordinator = _coordinator(config, bare_retrieval)
    parent = _parent(3)
    children = [_child(3, parent, "Use secure cookie flags.")]
    summary = AnalysisSummary()

    monkeypatch.setattr(
        coordinator,
        "get_parent_retrieval_result",
        lambda **kwargs: RetrievalResult(context_chunks=["sparse context"], source_block_ids=["p1_b1"]),
    )
    monkeypatch.setattr(
        "sdr.apps.ai.engine.debate.text_debate_coordinator.classify_parent_applicability",
        lambda **kwargs: ParentApplicabilityResult(
            applicable=False,
            confidence=0.5,
            reasoning="Ambiguous.",
            evidence=[],
            decision_mode="unclear",
        ),
    )

    applicable_parameters, _ = coordinator.apply_parent_applicability_gate(
        review=SimpleNamespace(id=3),
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1, version_no=1),
        parameters=children,
        indexes=SimpleNamespace(),
        tsd_document=SimpleNamespace(),
        summary=summary,
    )

    # No retrieve_for_parent_group on the stub -> retry must be skipped gracefully,
    # falling through to the existing fallback_mode behavior instead of raising.
    assert applicable_parameters == children
    assert summary.applicability["parents"][0]["retry_attempted"] is False


def test_extract_scope_terms_uses_all_children_not_just_first_four():
    child_requirements = [f"requirement about topic_{i}" for i in range(6)]
    terms = _extract_scope_terms(
        parent_title="Parent",
        parent_description="Description",
        child_requirements=child_requirements,
        query_details=None,
    )
    assert "topic_4" in terms
    assert "topic_5" in terms
