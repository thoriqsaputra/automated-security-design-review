from __future__ import annotations

import logging
from collections import deque
from types import SimpleNamespace

from sdr.apps.ai.engine.debate.category_analysis_coordinator import CategoryAnalysisCoordinator
from sdr.apps.ai.engine.debate.text_debate_coordinator import TextDebateCoordinator
from sdr.apps.ai.engine.dto import AnalysisSummary
from sdr.apps.ai.engine.preparation.retrieval_service import RetrievalService
from sdr.apps.ai.retrieval.core import AdvancedRetrievalConfig, RetrievalResult
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.core.config import settings


class _ImmediateFuture:
    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def result(self):
        return self._fn(*self._args, **self._kwargs)


class _ImmediateExecutor:
    seen_max_workers: list[int] = []

    def __init__(self, *, max_workers, thread_name_prefix):
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.__class__.seen_max_workers.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        return _ImmediateFuture(fn, *args, **kwargs)


def _parameter(parameter_id: int, *, stable_key: str | None = None, parent=None, text: str | None = None):
    return SimpleNamespace(
        id=parameter_id,
        stable_key=stable_key or f"child-{parameter_id}",
        parent=parent,
        requirement_text=text or f"Requirement {parameter_id}",
        requirement_text_normalized=text or f"Requirement {parameter_id}",
        details="",
        ordinal=parameter_id,
    )


def test_advanced_retrieval_config_reads_concurrency_settings(monkeypatch):
    monkeypatch.setattr(settings, "AI_RETRIEVAL_HYBRID_MAX_WORKERS", 5, raising=False)
    monkeypatch.setattr(settings, "AI_RETRIEVAL_GRAPH_LOCAL_MAX_WORKERS", 4, raising=False)
    monkeypatch.setattr(settings, "AI_RETRIEVAL_MANY_MAX_CONCURRENCY", 0, raising=False)

    config = AdvancedRetrievalConfig.from_settings()

    assert config.hybrid_max_workers == 5
    assert config.graph_local_max_workers == 4
    assert config.retrieve_many_max_concurrency == 1


def test_hybrid_executor_caps_worker_count_to_active_branches(monkeypatch):
    from sdr.apps.ai.retrieval.routing import executors as executors_module

    _ImmediateExecutor.seen_max_workers.clear()
    monkeypatch.setattr(executors_module, "ThreadPoolExecutor", _ImmediateExecutor)

    router = HybridRetrievalRouter.__new__(HybridRetrievalRouter)
    router.vector_top_k = 8
    router.raptor_top_k = 5
    router.graph_top_k = 6
    router.max_context_chunks = 12
    router.advanced_config = AdvancedRetrievalConfig(hybrid_max_workers=9)
    router._vector_searcher = SimpleNamespace(search=lambda **kwargs: SimpleNamespace(results=[], error=None))
    router._raptor_searcher = SimpleNamespace(search_collapsed_raptor=lambda **kwargs: None)
    router._keyword_searcher = SimpleNamespace(search=lambda **kwargs: [])
    router._vector_results_to_candidates = lambda response: []
    router._raptor_results_to_candidates = lambda response: []
    router._apply_keyword_coverage_boost = lambda candidates, keywords: candidates
    router._grade_and_filter_candidates = lambda candidates, **kwargs: (candidates, {})
    router._reranker = SimpleNamespace(rerank=lambda query, candidates, top_k: [])
    router._collect_candidate_block_ids = lambda candidates: []

    result = executors_module.RetrievalRouteExecutor().execute_hybrid(
        router,
        query_text="Use MFA",
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        raptor_tree=None,
        graph=None,
        query_embedding=[0.1],
        keywords=["mfa"],
        inferred_relations=set(),
    )

    assert result.error is None
    assert _ImmediateExecutor.seen_max_workers[-1] == 2


def test_retrieve_many_returns_results_and_per_parameter_errors():
    service = RetrievalService.__new__(RetrievalService)
    service.router = SimpleNamespace(advanced_config=AdvancedRetrievalConfig(retrieve_many_max_concurrency=1))
    service.logger = logging.getLogger("test.retrieve_many")

    def _retrieve(**kwargs):
        parameter = kwargs["parameter"]
        if parameter.id == 2:
            raise RuntimeError("backend down")
        return RetrievalResult(context_chunks=[f"ctx-{parameter.id}"])

    service.retrieve_for_parameter = _retrieve

    results = service.retrieve_many_for_parameters(
        parameters=[_parameter(1), _parameter(2)],
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        indexes=SimpleNamespace(),
        tsd_document=None,
    )

    assert results["1"].context_chunks == ["ctx-1"]
    assert results["2"].error == "backend down"


def test_rag_gate_children_uses_retrieve_many_and_preserves_error_routing():
    persisted = []
    debated = []
    coordinator = CategoryAnalysisCoordinator(
        config=SimpleNamespace(batch_debate_enabled=True),
        workflow_repository=SimpleNamespace(),
        progress_service=SimpleNamespace(),
        run_state_service=SimpleNamespace(),
        text_debate_coordinator=SimpleNamespace(
            retrieval=SimpleNamespace(
                retrieve_many_for_parameters=lambda **kwargs: {
                    "1": RetrievalResult(context_chunks=[]),
                    "2": RetrievalResult(error="timeout"),
                }
            ),
            run_batched_analysis_for_category=lambda **kwargs: debated.extend(
                parameter.id for parameter in kwargs["parameters"]
            ),
        ),
        diagram_analysis_coordinator=SimpleNamespace(),
    )
    coordinator._persist_rag_gate_not_met = lambda **kwargs: persisted.append(kwargs["child"].id)
    coordinator.workflow_repository.list_control_summary_requirements = lambda **kwargs: []

    coordinator._rag_gate_children(
        review=SimpleNamespace(id=1),
        category=SimpleNamespace(id=7, code="web_application"),
        ingestion_job=SimpleNamespace(id=11),
        applicable_cfsrs=[],
        raw_parameters=[_parameter(1), _parameter(2)],
        indexes=SimpleNamespace(),
        tsd_document=SimpleNamespace(),
        summary=AnalysisSummary(),
        category_code="web_application",
        killed_assumptions_memory=deque(),
        parent_context_cache={},
    )

    assert persisted == [1]
    assert debated == [2]


def test_parent_applicability_gate_prefetches_without_changing_outcome(monkeypatch):
    from sdr.apps.ai.engine.debate import text_debate_coordinator as text_debate_module

    parent_one = SimpleNamespace(id=10, title="Auth", description="Auth controls")
    parent_two = SimpleNamespace(id=20, title="Crypto", description="Crypto controls")
    child_one = _parameter(1, parent=parent_one, text="Use MFA")
    child_two = _parameter(2, parent=parent_two, text="Encrypt data")
    retrieval_calls = []

    coordinator = TextDebateCoordinator(
        config=SimpleNamespace(
            parent_applicability_enabled=True,
            parent_applicability_confidence_threshold=0.7,
            parent_applicability_fallback_mode="skip",
            parent_applicability_max_child_texts=8,
        ),
        retrieval_service=SimpleNamespace(get_retrieve_many_max_concurrency=lambda override=None: 2),
        debate_service=SimpleNamespace(),
        persistence_service=SimpleNamespace(),
        debate_input_factory=SimpleNamespace(),
        progress_service=SimpleNamespace(),
        run_state_service=SimpleNamespace(raise_if_cancelled=lambda *args, **kwargs: None),
        mediator_agent_factory=lambda: None,
    )
    coordinator.build_parent_retrieval_query_details = lambda parent, child_parameters: {"parent_title": parent.title}
    def _get_parent_retrieval_result(**kwargs):
        parent = kwargs["parent"]
        cache = kwargs["cache"]
        key = (
            getattr(kwargs["ingestion_job"], "id", None),
            getattr(kwargs["category"], "id", None),
            getattr(parent, "id", None),
        )
        result = RetrievalResult(context_chunks=[f"context-{parent.id}"])
        cache[key] = result
        retrieval_calls.append(parent.id)
        return result

    coordinator.get_parent_retrieval_result = _get_parent_retrieval_result
    coordinator.persist_parent_not_applicable_children = lambda **kwargs: None
    monkeypatch.setattr(text_debate_module, "classify_parent_applicability", lambda **kwargs: SimpleNamespace(
        applicable=True,
        confidence=0.9,
        reasoning="Applicable",
        evidence=["context"],
        decision_mode="allow",
        error=None,
    ))

    summary = AnalysisSummary()
    applicable, cache = coordinator.apply_parent_applicability_gate(
        review=SimpleNamespace(id=99),
        category=SimpleNamespace(id=7, code="web_application"),
        ingestion_job=SimpleNamespace(id=11, version_no=1),
        parameters=[child_one, child_two],
        indexes=SimpleNamespace(),
        tsd_document=SimpleNamespace(),
        summary=summary,
    )

    assert [parameter.id for parameter in applicable] == [1, 2]
    assert set(retrieval_calls) == {10, 20}
    assert len(cache) == 2
