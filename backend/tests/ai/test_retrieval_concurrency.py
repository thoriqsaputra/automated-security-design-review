from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from types import SimpleNamespace

from sdr.apps.ai.engine.debate.category_analysis_coordinator import CategoryAnalysisCoordinator
from sdr.apps.ai.engine.debate.text_debate_coordinator import TextDebateCoordinator
from sdr.apps.ai.engine.dto import AnalysisSummary
from sdr.apps.ai.engine.persistence.progress_tracker import SummaryProgressService
from sdr.apps.ai.engine.persistence.review_run_state_service import ReviewRunStateService
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
    monkeypatch.setattr(settings, "AI_RETRIEVAL_MANY_MAX_CONCURRENCY", 0, raising=False)
    monkeypatch.setattr(settings, "AI_RETRIEVAL_ENABLE_CROSS_ENCODER_RERANK", True, raising=False)

    config = AdvancedRetrievalConfig.from_settings()

    assert config.hybrid_max_workers == 5
    assert config.retrieve_many_max_concurrency == 1
    assert config.enable_cross_encoder_rerank is True


def test_hybrid_executor_caps_worker_count_to_active_branches(monkeypatch):
    from sdr.apps.ai.retrieval.routing import executors as executors_module

    _ImmediateExecutor.seen_max_workers.clear()
    monkeypatch.setattr(executors_module, "ThreadPoolExecutor", _ImmediateExecutor)

    router = HybridRetrievalRouter.__new__(HybridRetrievalRouter)
    router.vector_top_k = 8
    router.raptor_top_k = 5
    router.max_context_chunks = 12
    router.advanced_config = AdvancedRetrievalConfig(hybrid_max_workers=9)
    router._raptor_searcher = SimpleNamespace(search_collapsed_raptor=lambda **kwargs: None)
    router._keyword_searcher = SimpleNamespace(search=lambda **kwargs: [])
    router._dense_tsd_results_to_candidates = lambda response: []
    router._raptor_results_to_candidates = lambda response: []
    router._apply_keyword_coverage_boost = lambda candidates, keywords: candidates
    router._grade_and_filter_candidates = lambda candidates, **kwargs: (candidates, {})
    router._reranker = SimpleNamespace(rerank=lambda query, candidates, top_k, **kwargs: [])
    router._collect_candidate_block_ids = lambda candidates: []

    result = executors_module.RetrievalRouteExecutor().execute_hybrid(
        router,
        query_text="Use MFA",
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        raptor_tree=None,
        query_embedding=[0.1],
        keywords=["mfa"],
        query_variants=[],
    )

    assert result.error is None
    assert _ImmediateExecutor.seen_max_workers[-1] == 1


def test_hybrid_executor_includes_dense_and_raptor_branches_when_tree_available(monkeypatch):
    from sdr.apps.ai.retrieval.routing import executors as executors_module
    from sdr.apps.ai.retrieval.searchers.raptor import RAPTORSearchResponse, RAPTORSearchResult

    _ImmediateExecutor.seen_max_workers.clear()
    monkeypatch.setattr(executors_module, "ThreadPoolExecutor", _ImmediateExecutor)

    leaf_node = SimpleNamespace(
        node_id="leaf-1",
        level=0,
        text="The gateway enforces MFA on all logins.",
        page_numbers=[1],
        section_heading="Authentication",
        token_estimate=10,
    )
    leaf_result = RAPTORSearchResult(node=leaf_node, cosine_similarity=0.9, source_block_ids=["b1"])
    dense_response = RAPTORSearchResponse(results=[leaf_result])

    router = HybridRetrievalRouter.__new__(HybridRetrievalRouter)
    router.vector_top_k = 8
    router.raptor_top_k = 5
    router.max_context_chunks = 12
    router.advanced_config = AdvancedRetrievalConfig(hybrid_max_workers=9)
    router._raptor_searcher = SimpleNamespace(
        search_collapsed_raptor=lambda **kwargs: dense_response,
        search_multi_level=lambda **kwargs: dense_response,
    )
    router._keyword_searcher = SimpleNamespace(search=lambda **kwargs: [])
    router._apply_keyword_coverage_boost = lambda candidates, keywords: candidates
    router._grade_and_filter_candidates = lambda candidates, **kwargs: (candidates, {})
    router._reranker = SimpleNamespace(
        rerank=lambda query, candidates, top_k, **kwargs: candidates
    )
    router._collect_candidate_block_ids = lambda candidates: []

    result = executors_module.RetrievalRouteExecutor().execute_hybrid(
        router,
        query_text="Use MFA",
        category=SimpleNamespace(id=1, code="web_application"),
        ingestion_job=SimpleNamespace(id=1),
        raptor_tree=SimpleNamespace(is_empty=lambda: False),
        query_embedding=[0.1],
        keywords=["mfa"],
        query_variants=[],
    )

    assert result.error is None
    dense_candidates = [c for c in result.context_chunks]
    assert any("enforces MFA" in chunk for chunk in dense_candidates)


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

def test_run_single_analysis_for_category_runs_parameters_concurrently():
    import threading
    import time

    coordinator = TextDebateCoordinator(
        config=SimpleNamespace(batch_debate_max_concurrency=3),
        retrieval_service=SimpleNamespace(),
        debate_service=SimpleNamespace(),
        persistence_service=SimpleNamespace(),
        debate_input_factory=SimpleNamespace(),
        progress_service=SimpleNamespace(
            register_analysis_work=lambda **kwargs: None,
        ),
        run_state_service=SimpleNamespace(
            raise_if_cancelled=lambda *args, **kwargs: None,
            is_cancelled=lambda _review: False,
            persist_summary_snapshot=lambda *args, **kwargs: None,
        ),
        mediator_agent_factory=lambda: None,
    )
    coordinator.seed_live_debates = lambda **kwargs: None
    coordinator.publish_work_phase = lambda **kwargs: None
    coordinator.extract_killed_assumptions_from_output = lambda output, parameter: []

    lock = threading.Lock()
    in_flight = 0
    peak_in_flight = 0
    persisted_ids = []

    def _slow_analyze(**kwargs):
        nonlocal in_flight, peak_in_flight
        with lock:
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
        time.sleep(0.2)
        with lock:
            in_flight -= 1
        parameter = kwargs["parameter"]
        return SimpleNamespace(analysis_trace={}, mediator_result=None, hunter_result=None, critic_result=None, parameter_id=parameter.id)

    coordinator.analyze_single_child = _slow_analyze
    coordinator.persist_debate_output = lambda **kwargs: persisted_ids.append(kwargs["parameter"].id)
    coordinator.record_debate_progress = lambda **kwargs: None
    coordinator.record_persistence_progress = lambda **kwargs: None

    parameters = [_parameter(i) for i in range(1, 7)]
    summary = AnalysisSummary()

    started_at = time.monotonic()
    coordinator.run_single_analysis_for_category(
        review=SimpleNamespace(id=1),
        category=SimpleNamespace(id=7, code="web_application"),
        ingestion_job=SimpleNamespace(id=11),
        parameters=parameters,
        indexes=SimpleNamespace(),
        tsd_document=SimpleNamespace(),
        summary=summary,
        killed_assumptions_memory=deque(),
    )
    elapsed = time.monotonic() - started_at

    assert peak_in_flight > 1, "parameters should have overlapped instead of running strictly sequentially"
    assert peak_in_flight <= 3
    assert elapsed < 0.2 * len(parameters), "concurrent run should be faster than the fully sequential baseline"
    assert sorted(persisted_ids) == [1, 2, 3, 4, 5, 6]


def test_register_analysis_work_is_additive_across_concurrent_branches():
    service = SummaryProgressService()
    summary = AnalysisSummary()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                service.register_analysis_work,
                summary=summary,
                category_code="web_application",
                total_count=count,
            )
            for count in (58, 3)
        ]
        for future in futures:
            future.result()

    stats = summary.category_stats["web_application"]
    assert stats["debate_total_count"] == 61
    assert stats["debate_remaining_count"] == 61
    assert stats["persistence_total_count"] == 61
    assert stats["persistence_remaining_count"] == 61
    assert summary.debate_total_parameters == 61
    assert summary.persistence_total_parameters == 61


def test_summary_snapshot_cannot_commit_stale_text_total_after_diagram_total():
    import threading

    first_save_started = threading.Event()
    release_first_save = threading.Event()
    diagram_thread_started = threading.Event()
    saved_totals = []

    class _Repository:
        def save_summary_snapshot(self, _review_id, *, summary):
            total = summary["debate_total_parameters"]
            if total == 58:
                first_save_started.set()
                assert release_first_save.wait(timeout=2)
            saved_totals.append(total)

    summary = AnalysisSummary(debate_total_parameters=58, debate_remaining_parameters=58)
    service = ReviewRunStateService(workflow_repository=_Repository())
    review = SimpleNamespace(id=1, summary_json=None)
    progress = SummaryProgressService()

    text_thread = threading.Thread(target=service.persist_summary_snapshot, args=(review, summary))

    def _register_and_persist_diagrams():
        diagram_thread_started.set()
        progress.register_analysis_work(
            summary=summary,
            category_code="web_application",
            total_count=3,
        )
        service.persist_summary_snapshot(review, summary)

    diagram_thread = threading.Thread(target=_register_and_persist_diagrams)
    text_thread.start()
    assert first_save_started.wait(timeout=2)
    diagram_thread.start()
    assert diagram_thread_started.wait(timeout=2)
    release_first_save.set()
    text_thread.join(timeout=2)
    diagram_thread.join(timeout=2)

    assert not text_thread.is_alive()
    assert not diagram_thread.is_alive()
    assert saved_totals == [58, 61]
    assert review.summary_json["debate_total_parameters"] == 61
