from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from sdr.apps.ai.client import usage_tracker
from sdr.apps.ai.client.session import capture_current_context, job_session_context


def test_record_accumulates_tokens_and_duration_for_a_session():
    session_id = "test_session_accumulate"
    usage_tracker.clear(session_id)
    try:
        usage_tracker.record(session_id, {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, 1.5)
        usage_tracker.record(session_id, {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}, 0.5)

        snapshot = usage_tracker.snapshot(session_id)

        assert snapshot["call_count"] == 2
        assert snapshot["prompt_tokens"] == 150
        assert snapshot["completion_tokens"] == 30
        assert snapshot["total_tokens"] == 180
        assert snapshot["duration_seconds"] == 2.0
        assert snapshot["error_count"] == 0
    finally:
        usage_tracker.clear(session_id)


def test_record_tracks_errors_and_missing_usage_gracefully():
    session_id = "test_session_errors"
    usage_tracker.clear(session_id)
    try:
        usage_tracker.record(session_id, None, 0.3, error="rate_limited")
        usage_tracker.record(session_id, {}, 0.2)

        snapshot = usage_tracker.snapshot(session_id)

        assert snapshot["call_count"] == 2
        assert snapshot["error_count"] == 1
        assert snapshot["prompt_tokens"] == 0
        assert snapshot["total_tokens"] == 0
        assert abs(snapshot["duration_seconds"] - 0.5) < 1e-9
    finally:
        usage_tracker.clear(session_id)


def test_record_ignores_calls_with_no_session_id():
    # A falsy session_id (no active job_session_context) must not create a
    # shared/None bucket that different untagged calls would collide into.
    usage_tracker.record(None, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 1.0)
    usage_tracker.record("", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 1.0)

    assert usage_tracker.snapshot(None) == usage_tracker.snapshot(None)
    assert usage_tracker.snapshot(None)["call_count"] == 0


def test_snapshot_is_isolated_per_session():
    session_a = "test_session_a"
    session_b = "test_session_b"
    usage_tracker.clear(session_a)
    usage_tracker.clear(session_b)
    try:
        usage_tracker.record(session_a, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, 1.0)
        usage_tracker.record(session_b, {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200}, 5.0)

        assert usage_tracker.snapshot(session_a)["total_tokens"] == 2
        assert usage_tracker.snapshot(session_b)["total_tokens"] == 200
    finally:
        usage_tracker.clear(session_a)
        usage_tracker.clear(session_b)


def test_clear_removes_a_sessions_totals():
    session_id = "test_session_clear"
    usage_tracker.record(session_id, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, 1.0)
    assert usage_tracker.snapshot(session_id)["call_count"] == 1

    usage_tracker.clear(session_id)

    assert usage_tracker.snapshot(session_id)["call_count"] == 0


def test_concurrent_calls_from_a_thread_pool_do_not_lose_updates():
    # Reproduces the exact scenario the context-propagation fix exists for:
    # many worker threads all recording against the SAME session_id
    # concurrently must not race-drop any updates.
    session_id = "test_session_concurrent"
    usage_tracker.clear(session_id)
    try:
        def _record_one(_idx: int) -> None:
            usage_tracker.record(session_id, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, 0.01)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_record_one, idx) for idx in range(200)]
            for future in as_completed(futures):
                future.result()

        snapshot = usage_tracker.snapshot(session_id)
        assert snapshot["call_count"] == 200
        assert snapshot["total_tokens"] == 400
    finally:
        usage_tracker.clear(session_id)


def test_capture_current_context_propagates_session_id_into_worker_thread():
    # The actual bug this feature depends on being fixed: contextvars set by
    # job_session_context on the calling thread must survive into a function
    # submitted to a ThreadPoolExecutor when wrapped with
    # capture_current_context — otherwise usage recorded inside the pool
    # would be silently untracked (no session_id available there).
    from sdr.apps.ai.client.session import get_current_request_metadata

    session_id = "test_session_context_propagation"

    def _read_session_id_on_worker_thread() -> str:
        return get_current_request_metadata().get("session_id", "")

    with job_session_context(session_id=session_id, job_type="test", job_id=1):
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Unwrapped: context is NOT expected to propagate.
            unwrapped_result = executor.submit(_read_session_id_on_worker_thread).result()
            # Wrapped: context IS expected to propagate.
            wrapped_result = executor.submit(capture_current_context(_read_session_id_on_worker_thread)).result()

    assert unwrapped_result == ""
    assert wrapped_result == session_id
