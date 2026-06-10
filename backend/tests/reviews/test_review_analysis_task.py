from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from sdr.apps.designs.models import Design  # noqa: F401
from sdr.apps.reviews import tasks as review_tasks
from sdr.apps.reviews.models import Review
from sdr.apps.reviews.models.choices import ReviewStatus
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob  # noqa: F401


def _build_review(review_id: int = 12):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=review_id,
        design=SimpleNamespace(name="Design A"),
        selected_categories=[],
        ingestion_job=None,
        status=ReviewStatus.PENDING.value,
        error_message=None,
        created_at=now,
        updated_at=now,
    )


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalars(self):
        return self

    def first(self):
        return self._value


class _Session:
    def __init__(self, *, execute_value=None, get_value=None):
        self.execute_value = execute_value
        self.get_value = get_value
        self.expunge_all_called = False
        self.committed = False

    def execute(self, _statement):
        return _ScalarResult(self.execute_value)

    def expunge_all(self):
        self.expunge_all_called = True

    def get(self, model, review_id):
        assert model is Review
        if self.get_value and self.get_value.id == review_id:
            return self.get_value
        return None

    def commit(self):
        self.committed = True


class _SessionContext:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class _SessionFactory:
    def __init__(self, sessions):
        self._sessions = list(sessions)

    def __call__(self):
        return _SessionContext(self._sessions.pop(0))


def test_dispatch_review_analysis_task_runs_pipeline(monkeypatch):
    review = _build_review()
    load_session = _Session(execute_value=review)
    monkeypatch.setattr(
        review_tasks,
        "SessionLocal",
        _SessionFactory([load_session]),
    )
    called = {}
    monkeypatch.setattr(
        "sdr.apps.ai.services.analysis.run_tsd_analysis",
        lambda loaded_review: called.setdefault("review", loaded_review),
    )

    review_tasks.dispatch_review_analysis_task.run(review.id)

    assert called["review"] is review
    assert load_session.expunge_all_called is True


def test_dispatch_review_analysis_task_marks_failed_and_retries(monkeypatch):
    review = _build_review()
    persisted_review = _build_review(review.id)
    load_session = _Session(execute_value=review)
    save_session = _Session(get_value=persisted_review)
    monkeypatch.setattr(
        review_tasks,
        "SessionLocal",
        _SessionFactory([load_session, save_session]),
    )

    def _boom(_review):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr("sdr.apps.ai.services.analysis.run_tsd_analysis", _boom)

    class RetryRaised(Exception):
        pass

    def _retry(*, exc, countdown):
        raise RetryRaised((str(exc), countdown))

    monkeypatch.setattr(review_tasks.dispatch_review_analysis_task, "retry", _retry)

    with pytest.raises(RetryRaised) as exc_info:
        review_tasks.dispatch_review_analysis_task.run(review.id)

    assert exc_info.value.args[0] == ("pipeline failed", 60)
    assert persisted_review.status == ReviewStatus.FAILED.value
    assert persisted_review.error_message == "pipeline failed"
    assert save_session.committed is True
