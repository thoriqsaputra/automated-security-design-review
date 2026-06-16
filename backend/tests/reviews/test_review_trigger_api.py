from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import sdr.apps.designs.models  # noqa: F401

from sdr.apps.reviews.models import Review
from sdr.apps.reviews.routers import reviews as reviews_router_module
from sdr.apps.reviews.schemas import ReviewCreateSchema, ReviewTriggerSchema


def _build_review(*, review_id: int = 7, status: str = Review.STATUS_PENDING):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=review_id,
        design_id=11,
        status=status,
        celery_task_id=None,
        started_at=None,
        completed_at=None,
        error_message=None,
        summary_json={},
        retrieval_snapshot_json={"status": "ready"},
        overview=None,
        asvs_level_override=None,
        analysis_mode=Review.ANALYSIS_MODE_DEFAULT,
        created_at=now,
        updated_at=now,
    )


class _FakeSession:
    def __init__(self, review):
        self.review = review
        self.committed = False
        self.rolled_back = False
        self.refreshed = []
        self.executed = []

    def get(self, model, review_id):
        assert model is Review
        if self.review and self.review.id == review_id:
            return self.review
        return None

    def commit(self):
        self.committed = True

    def execute(self, statement):
        self.executed.append(statement)

    def rollback(self):
        self.rolled_back = True

    def refresh(self, obj):
        self.refreshed.append(obj)


def test_trigger_review_returns_404_for_missing_review(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        reviews_router_module.trigger_review(404, db=_FakeSession(review=None))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Review not found"


def test_trigger_review_rejects_running_review(monkeypatch):
    review = _build_review(status=Review.STATUS_RUNNING)

    with pytest.raises(HTTPException) as exc_info:
        reviews_router_module.trigger_review(review.id, db=_FakeSession(review=review))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Review cannot be triggered in its current state"


def test_trigger_review_persists_task_id(monkeypatch):
    review = _build_review()
    fake_db = _FakeSession(review=review)
    monkeypatch.setattr(
        reviews_router_module,
        "dispatch_review_analysis",
        lambda review_id: {"mode": "async", "task_id": f"task-{review_id}"},
    )

    result = reviews_router_module.trigger_review(review.id, db=fake_db)

    assert result is review
    assert review.celery_task_id == f"task-{review.id}"
    assert review.retrieval_snapshot_json is None
    assert fake_db.committed is True
    assert fake_db.refreshed == [review]


def test_trigger_review_updates_analysis_mode_when_provided(monkeypatch):
    review = _build_review()
    fake_db = _FakeSession(review=review)
    monkeypatch.setattr(
        reviews_router_module,
        "dispatch_review_analysis",
        lambda review_id: {"mode": "async", "task_id": f"task-{review_id}"},
    )

    result = reviews_router_module.trigger_review(
        review.id,
        payload=ReviewTriggerSchema(analysis_mode="diagram_only"),
        db=fake_db,
    )

    assert result is review
    assert review.analysis_mode == Review.ANALYSIS_MODE_DIAGRAM_ONLY
    assert fake_db.committed is True


def test_review_create_schema_defaults_analysis_mode():
    payload = ReviewCreateSchema(design_id=1, category_id=2)

    assert payload.analysis_mode.value == Review.ANALYSIS_MODE_DEFAULT


def test_review_trigger_schema_rejects_invalid_mode():
    with pytest.raises(ValidationError):
        ReviewTriggerSchema(analysis_mode="bad_mode")


def test_trigger_review_returns_500_when_dispatch_fails(monkeypatch):
    review = _build_review()

    def _boom(_review_id):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(reviews_router_module, "dispatch_review_analysis", _boom)

    with pytest.raises(HTTPException) as exc_info:
        reviews_router_module.trigger_review(review.id, db=_FakeSession(review=review))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to trigger review: queue unavailable"


def test_cancel_review_marks_cancelled_and_revokes_task(monkeypatch):
    review = _build_review(status=Review.STATUS_RUNNING)
    review.celery_task_id = "task-7"
    fake_db = _FakeSession(review=review)
    revoke_calls = []

    class _Control:
        def revoke(self, task_id, terminate=False, signal=None):
            revoke_calls.append((task_id, terminate, signal))

    monkeypatch.setattr(
        "sdr.celery_app.celery_app.control",
        _Control(),
    )

    result = reviews_router_module.cancel_review(review.id, db=fake_db)

    assert result is review
    assert review.status == "cancelled"
    assert review.error_message == "Analysis was cancelled by user."
    assert review.completed_at is not None
    assert fake_db.committed is True
    assert fake_db.refreshed == [review]
    assert revoke_calls == [("task-7", True, "SIGTERM")]


def test_get_review_retrieval_visualization_returns_pending_when_missing():
    review = _build_review()
    review.retrieval_snapshot_json = None

    payload = reviews_router_module.get_review_retrieval_visualization(review.id, db=_FakeSession(review=review))

    assert payload["status"] == "pending"
    assert payload["raptor"] is None
    assert payload["graph"] is None


def test_get_review_retrieval_visualization_returns_snapshot():
    review = _build_review()
    review.retrieval_snapshot_json = {
        "status": "ready",
        "generated_at": "2026-06-10T00:00:00+00:00",
        "raptor": {"status": "ready", "nodes": []},
        "graph": {"status": "ready", "nodes": [], "edges": []},
    }

    payload = reviews_router_module.get_review_retrieval_visualization(review.id, db=_FakeSession(review=review))

    assert payload is review.retrieval_snapshot_json


def test_review_progress_exposes_live_debate_and_persistence_counts():
    review = Review(status=Review.STATUS_RUNNING)
    review.summary_json = {
        "debate_total_parameters": 10,
        "debate_completed_parameters": 4,
        "debate_remaining_parameters": 6,
        "persistence_total_parameters": 10,
        "persistence_completed_parameters": 2,
        "persistence_remaining_parameters": 8,
        "error_count": 1,
        "applicability": {"children_marked_na_by_parent": 3},
        "asvs": {"categories": {"web_application": {"debate_total_count": 10}}},
    }

    progress = review.progress

    assert progress is not None
    assert progress["stage"] == "debate"
    assert progress["completed_items"] == 4
    assert progress["remaining_items"] == 6
    assert progress["failed_items"] == 1
    assert progress["preparation"]["debate"]["total"] == 10
    assert progress["preparation"]["persistence"]["completed"] == 2
    assert progress["preparation"]["skipped_by_parent_applicability"] == 3
