from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sdr.apps.reviews.models import Review
from sdr.apps.reviews.routers import reviews as reviews_router_module


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
        overview=None,
        created_at=now,
        updated_at=now,
    )


class _FakeSession:
    def __init__(self, review):
        self.review = review
        self.committed = False
        self.refreshed = []

    def get(self, model, review_id):
        assert model is Review
        if self.review and self.review.id == review_id:
            return self.review
        return None

    def commit(self):
        self.committed = True

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
    assert fake_db.committed is True
    assert fake_db.refreshed == [review]


def test_trigger_review_returns_500_when_dispatch_fails(monkeypatch):
    review = _build_review()

    def _boom(_review_id):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(reviews_router_module, "dispatch_review_analysis", _boom)

    with pytest.raises(HTTPException) as exc_info:
        reviews_router_module.trigger_review(review.id, db=_FakeSession(review=review))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to trigger review: queue unavailable"
