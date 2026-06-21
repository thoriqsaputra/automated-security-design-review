from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from sdr.apps.reviews.services.debate_events import ReviewDebateEventStore, build_debate_id


class _FakeLock:
    def acquire(self) -> bool:
        return True

    def release(self) -> None:
        pass


class _FakeRedis:
    """Minimal in-memory stand-in for redis.Redis, used to exercise sequential
    _mutate() calls without a real Redis server."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str):
        return self._store.get(key)

    def setex(self, key: str, _ttl: int, value: str) -> None:
        self._store[key] = value

    def delete(self, *keys: str) -> None:
        for key in keys:
            self._store.pop(key, None)

    def lock(self, *_args, **_kwargs) -> _FakeLock:
        return _FakeLock()

    def xadd(self, *_args, **_kwargs) -> str:
        return "0-1"

    def expire(self, *_args, **_kwargs) -> None:
        pass


def test_build_debate_id_prefers_parameter_id():
    assert build_debate_id(17, "V1.2.3") == "text:17"


def test_build_completed_snapshot_uses_persisted_reasoning():
    now = datetime.now(timezone.utc)
    finding = SimpleNamespace(
        id=41,
        child_parameter_id=17,
        requirement_reference="V1.2.3",
        requirement_text="Verify boundary protection is documented.",
        parent_parameter_title="Architecture",
        category_code="web_application",
        hunter_reasoning="Hunter summary",
        critic_reasoning="Critic summary",
        mediator_reasoning="Mediator summary",
        requirement_metadata={"section": "Architecture"},
        description="Fallback description",
        title="Finding title",
        created_at=now,
        updated_at=now,
    )

    snapshot = ReviewDebateEventStore().build_completed_snapshot(
        review_id=9,
        review_status="completed_with_findings",
        findings=[finding],
    )

    assert snapshot["review_id"] == 9
    assert snapshot["review_status"] == "completed_with_findings"
    assert len(snapshot["debates"]) == 1
    debate = snapshot["debates"][0]
    assert debate["debate_id"] == "text:17"
    assert debate["status"] == "completed"
    assert debate["finding_id"] == 41
    assert [message["agent"] for message in debate["transcript"]] == ["hunter", "critic", "mediator"]
    assert debate["last_snippet"] == "Mediator summary"


def test_mutate_survives_sequential_calls_without_list_vs_dict_crash():
    """Regression test: _mutate() persists "debates" as a list (the external
    wire shape) on every write, so any call after the first must reload that
    list-shaped snapshot and still treat "debates" as a dict internally.
    Previously this raised AttributeError: 'list' object has no attribute 'get'
    starting on the second sequential mutation for a review."""
    store = ReviewDebateEventStore()
    store._redis_client = _FakeRedis()
    review_id = 18

    debate = {
        "debate_id": "text:17",
        "parameter_id": 17,
        "requirement_reference": "V1.2.3",
        "requirement_text": "Verify boundary protection is documented.",
        "section_title": "Architecture",
        "category_code": "web_application",
        "execution_mode": "single",
    }

    store.seed_debates(review_id, review_status="running", debates=[debate])
    store.start_agent(
        review_id,
        debate=debate,
        agent="hunter",
        execution_mode="single",
        content="Hunter reasoning",
        progress_percent=10,
    )
    store.append_agent_chunk(review_id, debate_id="text:17", agent="hunter", chunk=" continued")
    store.complete_agent(
        review_id,
        debate_id="text:17",
        agent="hunter",
        content="Hunter final",
        progress_percent=40,
    )
    store.complete_debate(review_id, debate_id="text:17", finding_id=99, last_snippet="Resolved")

    snapshot = store.load_snapshot(review_id)
    assert snapshot is not None
    assert isinstance(snapshot["debates"], list)
    assert len(snapshot["debates"]) == 1
    persisted = snapshot["debates"][0]
    assert persisted["debate_id"] == "text:17"
    assert persisted["status"] == "completed"
    assert persisted["finding_id"] == 99


def test_normalize_snapshot_is_idempotent_on_already_list_debates():
    """Regression test: build_completed_snapshot() returns a list-shaped
    snapshot; save_snapshot() then normalizes it again. Previously this raised
    AttributeError: 'list' object has no attribute 'values'."""
    store = ReviewDebateEventStore()
    once = store._normalize_snapshot(
        {
            "review_id": 18,
            "review_status": "completed",
            "debates": {"text:1": {"debate_id": "text:1", "status": "completed"}},
        }
    )
    twice = store._normalize_snapshot(once)
    assert twice["debates"] == once["debates"]
