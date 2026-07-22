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


def test_build_completed_snapshot_uses_extractor_reasoner_labels_for_extract_reason_diagrams():
    now = datetime.now(timezone.utc)
    finding = SimpleNamespace(
        id=42,
        finding_type="diagram",
        child_parameter_id=None,
        diagram_id="diag-1",
        requirement_reference=None,
        requirement_text=None,
        diagram_caption="Network diagram",
        parent_parameter_title=None,
        category_code="web_application",
        hunter_reasoning="Extraction summary text",
        critic_reasoning=None,
        mediator_reasoning="Reasoner final verdict text",
        mediator_thought_process="Reasoner cot",
        requirement_metadata={"pipeline_mode": "extract_reason", "diagram_extraction": {"components": []}},
        description="Fallback description",
        title="Diagram finding",
        created_at=now,
        updated_at=now,
    )

    snapshot = ReviewDebateEventStore().build_completed_snapshot(
        review_id=10,
        review_status="completed_with_findings",
        findings=[finding],
    )

    debate = snapshot["debates"][0]
    assert [message["agent"] for message in debate["transcript"]] == ["extractor", "reasoner"]
    assert debate["active_agent"] == "reasoner"
    assert debate["pipeline_mode"] == "extract_reason"
    assert debate["diagram_extraction"] == {"components": []}


def test_build_completed_snapshot_defaults_diagram_to_hunter_critic_mediator():
    now = datetime.now(timezone.utc)
    finding = SimpleNamespace(
        id=43,
        finding_type="diagram",
        child_parameter_id=None,
        diagram_id="diag-2",
        requirement_reference=None,
        requirement_text=None,
        diagram_caption="Network diagram",
        parent_parameter_title=None,
        category_code="web_application",
        hunter_reasoning="Hunter summary",
        hunter_thought_process=None,
        critic_reasoning="Critic summary",
        critic_thought_process=None,
        mediator_reasoning="Mediator summary",
        mediator_thought_process=None,
        requirement_metadata={"source": "diagram_debate_service"},
        description="Fallback description",
        title="Diagram finding",
        created_at=now,
        updated_at=now,
    )

    snapshot = ReviewDebateEventStore().build_completed_snapshot(
        review_id=11,
        review_status="completed_with_findings",
        findings=[finding],
    )

    debate = snapshot["debates"][0]
    assert [message["agent"] for message in debate["transcript"]] == ["hunter", "critic", "mediator"]
    assert debate["active_agent"] == "mediator"
    assert debate["pipeline_mode"] == "debate"


def test_complete_debate_sets_custom_terminal_agent():
    store = ReviewDebateEventStore()
    store._redis_client = _FakeRedis()
    review_id = 20
    debate = {"debate_id": "diagram:d1", "diagram_id": "d1", "execution_mode": "single", "pipeline_mode": "extract_reason"}

    store.seed_debates(review_id, review_status="running", debates=[debate])
    store.start_agent(
        review_id, debate=debate, agent="extractor", execution_mode="single", content="Extracting", progress_percent=10,
    )
    store.complete_debate(review_id, debate_id="diagram:d1", finding_id=7, last_snippet="Done", terminal_agent="reasoner")

    snapshot = store.load_snapshot(review_id)
    persisted = snapshot["debates"][0]
    assert persisted["active_agent"] == "reasoner"
    assert persisted["pipeline_mode"] == "extract_reason"


def test_work_phase_updates_without_creating_a_transcript_agent():
    store = ReviewDebateEventStore()
    store._redis_client = _FakeRedis()
    review_id = 22
    debate = {
        "debate_id": "text:22",
        "parameter_id": 22,
        "execution_mode": "single",
        "work_phase": "queued",
    }

    store.seed_debates(review_id, review_status="running", debates=[debate])
    store.set_work_phase(
        review_id,
        debate_id="text:22",
        work_phase="retrieval",
        last_snippet="Retrieving and ranking supporting evidence.",
        progress_percent=2,
    )

    persisted = store.load_snapshot(review_id)["debates"][0]
    assert persisted["status"] == "running"
    assert persisted["work_phase"] == "retrieval"
    assert persisted["progress_percent"] == 2
    assert persisted["transcript"] == []

    store.start_agent(
        review_id,
        debate=debate,
        agent="hunter",
        execution_mode="single",
        content="Hunter is analyzing this requirement.",
        progress_percent=5,
    )
    persisted = store.load_snapshot(review_id)["debates"][0]
    assert persisted["work_phase"] == "debate"
    assert [message["agent"] for message in persisted["transcript"]] == ["hunter"]


def test_complete_agent_merges_extra_fields():
    store = ReviewDebateEventStore()
    store._redis_client = _FakeRedis()
    review_id = 21
    debate = {"debate_id": "diagram:d2", "diagram_id": "d2", "execution_mode": "single", "pipeline_mode": "extract_reason"}

    store.seed_debates(review_id, review_status="running", debates=[debate])
    store.start_agent(
        review_id, debate=debate, agent="extractor", execution_mode="single", content="Extracting", progress_percent=10,
    )
    store.complete_agent(
        review_id,
        debate_id="diagram:d2",
        agent="extractor",
        content="Extracted",
        progress_percent=55,
        extra_fields={"diagram_extraction": {"components": [{"id": "c1", "name": "API", "type": "service"}]}},
    )

    snapshot = store.load_snapshot(review_id)
    persisted = snapshot["debates"][0]
    assert persisted["diagram_extraction"] == {"components": [{"id": "c1", "name": "API", "type": "service"}]}


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
