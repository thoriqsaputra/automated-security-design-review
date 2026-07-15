from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import redis

from sdr.core.config import settings

logger = logging.getLogger(__name__)

_STREAM_MAXLEN = 2000
_SNAPSHOT_TTL_SECONDS = 60 * 60 * 24
_STREAM_TTL_SECONDS = 60 * 60 * 24
_LOCK_TIMEOUT_SECONDS = 10
_LOCK_BLOCKING_TIMEOUT_SECONDS = 5


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_debate_id(
    parameter_id: Any,
    requirement_reference: Optional[str] = None,
    *,
    diagram_id: Optional[str] = None,
) -> str:
    if diagram_id is not None:
        identifier = str(diagram_id).strip() or "unknown"
        return f"diagram:{identifier}"
    if parameter_id is not None:
        return f"text:{parameter_id}"
    reference = (requirement_reference or "unknown").strip() or "unknown"
    return f"text:{reference}"


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _extract_critic_outcome(analysis_trace: Any, *, is_diagram: bool) -> Optional[str]:
    """Reads the persisted critic outcome out of a finding's analysis_trace.

    Text findings: last round of debate_history[].critic.outcome (UPHOLD/OVERTURN/PARTIAL).
    Diagram findings: critic_result.outcome (uphold/overturn only, no partial) — normalized
    to the same uppercase vocabulary as text so a single frontend filter covers both.
    """
    if not isinstance(analysis_trace, dict):
        return None
    if is_diagram:
        critic_result = analysis_trace.get("critic_result")
        if not isinstance(critic_result, dict):
            return None
        outcome = str(critic_result.get("outcome") or "").strip().upper()
        return outcome or None
    debate_history = analysis_trace.get("debate_history")
    if not isinstance(debate_history, list) or not debate_history:
        return None
    last_round = debate_history[-1]
    if not isinstance(last_round, dict):
        return None
    critic = last_round.get("critic")
    if not isinstance(critic, dict):
        return None
    outcome = str(critic.get("outcome") or "").strip().upper()
    return outcome or None


def _debate_sort_key(item: Dict[str, Any]) -> tuple[int, str, str]:
    status = str(item.get("status") or "pending").strip().lower()
    rank = {
        "running": 0,
        "pending": 1,
        "completed": 2,
        "failed": 3,
        "cancelled": 4,
    }.get(status, 5)
    updated_at = str(item.get("updated_at") or "")
    debate_id = str(item.get("debate_id") or "")
    return (rank, updated_at, debate_id)


class ReviewDebateEventStore:
    def __init__(self) -> None:
        self._redis_client: Optional[redis.Redis] = None

    def _redis(self) -> Optional[redis.Redis]:
        if self._redis_client is not None:
            return self._redis_client
        try:
            self._redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
            )
        except Exception as exc:
            logger.warning("ReviewDebateEventStore: failed to initialize Redis client: %s", exc)
            self._redis_client = None
        return self._redis_client

    def _stream_key(self, review_id: int) -> str:
        return f"reviews:{review_id}:debates:stream"

    def _snapshot_key(self, review_id: int) -> str:
        return f"reviews:{review_id}:debates:snapshot"

    def _lock_key(self, review_id: int) -> str:
        return f"reviews:{review_id}:debates:lock"

    def reset_review(self, review_id: int) -> None:
        client = self._redis()
        if client is None:
            return
        try:
            client.delete(self._stream_key(review_id), self._snapshot_key(review_id), self._lock_key(review_id))
        except Exception as exc:
            logger.warning("ReviewDebateEventStore.reset_review: review_id=%s failed: %s", review_id, exc)

    def load_snapshot(self, review_id: int) -> Optional[Dict[str, Any]]:
        client = self._redis()
        if client is None:
            return None
        try:
            raw = client.get(self._snapshot_key(review_id))
        except Exception as exc:
            logger.warning("ReviewDebateEventStore.load_snapshot: review_id=%s failed: %s", review_id, exc)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return self._normalize_snapshot(data)

    def read_events(
        self,
        review_id: int,
        *,
        last_event_id: str,
        block_ms: int = 15000,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        client = self._redis()
        if client is None:
            return []
        try:
            records = client.xread(
                {self._stream_key(review_id): last_event_id},
                count=100,
                block=block_ms,
            )
        except Exception as exc:
            logger.warning("ReviewDebateEventStore.read_events: review_id=%s failed: %s", review_id, exc)
            return []
        parsed: List[Tuple[str, Dict[str, Any]]] = []
        for _stream_name, entries in records or []:
            for event_id, fields in entries:
                payload_raw = fields.get("payload")
                if not payload_raw:
                    continue
                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError:
                    continue
                parsed.append((event_id, payload))
        return parsed

    def publish_review_status(
        self,
        review_id: int,
        *,
        review_status: str,
        error_message: Optional[str] = None,
    ) -> None:
        def mutate(snapshot: Dict[str, Any]) -> Dict[str, Any]:
            snapshot["review_status"] = review_status
            if error_message:
                snapshot["error_message"] = error_message
            elif "error_message" in snapshot:
                snapshot.pop("error_message", None)
            return {
                "type": "review.status",
                "review_id": review_id,
                "review_status": review_status,
                "error_message": error_message,
            }

        self._mutate(review_id, mutate)

    def seed_debates(
        self,
        review_id: int,
        *,
        review_status: str,
        debates: Iterable[Dict[str, Any]],
    ) -> None:
        debate_list = list(debates)
        if not debate_list:
            return

        def mutate(snapshot: Dict[str, Any]) -> Dict[str, Any]:
            snapshot["review_status"] = review_status
            changed: List[Dict[str, Any]] = []
            debate_map = snapshot.setdefault("debates", {})
            for debate in debate_list:
                debate_id = str(debate["debate_id"])
                existing = debate_map.get(debate_id) or {}
                transcript = list(existing.get("transcript") or [])
                finding_id = existing.get("finding_id")
                merged = {
                    "debate_id": debate_id,
                    "finding_type": debate.get("finding_type") or existing.get("finding_type") or "requirement",
                    "parameter_id": debate.get("parameter_id"),
                    "diagram_id": debate.get("diagram_id") or existing.get("diagram_id"),
                    "requirement_reference": debate.get("requirement_reference"),
                    "requirement_text": debate.get("requirement_text"),
                    "section_title": debate.get("section_title"),
                    "category_code": debate.get("category_code"),
                    "status": existing.get("status") or "pending",
                    "active_agent": existing.get("active_agent"),
                    "execution_mode": debate.get("execution_mode") or existing.get("execution_mode") or "single",
                    "pipeline_mode": debate.get("pipeline_mode") or existing.get("pipeline_mode") or "debate",
                    "progress_percent": int(existing.get("progress_percent") or 0),
                    "last_snippet": existing.get("last_snippet") or "",
                    "updated_at": utc_now_iso(),
                    "finding_id": finding_id,
                    "transcript": transcript,
                }
                if existing.get("diagram_extraction") is not None:
                    merged["diagram_extraction"] = existing["diagram_extraction"]
                debate_map[debate_id] = merged
                changed.append(merged)
            return {
                "type": "debates.seeded",
                "review_id": review_id,
                "review_status": review_status,
                "debates": changed,
            }

        self._mutate(review_id, mutate)

    def start_agent(
        self,
        review_id: int,
        *,
        debate: Dict[str, Any],
        agent: str,
        execution_mode: str,
        content: str,
        progress_percent: int,
        round_number: Optional[int] = None,
    ) -> None:
        debate_id = str(debate["debate_id"])

        def mutate(snapshot: Dict[str, Any]) -> Dict[str, Any]:
            debate_state = self._ensure_debate(snapshot, debate, execution_mode=execution_mode)
            message_id = f"{agent}:{len(debate_state['transcript']) + 1}"
            now = utc_now_iso()
            message = {
                "message_id": message_id,
                "agent": agent,
                "kind": "reasoning",
                "status": "running",
                "content": content,
                "started_at": now,
                "completed_at": None,
                "updated_at": now,
                "round": round_number,
            }
            debate_state["status"] = "running"
            debate_state["active_agent"] = agent
            debate_state["progress_percent"] = max(0, min(100, int(progress_percent)))
            debate_state["last_snippet"] = content
            debate_state["updated_at"] = now
            debate_state["active_message_id"] = message_id
            debate_state.setdefault("transcript", []).append(message)
            return {
                "type": "debate.updated",
                "review_id": review_id,
                "debate": self._strip_internal_fields(debate_state),
            }

        self._mutate(review_id, mutate)

    def append_agent_chunk(
        self,
        review_id: int,
        *,
        debate_id: str,
        agent: str,
        chunk: str,
    ) -> None:
        chunk = chunk or ""
        if not chunk:
            return

        def mutate(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            debate_state = (snapshot.get("debates") or {}).get(debate_id)
            if not debate_state:
                return None
            message = self._find_active_message(debate_state, agent)
            if message is None:
                return None
            message["content"] = f"{message.get('content', '')}{chunk}"
            message["updated_at"] = utc_now_iso()
            debate_state["last_snippet"] = message["content"][-280:]
            debate_state["updated_at"] = message["updated_at"]
            return {
                "type": "debate.updated",
                "review_id": review_id,
                "debate": self._strip_internal_fields(debate_state),
            }

        self._mutate(review_id, mutate)

    def complete_agent(
        self,
        review_id: int,
        *,
        debate_id: str,
        agent: str,
        content: str,
        progress_percent: int,
        execution_mode: Optional[str] = None,
        critic_outcome: Optional[str] = None,
        requires_rebuttal: Optional[bool] = None,
        round_number: Optional[int] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        def mutate(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            debate_state = (snapshot.get("debates") or {}).get(debate_id)
            if not debate_state:
                return None
            if execution_mode:
                debate_state["execution_mode"] = execution_mode
            if extra_fields:
                debate_state.update(extra_fields)
            message = self._find_active_message(debate_state, agent)
            now = utc_now_iso()
            if message is None:
                message = {
                    "message_id": f"{agent}:{len(debate_state.get('transcript') or []) + 1}",
                    "agent": agent,
                    "kind": "reasoning",
                    "status": "completed",
                    "content": content,
                    "started_at": now,
                    "completed_at": now,
                    "updated_at": now,
                    "round": round_number,
                }
                debate_state.setdefault("transcript", []).append(message)
            else:
                message["content"] = content
                message["status"] = "completed"
                message["completed_at"] = now
                message["updated_at"] = now
                if round_number is not None:
                    message["round"] = round_number
            if critic_outcome is not None:
                message["critic_outcome"] = critic_outcome
                debate_state["critic_outcome"] = critic_outcome
            if requires_rebuttal is not None:
                message["requires_rebuttal"] = requires_rebuttal
                debate_state["requires_rebuttal"] = requires_rebuttal
            debate_state["status"] = "running"
            debate_state["progress_percent"] = max(0, min(100, int(progress_percent)))
            debate_state["last_snippet"] = _clean_text(content)[-280:]
            debate_state["updated_at"] = now
            debate_state.pop("active_message_id", None)
            return {
                "type": "debate.updated",
                "review_id": review_id,
                "debate": self._strip_internal_fields(debate_state),
            }

        self._mutate(review_id, mutate)

    def fail_agent(
        self,
        review_id: int,
        *,
        debate_id: str,
        agent: str,
        error_message: str,
    ) -> None:
        def mutate(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            debate_state = (snapshot.get("debates") or {}).get(debate_id)
            if not debate_state:
                return None
            message = self._find_active_message(debate_state, agent)
            now = utc_now_iso()
            if message is None:
                message = {
                    "message_id": f"{agent}:{len(debate_state.get('transcript') or []) + 1}",
                    "agent": agent,
                    "kind": "reasoning",
                    "status": "failed",
                    "content": error_message,
                    "started_at": now,
                    "completed_at": now,
                    "updated_at": now,
                }
                debate_state.setdefault("transcript", []).append(message)
            else:
                message["content"] = error_message
                message["status"] = "failed"
                message["completed_at"] = now
                message["updated_at"] = now
            debate_state["status"] = "failed"
            debate_state["last_snippet"] = error_message[-280:]
            debate_state["updated_at"] = now
            debate_state["active_agent"] = agent
            debate_state.pop("active_message_id", None)
            return {
                "type": "debate.updated",
                "review_id": review_id,
                "debate": self._strip_internal_fields(debate_state),
            }

        self._mutate(review_id, mutate)

    def complete_debate(
        self,
        review_id: int,
        *,
        debate_id: str,
        finding_id: Optional[int],
        last_snippet: str,
        terminal_agent: str = "mediator",
    ) -> None:
        def mutate(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            debate_state = (snapshot.get("debates") or {}).get(debate_id)
            if not debate_state:
                return None
            debate_state["status"] = "completed"
            debate_state["active_agent"] = terminal_agent
            debate_state["progress_percent"] = 100
            debate_state["finding_id"] = finding_id
            debate_state["last_snippet"] = last_snippet[-280:]
            debate_state["updated_at"] = utc_now_iso()
            debate_state.pop("active_message_id", None)
            return {
                "type": "debate.updated",
                "review_id": review_id,
                "debate": self._strip_internal_fields(debate_state),
            }

        self._mutate(review_id, mutate)

    def mark_debates_cancelled(self, review_id: int, *, error_message: str) -> None:
        def mutate(snapshot: Dict[str, Any]) -> Dict[str, Any]:
            now = utc_now_iso()
            for debate in (snapshot.get("debates") or {}).values():
                if str(debate.get("status") or "").lower() in {"completed", "failed", "cancelled"}:
                    continue
                debate["status"] = "cancelled"
                debate["last_snippet"] = error_message
                debate["updated_at"] = now
                debate["active_agent"] = debate.get("active_agent") or "system"
                debate.pop("active_message_id", None)
            snapshot["review_status"] = "cancelled"
            snapshot["error_message"] = error_message
            return {
                "type": "review.status",
                "review_id": review_id,
                "review_status": "cancelled",
                "error_message": error_message,
            }

        self._mutate(review_id, mutate)

    def build_completed_snapshot(
        self,
        *,
        review_id: int,
        review_status: str,
        findings: Iterable[Any],
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        debates: Dict[str, Dict[str, Any]] = {}
        now = utc_now_iso()
        for finding in findings:
            is_diagram = getattr(finding, "finding_type", None) == "diagram"
            requirement_reference = _clean_text(getattr(finding, "requirement_reference", None)) or None
            diagram_id = _clean_text(getattr(finding, "diagram_id", None)) or None if is_diagram else None
            debate_id = build_debate_id(
                getattr(finding, "child_parameter_id", None),
                requirement_reference,
                diagram_id=diagram_id,
            )
            requirement_metadata = getattr(finding, "requirement_metadata", None) or {}
            analysis_trace = requirement_metadata.get("analysis_trace") if isinstance(requirement_metadata, dict) else None
            critic_outcome = _extract_critic_outcome(analysis_trace, is_diagram=is_diagram)
            pipeline_mode = (
                requirement_metadata.get("pipeline_mode") if isinstance(requirement_metadata, dict) else None
            ) or "debate"
            terminal_agent = "reasoner" if (is_diagram and pipeline_mode == "extract_reason") else "mediator"

            started_at = getattr(finding, "created_at", None).isoformat() if getattr(finding, "created_at", None) else now
            completed_at = getattr(finding, "updated_at", None).isoformat() if getattr(finding, "updated_at", None) else now
            debate_history = analysis_trace.get("debate_history") if isinstance(analysis_trace, dict) else None
            transcript: List[Dict[str, Any]] = []

            if not is_diagram and isinstance(debate_history, list) and debate_history:
                # Replay every round (including rebuttals) instead of only the
                # final round's *_thought_process columns, so completed reviews
                # show the same back-and-forth conversation the live SSE view does.
                for round_entry in debate_history:
                    if not isinstance(round_entry, dict):
                        continue
                    round_number = round_entry.get("round")
                    hunter_round = round_entry.get("hunter") if isinstance(round_entry.get("hunter"), dict) else {}
                    critic_round = round_entry.get("critic") if isinstance(round_entry.get("critic"), dict) else {}

                    hunter_text = _clean_text(hunter_round.get("reasoning"))
                    if hunter_text:
                        transcript.append(
                            {
                                "message_id": f"hunter:{len(transcript) + 1}",
                                "agent": "hunter",
                                "kind": "reasoning",
                                "status": "completed",
                                "content": hunter_text,
                                "started_at": started_at,
                                "completed_at": completed_at,
                                "updated_at": completed_at,
                                "round": round_number,
                            }
                        )

                    critic_text = _clean_text(critic_round.get("reasoning"))
                    if critic_text:
                        round_outcome = str(critic_round.get("outcome") or "").strip().upper() or None
                        critic_message: Dict[str, Any] = {
                            "message_id": f"critic:{len(transcript) + 1}",
                            "agent": "critic",
                            "kind": "reasoning",
                            "status": "completed",
                            "content": critic_text,
                            "started_at": started_at,
                            "completed_at": completed_at,
                            "updated_at": completed_at,
                            "round": round_number,
                        }
                        if round_outcome:
                            critic_message["critic_outcome"] = round_outcome
                        if critic_round.get("requires_rebuttal") is not None:
                            critic_message["requires_rebuttal"] = bool(critic_round.get("requires_rebuttal"))
                        transcript.append(critic_message)

                mediator_text = _clean_text(getattr(finding, "mediator_thought_process", None)) or _clean_text(
                    getattr(finding, "mediator_reasoning", None)
                )
                if mediator_text:
                    transcript.append(
                        {
                            "message_id": f"mediator:{len(transcript) + 1}",
                            "agent": "mediator",
                            "kind": "reasoning",
                            "status": "completed",
                            "content": mediator_text,
                            "started_at": started_at,
                            "completed_at": completed_at,
                            "updated_at": completed_at,
                            "round": None,
                        }
                    )
            else:
                # Fallback (e.g. diagram findings, which don't use text debate_history):
                # one message per agent from the flat *_thought_process/*_reasoning columns.
                # The agent set/labels depend on which diagram pipeline produced this
                # finding (extract-then-reason vs classic hunter/critic/mediator debate).
                if is_diagram and pipeline_mode == "extract_reason":
                    agent_pairs = (
                        ("extractor", None, getattr(finding, "hunter_reasoning", None)),
                        (
                            "reasoner",
                            getattr(finding, "mediator_thought_process", None),
                            getattr(finding, "mediator_reasoning", None),
                        ),
                    )
                else:
                    agent_pairs = (
                        ("hunter", getattr(finding, "hunter_thought_process", None), getattr(finding, "hunter_reasoning", None)),
                        ("critic", getattr(finding, "critic_thought_process", None), getattr(finding, "critic_reasoning", None)),
                        ("mediator", getattr(finding, "mediator_thought_process", None), getattr(finding, "mediator_reasoning", None)),
                    )
                for agent, cot, reasoning in agent_pairs:
                    text = _clean_text(cot) or _clean_text(reasoning)
                    if not text:
                        continue
                    message = {
                        "message_id": f"{agent}:{len(transcript) + 1}",
                        "agent": agent,
                        "kind": "reasoning",
                        "status": "completed",
                        "content": text,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "updated_at": completed_at,
                    }
                    if agent == "critic" and critic_outcome:
                        message["critic_outcome"] = critic_outcome
                        message["requires_rebuttal"] = critic_outcome in ("OVERTURN", "PARTIAL")
                    transcript.append(message)

            section_title = None
            if isinstance(requirement_metadata, dict):
                section_title = requirement_metadata.get("section")
            last_snippet = transcript[-1]["content"][-280:] if transcript else _clean_text(getattr(finding, "description", None))[-280:]
            debate_state = {
                "debate_id": debate_id,
                "finding_type": "diagram" if is_diagram else "requirement",
                "parameter_id": getattr(finding, "child_parameter_id", None),
                "diagram_id": diagram_id,
                "requirement_reference": requirement_reference,
                "requirement_text": _clean_text(getattr(finding, "requirement_text", None)) or _clean_text(getattr(finding, "diagram_caption", None)) or _clean_text(getattr(finding, "title", None)),
                "section_title": _clean_text(section_title) or _clean_text(getattr(finding, "parent_parameter_title", None)) or None,
                "category_code": _clean_text(getattr(finding, "category_code", None)) or None,
                "status": "completed",
                "active_agent": terminal_agent,
                "execution_mode": "single",
                "pipeline_mode": pipeline_mode,
                "diagram_extraction": (
                    requirement_metadata.get("diagram_extraction") if isinstance(requirement_metadata, dict) else None
                ),
                "progress_percent": 100,
                "last_snippet": last_snippet,
                "updated_at": getattr(finding, "updated_at", None).isoformat() if getattr(finding, "updated_at", None) else now,
                "finding_id": getattr(finding, "id", None),
                "transcript": transcript,
            }
            if critic_outcome:
                debate_state["critic_outcome"] = critic_outcome
                debate_state["requires_rebuttal"] = critic_outcome in ("OVERTURN", "PARTIAL")
            debates[debate_id] = debate_state
        snapshot = self._normalize_snapshot(
            {
                "review_id": review_id,
                "review_status": review_status,
                "error_message": error_message,
                "updated_at": now,
                "last_event_id": None,
                "debates": debates,
            }
        )
        return snapshot

    def save_snapshot(self, review_id: int, snapshot: Dict[str, Any]) -> None:
        client = self._redis()
        if client is None:
            return
        payload = self._normalize_snapshot(snapshot)
        try:
            client.setex(self._snapshot_key(review_id), _SNAPSHOT_TTL_SECONDS, json.dumps(payload))
        except Exception as exc:
            logger.warning("ReviewDebateEventStore.save_snapshot: review_id=%s failed: %s", review_id, exc)

    def _ensure_debate(
        self,
        snapshot: Dict[str, Any],
        debate: Dict[str, Any],
        *,
        execution_mode: str,
    ) -> Dict[str, Any]:
        debate_map = snapshot.setdefault("debates", {})
        debate_id = str(debate["debate_id"])
        existing = debate_map.get(debate_id) or {}
        transcript = list(existing.get("transcript") or [])
        merged = {
            "debate_id": debate_id,
            "finding_type": debate.get("finding_type") or existing.get("finding_type") or "requirement",
            "parameter_id": debate.get("parameter_id"),
            "diagram_id": debate.get("diagram_id") or existing.get("diagram_id"),
            "requirement_reference": debate.get("requirement_reference"),
            "requirement_text": debate.get("requirement_text"),
            "section_title": debate.get("section_title"),
            "category_code": debate.get("category_code"),
            "status": existing.get("status") or "pending",
            "active_agent": existing.get("active_agent"),
            "execution_mode": execution_mode or existing.get("execution_mode") or "single",
            "pipeline_mode": debate.get("pipeline_mode") or existing.get("pipeline_mode") or "debate",
            "progress_percent": int(existing.get("progress_percent") or 0),
            "last_snippet": existing.get("last_snippet") or "",
            "updated_at": utc_now_iso(),
            "finding_id": existing.get("finding_id"),
            "transcript": transcript,
        }
        if existing.get("diagram_extraction") is not None:
            merged["diagram_extraction"] = existing["diagram_extraction"]
        if "active_message_id" in existing:
            merged["active_message_id"] = existing["active_message_id"]
        debate_map[debate_id] = merged
        return merged

    def _find_active_message(self, debate_state: Dict[str, Any], agent: str) -> Optional[Dict[str, Any]]:
        active_message_id = debate_state.get("active_message_id")
        transcript = debate_state.get("transcript") or []
        if active_message_id:
            for item in transcript:
                if item.get("message_id") == active_message_id:
                    return item
        for item in reversed(transcript):
            if item.get("agent") == agent and item.get("status") == "running":
                return item
        return None

    def _coerce_debate_map(self, debates: Any) -> Dict[str, Dict[str, Any]]:
        if isinstance(debates, dict):
            return debates
        if isinstance(debates, list):
            return {
                str(item["debate_id"]): item
                for item in debates
                if isinstance(item, dict) and item.get("debate_id")
            }
        return {}

    def _load_snapshot_for_mutation(self, review_id: int) -> Dict[str, Any]:
        default = {
            "review_id": review_id,
            "review_status": None,
            "updated_at": utc_now_iso(),
            "last_event_id": None,
            "debates": {},
        }
        client = self._redis()
        if client is None:
            return default
        try:
            raw = client.get(self._snapshot_key(review_id))
        except Exception as exc:
            logger.warning(
                "ReviewDebateEventStore._load_snapshot_for_mutation: review_id=%s failed: %s",
                review_id,
                exc,
            )
            return default
        if not raw:
            return default
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return default
        data["debates"] = self._coerce_debate_map(data.get("debates"))
        return data

    def _mutate(
        self,
        review_id: int,
        mutator: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    ) -> None:
        client = self._redis()
        if client is None:
            return
        lock = client.lock(
            self._lock_key(review_id),
            timeout=_LOCK_TIMEOUT_SECONDS,
            blocking_timeout=_LOCK_BLOCKING_TIMEOUT_SECONDS,
        )
        acquired = False
        try:
            acquired = lock.acquire()
            snapshot = self._load_snapshot_for_mutation(review_id)
            payload = mutator(snapshot)
            if payload is None:
                return
            snapshot["updated_at"] = utc_now_iso()
            encoded = json.dumps(self._normalize_snapshot(snapshot))
            event_id = client.xadd(
                self._stream_key(review_id),
                {"payload": json.dumps(payload)},
                maxlen=_STREAM_MAXLEN,
                approximate=True,
            )
            snapshot["last_event_id"] = event_id
            client.setex(self._snapshot_key(review_id), _SNAPSHOT_TTL_SECONDS, json.dumps(self._normalize_snapshot(snapshot)))
            client.expire(self._stream_key(review_id), _STREAM_TTL_SECONDS)
        except Exception as exc:
            logger.warning("ReviewDebateEventStore._mutate: review_id=%s failed: %s", review_id, exc)
        finally:
            if acquired:
                try:
                    lock.release()
                except Exception:
                    pass

    def _normalize_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        debate_map = self._coerce_debate_map(snapshot.get("debates"))
        debates = [
            self._strip_internal_fields(dict(item))
            for item in sorted(debate_map.values(), key=_debate_sort_key)
        ]
        return {
            "review_id": snapshot.get("review_id"),
            "review_status": snapshot.get("review_status"),
            "error_message": snapshot.get("error_message"),
            "updated_at": snapshot.get("updated_at") or utc_now_iso(),
            "last_event_id": snapshot.get("last_event_id"),
            "debates": debates,
        }

    def _strip_internal_fields(self, debate: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = dict(debate)
        cleaned.pop("active_message_id", None)
        cleaned["transcript"] = list(cleaned.get("transcript") or [])
        return cleaned


review_debate_event_store = ReviewDebateEventStore()
