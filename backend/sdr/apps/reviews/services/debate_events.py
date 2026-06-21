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


def build_debate_id(parameter_id: Any, requirement_reference: Optional[str] = None) -> str:
    if parameter_id is not None:
        return f"text:{parameter_id}"
    reference = (requirement_reference or "unknown").strip() or "unknown"
    return f"text:{reference}"


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


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
                    "parameter_id": debate.get("parameter_id"),
                    "requirement_reference": debate.get("requirement_reference"),
                    "requirement_text": debate.get("requirement_text"),
                    "section_title": debate.get("section_title"),
                    "category_code": debate.get("category_code"),
                    "status": existing.get("status") or "pending",
                    "active_agent": existing.get("active_agent"),
                    "execution_mode": debate.get("execution_mode") or existing.get("execution_mode") or "single",
                    "progress_percent": int(existing.get("progress_percent") or 0),
                    "last_snippet": existing.get("last_snippet") or "",
                    "updated_at": utc_now_iso(),
                    "finding_id": finding_id,
                    "transcript": transcript,
                }
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
    ) -> None:
        def mutate(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            debate_state = (snapshot.get("debates") or {}).get(debate_id)
            if not debate_state:
                return None
            if execution_mode:
                debate_state["execution_mode"] = execution_mode
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
                }
                debate_state.setdefault("transcript", []).append(message)
            else:
                message["content"] = content
                message["status"] = "completed"
                message["completed_at"] = now
                message["updated_at"] = now
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
    ) -> None:
        def mutate(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            debate_state = (snapshot.get("debates") or {}).get(debate_id)
            if not debate_state:
                return None
            debate_state["status"] = "completed"
            debate_state["active_agent"] = "mediator"
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
            requirement_reference = _clean_text(getattr(finding, "requirement_reference", None)) or None
            debate_id = build_debate_id(getattr(finding, "child_parameter_id", None), requirement_reference)
            transcript: List[Dict[str, Any]] = []
            for agent, content in (
                ("hunter", getattr(finding, "hunter_reasoning", None)),
                ("critic", getattr(finding, "critic_reasoning", None)),
                ("mediator", getattr(finding, "mediator_reasoning", None)),
            ):
                text = _clean_text(content)
                if not text:
                    continue
                transcript.append(
                    {
                        "message_id": f"{agent}:{len(transcript) + 1}",
                        "agent": agent,
                        "kind": "reasoning",
                        "status": "completed",
                        "content": text,
                        "started_at": getattr(finding, "created_at", None).isoformat() if getattr(finding, "created_at", None) else now,
                        "completed_at": getattr(finding, "updated_at", None).isoformat() if getattr(finding, "updated_at", None) else now,
                        "updated_at": getattr(finding, "updated_at", None).isoformat() if getattr(finding, "updated_at", None) else now,
                    }
                )
            requirement_metadata = getattr(finding, "requirement_metadata", None) or {}
            section_title = None
            if isinstance(requirement_metadata, dict):
                section_title = requirement_metadata.get("section")
            last_snippet = transcript[-1]["content"][-280:] if transcript else _clean_text(getattr(finding, "description", None))[-280:]
            debates[debate_id] = {
                "debate_id": debate_id,
                "parameter_id": getattr(finding, "child_parameter_id", None),
                "requirement_reference": requirement_reference,
                "requirement_text": _clean_text(getattr(finding, "requirement_text", None)) or _clean_text(getattr(finding, "title", None)),
                "section_title": _clean_text(section_title) or _clean_text(getattr(finding, "parent_parameter_title", None)) or None,
                "category_code": _clean_text(getattr(finding, "category_code", None)) or None,
                "status": "completed",
                "active_agent": "mediator",
                "execution_mode": "single",
                "progress_percent": 100,
                "last_snippet": last_snippet,
                "updated_at": getattr(finding, "updated_at", None).isoformat() if getattr(finding, "updated_at", None) else now,
                "finding_id": getattr(finding, "id", None),
                "transcript": transcript,
            }
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
            "parameter_id": debate.get("parameter_id"),
            "requirement_reference": debate.get("requirement_reference"),
            "requirement_text": debate.get("requirement_text"),
            "section_title": debate.get("section_title"),
            "category_code": debate.get("category_code"),
            "status": existing.get("status") or "pending",
            "active_agent": existing.get("active_agent"),
            "execution_mode": execution_mode or existing.get("execution_mode") or "single",
            "progress_percent": int(existing.get("progress_percent") or 0),
            "last_snippet": existing.get("last_snippet") or "",
            "updated_at": utc_now_iso(),
            "finding_id": existing.get("finding_id"),
            "transcript": transcript,
        }
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
