from __future__ import annotations

import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
_totals: Dict[str, Dict[str, Any]] = {}


def _empty() -> Dict[str, Any]:
    return {
        "call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "duration_seconds": 0.0,
        "error_count": 0,
    }


def record(
    session_id: Optional[str],
    usage: Optional[Dict[str, Any]],
    duration_seconds: Optional[float],
    error: Optional[str] = None,
) -> None:
    """Accumulate one LLM call's token usage and duration under session_id.

    Calls with no active session (session_id falsy) are dropped rather than
    tracked under a shared/None bucket — untagged usage isn't attributable to
    any of the three pipelines this exists to measure.
    """
    if not session_id:
        return
    usage = usage or {}
    with _lock:
        bucket = _totals.setdefault(session_id, _empty())
        bucket["call_count"] += 1
        bucket["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        bucket["total_tokens"] += int(usage.get("total_tokens") or 0)
        bucket["duration_seconds"] += float(duration_seconds or 0.0)
        if error:
            bucket["error_count"] += 1


def snapshot(session_id: Optional[str]) -> Dict[str, Any]:
    """Read the current totals for session_id without clearing them."""
    if not session_id:
        return _empty()
    with _lock:
        return dict(_totals.get(session_id, _empty()))


def clear(session_id: Optional[str]) -> None:
    """Drop a session's accumulated totals once its pipeline has finished."""
    if not session_id:
        return
    with _lock:
        _totals.pop(session_id, None)
