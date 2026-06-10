from __future__ import annotations

import contextlib
import contextvars
import math
from typing import Any, Callable, Dict, Iterator, Optional

_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "ai_usage_context",
    default={},
)


def get_ai_usage_context() -> Dict[str, Any]:
    return dict(_context.get() or {})


def capture_ai_usage_context() -> Dict[str, Any]:
    return get_ai_usage_context()


def run_with_ai_usage_context(
    context_snapshot: Optional[Dict[str, Any]],
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    token = _context.set(dict(context_snapshot or {}))
    try:
        return fn(*args, **kwargs)
    finally:
        _context.reset(token)


def set_ai_usage_context(**values: Any) -> contextvars.Token:
    current = get_ai_usage_context()
    merged = {**current, **{k: v for k, v in values.items() if v is not None}}
    return _context.set(merged)


def reset_ai_usage_context(token: contextvars.Token) -> None:
    _context.reset(token)


@contextlib.contextmanager
def ai_usage_context(**values: Any) -> Iterator[None]:
    current = get_ai_usage_context()
    merged = {**current, **{k: v for k, v in values.items() if v is not None}}
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def normalize_usage_payload(usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    usage = usage or {}
    input_tokens = _as_int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("input_token_count")
        or _nested_int(usage, "token_usage", "input_tokens")
    )
    output_tokens = _as_int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("output_token_count")
        or _nested_int(usage, "token_usage", "output_tokens")
    )
    cached_input_tokens = _as_int(
        usage.get("cached_input_tokens")
        or _nested_int(usage, "prompt_tokens_details", "cached_tokens")
        or _nested_int(usage, "token_usage", "cached_input_tokens")
    )
    reasoning_output_tokens = _as_int(
        usage.get("reasoning_output_tokens")
        or _nested_int(usage, "completion_tokens_details", "reasoning_tokens")
        or _nested_int(usage, "token_usage", "reasoning_output_tokens")
    )
    embedding_tokens = _as_int(
        usage.get("embedding_tokens")
        or usage.get("embedding_token_count")
        or _nested_int(usage, "token_usage", "embedding_tokens")
    )
    total_tokens = _as_int(
        usage.get("total_tokens")
        or _nested_int(usage, "token_usage", "total_tokens")
    )
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens + embedding_tokens
    if total_tokens <= 0 and usage:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": max(0, input_tokens),
        "output_tokens": max(0, output_tokens),
        "total_tokens": max(0, total_tokens),
        "cached_input_tokens": max(0, cached_input_tokens),
        "reasoning_output_tokens": max(0, reasoning_output_tokens),
        "embedding_tokens": max(0, embedding_tokens),
    }
