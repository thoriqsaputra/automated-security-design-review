from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sdr.core.config import settings

from sdr.apps.ai.client.base import AIProvider, AIResponse
from sdr.apps.ai.client.session import get_current_request_metadata

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _sanitize(value: str) -> str:
    cleaned = _SANITIZE_RE.sub("_", value or "").strip("_")
    return cleaned or "unknown"


def _fence_content(text: str) -> str:
    body = text or ""
    fence = "```"
    while fence in body:
        fence += "`"
    return f"{fence}text\n{body}\n{fence}"


def _render_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        parts.append(f"### {role}\n\n{_fence_content(content)}")
    return "\n\n".join(parts)


def log_llm_interaction(
    *,
    component: Optional[str],
    provider: AIProvider,
    request_kwargs: Dict[str, Any],
    response: Optional[AIResponse] = None,
    streamed_content: Optional[str] = None,
    error: Optional[str] = None,
    duration_seconds: float = 0.0,
) -> None:
    if not getattr(settings, "AI_LLM_LOG_ENABLED", False):
        return

    try:
        _write_llm_interaction(
            component=component,
            provider=provider,
            request_kwargs=request_kwargs,
            response=response,
            streamed_content=streamed_content,
            error=error,
            duration_seconds=duration_seconds,
        )
    except Exception:
        logger.warning("log_llm_interaction: failed to write LLM interaction log.", exc_info=True)


def _write_llm_interaction(
    *,
    component: Optional[str],
    provider: AIProvider,
    request_kwargs: Dict[str, Any],
    response: Optional[AIResponse],
    streamed_content: Optional[str],
    error: Optional[str],
    duration_seconds: float,
) -> None:
    request_metadata = get_current_request_metadata()
    session_id = request_metadata.get("session_id") or "adhoc"

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    millis = f"{now.microsecond // 1000:03d}"
    filename = f"{timestamp}_{millis}_{_sanitize(session_id)}.md"

    log_dir = Path(getattr(settings, "AI_LLM_LOG_DIR")) / _sanitize(component or "unknown")
    log_dir.mkdir(parents=True, exist_ok=True)

    content_text = streamed_content if streamed_content is not None else (response.content if response else "")
    finish_reason = getattr(response, "finish_reason", None) if response else None
    usage = getattr(response, "raw_usage", None) or getattr(response, "usage", None) if response else None
    response_error = error or (response.error if response else None)
    model_name = request_kwargs.get("model") or (response.model if response else "")

    metadata_lines = [
        f"- component: {component or 'unknown'}",
        f"- provider: {provider.value if isinstance(provider, AIProvider) else provider}",
        f"- model: {model_name}",
        f"- temperature: {request_kwargs.get('temperature')}",
        f"- max_tokens: {request_kwargs.get('max_tokens')}",
        f"- streamed: {streamed_content is not None}",
        f"- duration_seconds: {duration_seconds:.3f}",
        f"- timestamp_utc: {now.isoformat()}",
        f"- session_id: {request_metadata.get('session_id', '')}",
        f"- job_type: {request_metadata.get('job_type', '')}",
        f"- job_id: {request_metadata.get('job_id', '')}",
        f"- finish_reason: {finish_reason}",
        f"- usage: {usage}",
        f"- error: {response_error}",
    ]

    sections = [
        "# LLM Interaction",
        "## Metadata\n\n" + "\n".join(metadata_lines),
        "## Messages\n\n" + _render_messages(request_kwargs.get("messages")),
        "## Response\n\n" + _fence_content(content_text or ""),
    ]

    log_path = log_dir / filename
    log_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
