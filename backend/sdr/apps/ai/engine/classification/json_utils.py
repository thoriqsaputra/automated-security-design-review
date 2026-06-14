from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional, Tuple

from sdr.apps.ai.prompts.extraction import build_json_repair_prompt
from sdr.apps.ai.utils.parsing import strip_markdown_code_blocks, strip_thinking_block

logger = logging.getLogger(__name__)


def extract_json_payload(text: str) -> str:
    cleaned = strip_markdown_code_blocks(strip_thinking_block(text or "{}")).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()
    return cleaned


def parse_json_with_repair(
    raw_text: str,
    *,
    component: str,
    max_tokens: int,
    chat_completion_fn: Callable[..., Any],
) -> Tuple[Optional[Any], Optional[str]]:
    content = extract_json_payload(raw_text or "{}")
    try:
        return json.loads(content), None
    except json.JSONDecodeError as exc:
        logger.warning(
            "%s: initial JSON decode failed at pos=%s; attempting repair: %s",
            component,
            getattr(exc, "pos", "unknown"),
            exc,
        )

    repair_prompt = build_json_repair_prompt(content)
    repair_resp = chat_completion_fn(
        messages=[{"role": "user", "content": repair_prompt}],
        component="fallback",
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    if repair_resp.error or not repair_resp.content:
        error = str(repair_resp.error or "empty_repair_response")
        logger.warning("%s: JSON repair failed: %s", component, error)
        return None, error

    repaired_content = extract_json_payload(repair_resp.content or "{}")
    try:
        return json.loads(repaired_content), None
    except json.JSONDecodeError as exc:
        error = f"repair_decode_failed: {exc}"
        logger.warning("%s: repaired JSON still invalid: %s", component, exc)
        return None, error
