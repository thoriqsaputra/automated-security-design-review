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


def _sanitize_json_payload(text: str) -> str:
    sanitized = (text or "{}")
    sanitized = sanitized.replace("\u201c", '"').replace("\u201d", '"')
    sanitized = sanitized.replace("\u2018", "'").replace("\u2019", "'")
    sanitized = _escape_control_chars_in_json_strings(sanitized)
    sanitized = _strip_trailing_commas(sanitized)
    return sanitized


def _escape_control_chars_in_json_strings(text: str) -> str:
    chars = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            chars.append(ch)
            escape = False
            continue
        if ch == "\\":
            chars.append(ch)
            escape = True
            continue
        if ch == '"':
            chars.append(ch)
            in_string = not in_string
            continue
        if in_string:
            if ch == "\n":
                chars.append("\\n")
                continue
            if ch == "\r":
                chars.append("\\r")
                continue
            if ch == "\t":
                chars.append("\\t")
                continue
        chars.append(ch)
    return "".join(chars)


def _strip_trailing_commas(text: str) -> str:
    chars = []
    in_string = False
    escape = False
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if escape:
            chars.append(ch)
            escape = False
            idx += 1
            continue
        if ch == "\\":
            chars.append(ch)
            escape = True
            idx += 1
            continue
        if ch == '"':
            chars.append(ch)
            in_string = not in_string
            idx += 1
            continue
        if not in_string and ch == ",":
            lookahead = idx + 1
            while lookahead < length and text[lookahead] in " \t\r\n":
                lookahead += 1
            if lookahead < length and text[lookahead] in "}]":
                idx += 1
                continue
        chars.append(ch)
        idx += 1
    return "".join(chars)


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
        raw_error = exc

    sanitized_content = _sanitize_json_payload(content)
    if sanitized_content != content:
        try:
            parsed = json.loads(sanitized_content)
            logger.info(
                "%s: recovered malformed JSON via deterministic sanitizer after raw decode failed at pos=%s.",
                component,
                getattr(raw_error, "pos", "unknown"),
            )
            return parsed, None
        except json.JSONDecodeError:
            pass

    logger.warning(
        "%s: initial JSON decode failed at pos=%s; attempting repair: %s",
        component,
        getattr(raw_error, "pos", "unknown"),
        raw_error,
    )

    repair_prompt = build_json_repair_prompt(sanitized_content)
    repair_resp = chat_completion_fn(
        messages=[{"role": "user", "content": repair_prompt}],
        component="fallback",
        temperature=0.0,
        max_tokens=max_tokens,
        reasoning={"effort": "low"},
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
