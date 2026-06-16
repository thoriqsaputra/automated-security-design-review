from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

import tiktoken

from sdr.apps.ai.prompts.extraction import build_json_repair_prompt
from sdr.apps.ai.utils.parsing import strip_markdown_code_blocks, strip_thinking_block

logger = logging.getLogger(__name__)

_token_encoder = None
_TOC_HEADING_RE = re.compile(
    r"^\s*(daftar\s+isi|table\s+of\s+contents|contents)\s*$",
    re.IGNORECASE,
)
_TOC_ENTRY_DOTTED_RE = re.compile(r"\.{2,}\s*\d{1,4}\s*$")
_TOC_ENTRY_SPACED_RE = re.compile(r".{6,}\s{2,}\d{1,4}\s*$")
_TOC_LABEL_RE = re.compile(r"^\s*(halaman|page|pages?)\s*$", re.IGNORECASE)
_NON_REQUIREMENT_NOTE_RE = re.compile(r"^\s*note\s*[:\-]\s*", re.IGNORECASE)
_OWASP_SUBSECTION_HEADING_ONLY_RE = re.compile(r"^\s*V\d+\.\d+\s+[A-Z].*$", re.IGNORECASE)


def _count_tokens(text: str) -> int:
    global _token_encoder
    if _token_encoder is None:
        try:
            _token_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return max(1, len(text) // 4)
    return len(_token_encoder.encode(text))


def _looks_like_toc_entry(line: str) -> bool:
    text = (line or "").strip()
    if len(text) < 4:
        return False
    if _TOC_ENTRY_DOTTED_RE.search(text) or _TOC_ENTRY_SPACED_RE.search(text):
        return True
    return bool(
        re.match(
            r"^(?:bab|chapter|section)?\s*[ivxlcdm\d]+(?:\.\d+)*\.?\s+.{3,}\s+\d{1,4}$",
            text,
            re.IGNORECASE,
        )
    )


def _remove_table_of_contents(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines: List[str] = []
    in_toc = False
    skipped_lines = 0

    for line in lines:
        stripped = line.strip()
        if _TOC_HEADING_RE.match(stripped):
            in_toc = True
            skipped_lines += 1
            continue
        if in_toc:
            if not stripped or _TOC_LABEL_RE.match(stripped) or _looks_like_toc_entry(stripped):
                skipped_lines += 1
                continue
            in_toc = False
        cleaned_lines.append(line)

    if skipped_lines:
        logger.info(
            "_remove_table_of_contents: removed %d table-of-contents line(s) before extraction.",
            skipped_lines,
        )
    return "\n".join(cleaned_lines)


def _extract_json_payload(text: str) -> str:
    cleaned = strip_markdown_code_blocks(strip_thinking_block(text or "{}")).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()
    return cleaned


def _sanitize_json_payload(text: str) -> str:
    sanitized = text or "{}"
    sanitized = sanitized.replace("\u201c", '"').replace("\u201d", '"')
    sanitized = sanitized.replace("\u2018", "'").replace("\u2019", "'")
    sanitized = re.sub(r",(\s*[}\]])", r"\1", sanitized)
    return sanitized


def parse_json_response(raw_text: str) -> Any:
    payload = _extract_json_payload(raw_text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return json.loads(_sanitize_json_payload(payload))


def parse_json_with_repair(raw_text: str, *, llm_client, max_tokens: int) -> Any:
    content = _extract_json_payload(raw_text or "{}")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            return json.loads(_sanitize_json_payload(content))
        except json.JSONDecodeError:
            pass
        repair_prompt = build_json_repair_prompt(content)
        repair_resp = llm_client.repair_json(user_prompt=repair_prompt, max_tokens=max_tokens)
        if repair_resp.error:
            raise ValueError(f"JSON repair API error: {repair_resp.error}")
        return json.loads(_extract_json_payload(repair_resp.content or "{}"))


def _identity(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("requirement", "")).strip()
    return str(item).strip()


def _extract_logical_id(text: str) -> str:
    match = re.match(r"^(?:v?\d+(?:\.\d+)*\s*-\s*)?v?(\d+(?:\.\d+)*)\b", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return text


def _coerce_asvs_level(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and value in (1, 2, 3):
        return value
    text = str(value).strip().upper()
    if text.startswith("L"):
        text = text[1:].strip()
    if text in {"1", "2", "3"}:
        return int(text)
    return None


def _clean_asvs_level_definitions(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, dict):
        raw_items = parsed.get("levels") or parsed.get("asvs_levels") or []
    elif isinstance(parsed, list):
        raw_items = parsed
    else:
        raw_items = []

    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        level = _coerce_asvs_level(item.get("level") or item.get("code"))
        if level is None or level in seen:
            continue
        name = str(item.get("name") or f"Level {level}").strip()
        description = str(item.get("description") or "").strip()
        guidance = str(
            item.get("classification_guidance")
            or item.get("guidance")
            or item.get("classification")
            or description
        ).strip()
        if not guidance:
            continue
        cleaned.append(
            {
                "level": level,
                "code": f"L{level}",
                "name": name,
                "description": description or guidance,
                "classification_guidance": guidance,
                "source_quote": str(item.get("source_quote") or item.get("verbatim_quote") or "").strip(),
                "context_marker": str(item.get("context_marker") or "").strip(),
            }
        )
        seen.add(level)
    return sorted(cleaned, key=lambda item: item["level"])


def _is_non_requirement_note(*values: str) -> bool:
    for value in values:
        text = str(value or "").strip()
        if text and _NON_REQUIREMENT_NOTE_RE.match(text):
            return True
    return False


def _is_heading_only_requirement(requirement: str) -> bool:
    text = str(requirement or "").strip()
    if not text:
        return False
    if _OWASP_SUBSECTION_HEADING_ONLY_RE.match(text) and " - " not in text:
        if re.search(r"\b\d+\.\d+\.\d+\b", text):
            return False
        return True
    return False


def clean_structured_requirements(parsed: Any) -> Dict[str, List[Any]]:
    if not isinstance(parsed, dict):
        return {}
    cleaned_dict: Dict[str, List[Any]] = {}
    for section, raw_requirements in parsed.items():
        if not isinstance(raw_requirements, list):
            continue
        cleaned_reqs: List[Any] = []
        for item in raw_requirements:
            if isinstance(item, dict):
                req = str(item.get("requirement", "")).strip()
                details = str(item.get("details", "")).strip()
                verbatim_quote = str(item.get("verbatim_quote", "")).strip()
                context_marker = str(item.get("context_marker", "")).strip()
                if len(req) < 8 and len(details) < 8:
                    continue
                if _is_non_requirement_note(req, details, verbatim_quote):
                    continue
                if _is_heading_only_requirement(req):
                    continue
                cleaned_reqs.append(
                    {
                        "requirement": req,
                        "details": details,
                        "verbatim_quote": verbatim_quote,
                        "context_marker": context_marker,
                        "asvs_level": _coerce_asvs_level(item.get("asvs_level")),
                    }
                )
                continue
            if isinstance(item, str):
                text = item.strip()
                if len(text) >= 8 and not _is_non_requirement_note(text):
                    cleaned_reqs.append(text)
        if cleaned_reqs:
            cleaned_dict[section] = cleaned_reqs
    return cleaned_dict


def _get_item_length(item: Any) -> int:
    if isinstance(item, dict):
        return len(str(item.get("requirement", ""))) + len(str(item.get("details", "")))
    return len(str(item))


def _merge_requirement_metadata(primary: Any, secondary: Any) -> Any:
    if not isinstance(primary, dict) or not isinstance(secondary, dict):
        return primary
    merged = dict(primary)
    if merged.get("asvs_level") is None and secondary.get("asvs_level") is not None:
        merged["asvs_level"] = secondary.get("asvs_level")
    for field_name in ("verbatim_quote", "context_marker"):
        if not str(merged.get(field_name, "")).strip() and str(secondary.get(field_name, "")).strip():
            merged[field_name] = secondary.get(field_name)
    return merged


def _backfill_requirement_levels(
    requirements_by_section: Dict[str, List[Any]],
    logical_id_to_level: Dict[str, int],
) -> int:
    backfilled = 0
    if not logical_id_to_level:
        return backfilled
    for requirements in requirements_by_section.values():
        for item in requirements:
            if not isinstance(item, dict):
                continue
            if item.get("asvs_level") is not None:
                continue
            logical_id = _extract_logical_id(str(item.get("requirement", "")).strip())
            level = logical_id_to_level.get(logical_id)
            if level in (1, 2, 3):
                item["asvs_level"] = level
                backfilled += 1
    return backfilled


def _canonicalize_diagram_requirements(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    canonical_items: List[Dict[str, Any]] = []
    exact_seen = set()
    stable_key_counts: Dict[str, int] = {}
    used_stable_keys = set()
    for item in items:
        exact_fingerprint = (
            str(item.get("stable_key", "")).strip(),
            str(item.get("source_requirement_key", "")).strip(),
            str(item.get("requirement_text", "")).strip(),
            str(item.get("verification_hint", "")).strip(),
            item.get("asvs_level"),
            str(item.get("parent_section", "")).strip(),
        )
        if exact_fingerprint in exact_seen:
            continue
        exact_seen.add(exact_fingerprint)
        normalized_item = dict(item)
        base_stable_key = str(normalized_item.get("stable_key", "")).strip() or "diagram-requirement"
        occurrence = stable_key_counts.get(base_stable_key, 0) + 1
        candidate_key = f"{base_stable_key}-{occurrence}" if occurrence > 1 else base_stable_key
        while candidate_key in used_stable_keys:
            occurrence += 1
            candidate_key = f"{base_stable_key}-{occurrence}"
        stable_key_counts[base_stable_key] = occurrence
        normalized_item["stable_key"] = candidate_key
        used_stable_keys.add(candidate_key)
        canonical_items.append(normalized_item)
    return canonical_items


def _merge_requirements(
    base: Dict[str, List[Any]],
    incoming: Dict[str, List[Any]],
) -> Dict[str, List[Any]]:
    for section, new_reqs in incoming.items():
        if not new_reqs:
            continue
        if section not in base:
            base[section] = list(new_reqs)
            continue
        existing_items = base[section]
        existing_by_identity = {_identity(item): i for i, item in enumerate(existing_items)}
        existing_by_logical_id = {}
        for i, item in enumerate(existing_items):
            ident = _identity(item)
            log_id = _extract_logical_id(ident)
            if log_id != ident:
                existing_by_logical_id[log_id] = i
        for req in new_reqs:
            req_identity = _identity(req)
            if not req_identity:
                continue
            req_log_id = _extract_logical_id(req_identity)
            if req_identity in existing_by_identity:
                existing_idx = existing_by_identity[req_identity]
                existing_item = existing_items[existing_idx]
                if _get_item_length(req) > _get_item_length(existing_item):
                    existing_items[existing_idx] = _merge_requirement_metadata(req, existing_item)
                else:
                    existing_items[existing_idx] = _merge_requirement_metadata(existing_item, req)
                continue
            if req_log_id != req_identity and req_log_id in existing_by_logical_id:
                existing_idx = existing_by_logical_id[req_log_id]
                existing_item = existing_items[existing_idx]
                if _get_item_length(req) > _get_item_length(existing_item):
                    existing_items[existing_idx] = _merge_requirement_metadata(req, existing_item)
                    existing_by_identity[req_identity] = existing_idx
                    old_ident = _identity(existing_item)
                    if old_ident in existing_by_identity and old_ident != req_identity:
                        del existing_by_identity[old_ident]
                else:
                    existing_items[existing_idx] = _merge_requirement_metadata(existing_item, req)
                continue
            existing_items.append(req)
            new_idx = len(existing_items) - 1
            existing_by_identity[req_identity] = new_idx
            if req_log_id != req_identity:
                existing_by_logical_id[req_log_id] = new_idx
    return base
