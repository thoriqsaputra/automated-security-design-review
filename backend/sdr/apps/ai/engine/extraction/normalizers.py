from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

import tiktoken

from sdr.apps.ai.prompts.extraction import build_json_repair_prompt
from sdr.apps.ai.utils.parsing import strip_markdown_code_blocks, strip_thinking_block
from sdr.apps.standards.utils import build_parameter_analysis_text, normalize_requirement_text

logger = logging.getLogger(__name__)

_token_encoder = None
_TOC_HEADING_RE = re.compile(
    r"^\s*(daftar\s+isi|table\s+of\s+contents|contents)\s*$",
    re.IGNORECASE,
)
_TOC_ENTRY_DOTTED_RE = re.compile(r"\.{2,}\s*\d{1,4}\s*$")
_TOC_ENTRY_SPACED_RE = re.compile(r".{6,}\s{2,}\d{1,4}\s*$")
_TOC_LABEL_RE = re.compile(r"^\s*(halaman|page|pages?)\s*$", re.IGNORECASE)
_NOTE_PREFIX_RE = re.compile(r"^\s*note\s*:", re.IGNORECASE)
_OWASP_SUBSECTION_HEADING_RE = re.compile(
    r"^\s*V\d+\.\d+\b(?!\s*-\s*\d+\.\d+\.\d+\b)",
    re.IGNORECASE,
)
_CONTROL_ID_RE = re.compile(r"\b(?:[A-Z]{2,}(?:-[A-Z0-9]+)+|\d+\.\d+\.\d+(?:\.\d+)*)\b")
_DELETED_RESERVED_RE = re.compile(r"\[\s*(?:deleted|reserved|blank)\b", re.IGNORECASE)


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


def _extract_logical_id(text: str) -> str:
    match = re.match(r"^(?:v?\d+(?:\.\d+)*\s*-\s*)?v?(\d+(?:\.\d+)*)\b", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return text


def _is_deleted_reserved_or_blank(requirement: str, *, verbatim_quote: str = "") -> bool:
    combined = " ".join(part for part in (requirement, verbatim_quote) if part).strip()
    if not combined:
        return True
    return bool(_DELETED_RESERVED_RE.search(combined))


def _looks_like_owasp_subsection_heading(requirement: str) -> bool:
    text = (requirement or "").strip()
    if not text:
        return False
    if not _OWASP_SUBSECTION_HEADING_RE.match(text):
        return False
    return _CONTROL_ID_RE.search(text) is None


def _extract_requirement_anchor(requirement: str) -> str:
    text = (requirement or "").strip()
    if not text:
        return ""
    match = _CONTROL_ID_RE.search(text)
    return match.group(0).lower() if match else ""


def _should_skip_requirement_item(item: Dict[str, Any]) -> bool:
    requirement = str(item.get("requirement", "")).strip()
    details = str(item.get("details", "")).strip()
    verbatim_quote = str(item.get("verbatim_quote", "")).strip()
    if not requirement and not details:
        return True
    if _NOTE_PREFIX_RE.match(requirement):
        return True
    if _is_deleted_reserved_or_blank(requirement, verbatim_quote=verbatim_quote):
        return True
    if _looks_like_owasp_subsection_heading(requirement):
        return True
    return False


def _canonical_requirement_key(item: Dict[str, Any]) -> str:
    requirement = str(item.get("requirement", "")).strip()
    details = str(item.get("details", "")).strip()
    anchor = _extract_requirement_anchor(requirement)
    if anchor:
        return f"id:{anchor}"
    analysis_text = build_parameter_analysis_text(requirement, details)
    return f"text:{normalize_requirement_text(analysis_text)}"


def _requirement_richness(item: Dict[str, Any]) -> tuple:
    return (
        len(str(item.get("details", "")).strip()),
        1 if str(item.get("context_marker", "")).strip() else 0,
        1 if str(item.get("verbatim_quote", "")).strip() else 0,
    )


def canonicalize_requirement_items(items: List[Any]) -> List[Dict[str, Any]]:
    canonical_items: List[Dict[str, Any]] = []
    index_by_key: Dict[str, int] = {}
    for item in items or []:
        if isinstance(item, dict):
            normalized_item = {
                "requirement": str(item.get("requirement", "")).strip(),
                "details": str(item.get("details", "")).strip(),
                "verbatim_quote": str(item.get("verbatim_quote", "")).strip(),
                "context_marker": str(item.get("context_marker", "")).strip(),
            }
        elif isinstance(item, str):
            text = str(item).strip()
            normalized_item = {
                "requirement": text,
                "details": "",
                "verbatim_quote": "",
                "context_marker": "",
            }
        else:
            continue
        if _should_skip_requirement_item(normalized_item):
            continue
        key = _canonical_requirement_key(normalized_item)
        existing_idx = index_by_key.get(key)
        if existing_idx is None:
            index_by_key[key] = len(canonical_items)
            canonical_items.append(normalized_item)
            continue
        if _requirement_richness(normalized_item) > _requirement_richness(canonical_items[existing_idx]):
            canonical_items[existing_idx] = normalized_item
    return canonical_items


def canonicalize_structured_requirements(
    requirements_by_section: Dict[str, List[Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    canonicalized: Dict[str, List[Dict[str, Any]]] = {}
    for section, raw_requirements in (requirements_by_section or {}).items():
        cleaned_items = canonicalize_requirement_items(raw_requirements)
        if cleaned_items:
            canonicalized[str(section).strip()] = cleaned_items
    return canonicalized


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
                context_marker = str(item.get("context_marker", "")).strip()
                if not req and not details:
                    continue
                cleaned_reqs.append(
                    {
                        "requirement": req,
                        "details": details,
                        "verbatim_quote": str(item.get("verbatim_quote", "")).strip(),
                        "context_marker": context_marker,
                    }
                )
                continue
            if isinstance(item, str):
                text = str(item).strip()
                if text:
                    cleaned_reqs.append(
                        {
                            "requirement": text,
                            "details": "",
                            "verbatim_quote": "",
                            "context_marker": "",
                        }
                    )
        if cleaned_reqs:
            cleaned_dict[str(section).strip()] = cleaned_reqs
    return canonicalize_structured_requirements(cleaned_dict)


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
