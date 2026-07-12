from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.client.session import capture_current_context
from sdr.apps.ai.prompts.extraction import (
    DIAGRAM_REQ_EXTRACTION_SYSTEM_PROMPT,
    REQUIREMENT_CATEGORY_VALIDATION_SYSTEM_PROMPT,
    VALID_DIAGRAM_TYPES,
    build_diagram_req_extraction_prompt,
    build_hierarchical_extraction_prompt,
    build_requirement_category_validation_prompt,
)
from sdr.apps.ai.utils.chunking import chunk_text_semantically
from sdr.apps.ai.utils.concurrency import ConcurrencyProbe
from sdr.apps.standards.models import StandardSourceDocument

from .config import ExtractionConfig
from .document_reader import StandardDocumentReader
from .normalizers import (
    _canonicalize_diagram_requirements,
    _count_tokens,
    clean_structured_requirements,
    parse_json_response,
    parse_json_with_repair,
)
from .screening import StandardScreeningService

logger = logging.getLogger(__name__)
_AI_RESPONSE_PREVIEW_LIMIT = 800

_CHAPTER_HEADING_RE = re.compile(r"^#{1,6}\s*(V\d+)(?!\.\d)\s+(.+?)\s*$", re.MULTILINE)
_CHUNK_BANNER_RE = re.compile(r"^--- DOCUMENT CHUNK \d+ OF \d+ ---\n\n")
_MID_CHAPTER_HEADING_LOOKAHEAD_CHARS = 300
_VALID_REQUIREMENT_CATEGORIES = {"design", "code", "infrastructure", "process"}
_CONTROL_ID_RE = re.compile(r"\b(?:[A-Z]{2,}(?:-[A-Z0-9]+)+|[Vv]?\d+\.\d+\.\d+(?:\.\d+)*)\b")


def _annotate_chunks_with_chapter_context(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    active_heading: Optional[str] = None
    for chunk in chunks:
        text = chunk.get("text", "")
        banner_match = _CHUNK_BANNER_RE.match(text)
        banner = banner_match.group(0) if banner_match else ""
        body = text[len(banner):]

        matches = list(_CHAPTER_HEADING_RE.finditer(body))
        first_heading_offset = matches[0].start() if matches else None
        opens_mid_chapter = active_heading is not None and (
            first_heading_offset is None
            or first_heading_offset > _MID_CHAPTER_HEADING_LOOKAHEAD_CHARS
        )

        if opens_mid_chapter:
            chapter_tag, _, chapter_title = active_heading.partition(" ")
            context_line = (
                f'[CONTEXT: This chunk continues the chapter titled "{chapter_title}" '
                f'(JSON section key: "{active_heading}"). Keep each requirement\'s own ID '
                f'in its original bare numeric form, e.g. "8.1.1" -- never fuse the chapter '
                f'letter "{chapter_tag}" onto it.]\n\n'
            )
            chunk["text"] = banner + context_line + body

        if matches:
            last_match = matches[-1]
            active_heading = f"{last_match.group(1)} {last_match.group(2)}".strip()

    return chunks


def _complete_json(
    *,
    chat_completion_fn: Callable[..., Any],
    system_prompt: str,
    user_prompt: str,
    component: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[Dict[str, Any]] = None,
) -> Any:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    kwargs: Dict[str, Any] = {
        "messages": messages,
        "component": component,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    return chat_completion_fn(**kwargs)


def _repair_json(
    *,
    chat_completion_fn: Callable[..., Any],
    user_prompt: str,
    max_tokens: int,
) -> Any:
    return chat_completion_fn(
        messages=[{"role": "user", "content": user_prompt}],
        component="fallback",
        temperature=0.0,
        max_tokens=max_tokens,
    )


def _extract_document_requirements(
    *,
    document_reader: StandardDocumentReader,
    structured_extractor: "StructuredRequirementExtractionService",
    config: ExtractionConfig,
    source_doc: StandardSourceDocument,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    progress_callback=None,
) -> Dict[str, List[Any]]:
    if progress_callback:
        progress_callback("Reading Document", 5)
    try:
        content = document_reader.read_source_document(
            source_doc,
            start_page=start_page,
            end_page=end_page,
        )
        source_doc_text = (content or {}).get("text") or ""
        logger.info(
            "extract_requirements_from_document: conversion_method=%s for standard '%s'.",
            (content or {}).get("conversion_method", "unknown"),
            source_doc.name,
        )
    except Exception as exc:
        logger.error(
            "extract_requirements_from_document: ✗ [INGEST] failed to access standard file '%s': %s (type=%s)",
            source_doc.name,
            exc,
            type(exc).__name__,
        )
        return {}

    if not source_doc_text.strip():
        logger.warning(
            "extract_requirements_from_document: ✗ [INGEST] document '%s' yielded no extractable text (text length=%d); skipping AI call.",
            source_doc.name,
            len(source_doc_text),
        )
        return {}

    if progress_callback:
        progress_callback("Screening Document", 10)
    
    screening_service = StandardScreeningService(chat_completion_fn=structured_extractor.chat_completion)
    screening_service.screen_document(source_doc_text)

    if progress_callback:
        progress_callback("Chunking Document", 15)
    chunking_started_at = time.monotonic()
    chunks = chunk_text_semantically(
        source_doc_text,
        chunk_size=config.standard_extraction_chunk_token_target,
    )
    logger.info(
        "extract_requirements_from_document: [CHUNK] produced %d chunk(s) in %.2fs for standard '%s'.",
        len(chunks),
        time.monotonic() - chunking_started_at,
        source_doc.name,
    )
    if not chunks:
        logger.warning(
            "extract_requirements_from_document: ✗ [CHUNK] chunker returned no chunks for standard '%s' (text length was %d chars). Aborting.",
            source_doc.name,
            len(source_doc_text),
        )
        return {}

    chunks = _annotate_chunks_with_chapter_context(chunks)

    total_chunks = len(chunks)
    merged: Dict[str, List[Any]] = {}
    successful_chunks = 0
    failed_chunks = 0
    total_sections_seen = 0
    total_reqs_seen = 0
    total_tokens_seen = 0

    def _process_chunk(idx: int, text: str):
        thread_name = threading.current_thread().name
        token_count = _count_tokens(text)
        logger.info(
            "extract_requirements_from_document: [MAP %d/%d] STARTING on thread %s (~%d tokens)",
            idx,
            total_chunks,
            thread_name,
            token_count,
        )
        chunk_started_at = time.monotonic()
        result = structured_extractor.extract(text)
        elapsed = time.monotonic() - chunk_started_at
        logger.info(
            "extract_requirements_from_document: [MAP %d/%d] FINISHED on thread %s in %.2fs",
            idx,
            total_chunks,
            thread_name,
            elapsed,
        )
        return result, token_count

    map_reduce_started_at = time.monotonic()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=config.standard_extraction_max_workers,
        thread_name_prefix="ExtractWorker",
    )
    try:
        future_to_chunk = {
            executor.submit(capture_current_context(_process_chunk), idx, chunk_dict["text"]): (idx, chunk_dict)
            for idx, chunk_dict in enumerate(chunks, start=1)
        }
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            idx, _chunk_dict = future_to_chunk[future]
            completed_count += 1
            if progress_callback:
                progress_callback(
                    f"Extracting chunk {completed_count} of {total_chunks}",
                    15 + int((completed_count / max(total_chunks, 1)) * 80),
                )
            try:
                chunk_result, chunk_tokens = future.result()
            except Exception as exc:
                logger.error(
                    "extract_requirements_from_document: [MAP %d/%d] ✗ chunk failed with exception %s for standard '%s'",
                    idx,
                    total_chunks,
                    exc,
                    source_doc.name,
                )
                chunk_result, chunk_tokens = {}, 0
            total_tokens_seen += chunk_tokens
            if not chunk_result:
                failed_chunks += 1
                continue
            total_sections_seen += len(chunk_result)
            total_reqs_seen += sum(len(v) for v in chunk_result.values())
            for section, reqs in chunk_result.items():
                if section not in merged:
                    merged[section] = []
                merged[section].extend(reqs)
            successful_chunks += 1
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    logger.info(
        "extract_requirements_from_document: ✓ [MAP-REDUCE] complete for standard '%s' in %.2fs. Chunks: %d succeeded, %d failed. Total sections seen=%d, total reqs seen=%d, total tokens=%d. Final merged: %d section(s), %d total requirement(s).",
        source_doc.name,
        time.monotonic() - map_reduce_started_at,
        successful_chunks,
        failed_chunks,
        total_sections_seen,
        total_reqs_seen,
        total_tokens_seen,
        len(merged),
        sum(len(v) for v in merged.values()),
    )

    return merged
class RequirementCategoryValidationService:
    def __init__(self, *, chat_completion_fn: Callable[..., Any], config: ExtractionConfig) -> None:
        self.chat_completion = chat_completion_fn
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate(self, requirements_by_section: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        indexed_items: list[tuple[str, int, Dict[str, Any]]] = []
        for section, requirements in requirements_by_section.items():
            for item_index, item in enumerate(requirements):
                if isinstance(item, dict):
                    indexed_items.append((section, item_index, item))

        for batch_start in range(0, len(indexed_items), self.config.standard_category_validation_batch_size):
            batch = indexed_items[
                batch_start : batch_start + self.config.standard_category_validation_batch_size
            ]
            payload = []
            for local_index, (_section, _item_index, item) in enumerate(batch):
                requirement_text = str(item.get("requirement", "")).strip()
                control_id_match = _CONTROL_ID_RE.search(requirement_text)
                payload.append(
                    {
                        "index": local_index,
                        "control_id": control_id_match.group(0) if control_id_match else "",
                        "requirement_text": requirement_text,
                        "extracted_category": str(item.get("requirement_category", "design")).lower(),
                    }
                )

            response = _complete_json(
                chat_completion_fn=self.chat_completion,
                system_prompt=REQUIREMENT_CATEGORY_VALIDATION_SYSTEM_PROMPT,
                user_prompt=build_requirement_category_validation_prompt(
                    json.dumps(payload, ensure_ascii=False)
                ),
                component="standard_category_validation",
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            if response.error:
                self.logger.warning(
                    "category validation failed for batch starting at %d: %s; preserving extracted labels",
                    batch_start,
                    response.error,
                )
                continue

            try:
                parsed = parse_json_response(response.content or "{}")
                validated = parsed.get("items") if isinstance(parsed, dict) else None
                if not isinstance(validated, list):
                    raise ValueError("response does not contain an items list")
                labels: Dict[int, str] = {}
                for result in validated:
                    if not isinstance(result, dict):
                        raise ValueError("response contains a non-object item")
                    index = result.get("index")
                    category = str(result.get("requirement_category", "")).lower().strip()
                    if not isinstance(index, int) or index in labels:
                        raise ValueError("response contains an invalid or duplicate index")
                    if category not in _VALID_REQUIREMENT_CATEGORIES:
                        raise ValueError(f"response contains invalid category {category!r}")
                    labels[index] = category
                if set(labels) != set(range(len(batch))):
                    raise ValueError("response does not classify every input item exactly once")
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.logger.warning(
                    "category validation returned unusable output for batch starting at %d: %s; preserving extracted labels",
                    batch_start,
                    exc,
                )
                continue

            for local_index, (_section, _item_index, item) in enumerate(batch):
                item["requirement_category"] = labels[local_index]

        for _section, _item_index, item in indexed_items:
            current_category = str(item.get("requirement_category", "design")).lower().strip()
            if current_category not in _VALID_REQUIREMENT_CATEGORIES:
                current_category = "design"
            item["requirement_category"] = current_category

        return requirements_by_section


class StructuredRequirementExtractionService:
    def __init__(
        self,
        *,
        chat_completion_fn: Callable[..., Any] = chat_completion,
        config: Optional[ExtractionConfig] = None,
    ) -> None:
        self.chat_completion = chat_completion_fn
        self.config = config or ExtractionConfig.from_settings()
        self.category_validator = RequirementCategoryValidationService(
            chat_completion_fn=chat_completion_fn,
            config=self.config,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def extract(self, source_doc_text: str) -> Dict[str, List[Any]]:
        self.logger.debug(
            "extract_structured_requirements: starting extraction. Text length=%d chars, %d lines.",
            len(source_doc_text),
            len(source_doc_text.split("\n")),
        )
        prompt = build_hierarchical_extraction_prompt(source_doc_text)
        try:
            response = _complete_json(
                chat_completion_fn=self.chat_completion,
                system_prompt=(
                    "You are an expert security analyst. "
                    "Extract ONLY actionable, technical security requirements from the real "
                    "standard body content and output strictly valid JSON. "
                    "Ignore table-of-contents, document scopes, "
                    "introductions, generic domains, and compliance/verification processes. "
                    "Do not output analysis, reasoning, or <thinking> tags."
                ),
                user_prompt=prompt,
                component="standard_extraction",
                temperature=0.05,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )

            response_preview = (response.content or "").replace("\n", " ").replace("\r", " ")
            if len(response_preview) > _AI_RESPONSE_PREVIEW_LIMIT:
                response_preview = response_preview[:_AI_RESPONSE_PREVIEW_LIMIT] + "..."
            self.logger.info(
                "extract_structured_requirements: AI response received. model=%s provider=%s usage=%s preview=%s",
                getattr(response, "model", "unknown"),
                getattr(getattr(response, "provider", None), "value", getattr(response, "provider", "unknown")),
                getattr(response, "usage", None),
                response_preview,
            )
            if response.error:
                self.logger.error("extract_structured_requirements: API returned error: %s", response.error)
                return {}
            parsed = parse_json_with_repair(
                response.content or "{}",
                repair_json=lambda **kwargs: _repair_json(
                    chat_completion_fn=self.chat_completion,
                    **kwargs,
                ),
                max_tokens=8192,
            )
            return self.category_validator.validate(clean_structured_requirements(parsed))
        except Exception as exc:
            self.logger.error(
                "extract_structured_requirements: unexpected error processing AI response: %s (type=%s)",
                exc,
                type(exc).__name__,
            )
            return {}


class RequirementDocumentExtractionService:
    def __init__(
        self,
        *,
        document_reader: StandardDocumentReader,
        structured_extractor: StructuredRequirementExtractionService,
        config: ExtractionConfig,
        ) -> None:
        self.document_reader = document_reader
        self.structured_extractor = structured_extractor
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def extract(
        self,
        source_doc: StandardSourceDocument,
        *,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        progress_callback=None,
    ) -> Dict[str, List[Any]]:
        self.logger.info(
            "extract_requirements_from_document: [ENTRY] standard_id=%s, name='%s', document='%s'",
            source_doc.id,
            source_doc.name,
            source_doc.document if source_doc.document else "None",
        )
        return _extract_document_requirements(
            document_reader=self.document_reader,
            structured_extractor=self.structured_extractor,
            config=self.config,
            source_doc=source_doc,
            start_page=start_page,
            end_page=end_page,
            progress_callback=progress_callback,
        )



class DiagramRequirementExtractionService:
    def __init__(
        self,
        *,
        chat_completion_fn: Callable[..., Any] = chat_completion,
        config: ExtractionConfig,
    ) -> None:
        self.chat_completion = chat_completion_fn
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def extract(self, *, parameters: list, category_id: int, ingestion_job_id: int) -> list:
        if not parameters:
            self.logger.info("extract_diagram_requirements: no parameters provided, skipping.")
            return []

        param_lines = []
        for param in parameters:
            stable_key_val = getattr(param, "stable_key", "")
            req_text = getattr(param, "requirement_text", "")
            parent_title = getattr(getattr(param, "parent", None), "title", "") or ""
            line = f"[{stable_key_val}] [{parent_title}] {req_text}"
            param_lines.append(line)

        batch_size = 20
        batched_lines = [
            param_lines[start : start + batch_size]
            for start in range(0, len(param_lines), batch_size)
        ]
        total_batches = len(batched_lines)
        max_concurrency = max(
            1,
            min(self.config.diagram_requirement_extraction_max_concurrency, total_batches),
        )
        probe = ConcurrencyProbe(max_concurrency=max_concurrency)
        probe.mark_submitted(total_batches)

        def _process_diagram_batch(batch_index: int, batch: List[str]) -> tuple[int, list]:
            thread_name = threading.current_thread().name
            self.logger.info(
                "extract_diagram_requirements: [BATCH %d/%d] START thread=%s params=%d",
                batch_index + 1,
                total_batches,
                thread_name,
                len(batch),
            )
            response = _complete_json(
                chat_completion_fn=self.chat_completion,
                system_prompt=DIAGRAM_REQ_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=build_diagram_req_extraction_prompt(
                    requirements_text="\n".join(batch)
                ),
                component="diagram_requirement_extraction",
                temperature=0.0,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            if response.error:
                self.logger.error(
                    "extract_diagram_requirements: LLM error for batch %d: %s",
                    batch_index,
                    response.error,
                )
                return batch_index, []
            try:
                parsed = parse_json_response(response.content or "{}")
            except Exception:
                self.logger.error(
                    "extract_diagram_requirements: JSON parse error for batch %d",
                    batch_index,
                )
                return batch_index, []

            batch_results = []
            for item_index, item in enumerate(parsed.get("diagram_requirements") or []):
                if not isinstance(item, dict):
                    continue
                req_text = str(item.get("requirement_text", "")).strip()
                if not req_text:
                    continue
                stable_key_val = str(item.get("stable_key", "")).strip() or f"D-batch{batch_index}-{item_index}"
                stable_key_val = f"job{ingestion_job_id}-{stable_key_val}"
                raw_diagram_type = str(item.get("diagram_type", "")).strip().lower()
                diagram_type = raw_diagram_type if raw_diagram_type in VALID_DIAGRAM_TYPES else ""
                batch_results.append(
                    {
                        "category_id": category_id,
                        "ingestion_job_id": ingestion_job_id,
                        "stable_key": stable_key_val,
                        "source_requirement_key": str(item.get("source_requirement_id", "composite")).strip(),
                        "requirement_text": req_text[:200],
                        "verification_hint": str(item.get("verification_hint", "")).strip(),
                        "parent_section": str(item.get("parent_section", "General")).strip()[:255],
                        "diagram_type": diagram_type,
                    }
                )
            self.logger.info(
                "extract_diagram_requirements: [BATCH %d/%d] FINISH thread=%s generated=%d",
                batch_index + 1,
                total_batches,
                thread_name,
                len(batch_results),
            )
            return batch_index, batch_results

        ordered_results: Dict[int, list] = {}
        wrapped_batch_processor = probe.wrap(_process_diagram_batch)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="DiagramReqExtract",
        ) as executor:
            future_map = {
                executor.submit(capture_current_context(wrapped_batch_processor), batch_index, batch): batch_index
                for batch_index, batch in enumerate(batched_lines)
            }
            for future in concurrent.futures.as_completed(future_map):
                batch_index = future_map[future]
                try:
                    completed_batch_index, batch_results = future.result()
                except Exception as exc:
                    self.logger.exception(
                        "extract_diagram_requirements: failed for batch %d: %s",
                        batch_index,
                        exc,
                    )
                    continue
                ordered_results[completed_batch_index] = batch_results

        all_results = []
        for batch_index in range(total_batches):
            all_results.extend(ordered_results.get(batch_index, []))
        all_results = _canonicalize_diagram_requirements(all_results)

        for ordinal, item in enumerate(all_results, start=1):
            item["ordinal"] = ordinal

        self.logger.info("extract_diagram_requirements: concurrency=%s", probe.snapshot().to_dict())
        self.logger.info(
            "extract_diagram_requirements: generated %d diagram requirements from %d text parameters (category_id=%s, job_id=%s)",
            len(all_results),
            len(parameters),
            category_id,
            ingestion_job_id,
        )
        return all_results
