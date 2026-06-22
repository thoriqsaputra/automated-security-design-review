from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Any, Dict, List, Optional

from sdr.apps.ai.prompts.extraction import (
    DIAGRAM_REQ_EXTRACTION_SYSTEM_PROMPT,
    build_diagram_req_extraction_prompt,
    build_hierarchical_extraction_prompt,
)
from sdr.apps.ai.utils.chunking import chunk_text_semantically
from sdr.apps.ai.utils.concurrency import ConcurrencyProbe
from sdr.apps.standards.models import StandardSourceDocument

from .config import ExtractionConfig
from .document_reader import StandardDocumentReader
from .llm_client import ExtractionLLMClient
from .normalizers import (
    _canonicalize_diagram_requirements,
    _count_tokens,
    _remove_table_of_contents,
    clean_structured_requirements,
    parse_json_response,
    parse_json_with_repair,
)

logger = logging.getLogger(__name__)
_AI_RESPONSE_PREVIEW_LIMIT = 800


def _extract_document_requirements(
    *,
    document_reader: StandardDocumentReader,
    structured_extractor: "StructuredRequirementExtractionService",
    requirement_level_detector: Optional[Any],
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

    source_doc_text = _remove_table_of_contents(source_doc_text)
    if not source_doc_text.strip():
        logger.warning(
            "extract_requirements_from_document: ✗ [PREPROCESS] document '%s' had no extractable body text after table-of-contents removal.",
            source_doc.name,
        )
        return {}

    if progress_callback:
        progress_callback("Chunking Document", 15)
    chunks = chunk_text_semantically(
        source_doc_text,
        chunk_size=config.standard_extraction_chunk_token_target,
    )
    if not chunks:
        logger.warning(
            "extract_requirements_from_document: ✗ [CHUNK] chunker returned no chunks for standard '%s' (text length was %d chars). Aborting.",
            source_doc.name,
            len(source_doc_text),
        )
        return {}

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
        try:
            result = structured_extractor.extract(text, source_name=source_doc.name)
        except TypeError:
            result = structured_extractor.extract(text)
        logger.info(
            "extract_requirements_from_document: [MAP %d/%d] FINISHED on thread %s",
            idx,
            total_chunks,
            thread_name,
        )
        return result, token_count

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=config.standard_extraction_max_workers,
        thread_name_prefix="ExtractWorker",
    )
    try:
        future_to_chunk = {
            executor.submit(_process_chunk, idx, chunk_dict["text"]): (idx, chunk_dict)
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
        "extract_requirements_from_document: ✓ [MAP-REDUCE] complete for standard '%s'. Chunks: %d succeeded, %d failed. Total sections seen=%d, total reqs seen=%d, total tokens=%d. Final merged: %d section(s), %d total requirement(s).",
        source_doc.name,
        successful_chunks,
        failed_chunks,
        total_sections_seen,
        total_reqs_seen,
        total_tokens_seen,
        len(merged),
        sum(len(v) for v in merged.values()),
    )

    return merged


class StructuredRequirementExtractionService:
    def __init__(self, *, llm_client: ExtractionLLMClient) -> None:
        self.llm_client = llm_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def extract(self, source_doc_text: str, *, source_name: str = "") -> Dict[str, List[Any]]:
        self.logger.debug(
            "extract_structured_requirements: starting extraction. Text length=%d chars, %d lines.",
            len(source_doc_text),
            len(source_doc_text.split("\n")),
        )
        prompt = build_hierarchical_extraction_prompt(source_doc_text)
        try:
            response = self.llm_client.complete_json(
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
                llm_client=self.llm_client,
                max_tokens=8192,
            )
            return clean_structured_requirements(parsed)
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
        requirement_level_detector: Optional[Any] = None,
        config: ExtractionConfig,
        ) -> None:
        self.document_reader = document_reader
        self.structured_extractor = structured_extractor
        self.requirement_level_detector = requirement_level_detector
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
            requirement_level_detector=self.requirement_level_detector,
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
        llm_client: ExtractionLLMClient,
        config: ExtractionConfig,
    ) -> None:
        self.llm_client = llm_client
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
            details = getattr(param, "details", "") or ""
            parent_title = getattr(getattr(param, "parent", None), "title", "") or ""
            line = f"[{stable_key_val}] [{parent_title}] {req_text}"
            if details:
                line += f" | {details[:120]}"
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
            response = self.llm_client.complete_json(
                system_prompt=DIAGRAM_REQ_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=build_diagram_req_extraction_prompt(requirements_text="\n".join(batch)),
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
                batch_results.append(
                    {
                        "category_id": category_id,
                        "ingestion_job_id": ingestion_job_id,
                        "stable_key": stable_key_val,
                        "source_requirement_key": str(item.get("source_requirement_id", "composite")).strip(),
                        "requirement_text": req_text[:200],
                        "verification_hint": str(item.get("verification_hint", "")).strip(),
                        "parent_section": str(item.get("parent_section", "General")).strip()[:255],
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
                executor.submit(wrapped_batch_processor, batch_index, batch): batch_index
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
