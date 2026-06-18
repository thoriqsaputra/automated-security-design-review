from __future__ import annotations

import concurrent.futures
import logging
import threading
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from sdr.apps.ai.prompts.extraction import (
    ASVS_LEVEL_DEFINITIONS_EXTRACTION_SYSTEM_PROMPT,
    CFSR_EXTRACTION_SYSTEM_PROMPT,
    DIAGRAM_REQ_EXTRACTION_SYSTEM_PROMPT,
    build_asvs_level_definitions_extraction_prompt,
    build_cfsr_extraction_prompt,
    build_diagram_req_extraction_prompt,
    build_hierarchical_extraction_prompt,
)
from sdr.apps.ai.utils.chunking import chunk_text_semantically
from sdr.apps.ai.utils.concurrency import ConcurrencyProbe
from sdr.apps.standards.models import StandardSourceDocument

from .config import ExtractionConfig
from .document_reader import StandardDocumentReader
from .llm_client import ExtractionLLMClient
from .page_detection import ASVSPageRangeDetectionService, ASVSRequirementLevelDetectionService
from .normalizers import (
    _backfill_requirement_levels,
    _canonicalize_diagram_requirements,
    _clean_asvs_level_definitions,
    _coerce_asvs_level,
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
    requirement_level_detector: Optional[ASVSRequirementLevelDetectionService],
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

    if requirement_level_detector and merged:
        detected = requirement_level_detector.detect(
            source_doc,
            start_page=start_page,
            end_page=end_page,
        )
        backfilled = _backfill_requirement_levels(merged, detected.levels)
        logger.info(
            "extract_requirements_from_document: backfilled %d ASVS level(s) from deterministic PDF parsing. indexed_rows=%d matched_pages=%d source=%s",
            backfilled,
            len(detected.levels),
            detected.matched_pages,
            detected.source,
        )
    return merged


class ASVSLevelDefinitionExtractionService:
    def __init__(
        self,
        *,
        document_reader: StandardDocumentReader,
        llm_client: ExtractionLLMClient,
    ) -> None:
        self.document_reader = document_reader
        self.llm_client = llm_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def extract(
        self,
        source_doc: StandardSourceDocument,
        *,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        self.logger.info(
            "extract_asvs_level_definitions_from_document: [ENTRY] standard_id=%s name='%s' pages=%s-%s",
            source_doc.id,
            source_doc.name,
            start_page,
            end_page,
        )
        try:
            content = self.document_reader.read_source_document(
                source_doc,
                start_page=start_page,
                end_page=end_page,
            )
            source_doc_text = (content or {}).get("text") or ""
        except Exception as exc:
            self.logger.exception("extract_asvs_level_definitions_from_document: failed to read document: %s", exc)
            return []

        source_doc_text = _remove_table_of_contents(source_doc_text)
        if not source_doc_text.strip():
            self.logger.warning("extract_asvs_level_definitions_from_document: no text extracted")
            return []

        prompt = build_asvs_level_definitions_extraction_prompt(source_doc_text)
        try:
            response = self.llm_client.complete_json(
                system_prompt=ASVS_LEVEL_DEFINITIONS_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                component="standard_extraction",
                temperature=0.0,
                max_tokens=1800,
                response_format={"type": "json_object"},
            )
            if response.error or not response.content:
                self.logger.warning(
                    "extract_asvs_level_definitions_from_document: LLM error=%s",
                    response.error,
                )
                return []
            parsed = parse_json_with_repair(
                response.content or "{}",
                llm_client=self.llm_client,
                max_tokens=1800,
            )
            return _clean_asvs_level_definitions(parsed)
        except Exception as exc:
            self.logger.exception("extract_asvs_level_definitions_from_document: failed: %s", exc)
            return []


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
        requirement_level_detector: Optional[ASVSRequirementLevelDetectionService] = None,
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
            level = getattr(param, "asvs_level", None) or "unknown"
            parent_title = getattr(getattr(param, "parent", None), "title", "") or ""
            line = f"[{stable_key_val}] (L{level}) [{parent_title}] {req_text}"
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
                        "asvs_level": _coerce_asvs_level(item.get("asvs_level")),
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

        ordinal_counters = {}
        for item in all_results:
            level = item.get("asvs_level") or 0
            ordinal_counters[level] = ordinal_counters.get(level, 0) + 1
            item["ordinal"] = ordinal_counters[level]

        self.logger.info("extract_diagram_requirements: concurrency=%s", probe.snapshot().to_dict())
        self.logger.info(
            "extract_diagram_requirements: generated %d diagram requirements from %d text parameters (category_id=%s, job_id=%s)",
            len(all_results),
            len(parameters),
            category_id,
            ingestion_job_id,
        )
        return all_results


class ControlFamilySummaryExtractionService:
    """
    Distills raw CategoryParameterChild records into 3-5 per-family summary
    requirements (CFSRs) per ASVS level, optimised for text-based TSD debate.

    Groups parameters by parent (control family), calls the LLM once per group,
    and returns a list of dicts ready for DB insertion.
    """

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
            self.logger.info("ControlFamilySummaryExtractionService.extract: no parameters provided.")
            return []

        # Group by parent id
        parents_map: Dict[Any, Any] = {}
        for param in parameters:
            parent = getattr(param, "parent", None)
            parent_id = getattr(parent, "id", None)
            if parent_id not in parents_map:
                parents_map[parent_id] = (parent, [])
            parents_map[parent_id][1].append(param)

        parent_groups = list(parents_map.values())
        total_groups = len(parent_groups)
        max_concurrency = max(
            1,
            min(self.config.cfsr_extraction_max_concurrency, total_groups),
        )
        probe = ConcurrencyProbe(max_concurrency=max_concurrency)
        probe.mark_submitted(total_groups)

        def _process_parent_group(group_index: int, parent, children: list) -> tuple:
            thread_name = threading.current_thread().name
            parent_section = getattr(parent, "title", "General") or "General"
            self.logger.info(
                "ControlFamilySummaryExtractionService: [GROUP %d/%d] START thread=%s parent='%s' children=%d",
                group_index + 1,
                total_groups,
                thread_name,
                parent_section,
                len(children),
            )

            param_lines = []
            child_id_to_stable_key: Dict[str, str] = {}
            for child_index, param in enumerate(children):
                real_key = getattr(param, "stable_key", "")
                prompt_id = f"child-{child_index + 1:03d}"
                child_id_to_stable_key[prompt_id] = real_key
                text = getattr(param, "requirement_text", "")
                details = getattr(param, "details", "") or ""
                level = getattr(param, "asvs_level", None) or "unknown"
                line = f"[{prompt_id}] (L{level}) {text}"
                if details:
                    line += f" | {details[:120]}"
                param_lines.append(line)

            requirements_text = "\n".join(param_lines)
            response = self.llm_client.complete_json(
                system_prompt=CFSR_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=build_cfsr_extraction_prompt(
                    requirements_text=requirements_text,
                    parent_section=parent_section,
                    max_count=self.config.cfsr_max_per_parent,
                ),
                component="cfsr_extraction",
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            if response.error:
                self.logger.error(
                    "ControlFamilySummaryExtractionService: LLM error for group %d parent='%s': %s",
                    group_index,
                    parent_section,
                    response.error,
                )
                return group_index, []
            try:
                parsed = parse_json_response(response.content or "{}")
            except Exception as exc:
                self.logger.error(
                    "ControlFamilySummaryExtractionService: JSON parse error for group %d: %s",
                    group_index,
                    exc,
                )
                return group_index, []

            parent_id = getattr(parent, "id", None)
            results = []
            ordinal_counters: Dict[Any, int] = {}
            seen_keys_in_group: set = set()
            for item in parsed.get("summary_requirements") or []:
                if not isinstance(item, dict):
                    continue
                req_text = str(item.get("requirement_text", "")).strip()
                if not req_text:
                    continue
                level = _coerce_asvs_level(item.get("asvs_level"))
                ordinal_counters[level] = ordinal_counters.get(level, 0) + 1
                # Always include group_index so keys from different parent groups
                # never collide even if the LLM produces the same raw stable_key.
                raw_key = str(item.get("stable_key", "")).strip() or f"cfsr-L{level}-{ordinal_counters[level]}"
                stable_key_val = f"job{ingestion_job_id}-g{group_index}-{raw_key}"
                # Deduplicate within a group in case the LLM emits the same key twice
                if stable_key_val in seen_keys_in_group:
                    stable_key_val = f"{stable_key_val}-dup{ordinal_counters[level]}"
                seen_keys_in_group.add(stable_key_val)
                covered_raw = item.get("covered_child_keys")
                if not isinstance(covered_raw, list):
                    covered_raw = []
                covered = []
                for ck in covered_raw:
                    ck_str = str(ck).strip()
                    resolved = child_id_to_stable_key.get(ck_str)
                    if resolved:
                        covered.append(resolved)
                    elif ck_str in child_id_to_stable_key.values():
                        covered.append(ck_str)
                    else:
                        self.logger.debug(
                            "ControlFamilySummaryExtractionService: group %d parent='%s' "
                            "covered_child_keys entry '%s' did not resolve to any known child id",
                            group_index,
                            parent_section,
                            ck_str,
                        )
                results.append({
                    "category_id": category_id,
                    "ingestion_job_id": ingestion_job_id,
                    "parent_id": parent_id,
                    "asvs_level": level,
                    "stable_key": stable_key_val,
                    "requirement_text": req_text[:200],
                    "analysis_hint": str(item.get("analysis_hint", "")).strip(),
                    "covered_child_keys": covered,
                    "ordinal": ordinal_counters.get(level, 1),
                })

            # Pass A: merge near-duplicate CFSRs (same security concept, different wording)
            # Runs BEFORE the cfsr_max_per_parent cap so two near-duplicates that
            # straddle the cutoff get folded into one (combining their
            # covered_child_keys) instead of one being truncated away unmerged.
            _DEDUP_THRESHOLD = 0.65
            merged: list = []
            for r in results:
                matched = False
                for existing in merged:
                    ratio = SequenceMatcher(
                        None,
                        r["requirement_text"].lower(),
                        existing["requirement_text"].lower(),
                    ).ratio()
                    if ratio >= _DEDUP_THRESHOLD:
                        for ck in r["covered_child_keys"]:
                            if ck not in existing["covered_child_keys"]:
                                existing["covered_child_keys"].append(ck)
                        self.logger.info(
                            "ControlFamilySummaryExtractionService: group %d merged near-duplicate CFSR "
                            "(ratio=%.2f): '%s' → '%s'",
                            group_index,
                            ratio,
                            r["requirement_text"][:60],
                            existing["requirement_text"][:60],
                        )
                        matched = True
                        break
                if not matched:
                    merged.append(r)
            results = merged

            cap = getattr(self.config, "cfsr_max_per_parent", 5)
            if len(results) > cap:
                self.logger.warning(
                    "ControlFamilySummaryExtractionService: group %d parent='%s' produced %d CFSRs "
                    "after dedup, capping to %d",
                    group_index,
                    parent_section,
                    len(results),
                    cap,
                )
                results = results[:cap]

            # Pass B: enforce covered_child_keys exclusivity (each child in exactly one CFSR)
            _seen_child_keys: set = set()
            for r in results:
                unique = [ck for ck in r["covered_child_keys"] if ck not in _seen_child_keys]
                if len(unique) < len(r["covered_child_keys"]):
                    self.logger.debug(
                        "ControlFamilySummaryExtractionService: group %d removed %d overlapping child keys from cfsr=%s",
                        group_index,
                        len(r["covered_child_keys"]) - len(unique),
                        r.get("stable_key"),
                    )
                r["covered_child_keys"] = unique
                _seen_child_keys.update(unique)

            # Orphan coverage guarantee: assign any child the LLM missed to the closest CFSR
            if results:
                all_assigned: set = set()
                for r in results:
                    all_assigned.update(r.get("covered_child_keys", []))
                orphan_real_keys = [
                    real_key
                    for real_key in child_id_to_stable_key.values()
                    if real_key not in all_assigned
                ]
                if orphan_real_keys:
                    child_text_by_key = {
                        getattr(p, "stable_key", ""): getattr(p, "requirement_text", "")
                        for p in children
                    }
                    cfsr_texts = [r.get("requirement_text", "") for r in results]
                    for orphan_key in orphan_real_keys:
                        orphan_text = child_text_by_key.get(orphan_key, "")
                        best_idx = max(
                            range(len(cfsr_texts)),
                            key=lambda i: SequenceMatcher(None, orphan_text, cfsr_texts[i]).ratio(),
                        )
                        results[best_idx]["covered_child_keys"].append(orphan_key)
                        self.logger.debug(
                            "ControlFamilySummaryExtractionService: orphan child=%s reassigned to cfsr=%s",
                            orphan_key,
                            results[best_idx].get("stable_key"),
                        )
                    self.logger.info(
                        "ControlFamilySummaryExtractionService: group %d reassigned %d orphan children to existing CFSRs",
                        group_index,
                        len(orphan_real_keys),
                    )

            # Drop any CFSR that still covers nothing after orphan reassignment had
            # its chance to backfill coverage — these are pure LLM hallucinations.
            zero_coverage = [r for r in results if not r["covered_child_keys"]]
            if zero_coverage:
                self.logger.warning(
                    "ControlFamilySummaryExtractionService: group %d parent='%s' dropping %d CFSR(s) "
                    "with zero covered children: %s",
                    group_index,
                    parent_section,
                    len(zero_coverage),
                    [r.get("stable_key") for r in zero_coverage],
                )
                results = [r for r in results if r["covered_child_keys"]]

            self.logger.info(
                "ControlFamilySummaryExtractionService: [GROUP %d/%d] FINISH thread=%s generated=%d",
                group_index + 1,
                total_groups,
                thread_name,
                len(results),
            )
            return group_index, results

        ordered_results: Dict[int, list] = {}
        wrapped = probe.wrap(_process_parent_group)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="CFSRExtract",
        ) as executor:
            future_map = {
                executor.submit(wrapped, idx, parent, children): idx
                for idx, (parent, children) in enumerate(parent_groups)
            }
            for future in concurrent.futures.as_completed(future_map):
                idx = future_map[future]
                try:
                    completed_idx, results = future.result()
                except Exception as exc:
                    self.logger.exception(
                        "ControlFamilySummaryExtractionService: failed group %d: %s",
                        idx,
                        exc,
                    )
                    continue
                ordered_results[completed_idx] = results

        all_results = []
        seen_keys_global: set = set()
        for idx in range(total_groups):
            for item in ordered_results.get(idx, []):
                key = item["stable_key"]
                if key in seen_keys_global:
                    # Should not happen with group_index prefix, but guard defensively
                    self.logger.warning(
                        "ControlFamilySummaryExtractionService: duplicate stable_key '%s' dropped", key
                    )
                    continue
                seen_keys_global.add(key)
                all_results.append(item)

        self.logger.info(
            "ControlFamilySummaryExtractionService: generated %d CFSRs from %d parent groups (category_id=%s, job_id=%s)",
            len(all_results),
            total_groups,
            category_id,
            ingestion_job_id,
        )
        return all_results
