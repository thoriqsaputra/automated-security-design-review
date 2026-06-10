import json
import logging
import re
import concurrent.futures
import threading
from typing import Dict, List, Any

import tiktoken

from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.prompts.extraction_prompts import build_hierarchical_extraction_prompt
from sdr.apps.standards.models import StandardSourceDocument
from sdr.apps.workspace.document_processing import get_local_file_path, get_document_content
from sdr.apps.ai.utils.chunking import chunk_text_with_context
from sdr.apps.ai.utils.parsing import strip_markdown_code_blocks, strip_thinking_block
from sdr.core.config import settings

logger = logging.getLogger(__name__)

_EMBEDDING_BULK_BATCH = 500
_EMBEDDING_MODEL_NAME = "bge-small-en-v1.5"
_AI_RESPONSE_PREVIEW_LIMIT = 800

# Lazily loaded tiktoken encoder for token counting in logging
_token_encoder = None

def _count_tokens(text: str) -> int:
    """Estimate token count using cl100k_base (GPT-4 / Kimi compatible)."""
    global _token_encoder
    if _token_encoder is None:
        try:
            _token_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback: rough estimate at 4 chars/token
            return max(1, len(text) // 4)
    return len(_token_encoder.encode(text))


_TOC_HEADING_RE = re.compile(
    r"^\s*(daftar\s+isi|table\s+of\s+contents|contents)\s*$",
    re.IGNORECASE,
)
_TOC_ENTRY_DOTTED_RE = re.compile(r"\.{2,}\s*\d{1,4}\s*$")
_TOC_ENTRY_SPACED_RE = re.compile(r".{6,}\s{2,}\d{1,4}\s*$")
_TOC_LABEL_RE = re.compile(r"^\s*(halaman|page|pages?)\s*$", re.IGNORECASE)


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
    """
    Remove explicit table-of-contents blocks before chunking so TOC outline
    entries are not treated as extractable security parameters.
    """
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
            if (
                not stripped
                or _TOC_LABEL_RE.match(stripped)
                or _looks_like_toc_entry(stripped)
            ):
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
    """
    Best-effort extraction of a JSON object from a model response that may
    contain preamble/epilogue text.
    """
    cleaned = strip_markdown_code_blocks(strip_thinking_block(text or "{}")).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1].strip()

    return cleaned


def _identity(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("requirement", "")).strip()
    return str(item).strip()


# ---------------------------------------------------------------------------
# Core AI Extraction
# ---------------------------------------------------------------------------


def extract_structured_requirements(source_doc_text: str) -> Dict[str, List[Any]]:
    """
    Calls the AI service with a standard's text to get a structured
    dictionary of requirements.

    Returns: { "Section Header": ["Req 1", "Req 2", ...] }

    NOTE: The prompt is intentionally untouched per the refactor specification.
    """
    logger.debug(
        "extract_structured_requirements: starting extraction. "
        "Text length=%d chars, %d lines.",
        len(source_doc_text),
        len(source_doc_text.split('\n')),
    )
    prompt = build_hierarchical_extraction_prompt(source_doc_text)
    logger.debug(
        "extract_structured_requirements: prompt built. Prompt length=%d chars.",
        len(prompt),
    )

    try:
        logger.debug("extract_structured_requirements: calling chat_completion API.")
        response = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert security analyst. "
                        "Extract ONLY actionable, technical security requirements from the real "
                        "standard body content and output strictly valid JSON. "
                        "If source text is not English, translate section names and "
                        "parameter text/details to English while preserving exact source-language "
                        "verbatim_quote values. Ignore table-of-contents, document scopes, "
                        "introductions, generic domains, and compliance/verification processes. "
                        "Do not output analysis, reasoning, or <thinking> tags."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            component="standard_extraction",
            temperature=0.05,
            max_tokens=4000,
        )
        logger.debug("extract_structured_requirements: API call completed.")

        response_preview = (response.content or "").replace("\n", " ").replace("\r", " ")
        if len(response_preview) > _AI_RESPONSE_PREVIEW_LIMIT:
            response_preview = response_preview[:_AI_RESPONSE_PREVIEW_LIMIT] + "..."
        logger.info(
            "extract_structured_requirements: AI response received. model=%s provider=%s "
            "usage=%s preview=%s",
            getattr(response, "model", "unknown"),
            getattr(getattr(response, "provider", None), "value", getattr(response, "provider", "unknown")),
            getattr(response, "usage", None),
            response_preview,
        )

        if response.error:
            logger.error(
                "extract_structured_requirements: API returned error: %s",
                response.error,
            )
            return {}

        logger.debug(
            "extract_structured_requirements: received response. "
            "Content length=%d chars.",
            len(response.content or ""),
        )
        content = _extract_json_payload(response.content or "{}")
        logger.debug(
            "extract_structured_requirements: extracted JSON payload. "
            "Clean content length=%d chars.",
            len(content),
        )
        try:
            parsed = json.loads(content)
            logger.debug(
                "extract_structured_requirements: JSON parsed successfully. "
                "Type=%s",
                type(parsed).__name__,
            )
        except json.JSONDecodeError as e:
            logger.warning(
                "extract_structured_requirements: JSON decode failed. "
                "Position=%s, message=%s. Attempting LLM repair.",
                getattr(e, 'pos', 'unknown'),
                str(e),
            )
            repair_prompt = (
                "The following JSON has syntax errors (e.g. missing commas, unescaped quotes). "
                "Fix it and return ONLY valid JSON without any markdown or conversational text.\n\n"
                f"{content}"
            )
            repair_resp = chat_completion(
                messages=[{"role": "user", "content": repair_prompt}],
                component="fallback",
                temperature=0.0,
                max_tokens=4000
            )
            if repair_resp.error:
                logger.error("extract_structured_requirements: JSON repair API error: %s", repair_resp.error)
                return {}
            content = _extract_json_payload(repair_resp.content or "{}")
            try:
                parsed = json.loads(content)
                logger.info("extract_structured_requirements: successfully repaired JSON via LLM fallback.")
            except Exception as repair_err:
                logger.error("extract_structured_requirements: JSON repair failed: %s", repair_err)
                return {}

        if not isinstance(parsed, dict):
            logger.warning(
                "extract_structured_requirements: AI returned JSON, "
                "but it was not a dictionary. Parsed type=%s",
                type(parsed).__name__,
            )
            return {}

        logger.debug(
            "extract_structured_requirements: cleaning parsed dictionary. "
            "Sections found=%d",
            len(parsed),
        )
        cleaned_dict: Dict[str, List[Any]] = {}
        total_items_seen = 0
        total_items_filtered = 0
        total_items_added = 0

        for section, raw_requirements in parsed.items():
            if not isinstance(raw_requirements, list):
                logger.debug(
                    "extract_structured_requirements: section '%s' requirements "
                    "not a list (type=%s). Skipping.",
                    section,
                    type(raw_requirements).__name__,
                )
                continue

            cleaned_reqs: List[Any] = []

            for item in raw_requirements:
                total_items_seen += 1
                if isinstance(item, dict):
                    req = str(item.get("requirement", "")).strip()
                    details = str(item.get("details", "")).strip()
                    quote = str(item.get("verbatim_quote", "")).strip()
                    marker = str(item.get("context_marker", "")).strip()

                    if len(req) < 8 and len(details) < 8:
                        total_items_filtered += 1
                        logger.debug(
                            "extract_structured_requirements: dict item in section '%s' "
                            "missing/short requirement and details fields. Filtering out.",
                            section,
                        )
                        continue

                    cleaned_reqs.append(
                        {
                            "requirement": req,
                            "details": details,
                            "verbatim_quote": quote,
                            "context_marker": marker,
                        }
                    )
                    total_items_added += 1
                    continue

                if not isinstance(item, str):
                    logger.debug(
                        "extract_structured_requirements: item in section '%s' "
                        "not a string (type=%s). Filtering out.",
                        section,
                        type(item).__name__,
                    )
                    total_items_filtered += 1
                    continue
                text = item.strip()
                if len(text) < 8:
                    logger.debug(
                        "extract_structured_requirements: item in section '%s' "
                        "too short (len=%d). Filtering out.",
                        section,
                        len(text),
                    )
                    total_items_filtered += 1
                    continue
                cleaned_reqs.append(text)
                total_items_added += 1

            if cleaned_reqs:
                cleaned_dict[section] = cleaned_reqs
                logger.debug(
                    "extract_structured_requirements: section '%s' → %d requirements.",
                    section,
                    len(cleaned_reqs),
                )
            else:
                logger.debug(
                    "extract_structured_requirements: section '%s' had no valid "
                    "requirements after cleaning. Excluded.",
                    section,
                )

        logger.info(
            "extract_structured_requirements: cleaning complete. "
            "Items seen=%d, filtered=%d, added=%d. Final sections=%d, "
            "total reqs=%d",
            total_items_seen,
            total_items_filtered,
            total_items_added,
            len(cleaned_dict),
            sum(len(v) for v in cleaned_dict.values()),
        )
        return cleaned_dict

    except Exception as e:
        logger.error(
            "extract_structured_requirements: unexpected error processing AI response: "
            "%s (type=%s)",
            e,
            type(e).__name__,
        )
        return {}


# ---------------------------------------------------------------------------
# Map-Reduce Merge Helper
# ---------------------------------------------------------------------------


def _merge_requirements(
    base: Dict[str, List[Any]],
    incoming: Dict[str, List[Any]],
) -> Dict[str, List[Any]]:
    """
    Reduce step: merge *incoming* chunk results into the *base* accumulator.

    Merging rules
    -------------
    * If a section key already exists in *base*, extend its requirement list
      with items from *incoming* that are **not already present** (exact string
      match after strip).  This eliminates duplicates introduced by chunk
      overlap without discarding genuinely distinct requirements that happen
      to appear in multiple sections.
    * If a section key is new, add it directly to *base*.

    Args:
        base:     The running accumulator built from previously processed chunks.
        incoming: The extraction result from the current chunk.

    Returns:
        The updated *base* dictionary (mutated in-place for efficiency and
        also returned for convenient chaining).
    """
    logger.debug(
        "_merge_requirements: starting merge. "
        "Incoming sections=%d, current base sections=%d",
        len(incoming),
        len(base),
    )
    sections_new = 0
    sections_updated = 0
    duplicates_skipped = 0
    new_reqs_total = 0

    for section, new_reqs in incoming.items():
        if not new_reqs:
            logger.debug(
                "_merge_requirements: section '%s' has no requirements. Skipping.",
                section,
            )
            continue

        if section not in base:
            # New section discovered in this chunk — add it wholesale.
            base[section] = list(new_reqs)
            logger.debug(
                "_merge_requirements: new section added → '%s' with %d requirements.",
                section,
                len(new_reqs),
            )
            sections_new += 1
            new_reqs_total += len(new_reqs)
        else:
            # Existing section — append only genuinely new requirements.
            existing_set = {_identity(item) for item in base[section]}  # O(1) lookup
            existing_count_before = len(base[section])
            added = 0
            for req in new_reqs:
                req_identity = _identity(req)
                if req_identity and req_identity not in existing_set:
                    base[section].append(req)
                    existing_set.add(req_identity)
                    added += 1
                else:
                    duplicates_skipped += 1
            
            if added:
                sections_updated += 1
                new_reqs_total += added
                logger.debug(
                    "_merge_requirements: section '%s' → appended %d new requirement(s) "
                    "(before=%d, after=%d, duplicates_skipped=%d).",
                    section,
                    added,
                    existing_count_before,
                    len(base[section]),
                    len(new_reqs) - added,
                )
            else:
                logger.debug(
                    "_merge_requirements: section '%s' → all %d incoming requirement(s) "
                    "were duplicates. No changes.",
                    section,
                    len(new_reqs),
                )
                duplicates_skipped += len(new_reqs)

    logger.info(
        "_merge_requirements: merge complete. "
        "New sections=%d, updated sections=%d, new reqs total=%d, "
        "duplicates skipped=%d. Final base: %d section(s), %d total reqs.",
        sections_new,
        sections_updated,
        new_reqs_total,
        duplicates_skipped,
        len(base),
        sum(len(v) for v in base.values()),
    )
    return base


# ---------------------------------------------------------------------------
# Orchestration: Extract → Save
# ---------------------------------------------------------------------------


def extract_requirements_from_document(
    source_doc: StandardSourceDocument,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    progress_callback=None,
) -> Dict[str, List[Any]]:
    """
    Public entry point. Extracts security requirements from a source
    source document using a **Map-Reduce** pattern to handle arbitrarily
    large documents without hitting LLM token limits or suffering from
    "lost in the middle" syndrome.

    Pipeline
    --------
    2. **Ingest**  — read and decode the document via ``get_document_content``.
    3. **Chunk**   — split the full text into token-bounded, context-annotated
                     chunks using ``chunk_text_with_context``.
    4. **Map**     — call ``extract_structured_requirements`` on every chunk
                     independently, so each AI call is well within token limits.
    5. **Reduce**  — merge all per-chunk dicts into a single de-duplicated
                     requirements dictionary using ``_merge_requirements``.
    Args:
        source_doc: A ``StandardSourceDocument`` instance whose ``.document``
                  points to the document to be processed.

    Returns:
        The merged ``{ "Section Header": ["Req 1", ...] }`` dictionary,
        or ``{}`` on any unrecoverable error.
    """
    logger.info(
        "extract_requirements_from_document: [ENTRY] "
        "standard_id=%s, name='%s', document='%s'",
        source_doc.id,
        source_doc.name,
        source_doc.document if source_doc.document else 'None',
    )
    # ------------------------------------------------------------------
    # 2. Ingest: read the document content
    # ------------------------------------------------------------------
    logger.debug(
        "extract_requirements_from_document: [INGEST PHASE] "
        "retrieving document content."
    )
    if progress_callback:
        progress_callback("Reading Document", 5)
    try:
        logger.debug(
            "extract_requirements_from_document: getting local file path "
            "for document '%s'.",
            source_doc.document if source_doc.document else 'None',
        )
        with get_local_file_path(source_doc.document) as source_doc_path:
            logger.debug(
                "extract_requirements_from_document: local path retrieved: %s. "
                "Extracting content.",
                source_doc_path,
            )
            content = get_document_content(source_doc_path, source_doc.document, start_page=start_page, end_page=end_page)
            logger.debug(
                "extract_requirements_from_document: document content extracted. "
                "Content keys=%s",
                list((content or {}).keys()),
            )
            logger.info(
                "extract_requirements_from_document: conversion_method=%s "
                "for standard '%s'.",
                (content or {}).get("conversion_method", "unknown"),
                source_doc.name,
            )
            source_doc_text: str = (content or {}).get("text") or ""
            logger.debug(
                "extract_requirements_from_document: extracted text. "
                "Text length=%d chars, %d lines.",
                len(source_doc_text),
                len(source_doc_text.split('\n')) if source_doc_text else 0,
            )
    except Exception as exc:
        logger.error(
            "extract_requirements_from_document: ✗ [INGEST] "
            "failed to access standard file '%s': %s (type=%s)",
            source_doc.name,
            exc,
            type(exc).__name__,
        )
        return {}

    if not source_doc_text.strip():
        logger.warning(
            "extract_requirements_from_document: ✗ [INGEST] "
            "document '%s' yielded no extractable text (text length=%d); "
            "skipping AI call.",
            source_doc.name,
            len(source_doc_text),
        )
        return {}

    source_doc_text = _remove_table_of_contents(source_doc_text)
    if not source_doc_text.strip():
        logger.warning(
            "extract_requirements_from_document: ✗ [PREPROCESS] "
            "document '%s' had no extractable body text after table-of-contents removal.",
            source_doc.name,
        )
        return {}

    # ------------------------------------------------------------------
    # 3. Chunk: split into token-aware, context-annotated chunks
    # ------------------------------------------------------------------
    logger.debug(
        "extract_requirements_from_document: [CHUNK PHASE] "
        "splitting text into context-aware chunks."
    )
    if progress_callback:
        progress_callback("Chunking Document", 15)
    chunks = chunk_text_with_context(source_doc_text)
    logger.debug(
        "extract_requirements_from_document: chunking complete. "
        "Chunks=%d",
        len(chunks) if chunks else 0,
    )

    if not chunks:
        logger.warning(
            "extract_requirements_from_document: ✗ [CHUNK] "
            "chunker returned no chunks for standard '%s' "
            "(text length was %d chars). Aborting.",
            source_doc.name,
            len(source_doc_text),
        )
        return {}

    total_chunks = len(chunks)
    logger.info(
        "extract_requirements_from_document: ✓ [CHUNK] "
        "processing standard '%s' across %d chunk(s).",
        source_doc.name,
        total_chunks,
    )

    # ------------------------------------------------------------------
    # 4 & 5. Map + Reduce: extract per chunk, then merge incrementally
    # ------------------------------------------------------------------
    logger.debug(
        "extract_requirements_from_document: [MAP-REDUCE PHASE] "
        "starting extraction across all chunks."
    )
    merged: Dict[str, List[Any]] = {}
    successful_chunks = 0
    failed_chunks = 0
    total_sections_seen = 0
    total_reqs_seen = 0
    total_tokens_seen = 0  # accumulated across all chunks

    def _process_chunk_with_logging(idx: int, text: str) -> Dict[str, List[Any]]:
        thread_name = threading.current_thread().name
        token_count = _count_tokens(text)
        logger.info(
            "extract_requirements_from_document: [MAP %d/%d] "
            "STARTING on thread %s (~%d tokens)",
            idx,
            total_chunks,
            thread_name,
            token_count,
        )
        result = extract_structured_requirements(text)
        logger.info(
            "extract_requirements_from_document: [MAP %d/%d] "
            "FINISHED on thread %s",
            idx,
            total_chunks,
            thread_name,
        )
        return result, token_count

    logger.info(
        "extract_requirements_from_document: [CONCURRENCY INIT] "
        "Dispatching %d chunks to ThreadPoolExecutor(max_workers=%d) "
        "for concurrent extraction of standard '%s'.",
        total_chunks,
        settings.AI_STANDARD_EXTRACTION_MAX_WORKERS,
        source_doc.name,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=settings.AI_STANDARD_EXTRACTION_MAX_WORKERS, thread_name_prefix="ExtractWorker") as executor:
        future_to_chunk = {
            executor.submit(_process_chunk_with_logging, idx, chunk_dict["text"]): (idx, chunk_dict)
            for idx, chunk_dict in enumerate(chunks, start=1)
        }

        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            idx, chunk_dict = future_to_chunk[future]
            completed_count += 1
            
            if progress_callback:
                progress_callback(
                    f"Extracting chunk {completed_count} of {total_chunks}", 
                    15 + int((completed_count / max(total_chunks, 1)) * 80)
                )

            chunk_text = chunk_dict["text"]
            chunk_len = len(chunk_text)

            try:
                chunk_result, chunk_tokens = future.result()
            except Exception as e:
                logger.error(
                    "extract_requirements_from_document: [MAP %d/%d] ✗ "
                    "chunk failed with exception %s for standard '%s'",
                    idx,
                    total_chunks,
                    e,
                    source_doc.name,
                )
                chunk_result = {}
                chunk_tokens = 0

            chunk_sections = len(chunk_result)
            chunk_reqs = sum(len(v) for v in chunk_result.values())
            total_tokens_seen += chunk_tokens

            if not chunk_result:
                logger.warning(
                    "extract_requirements_from_document: [MAP %d/%d] ✗ "
                    "chunk returned no requirements for standard '%s' "
                    "(chunk_len=%d chars); treating as failed.",
                    idx,
                    total_chunks,
                    source_doc.name,
                    chunk_len,
                )
                failed_chunks += 1
                continue

            logger.debug(
                "extract_requirements_from_document: [MAP %d/%d] ✓ "
                "chunk extraction succeeded. sections=%d, reqs=%d.",
                idx,
                total_chunks,
                chunk_sections,
                chunk_reqs,
            )

            total_sections_seen += chunk_sections
            total_reqs_seen += chunk_reqs

            # Reduce: merge this chunk's results into the running accumulator
            logger.debug(
                "extract_requirements_from_document: [REDUCE %d/%d] "
                "merging chunk results into accumulator.",
                idx,
                total_chunks,
            )
            _merge_requirements(merged, chunk_result)
            successful_chunks += 1
            logger.info(
                "extract_requirements_from_document: [REDUCE %d/%d] ✓ "
                "chunk merged. Accumulator now: sections=%d, reqs=%d.",
                idx,
                total_chunks,
                len(merged),
                sum(len(v) for v in merged.values()),
            )

    logger.info(
        "extract_requirements_from_document: ✓ [MAP-REDUCE] "
        "complete for standard '%s'. Chunks: %d succeeded, %d failed. "
        "Total sections seen=%d, total reqs seen=%d, total tokens=%d. "
        "Final merged: %d section(s), %d total requirement(s).",
        source_doc.name,
        successful_chunks,
        failed_chunks,
        total_sections_seen,
        total_reqs_seen,
        total_tokens_seen,
        len(merged),
        sum(len(v) for v in merged.values()),
    )

    logger.info(
        "extract_requirements_from_document: [EXIT SUCCESS] "
        "extraction complete for source_doc '%s'. Extracted %d section(s) "
        "with %d total requirement(s).",
        source_doc.name,
        len(merged),
        sum(len(v) for v in merged.values()),
    )
    return merged
