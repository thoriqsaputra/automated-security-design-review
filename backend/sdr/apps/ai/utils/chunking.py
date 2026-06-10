from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from typing import Dict, List

import logging
logger = logging.getLogger(__name__)


def chunk_text_with_context(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 150,
    encoding_name: str = "cl100k_base",
) -> List[Dict[str, str]]:
    """
    Splits a large document into semantically coherent, token-aware chunks
    and prepends positional metadata to each chunk so the LLM retains
    document-placement context during extraction.

    Args:
        text:           The full document text to be split.
        chunk_size:     Maximum number of tokens per chunk (default: 4 000).
        overlap:        Number of overlapping tokens between adjacent chunks
                        to preserve cross-boundary context (default: 400).
        encoding_name:  The tiktoken encoding used for token counting.
                        "cl100k_base" covers GPT-3.5 / GPT-4 / text-embedding
                        models and is a safe general-purpose default.

    Returns:
        A list of dicts, each with a single "text" key whose value is the
        chunk content prefixed with a positional banner, e.g.

            [
                {"text": "--- DOCUMENT CHUNK 1 OF 12 ---\\n\\n<chunk text>"},
                {"text": "--- DOCUMENT CHUNK 2 OF 12 ---\\n\\n<chunk text>"},
                ...
            ]
    """
    if not text or not text.strip():
        logger.warning("chunk_text_with_context: received empty text; returning empty list.")
        return []

    # Build a tiktoken-backed length function so chunk_size is measured in
    # *tokens* rather than characters, which directly maps to LLM context limits.
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        logger.error(
            "chunk_text_with_context: failed to load tiktoken encoding '%s': %s. "
            "Falling back to character-based splitting.",
            encoding_name,
            exc,
        )
        # Graceful degradation: approximate 1 token ≈ 4 chars
        encoding = None

    def _token_len(s: str) -> int:
        if encoding is not None:
            return len(encoding.encode(s))
        return len(s) // 4  # character fallback

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=_token_len,
        is_separator_regex=False,
    )

    raw_chunks: List[str] = splitter.split_text(text)
    total = len(raw_chunks)

    if total == 0:
        logger.warning("chunk_text_with_context: splitter produced 0 chunks.")
        return []

    logger.info(
        "chunk_text_with_context: split document into %d chunk(s) "
        "(chunk_size=%d tokens, overlap=%d tokens).",
        total,
        chunk_size,
        overlap,
    )

    return [
        {"text": f"--- DOCUMENT CHUNK {idx} OF {total} ---\n\n{chunk}"}
        for idx, chunk in enumerate[str](raw_chunks, start=1)
    ]