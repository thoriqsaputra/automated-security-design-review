from __future__ import annotations

import difflib
import re


def normalize_quote_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().casefold())


def is_quote_grounded(quoted_text: str, block_text: str) -> bool:
    normalized_quote = normalize_quote_text(quoted_text)
    if not normalized_quote:
        return True
    normalized_block = normalize_quote_text(block_text)
    if not normalized_block:
        return False
    if normalized_quote in normalized_block:
        return True
    # Fallback for minor OCR/transcription noise: the quote must be covered
    # by a single contiguous match in the source block, not merely similar
    # in aggregate (which would also accept a quote spliced together from a
    # neighboring chunk that happens to share most of its wording).
    matcher = difflib.SequenceMatcher(None, normalized_quote, normalized_block, autojunk=False)
    match = matcher.find_longest_match(0, len(normalized_quote), 0, len(normalized_block))
    coverage = match.size / len(normalized_quote)
    return coverage >= 0.85
