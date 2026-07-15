from __future__ import annotations

import re
from typing import List

# Simple keyword extractor used for BM25 coverage boost and follow-up query
# construction — splits on non-alpha characters and drops short / stopword
# tokens. Single shared implementation (was previously duplicated verbatim in
# routing/router.py and routing/executors.py).
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "as", "or", "and", "that",
    "this", "it", "its", "not", "all", "any", "if", "when", "which",
    "used", "use", "using", "verify", "ensure", "check", "confirm",
})

_WORD_SPLIT_RE = re.compile(r"[^a-zA-Z]+")


def extract_keywords(text: str) -> List[str]:
    words = _WORD_SPLIT_RE.split(text or "")
    return [w for w in words if len(w) >= 4 and w.lower() not in _STOPWORDS]


__all__ = ["extract_keywords"]
