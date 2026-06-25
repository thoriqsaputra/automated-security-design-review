from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


_NORMALIZE_RE = re.compile(r"\s+")

# Additive score boost when multiple independent searchers agree on the
# same chunk — capped so agreement alone can't dominate raw relevance.
_AGREEMENT_BOOST_PER_EXTRA_SOURCE = 0.08
_AGREEMENT_BOOST_CAP = 0.24


@dataclass
class RetrievalCandidate:
    id: str
    source_type: str
    text: str
    score: float
    block_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


def normalized_text(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", (text or "").strip().lower())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def _key_for_candidate(candidate: RetrievalCandidate) -> str:
    # Prefer the underlying node's own stable id — searchers that reference
    # the same RAPTOR/graph node (e.g. BM25 and dense both finding the same
    # leaf) set the identical id, so this still merges genuine cross-searcher
    # duplicates. Keying on block_ids[0] alone is unsound: a level-0 leaf and
    # an unrelated level-1+ summary node can share a first block id (the
    # summary's source_block_ids are a union starting with that same leaf's
    # ids), which would wrongly merge them and let the summary's much wider
    # block-id breadth leak onto the leaf's precise, literal candidate.
    if candidate.id:
        return f"id:{candidate.id}"
    if candidate.block_ids:
        return f"block:{candidate.block_ids[0]}"
    return f"txt:{text_hash(candidate.text)}"


def dedupe_candidates(candidates: Iterable[RetrievalCandidate]) -> List[RetrievalCandidate]:
    deduped: Dict[str, RetrievalCandidate] = {}
    order: List[str] = []

    for candidate in candidates:
        key = _key_for_candidate(candidate)
        if key not in deduped:
            deduped[key] = candidate
            order.append(key)
            continue

        existing = deduped[key]
        merged_block_ids = list(dict.fromkeys((existing.block_ids or []) + (candidate.block_ids or [])))
        merged_metadata = dict(existing.metadata or {})
        existing_sources = list(merged_metadata.get("merged_sources", []))
        if existing.source_type not in existing_sources:
            existing_sources.append(existing.source_type)
        if candidate.source_type not in existing_sources:
            existing_sources.append(candidate.source_type)
        merged_metadata.update(candidate.metadata or {})
        merged_metadata["merged_sources"] = existing_sources

        if candidate.score > existing.score:
            preferred = candidate
            preferred.block_ids = merged_block_ids
            preferred.metadata = merged_metadata
            deduped[key] = preferred
        else:
            existing.block_ids = merged_block_ids
            existing.metadata = merged_metadata

    for candidate in deduped.values():
        agreement_count = len(candidate.metadata.get("merged_sources", [candidate.source_type]))
        if agreement_count > 1:
            boost = min(_AGREEMENT_BOOST_PER_EXTRA_SOURCE * (agreement_count - 1), _AGREEMENT_BOOST_CAP)
            candidate.score = float(candidate.score) + boost
            candidate.metadata["agreement_count"] = agreement_count
            candidate.metadata["agreement_boost"] = boost

    return [deduped[key] for key in order]


def merge_candidates(*candidate_lists: Iterable[RetrievalCandidate]) -> List[RetrievalCandidate]:
    merged: List[RetrievalCandidate] = []
    for candidate_list in candidate_lists:
        for candidate in candidate_list:
            md = dict(candidate.metadata or {})
            md.setdefault("merged_sources", [candidate.source_type])
            candidate.metadata = md
            merged.append(candidate)
    return merged


__all__ = ["RetrievalCandidate", "dedupe_candidates", "merge_candidates"]
