from __future__ import annotations

import re
from typing import List, Optional, Set

from sdr.apps.ai.retrieval.core.types import QueryType, RetrievalStrategy
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree


def _marker_matches(marker: str, text: str) -> bool:
    """Word-boundary match for single-word markers; substring for phrases."""
    if " " in marker or "-" in marker:
        return marker in text
    return re.search(r"\b" + re.escape(marker) + r"\b", text) is not None


class RetrievalStrategySelector:
    def select_strategy(
        self,
        *,
        advanced_config,
        query_type: QueryType,
        keywords: List[str],
        inferred_relations: set,
        raptor_tree: Optional[RAPTORTree],
    ) -> RetrievalStrategy:
        has_raptor = raptor_tree is not None and not raptor_tree.is_empty()
        has_many_keywords = len(keywords) >= 4

        if query_type == QueryType.FACT_BASED:
            if has_raptor:
                return RetrievalStrategy.HYBRID
            return RetrievalStrategy.VECTOR_ONLY

        # Multi-hop, reasoning, and global architectural queries all benefit
        # from RAPTOR's hierarchical summaries — route to RAPTOR_HIGH when
        # available, otherwise fall back to HYBRID or VECTOR_ONLY.
        if query_type in (QueryType.MULTI_HOP_SECURITY, QueryType.REASONING_BASED, QueryType.GLOBAL_ARCHITECTURAL):
            if has_raptor:
                return RetrievalStrategy.RAPTOR_HIGH
            return RetrievalStrategy.VECTOR_ONLY

        if has_raptor and has_many_keywords:
            return RetrievalStrategy.RAPTOR_HIGH
        if has_raptor:
            return RetrievalStrategy.RAPTOR_LOW
        return RetrievalStrategy.VECTOR_ONLY

    def classify_query_type(
        self,
        query_text: str,
        keywords: List[str],
        inferred_relations: Set[str],
        query_entities: Optional[List[str]] = None,
    ) -> QueryType:
        text = query_text.lower()
        global_markers = (
            "across all", "overall", "end-to-end", "entire architecture", "global",
            "how does the architecture", "main security risks", "trust boundaries", "dependencies",
            "every service", "all components", "system-wide", "holistic",
        )
        multi_hop_markers = (
            "bypass", "trace", "audit this request path", "leak", "reach the", "request path",
            "permissions", "tenant data", "cross-tenant", "privilege escalation", "idor",
            "authorization bypass", "unauthorized access", "horizontal access", "vertical access",
            "isolation", "multi-tenant", "impersonat",
        )
        integrity_markers = (
            "manipulate", "manipulated", "manipulation", "tamper", "tampering", "tampered",
            "immutable", "modify", "modified", "modification",
        )
        reasoning_markers = ("between", "path", "flow", "across", "relationship", "all services")
        if any(_marker_matches(marker, text) for marker in global_markers):
            return QueryType.GLOBAL_ARCHITECTURAL
        if any(_marker_matches(marker, text) for marker in multi_hop_markers):
            return QueryType.MULTI_HOP_SECURITY
        if any(_marker_matches(marker, text) for marker in integrity_markers):
            return QueryType.MULTI_HOP_SECURITY
        if any(_marker_matches(marker, text) for marker in reasoning_markers) or len(inferred_relations) >= 1:
            return QueryType.REASONING_BASED
        if len(keywords) <= 4:
            return QueryType.FACT_BASED
        return QueryType.REASONING_BASED


__all__ = ["RetrievalStrategySelector"]
