from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sdr.apps.ai.client import get_embedding
from sdr.apps.ai.tsd_processing.raptor import (
    RAPTORNode,
    RAPTORTree,
    _compute_cosine_similarity,
)

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5
_MAX_TOP_K = 20
_MIN_COSINE_SIMILARITY = 0.6
_RELAXATION_MARGIN = 0.1
_EMBEDDING_DIMENSIONS = 1024
RAPTOR_LEVEL_LOW = 0
RAPTOR_LEVEL_MID = 1
RAPTOR_LEVEL_HIGH = 2
RAPTOR_LEVEL_ROOT = 3

@dataclass
class RAPTORSearchResult:
    node: RAPTORNode
    cosine_similarity: float
    source_block_ids: List[str] = field(default_factory=list)
    threshold_relaxed: bool = False

    @property
    def level(self) -> int:
        return self.node.level

    @property
    def text(self) -> str:
        return self.node.text

    @property
    def section_heading(self) -> Optional[str]:
        return self.node.section_heading

    @property
    def page_numbers(self) -> List[int]:
        return self.node.page_numbers

    @property
    def is_relevant(self) -> bool:
        return self.cosine_similarity >= _MIN_COSINE_SIMILARITY


@dataclass
class RAPTORSearchResponse:
    results: List[RAPTORSearchResult] = field(default_factory=list)
    query_embedding: List[float] = field(default_factory=list)
    total_found: int = 0
    query_text: str = ""
    level_searched: int = RAPTOR_LEVEL_LOW
    error: Optional[str] = None
    threshold_relaxed: bool = False

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0

    @property
    def all_source_block_ids(self) -> List[str]:
        seen: set = set()
        block_ids: List[str] = []
        for result in self.results:
            for block_id in result.source_block_ids:
                if block_id not in seen:
                    block_ids.append(block_id)
                    seen.add(block_id)
        return block_ids

    def get_context_chunks(self) -> List[str]:
        return [r.text for r in self.results if r.text]

    def get_context_chunk_block_ids(self) -> List[List[str]]:
        return [list(r.source_block_ids) for r in self.results if r.text]

    def get_context_chunk_levels(self) -> List[int]:
        return [r.level for r in self.results if r.text]

class RAPTORSearcher:
    def __init__(
        self,
        min_cosine_similarity: float = _MIN_COSINE_SIMILARITY,
        embedding_dimensions: int = _EMBEDDING_DIMENSIONS,
        default_top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self.min_cosine_similarity = min_cosine_similarity
        self.embedding_dimensions = embedding_dimensions
        self.default_top_k = default_top_k
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    def search(
        self,
        query_text: str,
        tree: RAPTORTree,
        level: int = RAPTOR_LEVEL_LOW,
        top_k: Optional[int] = None,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> RAPTORSearchResponse:
        error = self._validate_query(query_text, tree, "search", level)
        if error is not None:
            return error

        resolved_top_k = self._resolve_top_k(top_k)
        level, level_nodes = self._resolve_level_nodes(tree, level)
        if not level_nodes:
            return self._error_response(
                f"No nodes found at any level in tree '{tree.document_name}'.",
                query_text=query_text,
                level=level,
            )

        embeddable_nodes = self._get_embeddable_nodes(
            level_nodes,
            query_text=query_text,
            level=level,
            tree=tree,
        )
        if not embeddable_nodes:
            return self._error_response(
                (
                    f"No nodes with embeddings found at level {level} in "
                    f"tree '{tree.document_name}' — embeddings may have failed "
                    f"during tree construction."
                ),
                query_text=query_text,
                level=level,
            )

        query_vector = self._resolve_query_embedding(
            query_text,
            precomputed_embedding,
            method_name="search",
            level=level,
        )
        if not query_vector:
            return self._error_response(
                f"Failed to generate embedding for query '{query_text[:60]}...'.",
                query_text=query_text,
                level=level,
                log_level="error",
            )

        scored_results = self._score_nodes(query_vector=query_vector, nodes=embeddable_nodes)
        top_results, threshold_relaxed = self._select_top_results(
            scored_results,
            top_k=resolved_top_k,
            query_text=query_text,
            tree=tree,
            level=level,
        )

        self.logger.info(
            "RAPTORSearcher.search: level=%d found %d/%d node(s) above "
            "threshold=%.2f for query '%s...' in tree '%s'.",
            level,
            len(top_results),
            len(embeddable_nodes),
            self.min_cosine_similarity,
            query_text[:60],
            tree.document_name,
        )

        return RAPTORSearchResponse(
            results=top_results,
            query_embedding=query_vector,
            total_found=len(top_results),
            query_text=query_text,
            level_searched=level,
            error=None,
            threshold_relaxed=threshold_relaxed,
        )

    def search_multi_level(
        self,
        query_text: str,
        tree: RAPTORTree,
        levels: Optional[List[int]] = None,
        top_k_per_level: int = 3,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> RAPTORSearchResponse:
        error = self._validate_query(query_text, tree, "search_multi_level")
        if error is not None:
            return error

        resolved_levels = levels if levels is not None else [
            RAPTOR_LEVEL_LOW,
            RAPTOR_LEVEL_MID,
            RAPTOR_LEVEL_HIGH,
        ]
        query_vector = self._resolve_query_embedding(
            query_text,
            precomputed_embedding,
            method_name="search_multi_level",
        )
        if not query_vector:
            return self._error_response(
                f"Failed to generate embedding for query '{query_text[:60]}...'.",
                query_text=query_text,
                log_level="error",
            )

        all_results: List[RAPTORSearchResult] = []
        seen_node_ids: set = set()
        levels_searched: List[int] = []
        any_threshold_relaxed = False

        for level in resolved_levels:
            level_response = self.search(
                query_text=query_text,
                tree=tree,
                level=level,
                top_k=top_k_per_level,
                precomputed_embedding=query_vector,
            )

            if level_response.error:
                self.logger.debug(
                    "RAPTORSearcher.search_multi_level: level %d returned "
                    "error '%s' — skipping.",
                    level,
                    level_response.error,
                )
                continue

            levels_searched.append(level)
            any_threshold_relaxed = any_threshold_relaxed or level_response.threshold_relaxed

            for result in level_response.results:
                if result.node.node_id not in seen_node_ids:
                    all_results.append(result)
                    seen_node_ids.add(result.node.node_id)

        if not all_results:
            msg = (
                f"No results found across levels {resolved_levels} "
                f"in tree '{tree.document_name}'."
            )
            self.logger.warning(
                "RAPTORSearcher.search_multi_level: %s", msg
            )
            return RAPTORSearchResponse(
                error=msg,
                query_text=query_text,
                query_embedding=query_vector,
            )

        self._sort_mixed_level_results(all_results, include_relaxed=True)

        self.logger.info(
            "RAPTORSearcher.search_multi_level: merged %d result(s) "
            "across levels %s for query '%s...' in tree '%s'.",
            len(all_results),
            levels_searched,
            query_text[:60],
            tree.document_name,
        )

        reported_level = min(levels_searched) if levels_searched else RAPTOR_LEVEL_LOW

        return RAPTORSearchResponse(
            results=all_results,
            query_embedding=query_vector,
            total_found=len(all_results),
            query_text=query_text,
            level_searched=reported_level,
            error=None,
            threshold_relaxed=any_threshold_relaxed,
        )

    def search_collapsed_raptor(
        self,
        query_text: str,
        tree: RAPTORTree,
        top_k: int = 20,
        max_tokens: int = 4000,
        allowed_levels: Optional[List[int]] = None,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> RAPTORSearchResponse:
        error = self._validate_query(query_text, tree, "search_collapsed_raptor")
        if error is not None:
            return error

        query_vector = self._resolve_query_embedding(
            query_text,
            precomputed_embedding,
            method_name="search_collapsed_raptor",
        )
        if not query_vector:
            return RAPTORSearchResponse(error="Failed to generate embedding.", query_text=query_text)

        nodes = self._get_all_embeddable_nodes(tree, allowed_levels=allowed_levels)
        if not nodes:
            return RAPTORSearchResponse(error="No embeddable RAPTOR nodes found.", query_text=query_text)

        scored = self._score_nodes(query_vector=query_vector, nodes=nodes)
        self._sort_mixed_level_results(scored, include_relaxed=False)

        selected: List[RAPTORSearchResult] = []
        token_total = 0
        resolved_top_k = self._resolve_top_k(top_k)
        for result in scored:
            if len(selected) >= resolved_top_k:
                break
            next_tokens = result.node.token_estimate
            if selected and token_total + next_tokens > max_tokens:
                continue
            if not selected and next_tokens > max_tokens:
                selected.append(result)
                break
            selected.append(result)
            token_total += next_tokens

        return RAPTORSearchResponse(
            results=selected,
            query_embedding=query_vector,
            total_found=len(selected),
            query_text=query_text,
            level_searched=RAPTOR_LEVEL_LOW,
            error=None,
        )

    def _validate_query(
        self,
        query_text: str,
        tree: Optional[RAPTORTree],
        method_name: str,
        level: Optional[int] = None,
    ) -> Optional[RAPTORSearchResponse]:
        if not query_text or not query_text.strip():
            return self._error_response(
                "query_text is empty — cannot perform RAPTOR search.",
                query_text=query_text,
                level=level,
                method_name=method_name,
            )
        if tree is None or tree.is_empty():
            return self._error_response(
                "RAPTORTree is empty — no nodes to search.",
                query_text=query_text,
                level=level,
                method_name=method_name,
            )
        return None

    def _error_response(
        self,
        message: str,
        *,
        query_text: str,
        level: Optional[int] = None,
        method_name: str = "search",
        log_level: str = "warning",
        query_embedding: Optional[List[float]] = None,
    ) -> RAPTORSearchResponse:
        log_fn = getattr(self.logger, log_level, self.logger.warning)
        log_fn("RAPTORSearcher.%s: %s", method_name, message)
        return RAPTORSearchResponse(
            error=message,
            query_text=query_text,
            level_searched=level if level is not None else RAPTOR_LEVEL_LOW,
            query_embedding=query_embedding or [],
        )

    def _resolve_top_k(self, top_k: Optional[int]) -> int:
        return min(top_k if top_k is not None else self.default_top_k, _MAX_TOP_K)

    def _resolve_query_embedding(
        self,
        query_text: str,
        precomputed_embedding: Optional[List[float]],
        *,
        method_name: str,
        level: Optional[int] = None,
    ) -> List[float]:
        if precomputed_embedding:
            self.logger.debug(
                "RAPTORSearcher.%s: using precomputed embedding for query '%s...'",
                method_name,
                query_text[:60],
            )
            return precomputed_embedding
        return self._embed_query(query_text)

    def _resolve_level_nodes(
        self,
        tree: RAPTORTree,
        level: int,
    ) -> tuple[int, List[RAPTORNode]]:
        level_nodes = tree.get_nodes_at_level(level)
        if level_nodes:
            return level, level_nodes

        self.logger.warning(
            "RAPTORSearcher.search: level %d not found in tree '%s' "
            "(max_level=%d) — falling back to level 0.",
            level,
            tree.document_name,
            tree.max_level,
        )
        return RAPTOR_LEVEL_LOW, tree.get_nodes_at_level(RAPTOR_LEVEL_LOW)

    def _get_embeddable_nodes(
        self,
        nodes: List[RAPTORNode],
        *,
        query_text: str,
        level: int,
        tree: RAPTORTree,
    ) -> List[RAPTORNode]:
        return [node for node in nodes if node.has_embedding]

    def _get_all_embeddable_nodes(
        self,
        tree: RAPTORTree,
        *,
        allowed_levels: Optional[List[int]] = None,
    ) -> List[RAPTORNode]:
        nodes = tree.get_all_nodes()
        if allowed_levels is not None:
            allowed = set(allowed_levels)
            nodes = [node for node in nodes if node.level in allowed]
        return [node for node in nodes if node.has_embedding and node.embedding]

    def _select_top_results(
        self,
        scored_results: List[RAPTORSearchResult],
        *,
        top_k: int,
        query_text: str,
        tree: RAPTORTree,
        level: int,
    ) -> tuple[List[RAPTORSearchResult], bool]:
        relevant_results = [result for result in scored_results if result.is_relevant]
        relevant_results.sort(key=lambda result: result.cosine_similarity, reverse=True)
        top_results = relevant_results[:top_k]

        if not top_results and scored_results:
            scored_results.sort(key=lambda result: result.cosine_similarity, reverse=True)
            top_results = scored_results[:top_k]
            for result in top_results:
                result.threshold_relaxed = True
            self.logger.warning(
                "RAPTORSearcher.search: no nodes met threshold "
                "(min_cosine_similarity=%.2f) at level %d in tree '%s' for "
                "query '%s...' — relaxing to best available "
                "(top similarity=%.4f).",
                self.min_cosine_similarity,
                level,
                tree.document_name,
                query_text[:60],
                top_results[0].cosine_similarity,
            )
            return top_results, True

        if len(top_results) < top_k and scored_results:
            selected_ids = {result.node.node_id for result in top_results}
            near_miss = [
                result
                for result in scored_results
                if result.node.node_id not in selected_ids
                and result.cosine_similarity >= (self.min_cosine_similarity - _RELAXATION_MARGIN)
            ]
            near_miss.sort(key=lambda result: result.cosine_similarity, reverse=True)
            fill = near_miss[: top_k - len(top_results)]
            if fill:
                for result in fill:
                    result.threshold_relaxed = True
                self.logger.info(
                    "RAPTORSearcher.search: filled %d/%d remaining slot(s) "
                    "with near-miss candidates (within %.2f of floor=%.2f) "
                    "at level %d in tree '%s' for query '%s...'.",
                    len(fill),
                    top_k - len(top_results),
                    _RELAXATION_MARGIN,
                    self.min_cosine_similarity,
                    level,
                    tree.document_name,
                    query_text[:60],
                )
                return top_results + fill, True

        return top_results, False

    def _sort_mixed_level_results(
        self,
        results: List[RAPTORSearchResult],
        *,
        include_relaxed: bool,
    ) -> None:
        if include_relaxed:
            results.sort(key=lambda result: (result.level > 0, result.threshold_relaxed, -result.cosine_similarity))
            return
        results.sort(key=lambda result: (result.level > 0, -result.cosine_similarity))

    def _embed_query(self, query_text: str) -> List[float]:
        try:
            vector = get_embedding(
                text=query_text,
                dimensions=self.embedding_dimensions,
            )   # [4]

            if not vector:
                self.logger.error(
                    "RAPTORSearcher._embed_query: get_embedding() returned "
                    "empty vector for query '%s...'",
                    query_text[:60],
                )
                return []

            return vector

        except Exception as exc:
            self.logger.error(
                "RAPTORSearcher._embed_query: unexpected error: %s",
                exc,
            )
            return []

    def _score_nodes(
        self,
        query_vector: List[float],
        nodes: List[RAPTORNode],
    ) -> List[RAPTORSearchResult]:
        results: List[RAPTORSearchResult] = []

        for node in nodes:
            if not node.embedding:
                self.logger.debug(
                    "RAPTORSearcher._score_nodes: node '%s' has no "
                    "embedding — skipping.",
                    node.node_id,
                )
                continue

            similarity = _compute_cosine_similarity(
                query_vector,
                node.embedding,
            )

            results.append(
                RAPTORSearchResult(
                    node=node,
                    cosine_similarity=similarity,
                    source_block_ids=list(node.source_block_ids),
                )
            )

        return results

def search_raptor_tree(
    query_text: str,
    tree: RAPTORTree,
    level: int = RAPTOR_LEVEL_LOW,
    top_k: int = _DEFAULT_TOP_K,
    precomputed_embedding: Optional[List[float]] = None,
) -> RAPTORSearchResponse:
    searcher = RAPTORSearcher()
    return searcher.search(
        query_text=query_text,
        tree=tree,
        level=level,
        top_k=top_k,
        precomputed_embedding=precomputed_embedding,
    )


def search_raptor_tree_multi_level(
    query_text: str,
    tree: RAPTORTree,
    levels: Optional[List[int]] = None,
    top_k_per_level: int = 3,
    precomputed_embedding: Optional[List[float]] = None,
) -> RAPTORSearchResponse:
    searcher = RAPTORSearcher()
    return searcher.search_multi_level(
        query_text=query_text,
        tree=tree,
        levels=levels,
        top_k_per_level=top_k_per_level,
        precomputed_embedding=precomputed_embedding,
    )


def search_collapsed_raptor(
    query_text: str,
    tree: RAPTORTree,
    top_k: int = 20,
    max_tokens: int = 4000,
    allowed_levels: Optional[List[int]] = None,
    precomputed_embedding: Optional[List[float]] = None,
) -> RAPTORSearchResponse:
    searcher = RAPTORSearcher()
    return searcher.search_collapsed_raptor(
        query_text=query_text,
        tree=tree,
        top_k=top_k,
        max_tokens=max_tokens,
        allowed_levels=allowed_levels,
        precomputed_embedding=precomputed_embedding,
    )


__all__ = [
    "RAPTOR_LEVEL_LOW",
    "RAPTOR_LEVEL_MID",
    "RAPTOR_LEVEL_HIGH",
    "RAPTOR_LEVEL_ROOT",
    "RAPTORSearchResult",
    "RAPTORSearchResponse",
    "RAPTORSearcher",
    "search_raptor_tree",
    "search_raptor_tree_multi_level",
    "search_collapsed_raptor",
]
