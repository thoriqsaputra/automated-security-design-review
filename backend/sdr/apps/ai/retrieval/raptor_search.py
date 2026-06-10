"""
RAPTOR Search — level-aware similarity retrieval against the RAPTOR
summarisation tree built from a TSD document.

Responsibility:
    Given a query text (a security parameter from CategoryParameterChild [3]),
    find the most semantically relevant RAPTORNode instances at the requested
    abstraction level — leaf (raw chunks), section summaries, or chapter
    summaries — and return their source_block_ids for citation tracing.

Why RAPTOR search alongside vector search?
    Vector search (retrieval/vector_search.py) searches the security parameter
    knowledge base — it finds which parameters match a TSD chunk.
    RAPTOR search is the inverse — it searches the TSD tree to find which
    TSD passages are relevant to a given parameter. The router
    (retrieval/router.py) combines both directions for HYBRID strategy queries.

Level selection strategy:
    RAPTOR_LOW  (level=0) → precise evidence, single text blocks
                            used when the parameter needs exact citations
    RAPTOR_MID  (level=1) → section summaries
                            used when evidence may span multiple blocks
    RAPTOR_HIGH (level=2) → chapter summaries
                            used when the parameter is cross-cutting

Dependency chain:
    raptor.py              (RAPTORTree, RAPTORNode, _compute_cosine_similarity)
    client.py [4]          (get_embedding — query embedding generation)
         ↓
    raptor_search.py       ← YOU ARE HERE
         ↓
    retrieval/router.py
         ↓
    analysis_service.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sdr.apps.ai.client import get_embedding
from sdr.apps.ai.tsd_processing.raptor import (
    RAPTORNode,
    RAPTORTree,
    _compute_cosine_similarity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default number of nodes to return per search
_DEFAULT_TOP_K = 5

# Maximum top_k — prevents returning the entire tree on broad queries
_MAX_TOP_K = 20

# Minimum cosine similarity threshold — nodes below this are discarded.
# 0.6 similarity is a reasonable floor for security parameter relevance.
_MIN_COSINE_SIMILARITY = 0.6

# Embedding dimensions — consistent with vector_search.py and tasks.py [5]
_EMBEDDING_DIMENSIONS = 1024

# Level constants for external use by the router
RAPTOR_LEVEL_LOW = 0     # leaf nodes — raw text blocks
RAPTOR_LEVEL_MID = 1     # section summaries
RAPTOR_LEVEL_HIGH = 2    # chapter summaries
RAPTOR_LEVEL_ROOT = 3    # document root summary


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RAPTORSearchResult:
    """
    A single result from a RAPTOR tree similarity search.

    Wraps a RAPTORNode with its similarity score and the source_block_ids
    it covers — these block_ids map directly to CitationAnchor.block_id
    in the review models for click-to-source navigation.

    cosine_similarity is in range [0.0, 1.0]:
        1.0 = identical vectors
        0.0 = completely unrelated
    """
    node: RAPTORNode
    cosine_similarity: float
    # Convenience copy of node.source_block_ids — avoids repeated attribute access
    source_block_ids: List[str] = field(default_factory=list)

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
        """True if the result meets the minimum similarity threshold."""
        return self.cosine_similarity >= _MIN_COSINE_SIMILARITY


@dataclass
class RAPTORSearchResponse:
    """
    The complete response from a RAPTORSearcher.search() call.

    results are ordered by cosine_similarity descending (most similar first).
    query_embedding is stored for reuse by the router to avoid regenerating
    the same embedding for vector search or graph enrichment.

    all_source_block_ids is the union of source_block_ids across all results —
    used by analysis_service.py to retrieve the full set of TSD text blocks
    that are relevant to a parameter for context window construction.
    """
    results: List[RAPTORSearchResult] = field(default_factory=list)
    query_embedding: List[float] = field(default_factory=list)
    total_found: int = 0
    query_text: str = ""
    level_searched: int = RAPTOR_LEVEL_LOW
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0

    @property
    def all_source_block_ids(self) -> List[str]:
        """
        Returns deduplicated union of source_block_ids across all results.
        Preserves insertion order — results are already sorted by similarity
        so higher-relevance block_ids appear first.
        """
        seen: set = set()
        block_ids: List[str] = []
        for result in self.results:
            for block_id in result.source_block_ids:
                if block_id not in seen:
                    block_ids.append(block_id)
                    seen.add(block_id)
        return block_ids

    def get_context_chunks(self) -> List[str]:
        """
        Returns the text of each result as a list of context strings.
        Used by analysis_service.py to build the context window passed
        to the Hunter agent.
        """
        return [r.text for r in self.results if r.text]


# ---------------------------------------------------------------------------
# RAPTOR Searcher
# ---------------------------------------------------------------------------

class RAPTORSearcher:
    """
    Performs cosine similarity search against a RAPTORTree at a specified
    abstraction level.

    Unlike VectorSearcher which queries the database [3], RAPTORSearcher
    operates entirely in memory against a pre-built RAPTORTree produced by
    RAPTORTreeBuilder. The tree is passed in at search time — it is built
    once per TSD document per review and cached by analysis_service.py.

    Search strategy:
        1. Generate embedding for the query text via Amazon Titan [4].
        2. Retrieve all nodes at the requested level from the RAPTORTree.
        3. Compute cosine similarity between query and each node embedding.
        4. Filter by _MIN_COSINE_SIMILARITY threshold.
        5. Return top_k results sorted by similarity descending.

    Multi-level search:
        search_multi_level() searches across multiple levels simultaneously
        and returns a merged, deduplicated result set. Used by the router
        for RAPTOR_HIGH strategy where the parameter needs cross-cutting
        context from multiple abstraction levels.

    Usage:
        searcher = RAPTORSearcher()
        response = searcher.search(
            query_text="JWT token validation on all API endpoints",
            tree=raptor_tree,
            level=RAPTOR_LEVEL_MID,
            top_k=5,
        )
        for result in response.results:
            print(result.text, result.cosine_similarity)
    """

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

    # ------------------------------------------------------------------
    # Public search entry points
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        tree: RAPTORTree,
        level: int = RAPTOR_LEVEL_LOW,
        top_k: Optional[int] = None,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> RAPTORSearchResponse:
        """
        Searches for the most relevant RAPTORNodes at a specific tree level.

        Args:
            query_text:            The security parameter text to search for.
            tree:                  The RAPTORTree built from the TSD document.
            level:                 The tree level to search — use RAPTOR_LEVEL_*
                                   constants. Defaults to RAPTOR_LEVEL_LOW (leaves).
            top_k:                 Maximum number of results. Capped at _MAX_TOP_K.
            precomputed_embedding: Optional precomputed vector to skip the
                                   embedding API call. Passed by the router when
                                   reusing embeddings across strategies.

        Returns:
            RAPTORSearchResponse — never raises. Check .error for failures.
        """
        # ------------------------------------------------------------------
        # 1. Validate inputs
        # ------------------------------------------------------------------
        if not query_text or not query_text.strip():
            msg = "query_text is empty — cannot perform RAPTOR search."
            self.logger.warning("RAPTORSearcher.search: %s", msg)
            return RAPTORSearchResponse(
                error=msg,
                query_text=query_text,
                level_searched=level,
            )

        if tree.is_empty():
            msg = f"RAPTORTree for '{tree.document_name}' is empty — no nodes to search."
            self.logger.warning("RAPTORSearcher.search: %s", msg)
            return RAPTORSearchResponse(
                error=msg,
                query_text=query_text,
                level_searched=level,
            )

        resolved_top_k = min(
            top_k if top_k is not None else self.default_top_k,
            _MAX_TOP_K,
        )

        # ------------------------------------------------------------------
        # 2. Get nodes at the requested level
        # ------------------------------------------------------------------
        level_nodes = tree.get_nodes_at_level(level)

        if not level_nodes:
            # Requested level doesn't exist — fall back to the highest
            # available level gracefully rather than returning empty
            self.logger.warning(
                "RAPTORSearcher.search: level %d not found in tree '%s' "
                "(max_level=%d) — falling back to level 0.",
                level,
                tree.document_name,
                tree.max_level,
            )
            level = 0
            level_nodes = tree.get_nodes_at_level(0)

            if not level_nodes:
                msg = f"No nodes found at any level in tree '{tree.document_name}'."
                return RAPTORSearchResponse(
                    error=msg,
                    query_text=query_text,
                    level_searched=level,
                )

        # Filter to nodes that have valid embeddings
        embeddable_nodes = [n for n in level_nodes if n.has_embedding]

        if not embeddable_nodes:
            msg = (
                f"No nodes with embeddings found at level {level} in "
                f"tree '{tree.document_name}' — embeddings may have failed "
                f"during tree construction."
            )
            self.logger.warning("RAPTORSearcher.search: %s", msg)
            return RAPTORSearchResponse(
                error=msg,
                query_text=query_text,
                level_searched=level,
            )

        # ------------------------------------------------------------------
        # 3. Generate or reuse the query embedding
        # ------------------------------------------------------------------
        if precomputed_embedding:
            query_vector = precomputed_embedding
            self.logger.debug(
                "RAPTORSearcher.search: using precomputed embedding "
                "for query '%s...'",
                query_text[:60],
            )
        else:
            query_vector = self._embed_query(query_text)
            if not query_vector:
                msg = (
                    f"Failed to generate embedding for query "
                    f"'{query_text[:60]}...'."
                )
                self.logger.error("RAPTORSearcher.search: %s", msg)
                return RAPTORSearchResponse(
                    error=msg,
                    query_text=query_text,
                    level_searched=level,
                )

        # ------------------------------------------------------------------
        # 4. Score all nodes at this level
        # ------------------------------------------------------------------
        scored_results = self._score_nodes(
            query_vector=query_vector,
            nodes=embeddable_nodes,
        )

        # ------------------------------------------------------------------
        # 5. Filter by threshold, sort, and truncate
        # ------------------------------------------------------------------
        relevant_results = [
            r for r in scored_results
            if r.is_relevant
        ]
        relevant_results.sort(key=lambda r: r.cosine_similarity, reverse=True)
        top_results = relevant_results[:resolved_top_k]

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
        )

    def search_multi_level(
        self,
        query_text: str,
        tree: RAPTORTree,
        levels: Optional[List[int]] = None,
        top_k_per_level: int = 3,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> RAPTORSearchResponse:
        """
        Searches across multiple tree levels simultaneously and returns a
        merged, deduplicated result set ordered by similarity descending.

        Used by the router for RAPTOR_HIGH strategy — cross-cutting
        security parameters need context from both leaf nodes (specific
        statements) and chapter summaries (overall posture).

        Args:
            query_text:            The security parameter text to search for.
            tree:                  The RAPTORTree built from the TSD document.
            levels:                List of level integers to search. Defaults
                                   to [0, 1, 2].
            top_k_per_level:       Maximum results per level before merging.
            precomputed_embedding: Optional precomputed vector.

        Returns:
            A single merged RAPTORSearchResponse — never raises.
        """
        resolved_levels = levels if levels is not None else [
            RAPTOR_LEVEL_LOW,
            RAPTOR_LEVEL_MID,
            RAPTOR_LEVEL_HIGH,
        ]

        if not query_text or not query_text.strip():
            msg = "query_text is empty — cannot perform multi-level RAPTOR search."
            self.logger.warning("RAPTORSearcher.search_multi_level: %s", msg)
            return RAPTORSearchResponse(
                error=msg,
                query_text=query_text,
            )

        # ------------------------------------------------------------------
        # Generate embedding once and reuse across all level searches
        # ------------------------------------------------------------------
        if precomputed_embedding:
            query_vector = precomputed_embedding
        else:
            query_vector = self._embed_query(query_text)
            if not query_vector:
                msg = (
                    f"Failed to generate embedding for query "
                    f"'{query_text[:60]}...'."
                )
                self.logger.error(
                    "RAPTORSearcher.search_multi_level: %s", msg
                )
                return RAPTORSearchResponse(
                    error=msg,
                    query_text=query_text,
                )

        # ------------------------------------------------------------------
        # Search each level independently, reusing the same embedding
        # ------------------------------------------------------------------
        all_results: List[RAPTORSearchResult] = []
        seen_node_ids: set = set()
        levels_searched: List[int] = []

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

            # Deduplicate by node_id — a node exists at exactly one level
            # in a RAPTOR tree, but the guard is kept for correctness
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

        # Sort merged results by similarity descending
        all_results.sort(key=lambda r: r.cosine_similarity, reverse=True)

        self.logger.info(
            "RAPTORSearcher.search_multi_level: merged %d result(s) "
            "across levels %s for query '%s...' in tree '%s'.",
            len(all_results),
            levels_searched,
            query_text[:60],
            tree.document_name,
        )

        # Use the lowest level searched as the reported level
        reported_level = min(levels_searched) if levels_searched else RAPTOR_LEVEL_LOW

        return RAPTORSearchResponse(
            results=all_results,
            query_embedding=query_vector,
            total_found=len(all_results),
            query_text=query_text,
            level_searched=reported_level,
            error=None,
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
        """
        Searches RAPTOR nodes across all levels as one flat candidate set,
        then applies both top_k and cumulative token budget constraints.
        """
        if not query_text or not query_text.strip():
            return RAPTORSearchResponse(error="query_text is empty.", query_text=query_text)
        if tree is None or tree.is_empty():
            return RAPTORSearchResponse(error="RAPTORTree is empty.", query_text=query_text)

        query_vector = precomputed_embedding or self._embed_query(query_text)
        if not query_vector:
            return RAPTORSearchResponse(error="Failed to generate embedding.", query_text=query_text)

        nodes = tree.get_all_nodes()
        if allowed_levels is not None:
            allowed = set(allowed_levels)
            nodes = [n for n in nodes if n.level in allowed]
        nodes = [n for n in nodes if n.has_embedding and n.embedding]
        if not nodes:
            return RAPTORSearchResponse(error="No embeddable RAPTOR nodes found.", query_text=query_text)

        scored = self._score_nodes(query_vector=query_vector, nodes=nodes)
        scored.sort(key=lambda r: r.cosine_similarity, reverse=True)

        selected: List[RAPTORSearchResult] = []
        token_total = 0
        resolved_top_k = min(top_k, _MAX_TOP_K)
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query_text: str) -> List[float]:
        """
        Generates an embedding vector for the query text using Amazon
        Titan via get_embedding() from client.py [4].

        Returns an empty list if the API call fails — callers must check
        for empty list before proceeding. Consistent with the same pattern
        in VectorSearcher._embed_query() (retrieval/vector_search.py).

        Args:
            query_text: The text to embed.

        Returns:
            A list of floats (embedding vector), or [] on failure.
        """
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
        """
        Computes cosine similarity between the query vector and every
        node embedding, returning a list of RAPTORSearchResult instances.

        Uses _compute_cosine_similarity() from raptor.py — defined there
        so the retrieval layer can import it without a circular dependency
        through the processing layer.

        Nodes with empty embeddings are silently skipped — they were
        already filtered by the caller via has_embedding checks, but this
        guard is kept as a second line of defence.

        Args:
            query_vector: The embedding vector for the query text.
            nodes:        RAPTORNode instances at a single tree level,
                          all of which have has_embedding=True.

        Returns:
            List of RAPTORSearchResult instances — one per scored node.
            Not sorted — caller is responsible for sorting.
        """
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


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def search_raptor_tree(
    query_text: str,
    tree: RAPTORTree,
    level: int = RAPTOR_LEVEL_LOW,
    top_k: int = _DEFAULT_TOP_K,
    precomputed_embedding: Optional[List[float]] = None,
) -> RAPTORSearchResponse:
    """
    Module-level convenience wrapper around RAPTORSearcher.search().

    Instantiates a RAPTORSearcher with default settings and executes
    a single-level search. Used by retrieval/router.py for RAPTOR_LOW
    and RAPTOR_MID strategy queries.

    Args:
        query_text:            The security parameter text to search for.
        tree:                  The RAPTORTree built from the TSD document.
        level:                 Tree level — use RAPTOR_LEVEL_* constants.
        top_k:                 Maximum number of results.
        precomputed_embedding: Optional precomputed vector.

    Returns:
        RAPTORSearchResponse — never raises.
    """
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
    """
    Module-level convenience wrapper around RAPTORSearcher.search_multi_level().

    Used by retrieval/router.py for RAPTOR_HIGH strategy queries where
    cross-cutting parameters need context from multiple abstraction levels.

    Args:
        query_text:            The security parameter text to search for.
        tree:                  The RAPTORTree built from the TSD document.
        levels:                List of levels to search. Defaults to [0, 1, 2].
        top_k_per_level:       Maximum results per level before merging.
        precomputed_embedding: Optional precomputed vector.

    Returns:
        RAPTORSearchResponse — never raises.
    """
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


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Level constants — imported by retrieval/router.py
    "RAPTOR_LEVEL_LOW",
    "RAPTOR_LEVEL_MID",
    "RAPTOR_LEVEL_HIGH",
    "RAPTOR_LEVEL_ROOT",
    # Result dataclasses — imported by analysis_service.py
    "RAPTORSearchResult",
    "RAPTORSearchResponse",
    # Main searcher class
    "RAPTORSearcher",
    # Convenience functions — used by router.py
    "search_raptor_tree",
    "search_raptor_tree_multi_level",
    "search_collapsed_raptor",
]
