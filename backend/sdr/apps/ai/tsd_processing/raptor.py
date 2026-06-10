# apps/ai/tsd_processing/raptor.py

"""
RAPTOR — Recursive Abstractive Processing for Tree-Organised Retrieval.

Responsibility:
    Builds a multi-level summarisation tree from TSD text blocks so the
    Multi-Agent pipeline can retrieve context at the right level of
    abstraction — a single requirement sentence, a section summary, or
    a document-wide summary — without the middle of the document getting
    lost during analysis.

Why RAPTOR?
    Standard flat chunking loses document structure. A security parameter
    about "data encryption at rest" might need evidence from three separate
    sections of a TSD. RAPTOR's tree allows the retrieval router to pull
    a Level-2 chapter summary that synthesises all three sections into a
    single context window, then drill down to Level-0 leaf blocks for
    precise citation metadata.

Tree structure:
    Level 0 (leaves):  Individual TextBlock chunks (~400 tokens each)
                       ← raw text from TSDIngestor [ingestor.py]
    Level 1:           Section summaries (~800 tokens)
                       ← summarise clusters of Level-0 leaves
    Level 2:           Chapter summaries (~1200 tokens)
                       ← summarise clusters of Level-1 nodes
    Level 3 (root):    Document summary
                       ← summarise all Level-2 nodes

Each node stores:
    - Its summarised text
    - Its embedding vector (from Amazon Titan via client.py [4])
    - References to its child nodes (for drill-down)
    - The source block_ids it covers (for citation tracing)

Dependency chain:
    ingestor.py     (TSDDocument, TextBlock)
         ↓
    client.py [4]   (get_embedding — Amazon Titan)
         ↓
    raptor.py       ← YOU ARE HERE
         ↓
    retrieval/raptor_search.py
         ↓
    retrieval/router.py
         ↓
    analysis_service.py
"""

from __future__ import annotations

import logging
import math
import hashlib
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from sdr.core.config import settings

from sdr.apps.ai.client import chat_completion, get_embeddings
from sdr.apps.ai.tsd_processing.ingestor import TextBlock, TSDDocument
from sdr.apps.ai.utils.concurrency import ConcurrencyProbe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Token budget per level — each level allows progressively larger summaries
_LEVEL_TOKEN_BUDGETS = {
    0: 400,    # leaf: raw text chunk size
    1: 800,    # section summary
    2: 1200,   # chapter summary
    3: 2000,   # document root summary
}

# Minimum number of leaf nodes required to build a tree worth using.
# Below this threshold, the document is too short for RAPTOR to add value.
_MIN_LEAF_NODES = 3

# Target cluster size at each level — how many child nodes per parent.
# Kept small so summaries remain focused.
_TARGET_CLUSTER_SIZE = 5

# Minimum cluster size — clusters smaller than this are merged with neighbours.
_MIN_CLUSTER_SIZE = 2

# Maximum tree depth — prevents runaway recursion on very large documents.
_MAX_TREE_DEPTH = 3

# Embedding model dimensions — must match CategoryParameterEmbedding.model_dim [3]
_EMBEDDING_DIMENSIONS = 1024

# Summarisation model — Claude Sonnet via BedrockService [4]
_SUMMARY_TEMPERATURE = 0.1
_SUMMARY_MAX_TOKENS = 1024
_MAX_SUMMARY_WORKERS = 3
_MAX_EMBED_WORKERS = 4
_DEFAULT_EMBED_BATCH_SIZE = 32


def _bounded_worker_count(configured: int, default: int) -> int:
    try:
        return max(1, int(configured))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# RAPTOR node dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAPTORNode:
    """
    A single node in the RAPTOR summarisation tree.

    Level-0 nodes are leaf nodes — they contain raw TextBlock text and
    carry the original block_ids for citation tracing.

    Level 1+ nodes are summary nodes — they contain AI-generated summaries
    of their child nodes and carry the union of all descendant block_ids.

    The embedding vector enables cosine similarity search at any tree level
    via the retrieval layer (retrieval/raptor_search.py).
    """
    node_id: str                              # "level{L}_node{idx}"
    level: int                                # 0 = leaf, 1+ = summary
    text: str                                 # raw text or AI summary
    embedding: List[float] = field(default_factory=list)
    # block_ids this node covers — union of all descendant leaf block_ids
    source_block_ids: List[str] = field(default_factory=list)
    # Direct children in the tree (empty for leaf nodes)
    children: List[RAPTORNode] = field(default_factory=list)
    # Page numbers covered by this node (for context display)
    page_numbers: List[int] = field(default_factory=list)
    # Section heading associated with this node (from TSDPage.section_heading)
    section_heading: Optional[str] = None
    # Whether embedding was successfully generated
    has_embedding: bool = False

    @property
    def is_leaf(self) -> bool:
        return self.level == 0

    @property
    def token_estimate(self) -> int:
        """Rough token estimate: ~4 characters per token."""
        return len(self.text) // 4

    def to_retrieval_dict(self) -> Dict[str, Any]:
        """
        Serialises this node to a flat dict for use by the retrieval layer.
        Does NOT include the children list — retrieval operates on
        individual nodes, not the tree structure.
        """
        return {
            "node_id": self.node_id,
            "level": self.level,
            "text": self.text,
            "embedding": self.embedding,
            "source_block_ids": self.source_block_ids,
            "page_numbers": self.page_numbers,
            "section_heading": self.section_heading,
            "has_embedding": self.has_embedding,
            "token_estimate": self.token_estimate,
        }


# ---------------------------------------------------------------------------
# RAPTOR tree dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAPTORTree:
    """
    The complete RAPTOR summarisation tree for a single TSD document.

    Produced by RAPTORTreeBuilder.build() and passed to:
        - retrieval/raptor_search.py  (level-aware similarity search)
        - analysis_service.py         (context retrieval per parameter)

    The tree is organised as a list of levels — each level is a list of
    RAPTORNode instances at that depth. Level 0 is the leaves (raw chunks),
    higher levels are progressively coarser summaries.
    """
    document_name: str
    levels: List[List[RAPTORNode]] = field(default_factory=list)
    root_node: Optional[RAPTORNode] = None
    total_nodes: int = 0
    max_level: int = 0

    def get_nodes_at_level(self, level: int) -> List[RAPTORNode]:
        """Returns all nodes at the specified level. Empty list if out of range."""
        if level < 0 or level >= len(self.levels):
            return []
        return self.levels[level]

    def get_all_nodes(self) -> List[RAPTORNode]:
        """Returns all nodes across all levels in level-ascending order."""
        nodes = []
        for level_nodes in self.levels:
            nodes.extend(level_nodes)
        return nodes

    def get_leaf_nodes(self) -> List[RAPTORNode]:
        """Returns all Level-0 leaf nodes."""
        return self.get_nodes_at_level(0)

    def get_nodes_with_embeddings(self) -> List[RAPTORNode]:
        """Returns all nodes that have valid embedding vectors."""
        return [n for n in self.get_all_nodes() if n.has_embedding]

    def is_empty(self) -> bool:
        return self.total_nodes == 0


# ---------------------------------------------------------------------------
# RAPTOR Tree Builder
# ---------------------------------------------------------------------------

class RAPTORTreeBuilder:
    """
    Builds a RAPTOR summarisation tree from a TSDDocument.

    Algorithm:
        1. Create Level-0 leaf nodes from TSD text blocks — each node
           covers one or more TextBlocks grouped by section heading.
        2. Cluster Level-0 nodes into groups of ~_TARGET_CLUSTER_SIZE.
        3. Summarise each cluster using Claude [4] → Level-1 nodes.
        4. Repeat clustering and summarisation up to _MAX_TREE_DEPTH.
        5. Generate embeddings for every node using Amazon Titan [4].

    The resulting tree enables retrieval at multiple granularities:
        - Level 0: precise citation-level evidence
        - Level 1: section-level context
        - Level 2: chapter-level context
        - Level 3: document-level context

    Usage:
        builder = RAPTORTreeBuilder()
        tree = builder.build(tsd_document)
    """

    def __init__(
        self,
        target_cluster_size: int = _TARGET_CLUSTER_SIZE,
        min_cluster_size: int = _MIN_CLUSTER_SIZE,
        max_depth: int = _MAX_TREE_DEPTH,
        embedding_dimensions: int = _EMBEDDING_DIMENSIONS,
    ) -> None:
        self.target_cluster_size = target_cluster_size
        self.min_cluster_size = min_cluster_size
        self.max_depth = max_depth
        self.embedding_dimensions = embedding_dimensions
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.summary_max_concurrency = _bounded_worker_count(
            getattr(
                settings,
                "AI_RAPTOR_SUMMARY_MAX_CONCURRENCY",
                _MAX_SUMMARY_WORKERS,
            ),
            _MAX_SUMMARY_WORKERS,
        )
        self.embed_max_concurrency = _bounded_worker_count(
            getattr(
                settings,
                "AI_RAPTOR_EMBED_MAX_CONCURRENCY",
                _MAX_EMBED_WORKERS,
            ),
            _MAX_EMBED_WORKERS,
        )
        self.embed_batch_size = _bounded_worker_count(
            getattr(
                settings,
                "AI_RAPTOR_EMBED_BATCH_SIZE",
                _DEFAULT_EMBED_BATCH_SIZE,
            ),
            _DEFAULT_EMBED_BATCH_SIZE,
        )
        self._last_summary_concurrency_stats: Dict[str, Any] = {}
        self._last_embed_concurrency_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, tsd_document: TSDDocument, progress_callback=None) -> RAPTORTree:
        """
        Builds the complete RAPTOR tree from a TSDDocument.

        Args:
            tsd_document: The fully parsed TSDDocument from TSDIngestor [ingestor.py].

        Returns:
            A populated RAPTORTree ready for the retrieval layer.
            Returns an empty RAPTORTree if the document has too few
            text blocks to build a meaningful tree.
        """
        tree = RAPTORTree(document_name=tsd_document.document_name)

        all_blocks = tsd_document.all_text_blocks
        valid_blocks = [b for b in all_blocks if b.is_valid()]

        if len(valid_blocks) < _MIN_LEAF_NODES:
            self.logger.warning(
                "RAPTORTreeBuilder.build: document '%s' has only %d valid "
                "text block(s) — below minimum %d for RAPTOR tree. "
                "Returning empty tree.",
                tsd_document.document_name,
                len(valid_blocks),
                _MIN_LEAF_NODES,
            )
            if progress_callback:
                progress_callback(
                    {
                        "status": "skipped",
                        "progress_percent": 100,
                        "current_step": "RAPTOR skipped for short document",
                        "leaf_nodes": len(valid_blocks),
                        "summary_levels_completed": 0,
                        "total_nodes": 0,
                        "embedded_nodes": 0,
                    }
                )
            return tree

        self.logger.info(
            "RAPTORTreeBuilder.build: building tree for '%s' "
            "from %d valid text block(s) "
            "(summary_max_concurrency=%d, embed_max_concurrency=%d).",
            tsd_document.document_name,
            len(valid_blocks),
            self.summary_max_concurrency,
            self.embed_max_concurrency,
        )

        # ------------------------------------------------------------------
        # Step 1: Build Level-0 leaf nodes from text blocks
        # ------------------------------------------------------------------
        leaf_nodes = self._build_leaf_nodes(valid_blocks, tsd_document)
        tree.levels.append(leaf_nodes)
        if progress_callback:
            progress_callback(
                {
                    "status": "running",
                    "progress_percent": 20,
                    "current_step": "Creating RAPTOR leaf nodes",
                    "leaf_nodes": len(leaf_nodes),
                    "summary_levels_completed": 0,
                }
            )

        self.logger.info(
            "RAPTORTreeBuilder.build: Level 0 — %d leaf node(s) created.",
            len(leaf_nodes),
        )

        # ------------------------------------------------------------------
        # Steps 2–4: Iteratively cluster and summarise up the tree
        # ------------------------------------------------------------------
        current_level_nodes = leaf_nodes
        current_level = 0

        while (
            len(current_level_nodes) > 1
            and current_level < self.max_depth
        ):
            next_level = current_level + 1

            clusters = self._cluster_nodes(current_level_nodes)

            if not clusters:
                self.logger.warning(
                    "RAPTORTreeBuilder.build: clustering produced no "
                    "clusters at level %d — stopping.",
                    current_level,
                )
                break

            summary_nodes = self._summarise_clusters(
                clusters=clusters,
                level=next_level,
                progress_callback=progress_callback,
            )

            if not summary_nodes:
                self.logger.warning(
                    "RAPTORTreeBuilder.build: no summary nodes produced "
                    "at level %d — stopping.",
                    next_level,
                )
                break

            tree.levels.append(summary_nodes)
            self.logger.info(
                "RAPTORTreeBuilder.build: Level %d — %d summary node(s) "
                "from %d cluster(s).",
                next_level,
                len(summary_nodes),
                len(clusters),
            )
            if progress_callback:
                progress_callback(
                    {
                        "status": "running",
                        "progress_percent": min(65, 20 + next_level * 15),
                        "current_step": f"Summarizing RAPTOR level {next_level}",
                        "leaf_nodes": len(leaf_nodes),
                        "summary_levels_completed": next_level,
                        "total_nodes": sum(len(level_nodes) for level_nodes in tree.levels),
                    }
                )

            current_level_nodes = summary_nodes
            current_level = next_level

        # ------------------------------------------------------------------
        # Step 5: Set the root node — the last remaining node after
        #         recursive summarisation, or a synthesised root if
        #         multiple nodes remain at the top level
        # ------------------------------------------------------------------
        top_level_nodes = tree.levels[-1] if tree.levels else []

        if len(top_level_nodes) == 1:
            tree.root_node = top_level_nodes[0]
        elif len(top_level_nodes) > 1:
            # Multiple top-level nodes — create a single root summary
            root_node = self._synthesise_root(
                top_level_nodes=top_level_nodes,
                level=len(tree.levels),
                document_name=tsd_document.document_name,
            )
            if root_node:
                tree.levels.append([root_node])
                tree.root_node = root_node

        # ------------------------------------------------------------------
        # Step 6: Generate embeddings for all nodes
        # ------------------------------------------------------------------
        all_nodes = tree.get_all_nodes()
        if progress_callback:
            progress_callback(
                {
                    "status": "running",
                    "progress_percent": 70,
                    "current_step": "Embedding RAPTOR nodes",
                    "leaf_nodes": len(leaf_nodes),
                    "summary_levels_completed": max(0, len(tree.levels) - 1),
                    "total_nodes": len(all_nodes),
                    "embedded_nodes": 0,
                }
            )
        self._embed_all_nodes(all_nodes, progress_callback=progress_callback)

        # ------------------------------------------------------------------
        # Finalise tree metadata
        # ------------------------------------------------------------------
        tree.total_nodes = len(all_nodes)
        tree.max_level = len(tree.levels) - 1

        self.logger.info(
            "RAPTORTreeBuilder.build: tree complete for '%s' — "
            "%d level(s), %d total node(s), %d with embeddings.",
            tsd_document.document_name,
            len(tree.levels),
            tree.total_nodes,
            len(tree.get_nodes_with_embeddings()),
        )
        if progress_callback:
            progress_callback(
                {
                    "status": "completed",
                    "progress_percent": 100,
                    "current_step": "RAPTOR index ready",
                    "leaf_nodes": len(leaf_nodes),
                    "summary_levels_completed": max(0, len(tree.levels) - 1),
                    "total_nodes": tree.total_nodes,
                    "embedded_nodes": len(tree.get_nodes_with_embeddings()),
                }
            )

        return tree

    # ------------------------------------------------------------------
    # Level-0 leaf node construction
    # ------------------------------------------------------------------

    def _build_leaf_nodes(
        self,
        text_blocks: List[TextBlock],
        tsd_document: TSDDocument,
    ) -> List[RAPTORNode]:
        """
        Groups TextBlocks by section heading and creates one Level-0
        leaf node per section group.

        Grouping by section heading ensures that related blocks stay
        together at Level 0 — a security parameter about authentication
        retrieves the full authentication section, not just one block.

        If a section group exceeds _LEVEL_TOKEN_BUDGETS[0] tokens,
        it is further split into sub-chunks so no leaf node exceeds
        the budget.

        Args:
            text_blocks:  All valid TextBlock instances from the TSD [ingestor.py].
            tsd_document: The parent TSDDocument for section heading lookup.

        Returns:
            A list of Level-0 RAPTORNode instances ordered by document position.
        """
        # Group page-level markdown by section heading so retrieval uses the
        # markdown-first ingestion output, including image references.
        section_groups: List[Dict[str, Any]] = []

        for page in tsd_document.pages:
            page_text = page.all_text.strip()
            if not page_text:
                continue

            heading_key = page.section_heading or f"__page_{page.page_number}__"
            page_block_ids = [
                block.block_id for block in page.text_blocks if block.is_valid()
            ]

            if (
                section_groups
                and section_groups[-1]["heading_key"] == heading_key
                and not heading_key.startswith("__page_")
            ):
                section_groups[-1]["texts"].append(page_text)
                section_groups[-1]["block_ids"].extend(page_block_ids)
                section_groups[-1]["page_numbers"].append(page.page_number)
            else:
                section_groups.append(
                    {
                        "heading_key": heading_key,
                        "texts": [page_text],
                        "block_ids": list(page_block_ids),
                        "page_numbers": [page.page_number],
                    }
                )

        leaf_nodes: List[RAPTORNode] = []
        node_idx = 0
        token_budget = _LEVEL_TOKEN_BUDGETS[0]

        for group in section_groups:
            heading_key = group["heading_key"]
            section_heading = (
                None if heading_key.startswith("__page_") else heading_key
            )

            group_text = "\n\n".join(group["texts"])
            group_block_ids = list(dict.fromkeys(group["block_ids"]))
            group_page_numbers = sorted(set(group["page_numbers"]))

            # Split into sub-chunks if the group exceeds the token budget
            sub_chunks = _split_text_to_budget(group_text, token_budget)

            for chunk_text in sub_chunks:
                leaf_nodes.append(
                    RAPTORNode(
                        node_id=f"level0_node{node_idx}",
                        level=0,
                        text=chunk_text,
                        source_block_ids=group_block_ids,
                        page_numbers=group_page_numbers,
                        section_heading=section_heading,
                    )
                )
                node_idx += 1

        self.logger.debug(
            "RAPTORTreeBuilder._build_leaf_nodes: created %d leaf node(s) "
            "from %d markdown section group(s).",
            len(leaf_nodes),
            len(section_groups),
        )
        return leaf_nodes

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster_nodes(
        self,
        nodes: List[RAPTORNode],
    ) -> List[List[RAPTORNode]]:
        """
        Partitions a flat list of RAPTORNodes into clusters of approximately
        _target_cluster_size nodes each.

        Uses sequential (document-order) clustering rather than k-means
        because:
        - Document order preserves narrative/structural coherence —
          adjacent sections in a TSD are semantically related.
        - k-means requires embeddings which are generated AFTER clustering.
        - Sequential clustering is O(n) vs O(n×k×iterations) for k-means.

        Clusters smaller than _min_cluster_size are merged into their
        preceding cluster to avoid creating summary nodes from a single
        source node.

        Args:
            nodes: Flat list of RAPTORNode instances at the current level.

        Returns:
            A list of clusters, where each cluster is a non-empty list
            of RAPTORNode instances. Returns an empty list if input is empty.
        """
        if not nodes:
            return []

        # Single node — no clustering needed, return as one cluster
        if len(nodes) == 1:
            return [nodes]

        clusters: List[List[RAPTORNode]] = []
        current_cluster: List[RAPTORNode] = []

        for node in nodes:
            current_cluster.append(node)
            if len(current_cluster) >= self.target_cluster_size:
                clusters.append(current_cluster)
                current_cluster = []

        # Handle the remaining nodes
        if current_cluster:
            if (
                len(current_cluster) < self.min_cluster_size
                and clusters
            ):
                # Merge undersized trailing cluster into the last cluster
                self.logger.debug(
                    "RAPTORTreeBuilder._cluster_nodes: merging trailing "
                    "cluster of %d node(s) into previous cluster.",
                    len(current_cluster),
                )
                clusters[-1].extend(current_cluster)
            else:
                clusters.append(current_cluster)

        self.logger.debug(
            "RAPTORTreeBuilder._cluster_nodes: produced %d cluster(s) "
            "from %d node(s).",
            len(clusters),
            len(nodes),
        )
        return clusters

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def _summarise_clusters(
        self,
        clusters: List[List[RAPTORNode]],
        level: int,
        progress_callback=None,
    ) -> List[RAPTORNode]:
        """
        Summarises each cluster of child nodes into a single parent
        RAPTORNode at the specified level using Claude via client.py [4].

        Each summary node carries the union of all source_block_ids and
        page_numbers from its children — this is how citation traceability
        is maintained up the tree. A Level-2 node knows exactly which
        Level-0 leaf block_ids it covers.

        Args:
            clusters: List of node clusters from _cluster_nodes().
            level:    The tree level for the new summary nodes (1, 2, or 3).

        Returns:
            List of summary RAPTORNode instances — one per cluster.
            Clusters that fail summarisation are skipped with a warning.
        """
        token_budget = _LEVEL_TOKEN_BUDGETS.get(level, 1200)
        if not clusters:
            return []

        summary_nodes_by_cluster: Dict[int, RAPTORNode] = {}
        probe = ConcurrencyProbe(max_concurrency=self.summary_max_concurrency)
        self.logger.info(
            "RAPTORTreeBuilder._summarise_clusters: level=%d clusters=%d max_concurrency=%d",
            level,
            len(clusters),
            self.summary_max_concurrency,
        )
        with ThreadPoolExecutor(
            max_workers=min(self.summary_max_concurrency, len(clusters)),
            thread_name_prefix="ThreadPoolExecutor-1",
        ) as executor:
            futures = {
                executor.submit(
                    probe.wrap(self._summarise_cluster),
                    cluster_idx=cluster_idx,
                    cluster=cluster,
                    level=level,
                    token_budget=token_budget,
                ): cluster_idx
                for cluster_idx, cluster in enumerate(clusters)
            }
            probe.mark_submitted(len(futures))
            for future in as_completed(futures):
                cluster_idx = futures[future]
                try:
                    summary_node = future.result()
                except Exception as exc:
                    self.logger.error(
                        "RAPTORTreeBuilder._summarise_clusters: unexpected "
                        "error for cluster %d at level %d: %s",
                        cluster_idx,
                        level,
                        exc,
                    )
                    summary_node = None
                if summary_node is not None:
                    summary_nodes_by_cluster[cluster_idx] = summary_node
                if progress_callback:
                    completed_clusters = len(summary_nodes_by_cluster)
                    cluster_progress = int(round((completed_clusters / max(len(clusters), 1)) * 100))
                    progress_callback(
                        {
                            "status": "running",
                            "progress_percent": min(65, 20 + level * 15),
                            "current_step": f"Summarizing RAPTOR level {level} ({cluster_progress}% of clusters)",
                            "summary_levels_completed": max(0, level - 1),
                        }
                    )

        summary_nodes = [
            summary_nodes_by_cluster[cluster_idx]
            for cluster_idx in sorted(summary_nodes_by_cluster.keys())
        ]
        summary_stats = probe.snapshot().to_dict()
        self._last_summary_concurrency_stats = summary_stats

        self.logger.debug(
            "RAPTORTreeBuilder._summarise_clusters: produced %d summary "
            "node(s) from %d cluster(s) at level %d with max_concurrency=%d.",
            len(summary_nodes),
            len(clusters),
            level,
            self.summary_max_concurrency,
        )
        self.logger.info(
            "RAPTORTreeBuilder._summarise_clusters: level=%d submitted=%d "
            "completed=%d failed=%d peak_in_flight=%d max_concurrency=%d "
            "elapsed_seconds=%.4f",
            level,
            summary_stats["submitted"],
            summary_stats["completed"],
            summary_stats["failed"],
            summary_stats["peak_in_flight"],
            summary_stats["max_concurrency"],
            summary_stats["elapsed_seconds"],
        )
        return summary_nodes

    def _summarise_cluster(
        self,
        cluster_idx: int,
        cluster: List[RAPTORNode],
        level: int,
        token_budget: int,
    ) -> Optional[RAPTORNode]:
        node_id = f"level{level}_node{cluster_idx}"

        all_block_ids: List[str] = []
        all_page_numbers: List[int] = []
        section_heading = cluster[0].section_heading

        for child_node in cluster:
            all_block_ids.extend(child_node.source_block_ids)
            all_page_numbers.extend(child_node.page_numbers)

        seen_ids: set = set()
        unique_block_ids = []
        for bid in all_block_ids:
            if bid not in seen_ids:
                unique_block_ids.append(bid)
                seen_ids.add(bid)

        unique_page_numbers = sorted(set(all_page_numbers))
        summary_text = self._call_summarisation_llm(
            cluster=cluster,
            level=level,
            token_budget=token_budget,
        )

        if not summary_text:
            self.logger.warning(
                "RAPTORTreeBuilder._summarise_clusters: summarisation "
                "failed for cluster %d at level %d — skipping.",
                cluster_idx,
                level,
            )
            return None

        return RAPTORNode(
            node_id=node_id,
            level=level,
            text=summary_text,
            source_block_ids=unique_block_ids,
            children=list(cluster),
            page_numbers=unique_page_numbers,
            section_heading=section_heading,
        )

    def _call_summarisation_llm(
        self,
        cluster: List[RAPTORNode],
        level: int,
        token_budget: int,
    ) -> Optional[str]:
        """
        Calls Claude via chat_completion() [4] to summarise a cluster
        of RAPTORNodes into a single coherent summary.

        The prompt is calibrated by level — Level-1 summaries focus on
        section-level detail, Level-2 summaries on chapter themes, and
        Level-3 on the document's overall security posture.

        Args:
            cluster:      The nodes to summarise.
            level:        The target tree level — controls prompt verbosity.
            token_budget: Maximum tokens for the summary output.

        Returns:
            The summary string, or None if the LLM call fails.
        """
        combined_text = "\n\n---\n\n".join(
            f"[Node {n.node_id}]\n{n.text}" for n in cluster
        )

        level_instructions = {
            1: (
                "Produce a detailed section-level summary. Preserve specific "
                "security controls, protocol names, configuration details, "
                "and any explicit compliance statements. These details are "
                "critical for security requirement matching."
            ),
            2: (
                "Produce a chapter-level summary covering the main security "
                "themes. Identify which security domains are addressed "
                "(e.g. authentication, encryption, access control) and note "
                "any significant gaps or strengths."
            ),
            3: (
                "Produce a document-level executive summary of the overall "
                "security posture described in this Technical Software Document. "
                "Identify the primary system components, key security controls "
                "implemented, and any notable omissions."
            ),
        }

        instruction = level_instructions.get(
            level,
            level_instructions[1],
        )

        prompt = (
            f"You are summarising sections of a Technical Software Document "
            f"(TSD) for a security compliance review pipeline.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Target length: approximately {token_budget} tokens.\n\n"
            f"Content to summarise:\n\n{combined_text}\n\n"
            f"Output the summary as plain text only. "
            f"No JSON, no bullet points, no markdown headers."
        )

        try:
            response = chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a technical writer summarising security "
                            "documentation for a compliance review pipeline. "
                            "Be precise and preserve all security-relevant details."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                component="long_context",
                temperature=_SUMMARY_TEMPERATURE,
                max_tokens=_SUMMARY_MAX_TOKENS,
            )

            if response.error:
                self.logger.error(
                    "RAPTORTreeBuilder._call_summarisation_llm: LLM error "
                    "at level %d: %s",
                    level,
                    response.error,
                )
                return None

            summary = (response.content or "").strip()
            if not summary:
                self.logger.warning(
                    "RAPTORTreeBuilder._call_summarisation_llm: LLM returned "
                    "empty summary at level %d.",
                    level,
                )
                return None

            return summary

        except Exception as exc:
            self.logger.error(
                "RAPTORTreeBuilder._call_summarisation_llm: unexpected error "
                "at level %d: %s",
                level,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Root synthesis
    # ------------------------------------------------------------------

    def _synthesise_root(
        self,
        top_level_nodes: List[RAPTORNode],
        level: int,
        document_name: str,
    ) -> Optional[RAPTORNode]:
        """
        Creates a single root node from multiple top-level nodes when
        recursive clustering does not converge to a single node.

        This happens when the document is too large for the tree to
        converge within _MAX_TREE_DEPTH levels. The root synthesises
        all top-level nodes into one document-wide summary.

        Args:
            top_level_nodes: The remaining nodes at the highest completed level.
            level:           The level index for the new root node.
            document_name:   Used in logging only.

        Returns:
            A single root RAPTORNode, or None if summarisation fails.
        """
        self.logger.info(
            "RAPTORTreeBuilder._synthesise_root: synthesising root from "
            "%d top-level node(s) for '%s'.",
            len(top_level_nodes),
            document_name,
        )

        # Aggregate all block_ids and page numbers from top-level nodes
        all_block_ids: List[str] = []
        all_page_numbers: List[int] = []

        for node in top_level_nodes:
            all_block_ids.extend(node.source_block_ids)
            all_page_numbers.extend(node.page_numbers)

        seen: set = set()
        unique_block_ids = [
            bid for bid in all_block_ids
            if not (bid in seen or seen.add(bid))
        ]
        unique_page_numbers = sorted(set(all_page_numbers))

        # Use level 3 summarisation instruction for the root
        root_summary = self._call_summarisation_llm(
            cluster=top_level_nodes,
            level=3,
            token_budget=_LEVEL_TOKEN_BUDGETS.get(3, 2000),
        )

        if not root_summary:
            self.logger.warning(
                "RAPTORTreeBuilder._synthesise_root: failed to generate "
                "root summary for '%s' — tree will have multiple top nodes.",
                document_name,
            )
            return None

        return RAPTORNode(
            node_id=f"level{level}_root",
            level=level,
            text=root_summary,
            source_block_ids=unique_block_ids,
            children=list(top_level_nodes),
            page_numbers=unique_page_numbers,
            section_heading=None,   # root covers the entire document
        )

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def _embed_all_nodes(self, nodes: List[RAPTORNode], progress_callback=None) -> None:
        """
        Generates embedding vectors for every RAPTORNode in the tree
        using Amazon Titan via get_embedding() from client.py [4].

        Embeddings are stored in-place on each node. Nodes that fail
        embedding generation have has_embedding=False and an empty
        embedding list — they are still usable for text-based retrieval
        but will be excluded from vector similarity search by the
        retrieval layer (retrieval/raptor_search.py).

        This method mutates the nodes in-place and does not return a value.
        It never raises — individual embedding failures are logged and
        skipped without aborting the remaining nodes.

        Args:
            nodes: All RAPTORNode instances across all tree levels.
        """
        if not nodes:
            return

        success_count = 0
        failure_count = 0
        hash_to_text: Dict[str, str] = {}
        hash_to_nodes: Dict[str, List[RAPTORNode]] = defaultdict(list)

        for node in nodes:
            if not node.text or not node.text.strip():
                self.logger.debug(
                    "RAPTORTreeBuilder._embed_all_nodes: skipping node '%s' "
                    "— empty text, cannot generate embedding.",
                    node.node_id,
                )
                node.embedding = []
                node.has_embedding = False
                failure_count += 1
                continue

            text_hash = hashlib.sha256(
                node.text.encode("utf-8")
            ).hexdigest()
            hash_to_text[text_hash] = node.text
            hash_to_nodes[text_hash].append(node)

        pending_hashes = list(hash_to_nodes.keys())
        probe = ConcurrencyProbe(max_concurrency=self.embed_max_concurrency)
        self.logger.info(
            "RAPTORTreeBuilder._embed_all_nodes: unique_texts=%d max_concurrency=%d batch_size=%d",
            len(pending_hashes),
            self.embed_max_concurrency,
            self.embed_batch_size,
        )
        if pending_hashes:
            hash_batches = [
                pending_hashes[idx:idx + self.embed_batch_size]
                for idx in range(0, len(pending_hashes), self.embed_batch_size)
            ]
            with ThreadPoolExecutor(
                max_workers=min(
                    self.embed_max_concurrency,
                    len(hash_batches),
                ),
                thread_name_prefix="ThreadPoolExecutor-2",
            ) as executor:
                futures = {
                    executor.submit(
                        probe.wrap(get_embeddings),
                        texts=[hash_to_text[text_hash] for text_hash in hash_batch],
                        dimensions=self.embedding_dimensions,
                    ): hash_batch
                    for hash_batch in hash_batches
                }
                probe.mark_submitted(len(futures))
                for future in as_completed(futures):
                    hash_batch = futures[future]
                    try:
                        vectors = future.result() or []
                    except Exception as exc:
                        self.logger.error(
                            "RAPTORTreeBuilder._embed_all_nodes: unexpected "
                            "error embedding batch of %d text(s): %s",
                            len(hash_batch),
                            exc,
                        )
                        vectors = [[] for _ in hash_batch]

                    if len(vectors) != len(hash_batch):
                        self.logger.warning(
                            "RAPTORTreeBuilder._embed_all_nodes: vector count mismatch for batch: expected %d got %d.",
                            len(hash_batch),
                            len(vectors),
                        )
                        vectors = [[] for _ in hash_batch]

                    for text_hash, vector in zip(hash_batch, vectors):
                        grouped_nodes = hash_to_nodes[text_hash]
                        if vector:
                            for node in grouped_nodes:
                                node.embedding = list(vector)
                                node.has_embedding = True
                                success_count += 1
                        else:
                            self.logger.warning(
                                "RAPTORTreeBuilder._embed_all_nodes: empty vector "
                                "returned for %d node(s) sharing hash '%s' — "
                                "marking as no embedding.",
                                len(grouped_nodes),
                                text_hash[:12],
                            )
                            for node in grouped_nodes:
                                node.embedding = []
                                node.has_embedding = False
                                failure_count += 1
                    if progress_callback:
                        processed = success_count + failure_count
                        embed_progress = 70 + int(round((processed / max(len(nodes), 1)) * 30))
                        progress_callback(
                            {
                                "status": "running",
                                "progress_percent": min(99, embed_progress),
                                "current_step": "Embedding RAPTOR nodes",
                                "total_nodes": len(nodes),
                                "embedded_nodes": success_count,
                            }
                        )
        embed_stats = probe.snapshot().to_dict()
        self._last_embed_concurrency_stats = embed_stats

        self.logger.info(
            "RAPTORTreeBuilder._embed_all_nodes: embedding complete — "
            "%d succeeded, %d failed out of %d total node(s) "
            "(unique_texts=%d, max_concurrency=%d, peak_in_flight=%d, "
            "worker_failed=%d, elapsed_seconds=%.4f).",
            success_count,
            failure_count,
            len(nodes),
            len(hash_to_nodes),
            self.embed_max_concurrency,
            embed_stats["peak_in_flight"],
            embed_stats["failed"],
            embed_stats["elapsed_seconds"],
        )


# ---------------------------------------------------------------------------
# Module-level pure utility functions
# ---------------------------------------------------------------------------

def _split_text_to_budget(
    text: str,
    token_budget: int,
) -> List[str]:
    """
    Splits a text string into sub-chunks that each fit within the given
    token budget, using a simple word-boundary split strategy.

    Used by _build_leaf_nodes() to ensure no Level-0 RAPTOR node exceeds
    the leaf token budget (_LEVEL_TOKEN_BUDGETS[0] = 400 tokens).

    Why not use RecursiveCharacterTextSplitter here?
    This function runs during tree construction before any LLM calls —
    it must be fast and have zero external dependencies. The
    RecursiveCharacterTextSplitter from langchain is used in the standards
    extraction pipeline [1] where semantic coherence matters more. Here,
    word-boundary splitting is sufficient because RAPTOR's summarisation
    step (Level 1+) re-establishes semantic coherence across the chunks.

    Token estimation: 1 token ≈ 4 characters — consistent with the
    fallback in chunk_text_with_context() [1] when tiktoken is unavailable.

    Args:
        text:         The input text string to split.
        token_budget: Maximum number of tokens per chunk.

    Returns:
        A list of non-empty text chunks, each within the token budget.
        Returns a list with the original text if it already fits.
        Returns an empty list if the input text is empty.
    """
    if not text or not text.strip():
        return []

    # Fast path — text already fits within budget
    char_budget = token_budget * 4   # 1 token ≈ 4 chars
    if len(text) <= char_budget:
        return [text.strip()]

    # Split on paragraph boundaries first, then sentence boundaries,
    # then word boundaries as a last resort — preserves coherence
    words = text.split()
    chunks: List[str] = []
    current_words: List[str] = []
    current_char_count = 0

    for word in words:
        word_len = len(word) + 1   # +1 for the space

        if current_char_count + word_len > char_budget and current_words:
            # Flush the current chunk
            chunks.append(" ".join(current_words).strip())
            current_words = [word]
            current_char_count = word_len
        else:
            current_words.append(word)
            current_char_count += word_len

    # Flush the final chunk
    if current_words:
        chunks.append(" ".join(current_words).strip())

    # Filter any empty strings produced by edge cases
    return [c for c in chunks if c]


def _compute_cosine_similarity(
    vec_a: List[float],
    vec_b: List[float],
) -> float:
    """
    Computes the cosine similarity between two embedding vectors.

    Used by retrieval/raptor_search.py for level-aware similarity ranking.
    Defined here so it can be imported alongside the tree dataclasses
    without creating a circular import through the retrieval layer.

    Returns 0.0 if either vector is empty or has zero magnitude —
    consistent with the safe fallback pattern used across the codebase.

    Args:
        vec_a: First embedding vector (List[float]).
        vec_b: Second embedding vector (List[float]).

    Returns:
        Cosine similarity in range [0.0, 1.0] for normalised vectors.
        Vectors from Amazon Titan are normalised by default [4].
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Dataclasses — imported by retrieval/raptor_search.py
    # and analysis_service.py
    "RAPTORNode",
    "RAPTORTree",
    # Main builder class
    "RAPTORTreeBuilder",
    # Pure utilities — usable independently in tests and retrieval layer
    "_split_text_to_budget",
    "_compute_cosine_similarity",
]
