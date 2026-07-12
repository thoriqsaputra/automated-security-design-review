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
from sdr.apps.ai.client.session import capture_current_context
from sdr.apps.ai.tsd_processing.document_models import TextBlock, TSDDocument
from sdr.apps.ai.utils.concurrency import ConcurrencyProbe
from sdr.apps.ai.prompts.indexing import (
    RAPTOR_SUMMARISATION_SYSTEM_PROMPT,
    build_raptor_summarisation_prompt,
    build_raptor_leaf_context_prompt,
)

logger = logging.getLogger(__name__)

_LEVEL_TOKEN_BUDGETS = {
    0: 400,
    1: 800,
    2: 1200,
    3: 2000,
}
_MIN_LEAF_NODES = 3
_TARGET_CLUSTER_SIZE = 5
_MIN_CLUSTER_SIZE = 2
_MAX_TREE_DEPTH = 3
_LEVEL_PROGRESS_BANDS: Dict[int, Tuple[int, int, int]] = {
    1: (30, 30, 45),
    2: (55, 55, 70),
    3: (80, 80, 90),
}
_EMBEDDING_DIMENSIONS = 1024
_SUMMARY_TEMPERATURE = 0.1
_SUMMARY_MAX_TOKENS = 1024
_MAX_SUMMARY_WORKERS = 3
_MAX_EMBED_WORKERS = 4
_DEFAULT_EMBED_BATCH_SIZE = 32
_LEVEL_INSTRUCTIONS = {
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


def _bounded_worker_count(configured: int, default: int) -> int:
    try:
        return max(1, int(configured))
    except (TypeError, ValueError):
        return default


@dataclass
class RAPTORNode:
    node_id: str
    level: int
    text: str
    embedding: List[float] = field(default_factory=list)
    source_block_ids: List[str] = field(default_factory=list)
    children: List[RAPTORNode] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    section_heading: Optional[str] = None
    has_embedding: bool = False

    @property
    def is_leaf(self) -> bool:
        return self.level == 0

    @property
    def token_estimate(self) -> int:
        return len(self.text) // 4

    def to_retrieval_dict(self) -> Dict[str, Any]:
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


@dataclass
class RAPTORTree:
    document_name: str
    levels: List[List[RAPTORNode]] = field(default_factory=list)
    root_node: Optional[RAPTORNode] = None
    total_nodes: int = 0
    max_level: int = 0
    build_stats: Dict[str, Any] = field(default_factory=dict)

    def get_nodes_at_level(self, level: int) -> List[RAPTORNode]:
        if level < 0 or level >= len(self.levels):
            return []
        return self.levels[level]

    def get_all_nodes(self) -> List[RAPTORNode]:
        nodes = []
        for level_nodes in self.levels:
            nodes.extend(level_nodes)
        return nodes

    def get_leaf_nodes(self) -> List[RAPTORNode]:
        return self.get_nodes_at_level(0)

    def get_nodes_with_embeddings(self) -> List[RAPTORNode]:
        return [n for n in self.get_all_nodes() if n.has_embedding]

    def is_empty(self) -> bool:
        return self.total_nodes == 0


class RAPTORTreeBuilder:
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
        self.leaf_token_budget = int(
            getattr(settings, "AI_RAPTOR_LEAF_TOKEN_BUDGET", _LEVEL_TOKEN_BUDGETS[0])
        )
        self.leaf_max_pages = int(getattr(settings, "AI_RAPTOR_LEAF_MAX_PAGES", 1))
        self.contextual_enrichment_enabled = bool(
            getattr(settings, "AI_RAPTOR_CONTEXTUAL_ENRICHMENT_ENABLED", True)
        )
        self.context_concurrency = int(
            getattr(settings, "AI_RAPTOR_CONTEXT_CONCURRENCY", 8)
        )

    def build(
        self,
        tsd_document: TSDDocument,
        progress_callback=None,
    ) -> RAPTORTree:
        tree = RAPTORTree(document_name=tsd_document.document_name)
        valid_blocks = self._get_valid_blocks(tsd_document)

        if len(valid_blocks) < _MIN_LEAF_NODES:
            self.logger.warning(
                "RAPTORTreeBuilder.build: document '%s' has only %d valid "
                "text block(s) — below minimum %d for RAPTOR tree. "
                "Returning empty tree.",
                tsd_document.document_name,
                len(valid_blocks),
                _MIN_LEAF_NODES,
            )
            self._emit_progress(
                progress_callback,
                status="skipped",
                progress_percent=100,
                current_step="RAPTOR skipped for short document",
                leaf_nodes=len(valid_blocks),
                summary_levels_completed=0,
                total_nodes=0,
                embedded_nodes=0,
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

        leaf_nodes = self._build_leaf_nodes(valid_blocks, tsd_document)
        tree.levels.append(leaf_nodes)
        self._emit_progress(
            progress_callback,
            status="running",
            progress_percent=5,
            current_step="Creating RAPTOR leaf nodes",
            leaf_nodes=len(leaf_nodes),
            summary_levels_completed=0,
        )

        self.logger.info(
            "RAPTORTreeBuilder.build: Level 0 — %d leaf node(s) created.",
            len(leaf_nodes),
        )

        if self.contextual_enrichment_enabled and leaf_nodes:
            self._enrich_leaf_nodes_with_context(leaf_nodes, tsd_document)

        self._embed_all_nodes(
            leaf_nodes,
            progress_callback=progress_callback,
            progress_range=(5, 20),
            phase_label="Embedding leaf nodes",
        )

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
            summarize_end, embed_start, embed_end = _LEVEL_PROGRESS_BANDS.get(
                next_level, _LEVEL_PROGRESS_BANDS[max(_LEVEL_PROGRESS_BANDS)]
            )
            self._emit_progress(
                progress_callback,
                status="running",
                progress_percent=summarize_end,
                current_step=(
                    f"Summarizing RAPTOR level {next_level} of {self.max_depth} "
                    f"— {len(clusters)} cluster(s)"
                ),
                leaf_nodes=len(leaf_nodes),
                summary_levels_completed=next_level,
                total_nodes=sum(len(level_nodes) for level_nodes in tree.levels),
            )

            self._embed_all_nodes(
                summary_nodes,
                progress_callback=progress_callback,
                progress_range=(embed_start, embed_end),
                phase_label=f"Embedding RAPTOR level {next_level} nodes",
            )

            current_level_nodes = summary_nodes
            current_level = next_level

        top_level_nodes = tree.levels[-1] if tree.levels else []

        if len(top_level_nodes) == 1:
            tree.root_node = top_level_nodes[0]
        elif len(top_level_nodes) > 1:
            root_node = self._synthesise_root(
                top_level_nodes=top_level_nodes,
                level=len(tree.levels),
                document_name=tsd_document.document_name,
            )
            if root_node:
                tree.levels.append([root_node])
                tree.root_node = root_node

        all_nodes = tree.get_all_nodes()
        nodes_to_embed = [node for node in all_nodes if not node.has_embedding]
        self._emit_progress(
            progress_callback,
            status="running",
            progress_percent=90,
            current_step="Embedding remaining RAPTOR nodes",
            leaf_nodes=len(leaf_nodes),
            summary_levels_completed=max(0, len(tree.levels) - 1),
            total_nodes=len(all_nodes),
            embedded_nodes=len(all_nodes) - len(nodes_to_embed),
        )
        self._embed_all_nodes(
            nodes_to_embed,
            progress_callback=progress_callback,
            progress_range=(90, 99),
            phase_label="Embedding remaining RAPTOR nodes",
        )

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
        self._emit_progress(
            progress_callback,
            status="completed",
            progress_percent=100,
            current_step="RAPTOR index ready",
            leaf_nodes=len(leaf_nodes),
            summary_levels_completed=max(0, len(tree.levels) - 1),
            total_nodes=tree.total_nodes,
            embedded_nodes=len(tree.get_nodes_with_embeddings()),
        )

        return tree

    def _get_valid_blocks(self, tsd_document: TSDDocument) -> List[TextBlock]:
        return [
            block
            for page in tsd_document.pages
            for block in page.text_blocks
            if block.is_valid()
        ]

    def _emit_progress(self, progress_callback=None, **payload: Any) -> None:
        if progress_callback:
            progress_callback(payload)

    def _build_leaf_nodes(
        self,
        text_blocks: List[TextBlock],
        tsd_document: TSDDocument,
    ) -> List[RAPTORNode]:
        section_groups: List[Dict[str, Any]] = []
        allowed_block_ids = {block.block_id for block in text_blocks if block.is_valid()}

        for page in tsd_document.pages:
            page_text = page.all_text.strip()
            if not page_text:
                continue

            valid_blocks = [
                block
                for block in page.text_blocks
                if block.is_valid() and block.block_id in allowed_block_ids
            ]
            distinct_headings = {block.section_heading for block in valid_blocks}

            if len(distinct_headings) <= 1:
                heading_key = page.section_heading or f"__page_{page.page_number}__"
                page_entries = [
                    {
                        "heading_key": heading_key,
                        "text": page_text,
                        "block_ids": [block.block_id for block in valid_blocks],
                        "page_number": page.page_number,
                    }
                ]
            else:
                page_entries = self._split_page_entries_by_heading(
                    page.page_number,
                    valid_blocks,
                )

            for entry in page_entries:
                heading_key = entry["heading_key"]
                page_entry = {
                    "text": entry["text"],
                    "block_ids": entry["block_ids"],
                    "page_number": entry["page_number"],
                }
                if (
                    section_groups
                    and section_groups[-1]["heading_key"] == heading_key
                    and not heading_key.startswith("__page_")
                ):
                    section_groups[-1]["pages"].append(page_entry)
                else:
                    section_groups.append(
                        {
                            "heading_key": heading_key,
                            "pages": [page_entry],
                        }
                    )

        leaf_nodes: List[RAPTORNode] = []
        node_idx = 0
        token_budget = self.leaf_token_budget
        char_budget = token_budget * 4
        max_pages = max(1, self.leaf_max_pages)

        for group in section_groups:
            heading_key = group["heading_key"]
            section_heading = None if heading_key.startswith("__page_") else heading_key
            current_texts: List[str] = []
            current_block_ids: List[str] = []
            current_page_numbers: List[int] = []
            current_char_count = 0

            def flush() -> None:
                nonlocal node_idx
                if not current_texts:
                    return
                leaf_nodes.append(
                    RAPTORNode(
                        node_id=f"level0_node{node_idx}",
                        level=0,
                        text="\n\n".join(current_texts),
                        source_block_ids=list(dict.fromkeys(current_block_ids)),
                        page_numbers=sorted(set(current_page_numbers)),
                        section_heading=section_heading,
                    )
                )
                node_idx += 1

            for page_entry in group["pages"]:
                page_text = page_entry["text"]
                page_len = len(page_text)

                if page_len > char_budget:
                    flush()
                    current_texts, current_block_ids, current_page_numbers = [], [], []
                    current_char_count = 0
                    for chunk_text in _split_text_to_budget(page_text, token_budget):
                        leaf_nodes.append(
                            RAPTORNode(
                                node_id=f"level0_node{node_idx}",
                                level=0,
                                text=chunk_text,
                                source_block_ids=list(dict.fromkeys(page_entry["block_ids"])),
                                page_numbers=[page_entry["page_number"]],
                                section_heading=section_heading,
                            )
                        )
                        node_idx += 1
                    continue

                if current_texts and (
                    current_char_count + page_len > char_budget
                    or len(current_page_numbers) >= max_pages
                ):
                    flush()
                    current_texts, current_block_ids, current_page_numbers = [], [], []
                    current_char_count = 0

                current_texts.append(page_text)
                current_block_ids.extend(page_entry["block_ids"])
                current_page_numbers.append(page_entry["page_number"])
                current_char_count += page_len

            flush()

        self.logger.debug(
            "RAPTORTreeBuilder._build_leaf_nodes: created %d leaf node(s) "
            "from %d markdown section group(s).",
            len(leaf_nodes),
            len(section_groups),
        )
        return leaf_nodes

    def _split_page_entries_by_heading(
        self,
        page_number: int,
        valid_blocks: List[TextBlock],
    ) -> List[Dict[str, Any]]:
        page_entries: List[Dict[str, Any]] = []
        current_run: List[TextBlock] = []
        current_heading: Optional[str] = None

        for block in valid_blocks:
            if current_run and block.section_heading != current_heading:
                page_entries.append(
                    self._build_page_entry(page_number, current_heading, current_run, len(page_entries))
                )
                current_run = []
            current_heading = block.section_heading
            current_run.append(block)

        if current_run:
            page_entries.append(
                self._build_page_entry(page_number, current_heading, current_run, len(page_entries))
            )

        return page_entries

    def _build_page_entry(
        self,
        page_number: int,
        heading: Optional[str],
        blocks: List[TextBlock],
        entry_index: int,
    ) -> Dict[str, Any]:
        return {
            "heading_key": heading or f"__page_{page_number}_run{entry_index}__",
            "text": "\n".join(block.text for block in blocks),
            "block_ids": [block.block_id for block in blocks],
            "page_number": page_number,
        }

    def _enrich_leaf_nodes_with_context(
        self,
        leaf_nodes: List[RAPTORNode],
        tsd_document: TSDDocument,
    ) -> None:
        doc_title = getattr(tsd_document, "title", None) or tsd_document.document_name or "Technical Security Design Document"
        nodes_to_enrich = [n for n in leaf_nodes if n.text and n.text.strip()]
        if not nodes_to_enrich:
            return

        self.logger.info(
            "RAPTORTreeBuilder: enriching %d leaf node(s) with contextual prefix "
            "(concurrency=%d).",
            len(nodes_to_enrich),
            self.context_concurrency,
        )

        with ThreadPoolExecutor(
            max_workers=min(self.context_concurrency, len(nodes_to_enrich))
        ) as executor:
            futures = {
                executor.submit(capture_current_context(self._generate_leaf_context), node, doc_title): node
                for node in nodes_to_enrich
            }
            for future in as_completed(futures):
                node = futures[future]
                try:
                    prefix = future.result()
                    if prefix and prefix.strip():
                        node.text = f"{prefix.strip()}\n\n{node.text}"
                except Exception:
                    self.logger.warning(
                        "Context prefix generation failed for node %s; keeping raw text.",
                        node.node_id,
                    )

    def _generate_leaf_context(self, node: RAPTORNode, doc_title: str) -> str:
        prompt = build_raptor_leaf_context_prompt(
            doc_title=doc_title,
            section_heading=node.section_heading or "General",
            chunk_text=node.text[:1500],
        )
        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            component="tsd_ingestion",
            temperature=0.0,
            max_tokens=80,
        )
        return (response.content or "").strip() if not response.error else ""

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster_nodes(
        self,
        nodes: List[RAPTORNode],
    ) -> List[List[RAPTORNode]]:
        if not nodes:
            return []

        if len(nodes) == 1:
            return [nodes]

        if all(node.has_embedding for node in nodes):
            return self._cluster_nodes_by_similarity(nodes)

        self.logger.debug(
            "RAPTORTreeBuilder._cluster_nodes: not all %d node(s) have "
            "embeddings — falling back to sequential clustering.",
            len(nodes),
        )
        return self._cluster_nodes_sequential(nodes)

    def _cluster_nodes_sequential(
        self,
        nodes: List[RAPTORNode],
    ) -> List[List[RAPTORNode]]:
        clusters: List[List[RAPTORNode]] = []
        current_cluster: List[RAPTORNode] = []

        for node in nodes:
            current_cluster.append(node)
            if len(current_cluster) >= self.target_cluster_size:
                clusters.append(current_cluster)
                current_cluster = []

        if current_cluster:
            if (
                len(current_cluster) < self.min_cluster_size
                and clusters
            ):
                self.logger.debug(
                    "RAPTORTreeBuilder._cluster_nodes_sequential: merging "
                    "trailing cluster of %d node(s) into previous cluster.",
                    len(current_cluster),
                )
                clusters[-1].extend(current_cluster)
            else:
                clusters.append(current_cluster)

        self.logger.debug(
            "RAPTORTreeBuilder._cluster_nodes_sequential: produced %d "
            "cluster(s) from %d node(s).",
            len(clusters),
            len(nodes),
        )
        return clusters

    def _cluster_nodes_by_similarity(
        self,
        nodes: List[RAPTORNode],
    ) -> List[List[RAPTORNode]]:
        unclustered = list(nodes)
        clusters: List[List[RAPTORNode]] = []

        while unclustered:
            seed = unclustered.pop(0)
            cluster = [seed]

            if unclustered:
                scored = sorted(
                    unclustered,
                    key=lambda candidate: _compute_cosine_similarity(
                        seed.embedding, candidate.embedding
                    ),
                    reverse=True,
                )
                take = scored[: max(0, self.target_cluster_size - 1)]
                for node in take:
                    unclustered.remove(node)
                    cluster.append(node)

            clusters.append(cluster)

        if (
            len(clusters) > 1
            and len(clusters[-1]) < self.min_cluster_size
        ):
            self.logger.debug(
                "RAPTORTreeBuilder._cluster_nodes_by_similarity: merging "
                "trailing cluster of %d node(s) into previous cluster.",
                len(clusters[-1]),
            )
            trailing = clusters.pop()
            clusters[-1].extend(trailing)

        self.logger.debug(
            "RAPTORTreeBuilder._cluster_nodes_by_similarity: produced %d "
            "cluster(s) from %d node(s).",
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
                    capture_current_context(probe.wrap(self._summarise_cluster)),
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
                completed_clusters = len(summary_nodes_by_cluster)
                cluster_progress = int(round((completed_clusters / max(len(clusters), 1)) * 100))
                self._emit_progress(
                    progress_callback,
                    status="running",
                    progress_percent=min(65, 20 + level * 15),
                    current_step=f"Summarizing RAPTOR level {level} ({cluster_progress}% of clusters)",
                    summary_levels_completed=max(0, level - 1),
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
        import threading
        thread_name = threading.current_thread().name
        self.logger.info(
            "RAPTORTreeBuilder._summarise_cluster [%s]: processing cluster %d at level %d",
            thread_name,
            cluster_idx,
            level,
        )
        node_id = f"level{level}_node{cluster_idx}"
        source_block_ids, page_numbers = self._collect_cluster_metadata(cluster)
        section_heading = cluster[0].section_heading
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
            source_block_ids=source_block_ids,
            children=list(cluster),
            page_numbers=page_numbers,
            section_heading=section_heading,
        )

    def _collect_cluster_metadata(
        self,
        nodes: List[RAPTORNode],
    ) -> Tuple[List[str], List[int]]:
        block_ids: List[str] = []
        page_numbers: List[int] = []

        for node in nodes:
            block_ids.extend(node.source_block_ids)
            page_numbers.extend(node.page_numbers)

        return list(dict.fromkeys(block_ids)), sorted(set(page_numbers))

    def _call_summarisation_llm(
        self,
        cluster: List[RAPTORNode],
        level: int,
        token_budget: int,
    ) -> Optional[str]:
        combined_text = "\n\n---\n\n".join(
            f"[Node {n.node_id}]\n{n.text}" for n in cluster
        )

        instruction = _LEVEL_INSTRUCTIONS.get(level, _LEVEL_INSTRUCTIONS[1])

        prompt = build_raptor_summarisation_prompt(
            instruction=instruction,
            token_budget=token_budget,
            combined_text=combined_text,
        )

        try:
            response = chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": RAPTOR_SUMMARISATION_SYSTEM_PROMPT,
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

    def _synthesise_root(
        self,
        top_level_nodes: List[RAPTORNode],
        level: int,
        document_name: str,
    ) -> Optional[RAPTORNode]:
        self.logger.info(
            "RAPTORTreeBuilder._synthesise_root: synthesising root from "
            "%d top-level node(s) for '%s'.",
            len(top_level_nodes),
            document_name,
        )
        source_block_ids, page_numbers = self._collect_cluster_metadata(top_level_nodes)
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
            source_block_ids=source_block_ids,
            children=list(top_level_nodes),
            page_numbers=page_numbers,
            section_heading=None,
        )

    def _embed_all_nodes(
        self,
        nodes: List[RAPTORNode],
        progress_callback=None,
        *,
        progress_range: Tuple[int, int] = (70, 99),
        phase_label: str = "Embedding RAPTOR nodes",
    ) -> None:
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

            text_hash = hashlib.sha256(node.text.encode("utf-8")).hexdigest()
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
                        capture_current_context(probe.wrap(get_embeddings)),
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
                    processed = success_count + failure_count
                    low, high = progress_range
                    embed_progress = low + int(round((processed / max(len(nodes), 1)) * (high - low)))
                    self._emit_progress(
                        progress_callback,
                        status="running",
                        progress_percent=min(high, embed_progress),
                        current_step=f"{phase_label} ({processed} of {len(nodes)})",
                        total_nodes=len(nodes),
                        embedded_nodes=success_count,
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

def _split_text_to_budget(
    text: str,
    token_budget: int,
) -> List[str]:
    if not text or not text.strip():
        return []

    char_budget = token_budget * 4
    if len(text) <= char_budget:
        return [text.strip()]

    words = text.split()
    chunks: List[str] = []
    current_words: List[str] = []
    current_char_count = 0

    for word in words:
        word_len = len(word) + 1

        if current_char_count + word_len > char_budget and current_words:
            chunks.append(" ".join(current_words).strip())
            current_words = [word]
            current_char_count = word_len
        else:
            current_words.append(word)
            current_char_count += word_len

    if current_words:
        chunks.append(" ".join(current_words).strip())

    return [c for c in chunks if c]


def _compute_cosine_similarity(
    vec_a: List[float],
    vec_b: List[float],
) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)

__all__ = [
    "RAPTORNode",
    "RAPTORTree",
    "RAPTORTreeBuilder",
    "_split_text_to_budget",
    "_compute_cosine_similarity",
]
