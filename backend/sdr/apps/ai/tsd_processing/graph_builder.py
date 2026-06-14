from __future__ import annotations

import json
import re
import logging
import hashlib
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from sdr.core.config import settings

from sdr.apps.ai.client import chat_completion, get_embeddings
from sdr.apps.ai.tsd_processing.document_models import TSDDocument, TSDPage, TextBlock
from sdr.apps.ai.tsd_processing.graph_embedding_formatter import (
    build_entity_embedding_text,
    build_relation_embedding_text,
)
from sdr.apps.ai.tsd_processing.prepared_view import PreparedTSDView, prepare_tsd_view
from sdr.apps.ai.tsd_processing.raptor_graph_linker import RaptorGraphLinker
from sdr.apps.ai.utils.concurrency import ConcurrencyProbe
from sdr.apps.ai.prompts.indexing import (
    GRAPH_EXTRACTION_SYSTEM_PROMPT,
    build_graph_extraction_prompt,
    build_graph_retry_extraction_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# networkx availability guard
# ---------------------------------------------------------------------------

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.error(
        "networkx is not installed. Graph builder will not function. "
        "Install with: pip install networkx"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LLM settings for entity/relation extraction
_EXTRACTION_TEMPERATURE = 0.0   # fully deterministic — entity extraction must be consistent
_EXTRACTION_MAX_TOKENS = 2048
_MAX_EXTRACTION_RETRIES = 1

# Minimum entity name length — filters noise like "A", "it", "the"
_MIN_ENTITY_NAME_LENGTH = 2

# Maximum text length sent per page extraction call — pages longer than
# this are truncated to avoid exceeding the model's context window [4]
_MAX_PAGE_TEXT_CHARS = 6000

# Confidence threshold below which extracted relations are discarded
_MIN_RELATION_CONFIDENCE = 0.5
_DEFAULT_GRAPH_EXTRACTION_MODE = "llm"
_EMBEDDING_DIMENSIONS = 1024
_MAX_GRAPH_EMBED_WORKERS = 4
_MAX_ENTITY_EMBED_TEXT_CHARS = 512
_MAX_RELATION_EMBED_TEXT_CHARS = 768
_MAX_GRAPH_EXTRACTION_WORKERS = 4
_DEFAULT_GRAPH_EMBED_BATCH_SIZE = 32


def _bounded_worker_count(configured: int, default: int) -> int:
    try:
        return max(1, int(configured))
    except (TypeError, ValueError):
        return default

# Valid entity types — anything outside this set is discarded
_VALID_ENTITY_TYPES = frozenset({
    "service",
    "api",
    "database",
    "auth_mechanism",
    "user",
    "protocol",
    "component",
    "data_store",
    "external_system",
    "queue",
    "cache",
})

# Valid relation types — aligned with security review semantics
_VALID_RELATION_TYPES = frozenset({
    "authenticates_with",
    "authorises_via",
    "sends_data_to",
    "receives_data_from",
    "reads_from",
    "writes_to",
    "stores_in",
    "exposes_endpoint",
    "depends_on",
    "calls",
    "uses_protocol",
    "encrypted_by",
    "accessed_by",
    "deployed_on",
    "communicates_with",
})

_VALID_GRAPH_EXTRACTION_MODES = frozenset({"spacy", "llm", "hybrid"})

_SIMPLE_ENTITY_TYPE_HINTS = {
    "api": "api",
    "service": "service",
    "database": "database",
    "db": "database",
    "cache": "cache",
    "queue": "queue",
    "auth": "auth_mechanism",
    "oauth": "auth_mechanism",
    "jwt": "protocol",
    "tls": "protocol",
    "https": "protocol",
}

try:
    import spacy  # type: ignore
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Graph dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GraphEntity:
    entity_id: str                      # normalised slug: "api_gateway"
    name: str                           # original name: "API Gateway"
    entity_type: str                    # one of _VALID_ENTITY_TYPES
    description: Optional[str] = None  # short description from extraction
    # Source page numbers where this entity was mentioned
    source_pages: List[int] = field(default_factory=list)
    # Source block_ids where this entity was mentioned — for citation tracing
    source_block_ids: List[str] = field(default_factory=list)
    grounded_texts: List[Dict[str, Any]] = field(default_factory=list)
    sensitivity: str = "internal"
    tenant_id: Optional[str] = None
    embedding: List[float] = field(default_factory=list)
    has_embedding: bool = False

    def is_valid(self) -> bool:
        return (
            bool(self.entity_id)
            and len(self.name) >= _MIN_ENTITY_NAME_LENGTH
            and self.entity_type in _VALID_ENTITY_TYPES
        )

    @property
    def block_ids(self) -> List[str]:
        return self.source_block_ids


@dataclass
class GraphRelation:
    source_entity_id: str              # entity_id of the source node
    target_entity_id: str              # entity_id of the target node
    relation_type: str                 # one of _VALID_RELATION_TYPES
    description: Optional[str] = None # human-readable edge description
    is_encrypted: Optional[bool] = None  # True/False/None (unknown)
    requires_auth: Optional[bool] = None # True/False/None (unknown)
    protocol: Optional[str] = None    # e.g. "TLS 1.2", "mTLS", "HTTPS"
    confidence: float = 1.0           # LLM confidence in this relation
    source_pages: List[int] = field(default_factory=list)
    source_block_ids: List[str] = field(default_factory=list)
    grounded_texts: List[Dict[str, Any]] = field(default_factory=list)
    sensitivity: str = "internal"
    tenant_id: Optional[str] = None
    embedding: List[float] = field(default_factory=list)
    has_embedding: bool = False

    def is_valid(self) -> bool:
        return (
            bool(self.source_entity_id)
            and bool(self.target_entity_id)
            and self.source_entity_id != self.target_entity_id
            and self.relation_type in _VALID_RELATION_TYPES
            and self.confidence >= _MIN_RELATION_CONFIDENCE
        )

    @property
    def relation(self) -> str:
        return self.relation_type

    @property
    def weight(self) -> float:
        return self.confidence

    @property
    def block_ids(self) -> List[str]:
        return self.source_block_ids


@dataclass
class TSDGraph:
    document_name: str
    graph: "nx.DiGraph" = field(default_factory=lambda: nx.DiGraph() if NETWORKX_AVAILABLE else None)
    entities: Dict[str, GraphEntity] = field(default_factory=dict)
    total_entities: int = 0
    total_relations: int = 0
    entity_to_block_ids: Dict[str, Set[str]] = field(default_factory=dict)
    block_id_to_entities: Dict[str, Set[str]] = field(default_factory=dict)
    edge_to_block_ids: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    raptor_node_to_entities: Dict[str, Set[str]] = field(default_factory=dict)
    entity_to_raptor_node_ids: Dict[str, Set[str]] = field(default_factory=dict)
    object_embedding_cache: Dict[str, List[float]] = field(default_factory=dict)
    community_embedding_cache: Dict[str, List[float]] = field(default_factory=dict)
    embedding_stats: Dict[str, int] = field(default_factory=dict)
    build_stats: Dict[str, float] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return self.total_entities == 0

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        return self.entities.get(entity_id)

    def get_neighbours(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
    ) -> List[Tuple[GraphEntity, GraphRelation]]:
        if not NETWORKX_AVAILABLE or self.graph is None:
            return []
        if entity_id not in self.graph:
            return []

        results = []
        for _, target_id, edge_data in self.graph.out_edges(entity_id, data=True):
            if relation_type and edge_data.get("relation_type") != relation_type:
                continue
            target_entity = self.entities.get(target_id)
            if target_entity:
                relation = edge_data.get("relation_obj")
                if relation:
                    results.append((target_entity, relation))

        return results

    def get_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 4,
    ) -> List[List[str]]:
        if not NETWORKX_AVAILABLE or self.graph is None:
            return []
        if source_id not in self.graph or target_id not in self.graph:
            return []

        try:
            paths = list(
                nx.all_simple_paths(
                    self.graph,
                    source=source_id,
                    target=target_id,
                    cutoff=max_depth,
                )
            )
            return paths
        except (nx.NetworkXError, nx.NodeNotFound):
            return []

    def find_entities_by_type(
        self,
        entity_type: str,
    ) -> List[GraphEntity]:
        return [
            e for e in self.entities.values()
            if e.entity_type == entity_type
        ]

    def find_entities_by_name_fragment(
        self,
        fragment: str,
    ) -> List[GraphEntity]:
        fragment_lower = fragment.lower()
        return [
            e for e in self.entities.values()
            if fragment_lower in e.name.lower()
        ]


# ---------------------------------------------------------------------------
# TSD Graph Builder
# ---------------------------------------------------------------------------

class TSDGraphBuilder:
    def __init__(
        self,
        min_entity_name_length: int = _MIN_ENTITY_NAME_LENGTH,
        min_relation_confidence: float = _MIN_RELATION_CONFIDENCE,
        graph_extraction_mode: str = _DEFAULT_GRAPH_EXTRACTION_MODE,
    ) -> None:
        if not NETWORKX_AVAILABLE:
            raise RuntimeError(
                "TSDGraphBuilder requires networkx. "
                "Install with: pip install networkx"
            )

        self.min_entity_name_length = min_entity_name_length
        self.min_relation_confidence = min_relation_confidence
        self.graph_extraction_mode = graph_extraction_mode.lower().strip()
        if self.graph_extraction_mode not in _VALID_GRAPH_EXTRACTION_MODES:
            self.graph_extraction_mode = _DEFAULT_GRAPH_EXTRACTION_MODE
        self._nlp = None
        if SPACY_AVAILABLE and self.graph_extraction_mode in {"spacy", "hybrid"}:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = None
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.extraction_max_concurrency = _bounded_worker_count(
            getattr(
                settings,
                "AI_GRAPH_EXTRACTION_MAX_CONCURRENCY",
                _MAX_GRAPH_EXTRACTION_WORKERS,
            ),
            _MAX_GRAPH_EXTRACTION_WORKERS,
        )
        self._last_extraction_concurrency_stats: Dict[str, Any] = {}
        self.embed_max_concurrency = _bounded_worker_count(
            getattr(
                settings,
                "AI_GRAPH_EMBED_MAX_CONCURRENCY",
                _MAX_GRAPH_EMBED_WORKERS,
            ),
            _MAX_GRAPH_EMBED_WORKERS,
        )
        self.embed_batch_size = _bounded_worker_count(
            getattr(
                settings,
                "AI_GRAPH_EMBED_BATCH_SIZE",
                _DEFAULT_GRAPH_EMBED_BATCH_SIZE,
            ),
            _DEFAULT_GRAPH_EMBED_BATCH_SIZE,
        )
        self.linker = RaptorGraphLinker()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(
        self,
        tsd_document: TSDDocument,
        progress_callback=None,
        prepared_view: Optional[PreparedTSDView] = None,
    ) -> TSDGraph:
        tsd_graph = TSDGraph(document_name=tsd_document.document_name)
        build_started = time.monotonic()
        prepared = prepared_view or prepare_tsd_view(tsd_document)
        filtered_view = prepared.filtered_view
        tsd_graph.build_stats.update(
            {
                f"content_filter_{key}": value
                for key, value in prepared.stats.items()
            }
        )

        if not tsd_document.pages:
            self.logger.warning(
                "TSDGraphBuilder.build: document '%s' has no pages — "
                "returning empty graph.",
                tsd_document.document_name,
            )
            if progress_callback:
                progress_callback(
                    {
                        "status": "skipped",
                        "progress_percent": 100,
                        "current_step": "GraphRAG skipped because the document has no pages",
                        "pages_total": 0,
                        "pages_completed": 0,
                        "entities_total": 0,
                        "entities_embedded": 0,
                        "relations_total": 0,
                    }
                )
            return tsd_graph

        self.logger.info(
            "TSDGraphBuilder.build: extracting entities from '%s' "
            "across %d page(s) (mode=%s, extraction_max_concurrency=%d).",
            tsd_document.document_name,
            len(filtered_view.pages),
            self.graph_extraction_mode,
            self.extraction_max_concurrency,
        )
        if progress_callback:
            progress_callback(
                {
                    "status": "running",
                    "progress_percent": 5,
                    "current_step": "Scanning TSD pages for GraphRAG extraction",
                    "pages_total": len(filtered_view.pages),
                    "pages_completed": 0,
                }
            )

        # Accumulate entities and relations across all pages
        all_entities: Dict[str, GraphEntity] = {}
        all_relations: List[GraphRelation] = []

        pages_with_text: List[TSDPage] = list(prepared.pages_with_text)

        extracted_pages: List[Tuple[int, List[GraphEntity], List[GraphRelation]]] = []
        if pages_with_text:
            probe = ConcurrencyProbe(
                max_concurrency=self.extraction_max_concurrency,
            )
            self.logger.info(
                "TSDGraphBuilder.build: extraction phase pages=%d max_concurrency=%d",
                len(pages_with_text),
                self.extraction_max_concurrency,
            )
            with ThreadPoolExecutor(
                max_workers=min(
                    self.extraction_max_concurrency,
                    len(pages_with_text),
                ),
                thread_name_prefix="ThreadPoolExecutor-0",
            ) as executor:
                futures = {
                    executor.submit(probe.wrap(self._extract_from_page), page): page
                    for page in pages_with_text
                }
                probe.mark_submitted(len(futures))
                for future in as_completed(futures):
                    page = futures[future]
                    try:
                        page_entities, page_relations = future.result()
                    except Exception as exc:
                        self.logger.error(
                            "TSDGraphBuilder.build: unexpected extraction "
                            "error on page %d: %s",
                            page.page_number,
                            exc,
                        )
                        page_entities, page_relations = [], []
                    extracted_pages.append(
                        (page.page_number, page_entities, page_relations)
                    )
                    if progress_callback:
                        progress_callback(
                            {
                                "status": "running",
                                "progress_percent": min(
                                    60,
                                    10 + int(round((len(extracted_pages) / max(len(pages_with_text), 1)) * 50)),
                                ),
                                "current_step": "Extracting GraphRAG entities and relations",
                                "pages_total": len(pages_with_text),
                                "pages_completed": len(extracted_pages),
                            }
                        )
            extraction_stats = probe.snapshot().to_dict()
            self._last_extraction_concurrency_stats = extraction_stats
            self.logger.info(
                "TSDGraphBuilder.build: extraction phase submitted=%d "
                "completed=%d failed=%d peak_in_flight=%d max_concurrency=%d "
                "elapsed_seconds=%.4f",
                extraction_stats["submitted"],
                extraction_stats["completed"],
                extraction_stats["failed"],
                extraction_stats["peak_in_flight"],
                extraction_stats["max_concurrency"],
                extraction_stats["elapsed_seconds"],
            )

        for _, page_entities, page_relations in sorted(
            extracted_pages,
            key=lambda item: item[0],
        ):
            for entity in page_entities:
                if entity.entity_id in all_entities:
                    self._merge_entity_metadata(
                        existing=all_entities[entity.entity_id],
                        incoming=entity,
                    )
                else:
                    all_entities[entity.entity_id] = entity

            all_relations.extend(page_relations)

        if not all_entities:
            self.logger.warning(
                "TSDGraphBuilder.build: no entities extracted from '%s' — "
                "returning empty graph.",
                tsd_document.document_name,
            )
            if progress_callback:
                progress_callback(
                    {
                        "status": "skipped",
                        "progress_percent": 100,
                        "current_step": "GraphRAG found no entities to index",
                        "pages_total": len(pages_with_text),
                        "pages_completed": len(extracted_pages),
                        "entities_total": 0,
                        "entities_embedded": 0,
                        "relations_total": 0,
                    }
                )
            return tsd_graph

        # Build the networkx DiGraph and indexes
        self._populate_graph(tsd_graph, all_entities, all_relations)
        self._build_indexes(tsd_graph)
        if progress_callback:
            progress_callback(
                {
                    "status": "running",
                    "progress_percent": 70,
                    "current_step": "Merging GraphRAG entities and building graph indexes",
                    "pages_total": len(pages_with_text),
                    "pages_completed": len(extracted_pages),
                    "entities_total": tsd_graph.total_entities,
                    "relations_total": tsd_graph.total_relations,
                }
            )
        tsd_graph.build_stats["extraction_merge_seconds"] = (
            time.monotonic() - build_started
        )
        entity_embed_started = time.monotonic()
        self._embed_graph_entities(tsd_graph, progress_callback=progress_callback)
        tsd_graph.build_stats["entity_embedding_seconds"] = (
            time.monotonic() - entity_embed_started
        )
        tsd_graph.build_stats["relation_embedding_seconds"] = 0.0
        tsd_graph.build_stats["total_seconds"] = time.monotonic() - build_started

        self.logger.info(
            "TSDGraphBuilder.build: graph complete for '%s' — "
            "%d entity(ies), %d relation(s).",
            tsd_document.document_name,
            tsd_graph.total_entities,
            tsd_graph.total_relations,
        )
        if progress_callback:
            progress_callback(
                {
                    "status": "completed",
                    "progress_percent": 100,
                    "current_step": "GraphRAG index ready",
                    "pages_total": len(pages_with_text),
                    "pages_completed": len(extracted_pages),
                    "entities_total": tsd_graph.total_entities,
                    "entities_embedded": int(tsd_graph.embedding_stats.get("entity_succeeded", 0) or 0),
                    "relations_total": tsd_graph.total_relations,
                }
            )

        return tsd_graph

    def _embed_graph_entities(self, tsd_graph: TSDGraph, progress_callback=None) -> None:
        if tsd_graph is None or tsd_graph.is_empty():
            return

        entity_items: List[Tuple[GraphEntity, str]] = []

        for entity in tsd_graph.entities.values():
            text = self._entity_embedding_text(entity)
            if text:
                entity_items.append((entity, text))

        ent_succeeded, ent_failed = self._embed_objects_with_cache(
            items=entity_items,
            cache=tsd_graph.object_embedding_cache,
            object_type="entity",
            progress_callback=progress_callback,
            progress_total=len(entity_items),
        )

        tsd_graph.embedding_stats = {
            "entity_attempted": len(entity_items),
            "entity_succeeded": ent_succeeded,
            "entity_failed": ent_failed,
            "relation_attempted": 0,
            "relation_succeeded": 0,
            "relation_failed": 0,
        }
        self.logger.info(
            "TSDGraphBuilder._embed_graph_entities: entities %d/%d embedded; relation embeddings deferred to search.",
            ent_succeeded,
            len(entity_items),
        )

    def _embed_objects_with_cache(
        self,
        items: List[Tuple[Any, str]],
        cache: Dict[str, List[float]],
        object_type: str,
        progress_callback=None,
        progress_total: int = 0,
    ) -> Tuple[int, int]:
        if not items:
            return 0, 0

        hash_to_text: Dict[str, str] = {}
        hash_to_objects: Dict[str, List[Any]] = defaultdict(list)
        for obj, text in items:
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            hash_to_text[text_hash] = text
            hash_to_objects[text_hash].append(obj)

        succeeded = 0
        failed = 0

        for text_hash, grouped in hash_to_objects.items():
            cached = cache.get(text_hash)
            if cached:
                for obj in grouped:
                    obj.embedding = list(cached)
                    obj.has_embedding = True
                    succeeded += 1

        pending_hashes = [h for h in hash_to_objects.keys() if h not in cache]
        if pending_hashes:
            hash_batches = [
                pending_hashes[idx:idx + self.embed_batch_size]
                for idx in range(0, len(pending_hashes), self.embed_batch_size)
            ]
            with ThreadPoolExecutor(
                max_workers=min(self.embed_max_concurrency, len(hash_batches)),
                thread_name_prefix="ThreadPoolExecutor-0",
            ) as executor:
                futures = {
                    executor.submit(
                        get_embeddings,
                        texts=[hash_to_text[text_hash] for text_hash in hash_batch],
                        dimensions=_EMBEDDING_DIMENSIONS,
                    ): hash_batch
                    for hash_batch in hash_batches
                }
                for future in as_completed(futures):
                    hash_batch = futures[future]
                    try:
                        vectors = future.result() or []
                    except Exception as exc:
                        self.logger.warning(
                            "TSDGraphBuilder._embed_objects_with_cache: %s embedding batch failed: %s",
                            object_type,
                            exc,
                        )
                        vectors = [[] for _ in hash_batch]

                    if len(vectors) != len(hash_batch):
                        self.logger.warning(
                            "TSDGraphBuilder._embed_objects_with_cache: %s vector count mismatch for batch: expected %d got %d.",
                            object_type,
                            len(hash_batch),
                            len(vectors),
                        )
                        vectors = [[] for _ in hash_batch]

                    for text_hash, vector in zip(hash_batch, vectors):
                        grouped = hash_to_objects[text_hash]
                        if vector:
                            cache[text_hash] = list(vector)
                            for obj in grouped:
                                obj.embedding = list(vector)
                                obj.has_embedding = True
                                succeeded += 1
                        else:
                            for obj in grouped:
                                obj.embedding = []
                                obj.has_embedding = False
                                failed += 1
                    if progress_callback and object_type == "entity":
                        processed = succeeded + failed
                        progress_callback(
                            {
                                "status": "running",
                                "progress_percent": min(
                                    99,
                                    70 + int(round((processed / max(progress_total or len(items), 1)) * 30)),
                                ),
                                "current_step": "Embedding GraphRAG entities",
                                "entities_total": progress_total or len(items),
                                "entities_embedded": succeeded,
                            }
                        )

        return succeeded, failed

    def _entity_embedding_text(self, entity: GraphEntity) -> str:
        return build_entity_embedding_text(entity)

    def _relation_embedding_text(self, relation: GraphRelation) -> str:
        return build_relation_embedding_text(relation)

    # ------------------------------------------------------------------
    # Per-page entity/relation extraction
    # ------------------------------------------------------------------

    def _extract_from_page(
        self,
        page: TSDPage,
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:

        import threading
        thread_name = threading.current_thread().name
        self.logger.info(
            "TSDGraphBuilder._extract_from_page [%s]: extracting from page %d",
            thread_name,
            page.page_number,
        )
        if self.graph_extraction_mode == "spacy":
            return self._extract_from_page_deterministic(page)
        if self.graph_extraction_mode == "llm":
            return self._extract_from_page_llm(page)

        # hybrid
        det_entities, det_relations = self._extract_from_page_deterministic(page)
        llm_entities, llm_relations = self._extract_from_page_llm(page)
        merged_entities: Dict[str, GraphEntity] = {}
        for entity in det_entities + llm_entities:
            if entity.entity_id in merged_entities:
                self._merge_entity_metadata(merged_entities[entity.entity_id], entity)
                merged_entities[entity.entity_id].grounded_texts.extend(entity.grounded_texts)
            else:
                merged_entities[entity.entity_id] = entity
        merged_relations = det_relations + llm_relations
        return list(merged_entities.values()), merged_relations

    def _extract_from_page_llm(self, page: TSDPage) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        page_text = page.all_text.strip()
        if not page_text:
            return [], []

        if len(page_text) > _MAX_PAGE_TEXT_CHARS:
            page_text = page_text[:_MAX_PAGE_TEXT_CHARS]
            self.logger.debug(
                "TSDGraphBuilder._extract_from_page: page %d truncated to %d chars.",
                page.page_number,
                _MAX_PAGE_TEXT_CHARS,
            )

        parse_failure: Optional[Dict[str, Any]] = None
        for attempt in range(_MAX_EXTRACTION_RETRIES + 1):
            prompt = (
                self._build_retry_extraction_prompt(
                    page_text=page_text,
                    page_number=page.page_number,
                    previous_failure=parse_failure,
                )
                if attempt > 0
                else self._build_extraction_prompt(page_text, page.page_number)
            )
            try:
                response = chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": GRAPH_EXTRACTION_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    component="coding_graph",
                    temperature=_EXTRACTION_TEMPERATURE,
                    max_tokens=_EXTRACTION_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                if response.error:
                    self.logger.error(
                        "TSDGraphBuilder._extract_from_page: LLM error on page %d: %s",
                        page.page_number,
                        response.error,
                    )
                    return [], []
                entities, relations, parse_failure = self._parse_extraction_response(
                    response.content or "{}",
                    page=page,
                )
                if parse_failure is None:
                    return entities, relations
                if attempt < _MAX_EXTRACTION_RETRIES:
                    self.logger.debug(
                        "TSDGraphBuilder._extract_from_page_llm: retrying malformed JSON on page %d after %s at pos=%s.",
                        page.page_number,
                        parse_failure.get("message", "decode_error"),
                        parse_failure.get("pos", "unknown"),
                    )
                    continue
                self.logger.warning(
                    "TSDGraphBuilder._extract_from_page_llm: malformed JSON on page %d stage=%s pos=%s message=%s snippet=%r length=%s",
                    page.page_number,
                    "retry_parse" if attempt > 0 else "initial_parse",
                    parse_failure.get("pos", "unknown"),
                    parse_failure.get("message", "decode_error"),
                    parse_failure.get("snippet", ""),
                    parse_failure.get("content_length", 0),
                )
                return [], []
            except Exception as exc:
                self.logger.error(
                    "TSDGraphBuilder._extract_from_page: unexpected error on page %d: %s",
                    page.page_number,
                    exc,
                )
                return [], []
        return [], []

    def _build_extraction_prompt(
        self,
        page_text: str,
        page_number: int,
    ) -> str:
        entity_types = ", ".join(sorted(_VALID_ENTITY_TYPES))
        relation_types = ", ".join(sorted(_VALID_RELATION_TYPES))
        return build_graph_extraction_prompt(
            page_text=page_text,
            page_number=page_number,
            entity_types=entity_types,
            relation_types=relation_types,
        )

    def _build_retry_extraction_prompt(
        self,
        page_text: str,
        page_number: int,
        previous_failure: Optional[Dict[str, Any]] = None,
    ) -> str:
        entity_types = ", ".join(sorted(_VALID_ENTITY_TYPES))
        relation_types = ", ".join(sorted(_VALID_RELATION_TYPES))
        failure_hint = ""
        if previous_failure:
            failure_hint = (
                f"\nPrevious output was malformed JSON at position "
                f"{previous_failure.get('pos', 'unknown')}: "
                f"{previous_failure.get('message', 'decode_error')}.\n"
            )
        return build_graph_retry_extraction_prompt(
            page_text=page_text,
            page_number=page_number,
            entity_types=entity_types,
            relation_types=relation_types,
            failure_hint=failure_hint,
        )

    def _extract_from_page_deterministic(
        self,
        page: TSDPage,
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        entities: Dict[str, GraphEntity] = {}
        relations: List[GraphRelation] = []
        valid_blocks = [b for b in page.text_blocks if b.is_valid() and (b.text or "").strip()]
        if not valid_blocks:
            return [], []

        for block in valid_blocks:
            text = block.text.strip()
            for sentence_id, sentence, start, end in self._split_sentences(text):
                raw_entity_names = self._extract_candidate_entities(sentence)
                unique_names: List[str] = []
                seen_names: Set[str] = set()
                for name in raw_entity_names:
                    normalized = _normalise_entity_id(name)
                    if not normalized or normalized in seen_names:
                        continue
                    seen_names.add(normalized)
                    unique_names.append(name.strip())

                    grounded = {
                        "block_id": block.block_id,
                        "sentence_id": f"p{page.page_number}_{block.block_id}_{sentence_id}",
                        "text": sentence,
                        "source_path": f"page/{page.page_number}",
                        "start_char": start,
                        "end_char": end,
                    }
                    if normalized in entities:
                        existing = entities[normalized]
                        if block.block_id not in existing.source_block_ids:
                            existing.source_block_ids.append(block.block_id)
                        existing.grounded_texts.append(grounded)
                    else:
                        entities[normalized] = GraphEntity(
                            entity_id=normalized,
                            name=name.strip(),
                            entity_type=self._infer_entity_type(name),
                            description=None,
                            source_pages=[page.page_number],
                            source_block_ids=[block.block_id],
                            grounded_texts=[grounded],
                        )

                entity_ids = [_normalise_entity_id(n) for n in unique_names]
                for i, source_id in enumerate(entity_ids):
                    for target_id in entity_ids[i + 1:]:
                        if not source_id or not target_id or source_id == target_id:
                            continue
                        grounded = {
                            "block_id": block.block_id,
                            "sentence_id": f"p{page.page_number}_{block.block_id}_{sentence_id}",
                            "text": sentence,
                            "source_path": f"page/{page.page_number}",
                            "start_char": start,
                            "end_char": end,
                        }
                        relations.append(
                            GraphRelation(
                                source_entity_id=source_id,
                                target_entity_id=target_id,
                                relation_type="communicates_with",
                                description="Sentence-level co-occurrence",
                                confidence=1.0,
                                source_pages=[page.page_number],
                                source_block_ids=[block.block_id],
                                grounded_texts=[grounded],
                            )
                        )
                        relations.append(
                            GraphRelation(
                                source_entity_id=target_id,
                                target_entity_id=source_id,
                                relation_type="communicates_with",
                                description="Sentence-level co-occurrence",
                                confidence=1.0,
                                source_pages=[page.page_number],
                                source_block_ids=[block.block_id],
                                grounded_texts=[grounded],
                            )
                        )
        return list(entities.values()), relations

    def _split_sentences(self, text: str) -> List[Tuple[int, str, int, int]]:
        sentences: List[Tuple[int, str, int, int]] = []
        if self._nlp is not None:
            doc = self._nlp(text)
            for idx, sent in enumerate(doc.sents, start=1):
                s = sent.text.strip()
                if s:
                    sentences.append((idx, s, sent.start_char, sent.end_char))
            return sentences
        for idx, match in enumerate(re.finditer(r"[^.!?]+[.!?]?", text), start=1):
            s = match.group(0).strip()
            if s:
                sentences.append((idx, s, match.start(), match.end()))
        return sentences

    def _extract_candidate_entities(self, sentence: str) -> List[str]:
        if self._nlp is not None:
            doc = self._nlp(sentence)
            out = [ent.text for ent in doc.ents if len(ent.text.strip()) >= _MIN_ENTITY_NAME_LENGTH]
            # fallback noun chunks
            out.extend([chunk.text for chunk in doc.noun_chunks if len(chunk.text.strip()) >= _MIN_ENTITY_NAME_LENGTH])
            return out
        # fallback heuristic: capitalized terms or hint terms
        terms = re.findall(r"\b[A-Z][a-zA-Z0-9_-]{1,}\b", sentence)
        tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", sentence)
        for token in tokens:
            low = token.lower()
            if low in _SIMPLE_ENTITY_TYPE_HINTS:
                terms.append(token)
        return terms

    def _infer_entity_type(self, name: str) -> str:
        low = (name or "").lower()
        for hint, etype in _SIMPLE_ENTITY_TYPE_HINTS.items():
            if hint in low:
                return etype
        return "component"

    def _parse_extraction_response(
        self,
        raw_content: str,
        page: TSDPage,
    ) -> Tuple[List[GraphEntity], List[GraphRelation], Optional[Dict[str, Any]]]:

        parsed, parse_failure = self._decode_extraction_payload(raw_content)
        if parse_failure is not None:
            return [], [], parse_failure

        if not isinstance(parsed, dict):
            self.logger.warning(
                "TSDGraphBuilder._parse_extraction_response: response is "
                "not a dict on page %d.",
                page.page_number,
            )
            return [], [], {"message": "non_dict_response", "pos": "unknown", "snippet": "", "content_length": len(raw_content or "")}

        # Collect all block_ids from this page for source tracing
        valid_blocks = [b for b in page.text_blocks if b.is_valid()]
        page_block_ids = [b.block_id for b in valid_blocks]
        page_grounded = [
            {
                "block_id": b.block_id,
                "sentence_id": f"p{page.page_number}_{b.block_id}_0",
                "text": (b.text or "")[:600],
                "source_path": f"page/{page.page_number}",
            }
            for b in valid_blocks
        ]

        # --- Parse entities ---
        raw_entities = parsed.get("entities", [])
        entities: List[GraphEntity] = []

        if isinstance(raw_entities, list):
            for item in raw_entities:
                if not isinstance(item, dict):
                    continue
                entity = GraphEntity(
                    entity_id=_normalise_entity_id(
                        item.get("entity_id", "")
                    ),
                    name=str(item.get("name", "")).strip(),
                    entity_type=str(
                        item.get("entity_type", "")
                    ).strip().lower(),
                    description=item.get("description") or None,
                    source_pages=[page.page_number],
                    source_block_ids=list(page_block_ids),
                    grounded_texts=list(page_grounded),
                )
                if entity.is_valid():
                    entities.append(entity)
                else:
                    self.logger.debug(
                        "TSDGraphBuilder._parse_extraction_response: "
                        "skipping invalid entity '%s' on page %d.",
                        item.get("entity_id"),
                        page.page_number,
                    )

        # --- Parse relations ---
        raw_relations = parsed.get("relations", [])
        relations: List[GraphRelation] = []

        if isinstance(raw_relations, list):
            # Build a set of valid entity_ids from this page's extraction
            # so we can reject relations that reference unknown entities
            valid_entity_ids: Set[str] = {e.entity_id for e in entities}

            for item in raw_relations:
                if not isinstance(item, dict):
                    continue

                source_id = _normalise_entity_id(
                    item.get("source_entity_id", "")
                )
                target_id = _normalise_entity_id(
                    item.get("target_entity_id", "")
                )

                # Reject relations referencing entities not on this page
                if source_id not in valid_entity_ids:
                    self.logger.debug(
                        "TSDGraphBuilder._parse_extraction_response: "
                        "source_entity_id '%s' not in extracted entities "
                        "on page %d — skipping relation.",
                        source_id,
                        page.page_number,
                    )
                    continue
                if target_id not in valid_entity_ids:
                    self.logger.debug(
                        "TSDGraphBuilder._parse_extraction_response: "
                        "target_entity_id '%s' not in extracted entities "
                        "on page %d — skipping relation.",
                        target_id,
                        page.page_number,
                    )
                    continue

                confidence = _safe_float_clamp(
                    item.get("confidence", 1.0),
                    default=1.0,
                )

                relation = GraphRelation(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=str(
                        item.get("relation_type", "")
                    ).strip().lower(),
                    description=item.get("description") or None,
                    is_encrypted=_parse_nullable_bool(
                        item.get("is_encrypted")
                    ),
                    requires_auth=_parse_nullable_bool(
                        item.get("requires_auth")
                    ),
                    protocol=item.get("protocol") or None,
                    confidence=confidence,
                    source_pages=[page.page_number],
                    source_block_ids=list(page_block_ids),
                    grounded_texts=list(page_grounded),
                )

                if relation.is_valid():
                    relations.append(relation)
                else:
                    self.logger.debug(
                        "TSDGraphBuilder._parse_extraction_response: "
                        "skipping invalid relation '%s' → '%s' on page %d.",
                        source_id,
                        target_id,
                        page.page_number,
                    )

        self.logger.debug(
            "TSDGraphBuilder._parse_extraction_response: page %d → "
            "%d valid entity(ies), %d valid relation(s).",
            page.page_number,
            len(entities),
            len(relations),
        )

        return entities, relations, None

    def _decode_extraction_payload(
        self,
        raw_content: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        candidates = self._build_extraction_json_candidates(raw_content)
        last_failure: Optional[Dict[str, Any]] = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed, None
                last_failure = {
                    "message": "non_dict_response",
                    "pos": "unknown",
                    "snippet": self._build_parse_snippet(candidate),
                    "content_length": len(candidate),
                }
            except json.JSONDecodeError as exc:
                last_failure = {
                    "message": str(exc),
                    "pos": exc.pos,
                    "snippet": self._build_parse_snippet(candidate, exc.pos),
                    "content_length": len(candidate),
                }
        return None, last_failure

    def _build_extraction_json_candidates(self, raw_content: str) -> List[str]:
        base = self._strip_json_fences(raw_content)
        candidates: List[str] = []
        seen: Set[str] = set()

        def add(candidate: str) -> None:
            normalized = (candidate or "").strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        add(base)
        extracted = self._extract_outer_json_object(base)
        add(extracted)

        for candidate in list(candidates):
            add(self._normalize_loose_json(candidate))

        return candidates

    def _strip_json_fences(self, raw_content: str) -> str:
        content = (raw_content or "").strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    def _extract_outer_json_object(self, content: str) -> str:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return content
        return content[start:end + 1].strip()

    def _normalize_loose_json(self, content: str) -> str:
        translated = content.translate(
            str.maketrans(
                {
                    "\u2018": "'",
                    "\u2019": "'",
                    "\u201c": '"',
                    "\u201d": '"',
                }
            )
        )
        translated = re.sub(r"\bNone\b", "null", translated)
        translated = re.sub(r"\bTrue\b", "true", translated)
        translated = re.sub(r"\bFalse\b", "false", translated)
        translated = re.sub(r",(\s*[}\]])", r"\1", translated)
        return translated.strip()

    def _build_parse_snippet(self, content: str, pos: Optional[int] = None) -> str:
        if not content:
            return ""
        if pos is None or not isinstance(pos, int):
            return content[:160]
        start = max(0, pos - 80)
        end = min(len(content), pos + 80)
        return content[start:end]

    # ------------------------------------------------------------------
    # Entity merging
    # ------------------------------------------------------------------

    def _merge_entity_metadata(
        self,
        existing: GraphEntity,
        incoming: GraphEntity,
    ) -> None:
        # Merge source_pages — deduplicate and sort for determinism
        merged_pages = sorted(
            set(existing.source_pages) | set(incoming.source_pages)
        )
        existing.source_pages = merged_pages

        # Merge source_block_ids — preserve order, deduplicate
        seen_ids: Set[str] = set(existing.source_block_ids)
        for block_id in incoming.source_block_ids:
            if block_id not in seen_ids:
                existing.source_block_ids.append(block_id)
                seen_ids.add(block_id)

        # Fill description only if currently missing
        if existing.description is None and incoming.description:
            existing.description = incoming.description
        if incoming.grounded_texts:
            existing.grounded_texts.extend(incoming.grounded_texts)

    # ------------------------------------------------------------------
    # Graph population
    # ------------------------------------------------------------------

    def _populate_graph(
        self,
        tsd_graph: TSDGraph,
        entities: Dict[str, GraphEntity],
        relations: List[GraphRelation],
    ) -> None:
        # --- Add nodes first ---
        for entity_id, entity in entities.items():
            tsd_graph.entities[entity_id] = entity
            tsd_graph.graph.add_node(
                entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                source_pages=list(entity.source_pages),
                source_block_ids=list(entity.source_block_ids),
                grounded_texts=list(entity.grounded_texts),
                sensitivity=entity.sensitivity,
                tenant_id=entity.tenant_id,
                entity_obj=entity,
            )
        tsd_graph.total_entities = len(tsd_graph.entities)

        # --- Add edges ---
        added_relations = 0
        merged_relations = 0

        for relation in relations:
            # Guard: both endpoints must exist as nodes
            if relation.source_entity_id not in tsd_graph.graph:
                self.logger.debug(
                    "TSDGraphBuilder._populate_graph: source_entity_id '%s' "
                    "not in graph nodes — skipping relation.",
                    relation.source_entity_id,
                )
                continue

            if relation.target_entity_id not in tsd_graph.graph:
                self.logger.debug(
                    "TSDGraphBuilder._populate_graph: target_entity_id '%s' "
                    "not in graph nodes — skipping relation.",
                    relation.target_entity_id,
                )
                continue

            source_id = relation.source_entity_id
            target_id = relation.target_entity_id

            # Check if an edge already exists between this (source, target) pair
            if tsd_graph.graph.has_edge(source_id, target_id):
                # Merge source metadata into the existing edge rather than
                # creating a parallel edge — DiGraph only supports one edge
                # per (source, target) pair
                existing_data = tsd_graph.graph[source_id][target_id]
                existing_relation: GraphRelation = existing_data.get("relation_obj")

                if existing_relation:
                    # Union source_pages
                    existing_relation.source_pages = sorted(
                        set(existing_relation.source_pages)
                        | set(relation.source_pages)
                    )
                    # Union source_block_ids preserving order
                    seen_ids: Set[str] = set(existing_relation.source_block_ids)
                    for block_id in relation.source_block_ids:
                        if block_id not in seen_ids:
                            existing_relation.source_block_ids.append(block_id)
                            seen_ids.add(block_id)

                    # Take the higher confidence value
                    if relation.confidence > existing_relation.confidence:
                        existing_relation.confidence = relation.confidence

                    # Fill in missing security metadata if now known
                    if existing_relation.is_encrypted is None and relation.is_encrypted is not None:
                        existing_relation.is_encrypted = relation.is_encrypted
                    if existing_relation.requires_auth is None and relation.requires_auth is not None:
                        existing_relation.requires_auth = relation.requires_auth
                    if existing_relation.protocol is None and relation.protocol:
                        existing_relation.protocol = relation.protocol
                    if relation.grounded_texts:
                        existing_relation.grounded_texts.extend(relation.grounded_texts)

                merged_relations += 1
                continue

            # New edge — add to the DiGraph
            tsd_graph.graph.add_edge(
                source_id,
                target_id,
                relation_type=relation.relation_type,
                description=relation.description,
                is_encrypted=relation.is_encrypted,
                requires_auth=relation.requires_auth,
                protocol=relation.protocol,
                confidence=relation.confidence,
                weight=relation.weight,
                source_pages=relation.source_pages,
                source_block_ids=relation.source_block_ids,
                grounded_texts=relation.grounded_texts,
                sensitivity=relation.sensitivity,
                tenant_id=relation.tenant_id,
                relation_obj=relation,   # direct object reference for get_neighbours()
            )
            added_relations += 1

        tsd_graph.total_relations = tsd_graph.graph.number_of_edges()

        self.logger.debug(
            "TSDGraphBuilder._populate_graph: added %d edge(s), "
            "merged %d duplicate(s). Total edges in graph: %d.",
            added_relations,
            merged_relations,
            tsd_graph.total_relations,
        )

    def _build_indexes(self, tsd_graph: TSDGraph) -> None:
        entity_to_block_ids: Dict[str, Set[str]] = defaultdict(set)
        block_id_to_entities: Dict[str, Set[str]] = defaultdict(set)
        edge_to_block_ids: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for entity_id, entity in tsd_graph.entities.items():
            for block_id in entity.source_block_ids:
                entity_to_block_ids[entity_id].add(block_id)
                block_id_to_entities[block_id].add(entity_id)

        if tsd_graph.graph is not None:
            for source_id, target_id, edge_data in tsd_graph.graph.edges(data=True):
                relation: Optional[GraphRelation] = edge_data.get("relation_obj")
                if relation:
                    for block_id in relation.source_block_ids:
                        edge_to_block_ids[(source_id, target_id)].add(block_id)

        tsd_graph.entity_to_block_ids = dict(entity_to_block_ids)
        tsd_graph.block_id_to_entities = dict(block_id_to_entities)
        tsd_graph.edge_to_block_ids = dict(edge_to_block_ids)

    def link_raptor_entities(self, tsd_graph: TSDGraph, raptor_tree: Any) -> None:
        self.linker.link(tsd_graph, raptor_tree)


# ---------------------------------------------------------------------------
# Module-level pure utility functions
# ---------------------------------------------------------------------------

def _normalise_entity_id(raw: object) -> str:
    if raw is None:
        return ""

    try:
        slug = str(raw).strip().lower()
    except Exception:
        return ""

    if not slug:
        return ""

    # Replace common separators with underscores
    slug = slug.replace(" ", "_").replace("-", "_").replace(".", "_")

    # Remove characters that are not alphanumeric or underscore
    slug = re.sub(r"[^\w]", "", slug)

    # Collapse consecutive underscores
    slug = re.sub(r"_+", "_", slug)

    # Strip leading/trailing underscores
    slug = slug.strip("_")

    return slug


def _parse_nullable_bool(raw: object) -> Optional[bool]:
    if raw is None:
        return None

    if isinstance(raw, bool):
        return raw

    if isinstance(raw, int):
        if raw == 1:
            return True
        if raw == 0:
            return False
        return None

    if isinstance(raw, str):
        normalised = raw.strip().lower()
        if normalised in {"true", "yes", "1"}:
            return True
        if normalised in {"false", "no", "0"}:
            return False
        if normalised in {"null", "none", "unknown", "n/a", ""}:
            return None

    # Unrecognised type or value — treat as unknown
    return None


def _safe_float_clamp(
    value: object,
    default: float = 1.0,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float:
    if value is None:
        return default

    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default

    return max(min_val, min(max_val, coerced))


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Dataclasses — imported by retrieval/graph_search.py
    # and analysis_service.py
    "GraphEntity",
    "GraphRelation",
    "TSDGraph",
    # Main builder class
    "TSDGraphBuilder",
    # Pure utilities — independently testable, used internally
    "_normalise_entity_id",
    "_parse_nullable_bool",
    "_safe_float_clamp",
]
