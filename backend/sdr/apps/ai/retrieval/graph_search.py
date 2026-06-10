"""
Graph Search — relationship-aware retrieval against the TSDGraph built
by TSDGraphBuilder (tsd_processing/graph_builder.py).

Responsibility:
    Given a security parameter (CategoryParameterChild [3]), traverse the
    TSDGraph to find architectural evidence that pure vector search would
    miss — specifically inter-service relationships, authentication paths,
    encryption edges, and data flow chains that are implicit in the
    document's structure rather than stated in any single text block.

Why graph search alongside vector and RAPTOR search?
    Vector search (retrieval/vector_search.py) finds semantically similar
    text chunks — it answers "what does this section say about encryption?"
    RAPTOR search (retrieval/raptor_search.py) finds relevant TSD passages
    at multiple abstraction levels.
    Graph search answers a different class of question entirely:
    "Does Service A enforce authentication on ALL paths to Service B?" —
    this requires traversing the call graph, not matching text similarity.

Search strategies:
    ENTITY_MATCH    — find entities whose name matches parameter keywords
    RELATION_FILTER — find edges of a specific relation type
    PATH_ANALYSIS   — find all paths between two entities and audit them
    NEIGHBOURHOOD   — find the immediate neighbourhood of a key entity

Dependency chain:
    tsd_processing/graph_builder.py  (TSDGraph, GraphEntity, GraphRelation)
         ↓
    graph_search.py                  ← YOU ARE HERE
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
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from sdr.apps.ai.client import get_embeddings
from sdr.apps.ai.tsd_processing.graph_builder import (
    GraphEntity,
    GraphRelation,
    TSDGraph,
    TSDGraphBuilder,
)
from .policy import UserContext

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
        "networkx is not installed. Graph search will not function. "
        "Install with: pip install networkx"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default maximum number of entity results to return
_DEFAULT_TOP_K = 10

# Maximum path depth for path analysis queries
_DEFAULT_MAX_PATH_DEPTH = 4

# Minimum keyword length — single character keywords produce too much noise
_MIN_KEYWORD_LENGTH = 3

# Security-relevant relation types that are always included in
# neighbourhood queries regardless of the parameter keywords
_SECURITY_CRITICAL_RELATION_TYPES = frozenset({
    "authenticates_with",
    "authorises_via",
    "encrypted_by",
    "uses_protocol",
})

# Keyword → relation type mapping for parameter-driven relation filtering
# Used by _infer_relation_types_from_parameter() to map parameter keywords
# to graph edge types without an LLM call
_KEYWORD_TO_RELATION_MAP: Dict[str, Set[str]] = {
    "authenticat": {"authenticates_with", "authorises_via"},
    "authoris":    {"authorises_via", "authenticates_with"},
    "encrypt":     {"encrypted_by", "uses_protocol"},
    "tls":         {"uses_protocol", "encrypted_by"},
    "ssl":         {"uses_protocol", "encrypted_by"},
    "jwt":         {"authenticates_with", "uses_protocol"},
    "oauth":       {"authenticates_with", "authorises_via"},
    "access":      {"accessed_by", "authorises_via"},
    "stor":        {"stores_in", "writes_to", "reads_from"},
    "send":        {"sends_data_to", "communicates_with"},
    "receiv":      {"receives_data_from", "communicates_with"},
    "deploy":      {"deployed_on", "depends_on"},
    "depend":      {"depends_on"},
    "call":        {"calls", "communicates_with"},
    "communicat":  {"communicates_with", "calls"},
    "expos":       {"exposes_endpoint"},
    "database":    {"reads_from", "writes_to", "stores_in"},
    "db":          {"reads_from", "writes_to", "stores_in"},
}


# ---------------------------------------------------------------------------
# Search strategy enum
# ---------------------------------------------------------------------------

class GraphSearchStrategy(Enum):
    """
    The retrieval strategy used for a GraphSearch query.
    Selected by the router based on parameter type and keyword analysis.
    """
    ENTITY_MATCH    = "entity_match"     # match entities by keyword
    RELATION_FILTER = "relation_filter"  # filter edges by relation type
    PATH_ANALYSIS   = "path_analysis"    # traverse paths between entities
    NEIGHBOURHOOD   = "neighbourhood"    # expand entity neighbourhood
    COMBINED        = "combined"         # all strategies merged


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GraphSearchResult:
    """
    A single result from a graph search query.

    Carries the matched entity, all its outgoing relations relevant to
    the query, and the source_block_ids for citation tracing back to
    the TSD text that described this entity/relationship.
    """
    entity: GraphEntity
    # Relations from this entity that are relevant to the query
    relevant_relations: List[GraphRelation] = field(default_factory=list)
    # Score — higher is more relevant (0.0–1.0)
    relevance_score: float = 0.0
    # Strategy that produced this result
    strategy: GraphSearchStrategy = GraphSearchStrategy.ENTITY_MATCH
    # Union of source_block_ids from entity + all relevant relations
    source_block_ids: List[str] = field(default_factory=list)
    entity_similarity: float = 0.0
    relation_similarity: float = 0.0
    blended_score: float = 0.0

    @property
    def entity_type(self) -> str:
        return self.entity.entity_type

    @property
    def entity_name(self) -> str:
        return self.entity.name

    def has_security_gap(self, relation_type: str) -> bool:
        """
        Returns True if none of this entity's relevant relations match
        the specified security relation type.
        Used to identify missing controls — e.g. "authenticates_with"
        missing from a service that handles sensitive data.
        """
        return not any(
            r.relation_type == relation_type
            for r in self.relevant_relations
        )


@dataclass
class PathAnalysisResult:
    """
    Represents a single path between two entities in the TSDGraph.

    Used by path analysis queries to audit whether security controls
    exist on every hop in a call chain — e.g. does every path from
    the public API to the database pass through an auth service?
    """
    path_entities: List[GraphEntity]          # ordered list of entities in the path
    path_relations: List[GraphRelation]       # ordered list of relations between them
    has_auth: bool = False                    # True if any edge has requires_auth=True
    has_encryption: bool = False              # True if any edge has is_encrypted=True
    is_fully_secured: bool = False            # True if auth AND encryption on all edges
    source_block_ids: List[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.path_entities)

    @property
    def entity_names(self) -> List[str]:
        return [e.name for e in self.path_entities]


@dataclass
class GraphSearchResponse:
    """
    The complete response from a GraphSearcher.search() call.

    results are ordered by relevance_score descending.
    path_results are populated only for PATH_ANALYSIS strategy queries.
    all_source_block_ids is the union across all results for context
    window construction in analysis_service.py.
    """
    results: List[GraphSearchResult] = field(default_factory=list)
    path_results: List[PathAnalysisResult] = field(default_factory=list)
    strategy_used: GraphSearchStrategy = GraphSearchStrategy.ENTITY_MATCH
    query_keywords: List[str] = field(default_factory=list)
    inferred_relation_types: List[str] = field(default_factory=list)
    graph_node_ids: List[str] = field(default_factory=list)
    graph_edge_ids: List[str] = field(default_factory=list)
    grounded_texts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0 and len(self.path_results) == 0

    @property
    def all_source_block_ids(self) -> List[str]:
        """
        Returns deduplicated union of source_block_ids across all results.
        Used by analysis_service.py to build the context window for the
        Hunter agent alongside RAPTOR and vector results.
        """
        seen: Set[str] = set()
        block_ids: List[str] = []
        for result in self.results:
            for bid in result.source_block_ids:
                if bid not in seen:
                    block_ids.append(bid)
                    seen.add(bid)
        for path in self.path_results:
            for bid in path.source_block_ids:
                if bid not in seen:
                    block_ids.append(bid)
                    seen.add(bid)
        return block_ids

    def get_security_gaps(self, relation_type: str) -> List[GraphSearchResult]:
        """
        Returns results where the specified security relation type is
        absent from the entity's relevant relations.
        Used by analysis_service.py to surface missing controls directly.
        """
        return [r for r in self.results if r.has_security_gap(relation_type)]


@dataclass
class GraphTraversalConfig:
    max_hops: int = 2
    max_expanded_nodes: int = 25
    min_edge_weight: float = 1.0
    require_authorized: bool = True


# ---------------------------------------------------------------------------
# Graph Searcher
# ---------------------------------------------------------------------------

class GraphSearcher:
    """
    Performs relationship-aware traversal queries against a TSDGraph built
    by TSDGraphBuilder (tsd_processing/graph_builder.py).

    The GraphSearcher operates entirely in memory against a pre-built
    TSDGraph — like RAPTORSearcher, it requires no database queries.
    The graph is built once per TSD document per review and cached by
    analysis_service.py.

    Search entry points:
        search()             — primary entry point, auto-selects strategy
        search_by_entity()   — entity keyword matching
        search_by_relation() — relation type filtering
        analyse_paths()      — path analysis between two entities

    Usage:
        searcher = GraphSearcher()
        response = searcher.search(
            parameter_text="All inter-service communication must use mTLS",
            graph=tsd_graph,
        )
        for result in response.results:
            print(result.entity_name, result.relevant_relations)
    """

    def __init__(
        self,
        default_top_k: int = _DEFAULT_TOP_K,
        max_path_depth: int = _DEFAULT_MAX_PATH_DEPTH,
        traversal_config: Optional[GraphTraversalConfig] = None,
    ) -> None:
        if not NETWORKX_AVAILABLE:
            raise RuntimeError(
                "GraphSearcher requires networkx. "
                "Install with: pip install networkx"
            )
        self.default_top_k = default_top_k
        self.max_path_depth = max_path_depth
        self.traversal_config = traversal_config or GraphTraversalConfig()
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Primary search entry point
    # ------------------------------------------------------------------

    def search(
        self,
        parameter_text: str,
        graph: TSDGraph,
        query_embedding: Optional[List[float]] = None,
        strategy: Optional[GraphSearchStrategy] = None,
        top_k: Optional[int] = None,
    ) -> GraphSearchResponse:
        """
        Primary entry point for graph-based retrieval.

        Automatically selects the most appropriate strategy based on
        parameter keyword analysis if strategy is not specified.

        Pipeline:
            1. Validate inputs.
            2. Extract keywords from parameter_text.
            3. Infer relevant relation types from keywords.
            4. Select or confirm strategy.
            5. Execute the selected strategy.
            6. Return merged GraphSearchResponse.

        Args:
            parameter_text: The full requirement text from
                            CategoryParameterChild [3].
            graph:          The TSDGraph built from the TSD document.
            strategy:       Optional explicit strategy. If None, auto-selected
                            from keyword analysis.
            top_k:          Maximum number of entity results.

        Returns:
            GraphSearchResponse — never raises. Check .error for failures.
        """
        # ------------------------------------------------------------------
        # 1. Validate inputs
        # ------------------------------------------------------------------
        if not parameter_text or not parameter_text.strip():
            msg = "parameter_text is empty — cannot perform graph search."
            self.logger.warning("GraphSearcher.search: %s", msg)
            return GraphSearchResponse(error=msg)

        if graph.is_empty():
            msg = (
                f"TSDGraph for '{graph.document_name}' is empty — "
                f"no entities to search."
            )
            self.logger.warning("GraphSearcher.search: %s", msg)
            return GraphSearchResponse(error=msg)

        resolved_top_k = top_k if top_k is not None else self.default_top_k

        # ------------------------------------------------------------------
        # 2. Extract keywords
        # ------------------------------------------------------------------
        keywords = _extract_keywords(parameter_text)

        if not keywords:
            self.logger.debug(
                "GraphSearcher.search: no meaningful keywords extracted "
                "from parameter '%s...' — using entity match only.",
                parameter_text[:60],
            )

        # ------------------------------------------------------------------
        # 3. Infer relation types
        # ------------------------------------------------------------------
        inferred_relations = _infer_relation_types_from_parameter(keywords)

        # ------------------------------------------------------------------
        # 4. Select strategy
        # ------------------------------------------------------------------
        resolved_strategy = strategy or self._select_strategy(
            keywords=keywords,
            inferred_relations=inferred_relations,
        )

        self.logger.info(
            "GraphSearcher.search: strategy=%s keywords=%s "
            "inferred_relations=%s for parameter '%s...'",
            resolved_strategy.value,
            keywords[:5],
            list(inferred_relations)[:3],
            parameter_text[:60],
        )

        # ------------------------------------------------------------------
        # 5. Execute strategy
        # ------------------------------------------------------------------
        try:
            if resolved_strategy == GraphSearchStrategy.ENTITY_MATCH:
                results = self._entity_match(
                    graph=graph,
                    keywords=keywords,
                    inferred_relations=inferred_relations,
                    query_embedding=query_embedding,
                    top_k=resolved_top_k,
                )
                return GraphSearchResponse(
                    results=results,
                    strategy_used=resolved_strategy,
                    query_keywords=keywords,
                    inferred_relation_types=list(inferred_relations),
                )

            elif resolved_strategy == GraphSearchStrategy.RELATION_FILTER:
                results = self._relation_filter(
                    graph=graph,
                    relation_types=inferred_relations,
                    keywords=keywords,
                    query_embedding=query_embedding,
                    top_k=resolved_top_k,
                )
                return GraphSearchResponse(
                    results=results,
                    strategy_used=resolved_strategy,
                    query_keywords=keywords,
                    inferred_relation_types=list(inferred_relations),
                )

            elif resolved_strategy == GraphSearchStrategy.NEIGHBOURHOOD:
                results = self._neighbourhood_search(
                    graph=graph,
                    keywords=keywords,
                    inferred_relations=inferred_relations,
                    query_embedding=query_embedding,
                    top_k=resolved_top_k,
                )
                return GraphSearchResponse(
                    results=results,
                    strategy_used=resolved_strategy,
                    query_keywords=keywords,
                    inferred_relation_types=list(inferred_relations),
                )

            elif resolved_strategy == GraphSearchStrategy.COMBINED:
                return self._combined_search(
                    graph=graph,
                    keywords=keywords,
                    inferred_relations=inferred_relations,
                    query_embedding=query_embedding,
                    top_k=resolved_top_k,
                )

            else:
                # Default fallback — entity match
                results = self._entity_match(
                    graph=graph,
                    keywords=keywords,
                    inferred_relations=inferred_relations,
                    query_embedding=query_embedding,
                    top_k=resolved_top_k,
                )
                return GraphSearchResponse(
                                        results=results,
                    strategy_used=resolved_strategy,
                    query_keywords=keywords,
                    inferred_relation_types=list(inferred_relations),
                )

        except Exception as exc:
            msg = f"Graph search failed with strategy={resolved_strategy.value}: {exc}"
            self.logger.error("GraphSearcher.search: %s", msg)
            return GraphSearchResponse(
                error=msg,
                strategy_used=resolved_strategy,
                query_keywords=keywords,
                inferred_relation_types=list(inferred_relations),
            )

    # ------------------------------------------------------------------
    # Specialised public search methods
    # ------------------------------------------------------------------

    def search_by_entity(
        self,
        keywords: List[str],
        graph: TSDGraph,
        query_embedding: Optional[List[float]] = None,
        top_k: int = _DEFAULT_TOP_K,
    ) -> List[GraphSearchResult]:
        """
        Finds entities whose names contain any of the provided keywords
        and returns their full neighbourhood context.

        Used directly by analysis_service.py when a parameter explicitly
        names a component (e.g. "The API Gateway must enforce TLS").

        Args:
            keywords: List of lowercase keyword strings.
            graph:    The TSDGraph to search.
            top_k:    Maximum number of results.

        Returns:
            List of GraphSearchResult ordered by relevance_score descending.
        """
        if not keywords or graph.is_empty():
            return []

        matched_entities = []
        for keyword in keywords:
            if len(keyword) < _MIN_KEYWORD_LENGTH:
                continue
            matched_entities.extend(
                graph.find_entities_by_name_fragment(keyword)
            )

        # Deduplicate by entity_id
        seen: Set[str] = set()
        unique_entities: List[GraphEntity] = []
        for entity in matched_entities:
            if entity.entity_id not in seen:
                unique_entities.append(entity)
                seen.add(entity.entity_id)

        if not unique_entities:
            return []

        if query_embedding:
            self._ensure_relation_embeddings(
                graph=graph,
                relations=[
                    relation
                    for entity in unique_entities[:top_k]
                    for relation in self._get_all_relations(graph, entity.entity_id)
                ],
            )

        results: List[GraphSearchResult] = []
        for entity in unique_entities[:top_k]:
            relations = self._get_all_relations(graph, entity.entity_id)
            source_block_ids = self._aggregate_block_ids(entity, relations)
            score = self._score_entity_relevance(entity, relations, keywords)
            entity_sim, rel_sim, blended_score = self._blend_scores(
                query_embedding=query_embedding,
                entity=entity,
                relations=relations,
                heuristic_score=score,
            )

            results.append(
                GraphSearchResult(
                    entity=entity,
                    relevant_relations=relations,
                    relevance_score=blended_score,
                    strategy=GraphSearchStrategy.ENTITY_MATCH,
                    source_block_ids=source_block_ids,
                    entity_similarity=entity_sim,
                    relation_similarity=rel_sim,
                    blended_score=blended_score,
                )
            )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def search_by_relation(
        self,
        relation_types: Set[str],
        graph: TSDGraph,
        keywords: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
        top_k: int = _DEFAULT_TOP_K,
    ) -> List[GraphSearchResult]:
        """
        Finds all entities that participate in edges of the specified
        relation types. Used for parameters like "all services must
        authenticate" — we search for all authenticates_with edges.

        Args:
            relation_types: Set of relation type strings to match.
            graph:          The TSDGraph to search.
            keywords:       Optional keyword filter — if provided, only
                            entities whose names match are returned.
            top_k:          Maximum number of results.

        Returns:
            List of GraphSearchResult ordered by relevance_score descending.
        """
        if not relation_types or graph.is_empty():
            return []

        if not NETWORKX_AVAILABLE or graph.graph is None:
            return []

        matched_entity_ids: Set[str] = set()

        for source_id, target_id, edge_data in graph.graph.edges(data=True):
            edge_relation_type = edge_data.get("relation_type", "")
            if edge_relation_type in relation_types:
                matched_entity_ids.add(source_id)
                matched_entity_ids.add(target_id)

        if not matched_entity_ids:
            return []

        if query_embedding:
            candidate_relations = []
            for entity_id in matched_entity_ids:
                for _, _, edge_data in graph.graph.out_edges(entity_id, data=True):
                    relation = edge_data.get("relation_obj")
                    if relation is not None and edge_data.get("relation_type") in relation_types:
                        candidate_relations.append(relation)
            self._ensure_relation_embeddings(graph=graph, relations=candidate_relations)

        results: List[GraphSearchResult] = []
        for entity_id in matched_entity_ids:
            entity = graph.get_entity(entity_id)
            if not entity:
                continue

            # Optional keyword filter
            if keywords:
                name_lower = entity.name.lower()
                if not any(kw in name_lower for kw in keywords if len(kw) >= _MIN_KEYWORD_LENGTH):
                    # Still include — relation match is the primary signal
                    pass

            # Only return relations that match the requested types
            relevant_relations = [
                edge_data.get("relation_obj")
                for _, _, edge_data in graph.graph.out_edges(entity_id, data=True)
                if edge_data.get("relation_type") in relation_types
                and edge_data.get("relation_obj") is not None
            ]

            source_block_ids = self._aggregate_block_ids(entity, relevant_relations)
            score = self._score_entity_relevance(
                entity, relevant_relations, keywords or []
            )
            entity_sim, rel_sim, blended_score = self._blend_scores(
                query_embedding=query_embedding,
                entity=entity,
                relations=relevant_relations,
                heuristic_score=score,
            )

            results.append(
                GraphSearchResult(
                    entity=entity,
                    relevant_relations=relevant_relations,
                    relevance_score=blended_score,
                    strategy=GraphSearchStrategy.RELATION_FILTER,
                    source_block_ids=source_block_ids,
                    entity_similarity=entity_sim,
                    relation_similarity=rel_sim,
                    blended_score=blended_score,
                )
            )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    def analyse_paths(
        self,
        source_keyword: str,
        target_keyword: str,
        graph: TSDGraph,
        max_depth: Optional[int] = None,
    ) -> List[PathAnalysisResult]:
        """
        Finds all paths between entities matching source_keyword and
        target_keyword, then audits each path for security controls.

        Used to answer questions like "does every path from the public
        API to the database pass through an auth service?"

        Args:
            source_keyword: Keyword to match source entity names.
            target_keyword: Keyword to match target entity names.
            graph:          The TSDGraph to traverse.
            max_depth:      Maximum path length. Defaults to _DEFAULT_MAX_PATH_DEPTH.

        Returns:
            List of PathAnalysisResult — one per path found.
        """
        if graph.is_empty():
            return []

        resolved_depth = max_depth or self.max_path_depth

        source_entities = graph.find_entities_by_name_fragment(source_keyword)
        target_entities = graph.find_entities_by_name_fragment(target_keyword)

        if not source_entities or not target_entities:
            self.logger.debug(
                "GraphSearcher.analyse_paths: no entities found for "
                "source='%s' or target='%s'.",
                source_keyword,
                target_keyword,
            )
            return []

        path_results: List[PathAnalysisResult] = []

        for source_entity in source_entities:
            for target_entity in target_entities:
                if source_entity.entity_id == target_entity.entity_id:
                    continue

                paths = graph.get_all_paths(
                    source_id=source_entity.entity_id,
                    target_id=target_entity.entity_id,
                    max_depth=resolved_depth,
                )

                for path_entity_ids in paths:
                    path_result = self._analyse_single_path(
                        path_entity_ids=path_entity_ids,
                        graph=graph,
                    )
                    if path_result:
                        path_results.append(path_result)

        self.logger.debug(
            "GraphSearcher.analyse_paths: found %d path(s) between "
            "'%s' and '%s'.",
            len(path_results),
            source_keyword,
            target_keyword,
        )
        return path_results

    def search_local(
        self,
        query_entities: List[str],
        graph: TSDGraph,
        query_embedding: Optional[List[float]] = None,
        user_context: Optional[UserContext] = None,
        traversal_config: Optional[GraphTraversalConfig] = None,
    ) -> GraphSearchResponse:
        if graph.is_empty() or not query_entities:
            return GraphSearchResponse(error="Graph local search requires graph and query entities.")

        cfg = traversal_config or self.traversal_config
        ctx = user_context or UserContext()

        start_entities: List[GraphEntity] = []
        for q in query_entities:
            if q in graph.entities:
                start_entities.append(graph.entities[q])
            start_entities.extend(graph.find_entities_by_name_fragment(q))
        seen_start: Set[str] = set()
        dedup_start: List[GraphEntity] = []
        for entity in start_entities:
            if entity.entity_id not in seen_start:
                dedup_start.append(entity)
                seen_start.add(entity.entity_id)

        visited: Set[str] = set()
        queue: List[Tuple[str, int]] = [(e.entity_id, 0) for e in dedup_start]
        results: Dict[str, GraphSearchResult] = {}
        edge_ids: Set[str] = set()
        grounded_texts: List[Dict[str, Any]] = []
        expanded = 0
        if query_embedding:
            self._ensure_relation_embeddings(
                graph=graph,
                relations=[
                    relation
                    for entity in dedup_start
                    for relation in self._get_all_relations(graph, entity.entity_id)
                ],
            )

        while queue and expanded < cfg.max_expanded_nodes:
            entity_id, hops = queue.pop(0)
            if entity_id in visited or hops > cfg.max_hops:
                continue
            entity = graph.get_entity(entity_id)
            if entity is None:
                continue
            if cfg.require_authorized and not self._is_authorized(entity.sensitivity, entity.tenant_id, ctx):
                continue

            visited.add(entity_id)
            expanded += 1
            relations = self._get_all_relations(graph, entity_id)
            allowed_relations: List[GraphRelation] = []
            for rel in relations:
                if rel.weight < cfg.min_edge_weight:
                    continue
                if cfg.require_authorized and not self._is_authorized(rel.sensitivity, rel.tenant_id, ctx):
                    continue
                allowed_relations.append(rel)
                edge_ids.add(f"{rel.source_entity_id}->{rel.target_entity_id}")
                grounded_texts.extend(rel.grounded_texts or [])
                if hops + 1 <= cfg.max_hops:
                    queue.append((rel.target_entity_id, hops + 1))

            grounded_texts.extend(entity.grounded_texts or [])
            source_block_ids = self._aggregate_block_ids(entity, allowed_relations)
            results[entity_id] = GraphSearchResult(
                entity=entity,
                relevant_relations=allowed_relations,
                relevance_score=self._blend_scores(
                    query_embedding=query_embedding,
                    entity=entity,
                    relations=allowed_relations,
                    heuristic_score=1.0 / (1 + hops),
                )[2],
                strategy=GraphSearchStrategy.NEIGHBOURHOOD,
                source_block_ids=source_block_ids,
            )

        sorted_results = sorted(results.values(), key=lambda r: r.relevance_score, reverse=True)
        return GraphSearchResponse(
            results=sorted_results,
            strategy_used=GraphSearchStrategy.NEIGHBOURHOOD,
            query_keywords=query_entities,
            inferred_relation_types=[],
            graph_node_ids=[r.entity.entity_id for r in sorted_results],
            graph_edge_ids=sorted(edge_ids),
            grounded_texts=grounded_texts,
        )

    def _is_authorized(self, sensitivity: Optional[str], tenant_id: Optional[str], user_context: UserContext) -> bool:
        rank = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
        user_rank = rank.get(user_context.clearance, 1)
        candidate_rank = rank.get((sensitivity or "internal"), 1)
        if user_context.tenant_id and tenant_id and user_context.tenant_id != tenant_id:
            return False
        return candidate_rank <= user_rank

    # ------------------------------------------------------------------
    # Private strategy implementations
    # ------------------------------------------------------------------

    def _entity_match(
        self,
        graph: TSDGraph,
        keywords: List[str],
        inferred_relations: Set[str],
        query_embedding: Optional[List[float]],
        top_k: int,
    ) -> List[GraphSearchResult]:
        """
        Entity keyword match strategy — finds entities by name fragment
        and returns their security-relevant neighbourhood.
        """
        results = self.search_by_entity(keywords, graph, query_embedding=query_embedding, top_k=top_k)

        # Enrich with security-critical relation context even if not
        # directly matched by keywords
        for result in results:
            all_relations = self._get_all_relations(graph, result.entity.entity_id)
            security_relations = [
                r for r in all_relations
                if r.relation_type in _SECURITY_CRITICAL_RELATION_TYPES
                or r.relation_type in inferred_relations
            ]
            if security_relations:
                # Merge without duplicating
                existing_types = {r.relation_type for r in result.relevant_relations}
                for rel in security_relations:
                    if rel.relation_type not in existing_types:
                        result.relevant_relations.append(rel)

        return results

    def _relation_filter(
        self,
        graph: TSDGraph,
        relation_types: Set[str],
        keywords: List[str],
        query_embedding: Optional[List[float]],
        top_k: int,
    ) -> List[GraphSearchResult]:
        """
        Relation type filter strategy — finds all entities involved in
        edges of the inferred relation types.
        """
        return self.search_by_relation(
            relation_types=relation_types,
            graph=graph,
            keywords=keywords,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def _neighbourhood_search(
        self,
        graph: TSDGraph,
        keywords: List[str],
        inferred_relations: Set[str],
        query_embedding: Optional[List[float]],
        top_k: int,
    ) -> List[GraphSearchResult]:
        """
        Neighbourhood expansion strategy — starts from keyword-matched
        entities and expands to their full neighbourhood, including
        second-degree connections for security-critical relation types.
        """
        primary_results = self.search_by_entity(
            keywords=keywords,
            graph=graph,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        if not primary_results:
            return []

        # Expand to second-degree neighbours via security-critical edges
        expanded_entity_ids: Set[str] = {r.entity.entity_id for r in primary_results}
        additional_results: List[GraphSearchResult] = []

        for result in primary_results:
            neighbours = graph.get_neighbours(result.entity.entity_id)
            for neighbour_entity, relation in neighbours:
                if (
                    relation.relation_type in _SECURITY_CRITICAL_RELATION_TYPES
                    and neighbour_entity.entity_id not in expanded_entity_ids
                ):
                    neighbour_relations = self._get_all_relations(
                        graph, neighbour_entity.entity_id
                    )
                    source_block_ids = self._aggregate_block_ids(
                        neighbour_entity, neighbour_relations
                    )
                    additional_results.append(
                        GraphSearchResult(
                            entity=neighbour_entity,
                            relevant_relations=neighbour_relations,
                            relevance_score=self._blend_scores(
                                query_embedding=query_embedding,
                                entity=neighbour_entity,
                                relations=neighbour_relations,
                                heuristic_score=result.relevance_score * 0.7,
                            )[2],
                            strategy=GraphSearchStrategy.NEIGHBOURHOOD,
                            source_block_ids=source_block_ids,
                        )
                    )
                    expanded_entity_ids.add(neighbour_entity.entity_id)

        all_results = primary_results + additional_results
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)
        return all_results[:top_k]

    def _combined_search(
        self,
        graph: TSDGraph,
        keywords: List[str],
        inferred_relations: Set[str],
        query_embedding: Optional[List[float]],
        top_k: int,
    ) -> GraphSearchResponse:
        """
        Combined strategy — merges results from entity match, relation
        filter, and neighbourhood expansion. Used for complex parameters
        that touch multiple graph dimensions simultaneously.
        """
        entity_results = self._entity_match(graph, keywords, inferred_relations, query_embedding, top_k)
        relation_results = self._relation_filter(graph, inferred_relations, keywords, query_embedding, top_k)
        neighbourhood_results = self._neighbourhood_search(graph, keywords, inferred_relations, query_embedding, top_k)

        # Merge and deduplicate by entity_id — keep highest score per entity
        merged: Dict[str, GraphSearchResult] = {}

        for result in entity_results + relation_results + neighbourhood_results:
            eid = result.entity.entity_id
            if eid not in merged or result.relevance_score > merged[eid].relevance_score:
                merged[eid] = result

        final_results = sorted(merged.values(), key=lambda r: r.relevance_score, reverse=True)

        return GraphSearchResponse(
            results=final_results[:top_k],
            strategy_used=GraphSearchStrategy.COMBINED,
            query_keywords=keywords,
            inferred_relation_types=list(inferred_relations),
        )

    # ------------------------------------------------------------------
    # Private path analysis helper
    # ------------------------------------------------------------------

    def _analyse_single_path(
        self,
        path_entity_ids: List[str],
        graph: TSDGraph,
    ) -> Optional[PathAnalysisResult]:
        """
        Audits a single path for security controls — auth and encryption
        on every edge.
        """
        if len(path_entity_ids) < 2:
            return None

        path_entities: List[GraphEntity] = []
        path_relations: List[GraphRelation] = []
        all_block_ids: List[str] = []
        has_auth = False
        has_encryption = False
        auth_on_all_edges = True
        encryption_on_all_edges = True

        for i, entity_id in enumerate(path_entity_ids):
            entity = graph.get_entity(entity_id)
            if not entity:
                return None
            path_entities.append(entity)
            all_block_ids.extend(entity.source_block_ids)

            if i < len(path_entity_ids) - 1:
                next_id = path_entity_ids[i + 1]
                if not NETWORKX_AVAILABLE or graph.graph is None:
                    continue
                if graph.graph.has_edge(entity_id, next_id):
                    edge_data = graph.graph[entity_id][next_id]
                    relation: Optional[GraphRelation] = edge_data.get("relation_obj")
                    if relation:
                        path_relations.append(relation)
                        all_block_ids.extend(relation.source_block_ids)

                        if relation.requires_auth is True:
                            has_auth = True
                        elif relation.requires_auth is False:
                            auth_on_all_edges = False

                        if relation.is_encrypted is True:
                            has_encryption = True
                        elif relation.is_encrypted is False:
                            encryption_on_all_edges = False

        # Deduplicate block_ids
        seen: Set[str] = set()
        unique_block_ids = [
            bid for bid in all_block_ids
            if not (bid in seen or seen.add(bid))
        ]

        return PathAnalysisResult(
            path_entities=path_entities,
            path_relations=path_relations,
            has_auth=has_auth,
            has_encryption=has_encryption,
            is_fully_secured=auth_on_all_edges and encryption_on_all_edges and has_auth and has_encryption,
            source_block_ids=unique_block_ids,
        )

    # ------------------------------------------------------------------
    # Private scoring and aggregation helpers
    # ------------------------------------------------------------------

    def _get_all_relations(
        self,
        graph: TSDGraph,
        entity_id: str,
    ) -> List[GraphRelation]:
        """
        Returns all outgoing GraphRelation objects from an entity.
        Used to build the full neighbourhood context for a GraphSearchResult.

        Args:
            graph:     The TSDGraph to query.
            entity_id: The source entity_id to retrieve relations for.

        Returns:
            List of GraphRelation instances from outgoing edges.
            Empty list if entity has no outgoing edges or graph is unavailable.
        """
        if not NETWORKX_AVAILABLE or graph.graph is None:
            return []

        if entity_id not in graph.graph:
            return []

        relations: List[GraphRelation] = []
        for _, _, edge_data in graph.graph.out_edges(entity_id, data=True):
            relation = edge_data.get("relation_obj")
            if relation is not None:
                relations.append(relation)

        return relations

    def _aggregate_block_ids(
        self,
        entity: GraphEntity,
        relations: List[GraphRelation],
    ) -> List[str]:
        """
        Aggregates and deduplicates source_block_ids from an entity and
        its relevant relations into a single ordered list.

        Used to build the source_block_ids on GraphSearchResult so
        analysis_service.py can retrieve the exact TSD text blocks that
        describe this entity and its relationships for citation tracing.

        Args:
            entity:    The GraphEntity whose block_ids to include.
            relations: The relations whose block_ids to include.

        Returns:
            Deduplicated list of block_id strings preserving insertion order.
        """
        seen: Set[str] = set()
        block_ids: List[str] = []

        # Entity block_ids first — they are the primary source
        for bid in entity.source_block_ids:
            if bid not in seen:
                block_ids.append(bid)
                seen.add(bid)

        # Relation block_ids second — may overlap with entity block_ids
        for relation in relations:
            for bid in relation.source_block_ids:
                if bid not in seen:
                    block_ids.append(bid)
                    seen.add(bid)

        return block_ids

    def _score_entity_relevance(
        self,
        entity: GraphEntity,
        relations: List[GraphRelation],
        keywords: List[str],
    ) -> float:
        """
        Scores an entity's relevance to the query on a 0.0–1.0 scale.

        Scoring signals (additive, capped at 1.0):
            +0.4  if any keyword appears in the entity name (primary signal)
            +0.2  if entity type is security-relevant (auth, database, api)
            +0.1  per security-critical relation present (capped at +0.3)
            +0.1  if any relation has is_encrypted=True
            +0.1  if any relation has requires_auth=True

        This heuristic scoring avoids an LLM call while still ranking
        entities by their likely relevance to security parameters.

        Args:
            entity:    The entity to score.
            relations: The entity's relevant outgoing relations.
            keywords:  Lowercase keyword strings from the parameter text.

        Returns:
            A float in [0.0, 1.0].
        """
        score = 0.0
        name_lower = entity.name.lower()

        # Keyword match in entity name — strongest signal
        if keywords and any(
            kw in name_lower
            for kw in keywords
            if len(kw) >= _MIN_KEYWORD_LENGTH
        ):
            score += 0.4

        # Security-relevant entity types get a boost
        _HIGH_VALUE_ENTITY_TYPES = frozenset({
            "auth_mechanism", "api", "database", "data_store", "service"
        })
        if entity.entity_type in _HIGH_VALUE_ENTITY_TYPES:
            score += 0.2

        # Security-critical relations — capped at 0.3
        critical_relation_count = sum(
            1 for r in relations
            if r.relation_type in _SECURITY_CRITICAL_RELATION_TYPES
        )
        score += min(critical_relation_count * 0.1, 0.3)

        # Encryption present
        if any(r.is_encrypted is True for r in relations):
            score += 0.1

        # Auth required
        if any(r.requires_auth is True for r in relations):
            score += 0.1

        return min(score, 1.0)

    def _ensure_relation_embeddings(
        self,
        *,
        graph: TSDGraph,
        relations: List[GraphRelation],
    ) -> None:
        if not relations:
            return

        pending_by_hash: Dict[str, Tuple[GraphRelation, str]] = {}
        for relation in relations:
            if relation is None or getattr(relation, "has_embedding", False):
                continue
            text = TSDGraphBuilder._relation_embedding_text(self, relation)
            if not text:
                continue
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = graph.object_embedding_cache.get(text_hash)
            if cached:
                relation.embedding = list(cached)
                relation.has_embedding = True
                continue
            if text_hash not in pending_by_hash:
                pending_by_hash[text_hash] = (relation, text)

        if not pending_by_hash:
            return

        ordered_hashes = list(pending_by_hash.keys())
        texts = [pending_by_hash[text_hash][1] for text_hash in ordered_hashes]
        vectors = get_embeddings(texts=texts)
        if len(vectors) != len(ordered_hashes):
            self.logger.warning(
                "GraphSearcher._ensure_relation_embeddings: vector count mismatch for %d pending relation(s); expected %d got %d.",
                len(relations),
                len(ordered_hashes),
                len(vectors),
            )
            vectors = [[] for _ in ordered_hashes]

        for text_hash, vector in zip(ordered_hashes, vectors):
            if not vector:
                continue
            graph.object_embedding_cache[text_hash] = list(vector)

        for relation in relations:
            if relation is None or getattr(relation, "has_embedding", False):
                continue
            text = TSDGraphBuilder._relation_embedding_text(self, relation)
            if not text:
                continue
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = graph.object_embedding_cache.get(text_hash)
            if cached:
                relation.embedding = list(cached)
                relation.has_embedding = True

    def _blend_scores(
        self,
        query_embedding: Optional[List[float]],
        entity: GraphEntity,
        relations: List[GraphRelation],
        heuristic_score: float,
    ) -> Tuple[float, float, float]:
        if not query_embedding:
            return 0.0, 0.0, heuristic_score

        entity_sim = self._cosine_similarity(query_embedding, getattr(entity, "embedding", []))
        relation_sims = [
            self._cosine_similarity(query_embedding, getattr(rel, "embedding", []))
            for rel in relations
            if getattr(rel, "embedding", None)
        ]
        relation_sim = max(relation_sims) if relation_sims else 0.0

        components: List[Tuple[float, float]] = [(heuristic_score, 0.55)]
        if entity_sim > 0.0:
            components.append((entity_sim, 0.30))
        if relation_sim > 0.0:
            components.append((relation_sim, 0.15))

        total_weight = sum(weight for _, weight in components)
        if total_weight <= 0:
            return entity_sim, relation_sim, heuristic_score

        blended = sum(value * weight for value, weight in components) / total_weight
        return entity_sim, relation_sim, min(1.0, max(0.0, blended))

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def _select_strategy(
        self,
        keywords: List[str],
        inferred_relations: Set[str],
    ) -> GraphSearchStrategy:
        """
        Automatically selects the most appropriate GraphSearchStrategy
        based on keyword and relation type analysis.

        Selection rules (in priority order):
            COMBINED        → many inferred relations AND many keywords
                              (complex cross-cutting parameter)
            RELATION_FILTER → inferred relations present but few keywords
                              (parameter is relation-type focused)
            NEIGHBOURHOOD   → keywords present but no inferred relations
                              (parameter names a specific component)
            ENTITY_MATCH    → default fallback

        Args:
            keywords:          Extracted keyword strings.
            inferred_relations: Inferred relation types from keywords.

        Returns:
            The selected GraphSearchStrategy enum value.
        """
        has_keywords = len(keywords) >= 2
        has_relations = len(inferred_relations) >= 1
        has_many_relations = len(inferred_relations) >= 3

        if has_keywords and has_many_relations:
            return GraphSearchStrategy.COMBINED

        if has_relations and not has_keywords:
            return GraphSearchStrategy.RELATION_FILTER

        if has_keywords and not has_relations:
            return GraphSearchStrategy.NEIGHBOURHOOD

        if has_relations:
            return GraphSearchStrategy.RELATION_FILTER

        return GraphSearchStrategy.ENTITY_MATCH


# ---------------------------------------------------------------------------
# Module-level pure utility functions
# ---------------------------------------------------------------------------

def _extract_keywords(parameter_text: str) -> List[str]:
    """
    Extracts meaningful lowercase keyword strings from a security
    parameter text for use in entity name matching and relation inference.

    Steps:
        1. Lowercase and split on whitespace and common punctuation.
        2. Remove stop words and tokens shorter than _MIN_KEYWORD_LENGTH.
        3. Deduplicate while preserving insertion order.

    Args:
        parameter_text: The full requirement text from CategoryParameterChild [3].

    Returns:
        A deduplicated list of lowercase keyword strings.
    """
    if not parameter_text:
        return []

    # Common English stop words that add no signal for graph traversal
    _STOP_WORDS = frozenset({
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must", "shall",
        "that", "this", "these", "those", "it", "its", "all", "any", "both",
        "each", "than", "too", "very", "can", "not", "no", "nor", "so",
        "yet", "both", "either", "neither", "such", "when", "where", "which",
        "who", "whom", "how", "what", "why", "if", "then", "than", "as",
    })

    import re
    # Split on whitespace and punctuation, lowercase everything
    tokens = re.split(r"[\s\.,;:!\?\(\)\[\]{}\"\'/\\]+", parameter_text.lower())

    seen: Set[str] = set()
    keywords: List[str] = []

    for token in tokens:
        token = token.strip()
        if (
            len(token) >= _MIN_KEYWORD_LENGTH
            and token not in _STOP_WORDS
            and token not in seen
        ):
            keywords.append(token)
            seen.add(token)

    return keywords


def _infer_relation_types_from_parameter(
    keywords: List[str],
) -> Set[str]:
    """
    Maps parameter keywords to relevant GraphRelation types without
    an LLM call, using the _KEYWORD_TO_RELATION_MAP lookup table.

    Matching is by prefix/substring — "authenticat" matches both
    "authenticates" and "authentication". This is intentional — it
    casts a wider net than exact matching for security terminology.

    Always includes _SECURITY_CRITICAL_RELATION_TYPES as a baseline
    so security-critical edges are never missed regardless of keywords.

    Args:
        keywords: Lowercase keyword strings from _extract_keywords().

    Returns:
        Set of relation type strings from _VALID_RELATION_TYPES.
    """
    inferred: Set[str] = set()

    for keyword in keywords:
        for prefix, relation_types in _KEYWORD_TO_RELATION_MAP.items():
            if prefix in keyword or keyword in prefix:
                inferred.update(relation_types)

    return inferred


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def search_graph(
    parameter_text: str,
    graph: TSDGraph,
    strategy: Optional[GraphSearchStrategy] = None,
    top_k: int = _DEFAULT_TOP_K,
) -> GraphSearchResponse:
    """
    Module-level convenience wrapper around GraphSearcher.search().

    Instantiates a GraphSearcher with default settings and executes
    a search. Used by retrieval/router.py for GRAPH_TRAVERSE strategy
    queries and by analysis_service.py for direct graph lookups.

    Args:
        parameter_text: The full requirement text from CategoryParameterChild [3].
        graph:          The TSDGraph built from the TSD document.
        strategy:       Optional explicit strategy. Auto-selected if None.
        top_k:          Maximum number of results.

    Returns:
        GraphSearchResponse — never raises.
    """
    searcher = GraphSearcher()
    return searcher.search(
        parameter_text=parameter_text,
        graph=graph,
        strategy=strategy,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "GraphSearchStrategy",
    "GraphTraversalConfig",
    # Result dataclasses — imported by analysis_service.py
    "GraphSearchResult",
    "PathAnalysisResult",
    "GraphSearchResponse",
    # Main searcher class
    "GraphSearcher",
    # Pure utilities — independently testable
    "_extract_keywords",
    "_infer_relation_types_from_parameter",
    # Convenience function — used by router.py
    "search_graph",
]
