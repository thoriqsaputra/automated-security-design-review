from __future__ import annotations

from typing import List, Optional, Set

from sdr.apps.ai.retrieval.core.types import QueryType, RetrievalStrategy
from sdr.apps.ai.tsd_processing.graph_builder import TSDGraph, _normalise_entity_id
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree

try:
    import networkx as nx
except Exception:
    nx = None


class RetrievalStrategySelector:
    def select_strategy(
        self,
        *,
        advanced_config,
        query_type: QueryType,
        keywords: List[str],
        inferred_relations: set,
        query_entities: List[str],
        raptor_tree: Optional[RAPTORTree],
        graph: Optional[TSDGraph],
        has_community_summaries,
    ) -> RetrievalStrategy:
        has_raptor = raptor_tree is not None and not raptor_tree.is_empty()
        has_graph = graph is not None and not graph.is_empty()
        has_many_keywords = len(keywords) >= 4

        if query_type == QueryType.FACT_BASED:
            if has_raptor:
                return RetrievalStrategy.HYBRID
            return RetrievalStrategy.VECTOR_ONLY

        if query_type == QueryType.MULTI_HOP_SECURITY and has_graph:
            if advanced_config.enable_ir_cot:
                return RetrievalStrategy.IR_COT_GRAPH
            return RetrievalStrategy.GRAPH_LOCAL

        if query_type == QueryType.REASONING_BASED and has_graph:
            if self.graph_entities_are_close(graph, query_entities):
                return RetrievalStrategy.GRAPH_LOCAL
            if has_raptor:
                return RetrievalStrategy.HYBRID
            return RetrievalStrategy.GRAPH_TRAVERSE

        if query_type == QueryType.GLOBAL_ARCHITECTURAL:
            if has_graph and advanced_config.enable_graph_global and has_community_summaries(graph):
                if advanced_config.enable_ir_cot:
                    return RetrievalStrategy.IR_COT_GRAPH
                return RetrievalStrategy.GRAPH_GLOBAL
            if has_raptor:
                return RetrievalStrategy.HYBRID

        if has_raptor and has_many_keywords:
            return RetrievalStrategy.RAPTOR_HIGH
        if has_raptor:
            return RetrievalStrategy.RAPTOR_LOW
        return RetrievalStrategy.VECTOR_ONLY

    def extract_query_entities(self, query_text: str, keyword_extractor) -> List[str]:
        keywords = keyword_extractor(query_text)
        entities: List[str] = []
        seen: Set[str] = set()
        for keyword in keywords:
            normalized = _normalise_entity_id(keyword)
            if normalized and normalized not in seen:
                seen.add(normalized)
                entities.append(normalized)
        return entities

    def classify_query_type(self, query_text: str, keywords: List[str], inferred_relations: Set[str]) -> QueryType:
        text = query_text.lower()
        global_markers = (
            "across all", "overall", "end-to-end", "entire architecture", "global",
            "how does the architecture", "main security risks", "trust boundaries", "dependencies",
        )
        multi_hop_markers = (
            "bypass", "trace", "audit this request path", "leak", "reach the", "request path",
            "permissions", "tenant data",
        )
        reasoning_markers = ("between", "path", "flow", "across", "relationship", "all services")
        if any(marker in text for marker in global_markers):
            return QueryType.GLOBAL_ARCHITECTURAL
        if any(marker in text for marker in multi_hop_markers):
            return QueryType.MULTI_HOP_SECURITY
        if any(marker in text for marker in reasoning_markers) or len(inferred_relations) >= 1:
            return QueryType.REASONING_BASED
        if len(keywords) <= 4:
            return QueryType.FACT_BASED
        return QueryType.REASONING_BASED

    def graph_entities_are_close(self, graph: Optional[TSDGraph], query_entities: List[str]) -> bool:
        if graph is None or graph.is_empty() or len(query_entities) < 2 or graph.graph is None or nx is None:
            return False
        matching_ids: List[str] = []
        for entity_name in query_entities:
            normalized = _normalise_entity_id(entity_name)
            if normalized in graph.entities:
                matching_ids.append(normalized)
                continue
            matches = graph.find_entities_by_name_fragment(entity_name)
            if matches:
                matching_ids.append(matches[0].entity_id)
        if len(matching_ids) < 2:
            return False
        for i, source_id in enumerate(matching_ids):
            for target_id in matching_ids[i + 1 :]:
                try:
                    distance = nx.shortest_path_length(graph.graph.to_undirected(), source=source_id, target=target_id)
                    if distance <= 2:
                        return True
                except Exception:
                    continue
        return False


__all__ = ["RetrievalStrategySelector"]
