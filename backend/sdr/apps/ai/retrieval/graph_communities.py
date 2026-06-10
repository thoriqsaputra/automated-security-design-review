from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.tsd_processing.graph_builder import TSDGraph

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    NETWORKX_AVAILABLE = True
except Exception:
    NETWORKX_AVAILABLE = False


@dataclass
class GraphCommunity:
    community_id: str
    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
    block_ids: List[str] = field(default_factory=list)
    title: Optional[str] = None
    summary: Optional[str] = None
    sensitivity: str = "internal"
    tenant_id: Optional[str] = None


@dataclass
class CommunitySummary:
    community_id: str
    title: str
    summary: str
    key_entities: List[str] = field(default_factory=list)
    key_relationships: List[str] = field(default_factory=list)
    block_ids: List[str] = field(default_factory=list)
    sensitivity: str = "internal"
    source_metadata: Dict[str, Any] = field(default_factory=dict)


class GraphCommunityService:
    def __init__(self, enable_llm_summary: bool = False) -> None:
        self.enable_llm_summary = enable_llm_summary

    def detect_communities(self, graph: TSDGraph) -> List[GraphCommunity]:
        if not NETWORKX_AVAILABLE or graph is None or graph.graph is None or graph.is_empty():
            return []

        undirected = graph.graph.to_undirected()
        communities: List[Set[str]] = []

        # Prefer Leiden/Louvain if available (import guarded), fallback to networkx greedy modularity.
        try:
            import leidenalg  # type: ignore
            import igraph as ig  # type: ignore
            node_ids = list(undirected.nodes())
            idx = {n: i for i, n in enumerate(node_ids)}
            ig_graph = ig.Graph()
            ig_graph.add_vertices(len(node_ids))
            ig_graph.add_edges([(idx[u], idx[v]) for u, v in undirected.edges()])
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
            communities = [{node_ids[i] for i in part} for part in partition]
        except Exception:
            try:
                import community as community_louvain  # type: ignore
                partition = community_louvain.best_partition(undirected)
                bucket: Dict[int, Set[str]] = {}
                for node_id, cid in partition.items():
                    bucket.setdefault(cid, set()).add(node_id)
                communities = list(bucket.values())
            except Exception:
                communities = [set(c) for c in greedy_modularity_communities(undirected)]

        out: List[GraphCommunity] = []
        for idx, node_set in enumerate(communities, start=1):
            node_ids = sorted(node_set)
            edge_ids: List[str] = []
            block_ids: List[str] = []
            sensitivity = "public"
            tenant_id: Optional[str] = None

            for node_id in node_ids:
                entity = graph.get_entity(node_id)
                if entity is None:
                    continue
                block_ids.extend(entity.source_block_ids)
                sensitivity = _max_sensitivity(sensitivity, entity.sensitivity)
                if tenant_id is None:
                    tenant_id = entity.tenant_id
                elif entity.tenant_id and tenant_id != entity.tenant_id:
                    tenant_id = None

            for source_id, target_id in graph.graph.edges():
                if source_id in node_set and target_id in node_set:
                    edge_ids.append(f"{source_id}->{target_id}")
                    relation = graph.graph[source_id][target_id].get("relation_obj")
                    if relation is not None:
                        block_ids.extend(relation.source_block_ids)
                        sensitivity = _max_sensitivity(sensitivity, relation.sensitivity)

            unique_block_ids = list(dict.fromkeys(block_ids))
            title = self._derive_title(graph, node_ids)
            out.append(
                GraphCommunity(
                    community_id=f"community_{idx}",
                    node_ids=node_ids,
                    edge_ids=edge_ids,
                    block_ids=unique_block_ids,
                    title=title,
                    sensitivity=sensitivity,
                    tenant_id=tenant_id,
                )
            )
        return out

    def summarize_communities(self, graph: TSDGraph, communities: List[GraphCommunity]) -> List[CommunitySummary]:
        summaries: List[CommunitySummary] = []
        for community in communities:
            fallback = self._fallback_summary(graph, community)
            title = fallback["title"]
            summary_text = fallback["summary"]
            if self.enable_llm_summary:
                llm_summary = self._llm_summary(graph, community, fallback)
                if llm_summary:
                    title = llm_summary.get("title", title)
                    summary_text = llm_summary.get("summary", summary_text)
            summaries.append(
                CommunitySummary(
                    community_id=community.community_id,
                    title=title,
                    summary=summary_text,
                    key_entities=fallback["key_entities"],
                    key_relationships=fallback["key_relationships"],
                    block_ids=list(community.block_ids),
                    sensitivity=community.sensitivity,
                    source_metadata={
                        "edge_ids": list(community.edge_ids),
                        "tenant_id": community.tenant_id,
                    },
                )
            )
        return summaries

    def _derive_title(self, graph: TSDGraph, node_ids: List[str]) -> str:
        names = []
        for node_id in node_ids[:3]:
            entity = graph.get_entity(node_id)
            if entity:
                names.append(entity.name)
        return " / ".join(names) if names else "Architecture Community"

    def _fallback_summary(self, graph: TSDGraph, community: GraphCommunity) -> Dict[str, Any]:
        names: List[str] = []
        for node_id in community.node_ids:
            entity = graph.get_entity(node_id)
            if entity:
                names.append(entity.name)

        edge_rels: List[str] = []
        for edge_id in community.edge_ids:
            source_id, target_id = edge_id.split("->", 1)
            if graph.graph and graph.graph.has_edge(source_id, target_id):
                relation = graph.graph[source_id][target_id].get("relation_obj")
                if relation:
                    edge_rels.append(relation.relation_type)

        relation_counter = Counter(edge_rels)
        top_rels = [rel for rel, _ in relation_counter.most_common(4)]

        title = community.title or "Architecture Community"
        summary = (
            f"Community with key entities: {', '.join(names[:5]) or 'N/A'}. "
            f"Frequent relationships: {', '.join(top_rels) or 'N/A'}. "
            f"Linked source blocks: {', '.join(community.block_ids[:5]) or 'N/A'}."
        )

        return {
            "title": title,
            "summary": summary,
            "key_entities": names[:8],
            "key_relationships": top_rels,
        }

    def _llm_summary(self, graph: TSDGraph, community: GraphCommunity, fallback: Dict[str, Any]) -> Optional[Dict[str, str]]:
        try:
            prompt = (
                "Summarize this architecture community in concise report style JSON. "
                "Return keys: title, summary.\n"
                f"Entities: {fallback['key_entities']}\n"
                f"Relationships: {fallback['key_relationships']}\n"
                f"Blocks: {community.block_ids[:10]}"
            )
            response = chat_completion(
                messages=[
                    {"role": "system", "content": "You are a software security architecture summarizer. Output strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                component="coding_graph",
                temperature=0.1,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            if response.error or not response.content:
                return None
            import json
            parsed = json.loads(response.content)
            if not isinstance(parsed, dict):
                return None
            return {
                "title": str(parsed.get("title") or "").strip() or fallback["title"],
                "summary": str(parsed.get("summary") or "").strip() or fallback["summary"],
            }
        except Exception:
            logger.exception("GraphCommunityService._llm_summary failed")
            return None


def _max_sensitivity(current: str, incoming: Optional[str]) -> str:
    rank = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
    incoming_value = incoming or "internal"
    return current if rank.get(current, 1) >= rank.get(incoming_value, 1) else incoming_value


__all__ = [
    "GraphCommunity",
    "CommunitySummary",
    "GraphCommunityService",
]
