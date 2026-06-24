from __future__ import annotations

from typing import List

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate
from sdr.apps.ai.retrieval.searchers.graph import GraphSearchResponse
from sdr.apps.ai.retrieval.searchers.vector import VectorSearchResponse


def build_chunks_from_vector(vector_response: VectorSearchResponse) -> List[str]:
    chunks: List[str] = []
    for idx, result in enumerate(vector_response.results, start=1):
        child = result.child
        section = child.parent.title if hasattr(child, "parent") and child.parent else "Unknown Section"
        similarity_pct = round(result.cosine_similarity * 100, 1)
        chunk = (
            f"--- VECTOR RESULT {idx} "
            f"[similarity={similarity_pct}%] "
            f"[section={section}] ---\n\n"
            f"{child.requirement_text}"
        )
        chunks.append(chunk)
    return chunks


def build_chunks_from_graph(graph_response: GraphSearchResponse) -> List[str]:
    chunks: List[str] = []
    for idx, result in enumerate(graph_response.results, start=1):
        entity = result.entity
        lines = [
            f"--- GRAPH RESULT {idx} "
            f"[strategy={graph_response.strategy_used.value}] "
            f"[entity={entity.name}] "
            f"[type={entity.entity_type}] ---",
        ]
        if entity.description:
            lines.append(f"Description: {entity.description}")
        if result.relevant_relations:
            lines.append("Security-relevant relationships:")
            for rel in result.relevant_relations:
                security_flags: List[str] = []
                if rel.is_encrypted is True:
                    security_flags.append("encrypted=yes")
                elif rel.is_encrypted is False:
                    security_flags.append("encrypted=NO")
                if rel.requires_auth is True:
                    security_flags.append("auth=yes")
                elif rel.requires_auth is False:
                    security_flags.append("auth=NO")
                if rel.protocol:
                    security_flags.append(f"protocol={rel.protocol}")
                flags_str = f" [{', '.join(security_flags)}]" if security_flags else ""
                lines.append(
                    f"  → {rel.relation_type} → {rel.target_entity_id}: "
                    f"{(rel.description or rel.relation_type)}{flags_str}"
                )
        chunks.append("\n".join(lines))

    for idx, path_result in enumerate(graph_response.path_results, start=1):
        lines = [
            f"--- GRAPH PATH {idx} [length={path_result.length}] [secured={path_result.is_fully_secured}] ---",
            f"Path: {' → '.join(path_result.entity_names)}",
            f"Has authentication: {path_result.has_auth}",
            f"Has encryption: {path_result.has_encryption}",
            f"Fully secured: {path_result.is_fully_secured}",
        ]
        if path_result.path_relations:
            lines.append("Edge details:")
            for rel in path_result.path_relations:
                flags: List[str] = []
                if rel.is_encrypted is not None:
                    flags.append(f"encrypted={'yes' if rel.is_encrypted else 'NO'}")
                if rel.requires_auth is not None:
                    flags.append(f"auth={'yes' if rel.requires_auth else 'NO'}")
                if rel.protocol:
                    flags.append(f"protocol={rel.protocol}")
                flags_str = f" [{', '.join(flags)}]" if flags else ""
                lines.append(f"  {rel.source_entity_id} --[{rel.relation_type}]--> {rel.target_entity_id}{flags_str}")
        chunks.append("\n".join(lines))
    return chunks


def collect_block_ids_from_vector(vector_response: VectorSearchResponse) -> List[str]:
    return []


def build_chunk_block_ids_from_graph(graph_response: GraphSearchResponse) -> List[List[str]]:
    """Per-chunk source block ids, index-aligned with build_chunks_from_graph()."""
    block_ids: List[List[str]] = []
    for result in graph_response.results:
        block_ids.append(list(result.source_block_ids))
    for path_result in graph_response.path_results:
        block_ids.append(list(path_result.source_block_ids))
    return block_ids


def graph_response_to_candidates(graph_response: GraphSearchResponse) -> List[RetrievalCandidate]:
    candidates: List[RetrievalCandidate] = []
    for idx, result in enumerate(graph_response.results):
        entity = result.entity
        text_lines = [f"GRAPH NODE: {entity.name} ({entity.entity_type})"]
        for rel in result.relevant_relations:
            text_lines.append(f"{rel.source_entity_id} --{rel.relation_type}--> {rel.target_entity_id}")
        text = "\n".join(text_lines)
        candidates.append(
            RetrievalCandidate(
                id=f"graph:{entity.entity_id}:{idx}",
                source_type="graph",
                text=text,
                score=float(result.relevance_score),
                block_ids=list(result.source_block_ids),
                metadata={
                    "graph_node_id": entity.entity_id,
                    "graph_edge_ids": [f"{r.source_entity_id}->{r.target_entity_id}" for r in result.relevant_relations],
                    "grounded_texts": (entity.grounded_texts or []),
                    "entity_similarity": float(getattr(result, "entity_similarity", 0.0)),
                    "relation_similarity": float(getattr(result, "relation_similarity", 0.0)),
                    "blended_score": float(getattr(result, "blended_score", result.relevance_score)),
                    "sensitivity": entity.sensitivity,
                    "tenant_id": entity.tenant_id,
                },
                token_count=max(1, len(text) // 4),
            )
        )
    return candidates


__all__ = [
    "build_chunk_block_ids_from_graph",
    "build_chunks_from_graph",
    "build_chunks_from_vector",
    "collect_block_ids_from_vector",
    "graph_response_to_candidates",
]
