from __future__ import annotations

from typing import List

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


def collect_block_ids_from_vector(vector_response: VectorSearchResponse) -> List[str]:
    return []


__all__ = [
    "build_chunks_from_vector",
    "collect_block_ids_from_vector",
]
