import logging
from typing import Dict, List, Any

from sdr.core.config import settings
from sdr.core.database import SessionLocal
from sdr.apps.ai.client import get_ai_service, get_embeddings
from sdr.apps.standards.models import (
    CategoryDiagramRequirementEmbedding,
)

logger = logging.getLogger(__name__)

_EMBEDDING_DIMENSIONS = 1024                               
                                
# ---------------------------------------------------------------------------
# Batch embedding helper
# ---------------------------------------------------------------------------

def get_default_embedding_model_name() -> str:
    """
    Return the same default embedding model used by client.get_embedding().
    """
    service = get_ai_service()
    model_name = getattr(service, "model_embedding", None) if service else None
    return model_name or getattr(settings, "AI_MODEL_EMBEDDING", "mxbai-embed-large-v1")

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    return get_embeddings(texts=texts, dimensions=_EMBEDDING_DIMENSIONS)


def generate_and_store_diagram_requirement_embeddings(
    items_to_embed: List[Dict[str, Any]],
    job_id: str,
    summary: Dict[str, Any],
) -> None:
    try:
        texts: List[str] = [item["text"] for item in items_to_embed]

        logger.info(
            "generate_and_store_diagram_requirement_embeddings: requesting %d embedding(s) for job=%s.",
            len(texts),
            job_id,
        )

        vectors: List[List[float]] = generate_embeddings_batch(texts)
        embedding_model_name = get_default_embedding_model_name()

        if len(vectors) != len(items_to_embed):
            logger.error(
                "generate_and_store_diagram_requirement_embeddings: vector count mismatch "
                "(expected %d, got %d) for job=%s. Aborting embedding phase.",
                len(items_to_embed),
                len(vectors),
                job_id,
            )
            summary["diagram_requirement_embeddings_failed"] += len(items_to_embed)
            return

        embedding_objects: List[CategoryDiagramRequirementEmbedding] = []

        for item, vector in zip(items_to_embed, vectors):
            if not vector:
                logger.warning(
                    "generate_and_store_diagram_requirement_embeddings: skipping diagram_requirement_id=%s "
                    "for job=%s because embedding returned empty.",
                    item.get("diagram_requirement_id"),
                    job_id,
                )
                summary["diagram_requirement_embeddings_failed"] += 1
                continue

            embedding_objects.append(
                CategoryDiagramRequirementEmbedding(
                    diagram_requirement_id=item.get("diagram_requirement_id"),
                    model_name=embedding_model_name,
                    model_dim=len(vector),
                    embedding=vector,
                    content_hash=item["content_hash"],
                    is_active=True,
                )
            )

        if not embedding_objects:
            logger.warning(
                "generate_and_store_diagram_requirement_embeddings: no valid embeddings to persist for job=%s.",
                job_id,
            )
            return

        with SessionLocal() as db:
            db.add_all(embedding_objects)
            db.commit()
            created_count = len(embedding_objects)

        summary["diagram_requirement_embeddings_created"] += created_count

        logger.info(
            "generate_and_store_diagram_requirement_embeddings: persisted %d embedding(s) for job=%s (%d failed/skipped).",
            created_count,
            job_id,
            summary["diagram_requirement_embeddings_failed"],
        )

    except Exception as exc:
        logger.exception(
            "generate_and_store_diagram_requirement_embeddings: unexpected error during embedding phase "
            "for job=%s — ingestion result is unaffected. Error: %s",
            job_id,
            exc,
        )
        already_counted = (
            summary["diagram_requirement_embeddings_created"]
            + summary["diagram_requirement_embeddings_failed"]
        )
        summary["diagram_requirement_embeddings_failed"] += len(items_to_embed) - already_counted
