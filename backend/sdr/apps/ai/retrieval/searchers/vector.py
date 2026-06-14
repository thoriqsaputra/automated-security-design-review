from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from sdr.core.database import SessionLocal
from sdr.apps.ai.client import get_embedding   # [4]
from sdr.apps.standards.models import (        # [3]
    CategoryParameterChild,
    CategoryParameterParent,
    CategoryParameterEmbedding,
    StandardCategory,
    StandardIngestionJob,
)
from sdr.apps.standards.utils import build_parameter_analysis_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# pgvector availability guard
# ---------------------------------------------------------------------------

try:
    # Check if pgvector is available for SQLAlchemy
    import pgvector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    logger.error(
        "pgvector is not installed. Vector search will not function. "
        "Install with: pip install pgvector"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K = 10
_MAX_TOP_K = 50
_EMBEDDING_DIMENSIONS = 1024
_MAX_COSINE_DISTANCE = 0.5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VectorSearchResult:
    child: CategoryParameterChild
    cosine_distance: float
    embedding_id: Optional[int] = None
    content_hash: Optional[str] = None

    @property
    def cosine_similarity(self) -> float:
        return max(0.0, 1.0 - self.cosine_distance)

    @property
    def is_relevant(self) -> bool:
        return self.cosine_distance <= _MAX_COSINE_DISTANCE


@dataclass
class VectorSearchResponse:
    results: List[VectorSearchResult] = field(default_factory=list)
    query_embedding: List[float] = field(default_factory=list)
    total_found: int = 0
    query_text: str = ""
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0

    def get_children(self) -> List[CategoryParameterChild]:
        return [r.child for r in self.results]

    def get_block_ids_by_child_stable_key(self) -> dict:
        return {
            r.child.stable_key: r.cosine_similarity
            for r in self.results
        }


# ---------------------------------------------------------------------------
# Vector Searcher
# ---------------------------------------------------------------------------

class VectorSearcher:
    def __init__(
        self,
        embedding_dimensions: int = _EMBEDDING_DIMENSIONS,
        max_cosine_distance: float = _MAX_COSINE_DISTANCE,
        default_top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        if not PGVECTOR_AVAILABLE:
            raise RuntimeError(
                "VectorSearcher requires pgvector. "
                "Install with: pip install pgvector"
            )

        self.embedding_dimensions = embedding_dimensions
        self.max_cosine_distance = max_cosine_distance
        self.default_top_k = default_top_k
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    def search(
        self,
        query_text: str,
        category: StandardCategory,
        top_k: Optional[int] = None,
        ingestion_job: Optional[StandardIngestionJob] = None,
        precomputed_embedding: Optional[List[float]] = None,
    ) -> VectorSearchResponse:
        if not query_text or not query_text.strip():
            msg = "query_text is empty — cannot perform vector search."
            self.logger.warning("VectorSearcher.search: %s", msg)
            return VectorSearchResponse(error=msg, query_text=query_text)

        resolved_top_k = min(
            top_k if top_k is not None else self.default_top_k,
            _MAX_TOP_K,
        )

        resolved_job = ingestion_job or self._get_active_job(category)

        if resolved_job is None:
            msg = (
                f"No active ingestion job found for category "
                f"'{category.code}' — no embeddings to search."
            )
            self.logger.warning("VectorSearcher.search: %s", msg)
            return VectorSearchResponse(error=msg, query_text=query_text)

        if precomputed_embedding:
            query_vector = precomputed_embedding
            self.logger.debug(
                "VectorSearcher.search: using precomputed embedding "
                "for query '%s...'",
                query_text[:60],
            )
        else:
            query_vector = self._embed_query(query_text)
            if not query_vector:
                msg = f"Failed to generate embedding for query '{query_text[:60]}...'."
                self.logger.error("VectorSearcher.search: %s", msg)
                return VectorSearchResponse(
                    error=msg,
                    query_text=query_text,
                )

        try:
            results = self._execute_search(
                query_vector=query_vector,
                category=category,
                ingestion_job=resolved_job,
                top_k=resolved_top_k,
            )
        except Exception as exc:
            msg = f"pgvector search query failed: {exc}"
            self.logger.error("VectorSearcher.search: %s", msg)
            return VectorSearchResponse(
                error=msg,
                query_text=query_text,
                query_embedding=query_vector,
            )

        relevant_results = [r for r in results if r.is_relevant]

        self.logger.info(
            "VectorSearcher.search: found %d result(s) (%d above threshold) "
            "for query '%s...' in category '%s'.",
            len(results),
            len(relevant_results),
            query_text[:60],
            category.code,
        )

        return VectorSearchResponse(
            results=relevant_results,
            query_embedding=query_vector,
            total_found=len(relevant_results),
            query_text=query_text,
            error=None,
        )

    def search_by_parameter(
        self,
        parameter: CategoryParameterChild,
        category: StandardCategory,
        top_k: Optional[int] = None,
        exclude_self: bool = True,
    ) -> VectorSearchResponse:
        response = self.search(
            query_text=build_parameter_analysis_text(parameter),
            category=category,
            top_k=(top_k or self.default_top_k) + (1 if exclude_self else 0),
        )

        if exclude_self and not response.is_empty:
            response.results = [
                r for r in response.results
                if r.child.stable_key != parameter.stable_key
            ]
            response.total_found = len(response.results)

        return response

    def _get_active_job(
        self,
        category: StandardCategory,
    ) -> Optional[StandardIngestionJob]:
        with SessionLocal() as db:
            qs = (
                select(StandardIngestionJob)
                .where(
                    StandardIngestionJob.category_id == category.id,
                    StandardIngestionJob.is_active == True
                )
                .order_by(StandardIngestionJob.created_at.desc())
            )
            return db.execute(qs).scalars().first()

    def _embed_query(self, query_text: str) -> List[float]:
        try:
            vector = get_embedding(
                text=query_text,
                dimensions=self.embedding_dimensions,
            )

            if not vector:
                self.logger.error(
                    "VectorSearcher._embed_query: get_embedding() returned "
                    "empty vector for query '%s...'",
                    query_text[:60],
                )
                return []

            return vector

        except Exception as exc:
            self.logger.error(
                "VectorSearcher._embed_query: unexpected error: %s",
                exc,
            )
            return []

    def _execute_search(
        self,
        query_vector: List[float],
        category: StandardCategory,
        ingestion_job: StandardIngestionJob,
        top_k: int,
    ) -> List[VectorSearchResult]:
        
        param_type = getattr(CategoryParameterEmbedding, 'TYPE_CHILD', 'child')

        with SessionLocal() as db:
            distance_expr = CategoryParameterEmbedding.embedding.cosine_distance(query_vector).label('distance')
            
            qs = (
                select(CategoryParameterEmbedding, distance_expr)
                .options(
                    joinedload(CategoryParameterEmbedding.child)
                    .joinedload(CategoryParameterChild.parent)
                    .joinedload(CategoryParameterParent.category)
                )
                .join(CategoryParameterChild, CategoryParameterEmbedding.child_id == CategoryParameterChild.id)
                .join(CategoryParameterParent, CategoryParameterChild.parent_id == CategoryParameterParent.id)
                .where(
                    CategoryParameterEmbedding.parameter_type == param_type,
                    CategoryParameterEmbedding.is_active == True,
                    CategoryParameterParent.category_id == category.id,
                    CategoryParameterParent.ingestion_job_id == ingestion_job.id,
                )
                .order_by('distance')
                .limit(top_k)
            )

            rows = db.execute(qs).all()

            results: List[VectorSearchResult] = []

            for embedding_record, distance in rows:
                results.append(
                    VectorSearchResult(
                        child=embedding_record.child,
                        cosine_distance=float(distance),
                        embedding_id=embedding_record.id,
                        content_hash=embedding_record.content_hash,
                    )
                )

            self.logger.debug(
                "VectorSearcher._execute_search: raw query returned %d row(s) "
                "for category='%s' job_id=%s top_k=%d.",
                len(results),
                category.code,
                ingestion_job.id,
                top_k,
            )

            return results


def search_similar_parameters(
    query_text: str,
    category: StandardCategory,
    top_k: int = _DEFAULT_TOP_K,
    ingestion_job: Optional[StandardIngestionJob] = None,
    precomputed_embedding: Optional[List[float]] = None,
) -> VectorSearchResponse:
    searcher = VectorSearcher()
    return searcher.search(
        query_text=query_text,
        category=category,
        top_k=top_k,
        ingestion_job=ingestion_job,
        precomputed_embedding=precomputed_embedding,
    )

__all__ = [
    "VectorSearchResult",
    "VectorSearchResponse",
    "VectorSearcher",
    "search_similar_parameters",
]
