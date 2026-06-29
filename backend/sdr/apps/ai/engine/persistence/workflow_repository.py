from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import joinedload

from sdr.apps.reviews.models import Review
from sdr.apps.standards.models import (
    CategoryDiagramRequirementEmbedding,
    CategoryDiagramRequirement,
    CategoryParameterChild,
    CategoryParameterParent,
    StandardIngestionJob,
)
from sdr.core import database as core_database


class ReviewWorkflowRepository(ABC):
    @abstractmethod
    def get_latest_review(self, review_id: Any) -> Optional[Review]:
        raise NotImplementedError

    @abstractmethod
    def mark_review_running(self, review_id: Any, *, status: str, started_at: datetime) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_review_overview(self, review_id: Any, *, overview: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_summary_snapshot(self, review_id: Any, *, summary: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_retrieval_snapshot(self, review_id: Any, *, snapshot: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_review_completed(
        self,
        review_id: Any,
        *,
        status: str,
        completed_at: datetime,
        summary: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_review_failed(
        self,
        review_id: Any,
        *,
        status: str,
        completed_at: datetime,
        error_message: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_latest_active_ingestion_job(self, category_id: Any) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def list_category_parameters(self, *, category_id: Any, ingestion_job_id: Any) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def list_diagram_requirements(
        self,
        *,
        category_id: Any,
        ingestion_job_id: Any,
    ) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def search_diagram_requirements(
        self,
        *,
        category_id: Any,
        ingestion_job_id: Any,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Any]:
        raise NotImplementedError


class SqlAlchemyReviewWorkflowRepository(ReviewWorkflowRepository):
    def get_latest_review(self, review_id: Any) -> Optional[Review]:
        with core_database.SessionLocal() as db:
            return db.execute(select(Review).where(Review.id == review_id)).scalars().first()

    def mark_review_running(self, review_id: Any, *, status: str, started_at: datetime) -> None:
        with core_database.SessionLocal() as db:
            db.execute(
                update(Review).where(Review.id == review_id).values(
                    status=status,
                    started_at=started_at,
                )
            )
            db.commit()

    def save_review_overview(self, review_id: Any, *, overview: str) -> None:
        with core_database.SessionLocal() as db:
            db.execute(update(Review).where(Review.id == review_id).values(overview=overview))
            db.commit()

    def save_summary_snapshot(self, review_id: Any, *, summary: Dict[str, Any]) -> None:
        with core_database.SessionLocal() as db:
            db.execute(update(Review).where(Review.id == review_id).values(summary_json=summary))
            db.commit()

    def save_retrieval_snapshot(self, review_id: Any, *, snapshot: Dict[str, Any]) -> None:
        with core_database.SessionLocal() as db:
            db.execute(
                update(Review).where(Review.id == review_id).values(retrieval_snapshot_json=snapshot)
            )
            db.commit()

    def mark_review_completed(
        self,
        review_id: Any,
        *,
        status: str,
        completed_at: datetime,
        summary: Dict[str, Any],
    ) -> None:
        with core_database.SessionLocal() as db:
            db.execute(
                update(Review).where(Review.id == review_id).values(
                    status=status,
                    completed_at=completed_at,
                    summary_json=summary,
                )
            )
            db.commit()

    def mark_review_failed(
        self,
        review_id: Any,
        *,
        status: str,
        completed_at: datetime,
        error_message: str,
    ) -> None:
        with core_database.SessionLocal() as db:
            db.execute(
                update(Review).where(Review.id == review_id).values(
                    status=status,
                    completed_at=completed_at,
                    error_message=error_message,
                )
            )
            db.commit()

    def get_latest_active_ingestion_job(self, category_id: Any) -> Optional[Any]:
        with core_database.SessionLocal() as db:
            return db.execute(
                select(StandardIngestionJob)
                .where(
                    StandardIngestionJob.category_id == category_id,
                    StandardIngestionJob.is_active == True,
                )
                .order_by(StandardIngestionJob.created_at.desc())
            ).scalars().first()

    def list_category_parameters(self, *, category_id: Any, ingestion_job_id: Any) -> List[Any]:
        with core_database.SessionLocal() as db:
            return db.execute(
                select(CategoryParameterChild)
                .options(
                    joinedload(CategoryParameterChild.parent).joinedload(
                        CategoryParameterParent.category
                    )
                )
                .join(CategoryParameterParent, CategoryParameterChild.parent_id == CategoryParameterParent.id)
                .where(
                    CategoryParameterParent.category_id == category_id,
                    CategoryParameterParent.ingestion_job_id == ingestion_job_id,
                    CategoryParameterChild.requirement_category == "design",
                    CategoryParameterParent.is_active == True,
                    CategoryParameterChild.is_active == True,
                )
                .order_by(CategoryParameterParent.title, CategoryParameterChild.ordinal)
            ).scalars().all()

    def list_diagram_requirements(
        self,
        *,
        category_id: Any,
        ingestion_job_id: Any,
    ) -> List[Any]:
        with core_database.SessionLocal() as db:
            return db.execute(
                select(CategoryDiagramRequirement)
                .where(
                    CategoryDiagramRequirement.category_id == category_id,
                    CategoryDiagramRequirement.ingestion_job_id == ingestion_job_id,
                )
                .order_by(CategoryDiagramRequirement.ordinal)
            ).scalars().all()

    def search_diagram_requirements(
        self,
        *,
        category_id: Any,
        ingestion_job_id: Any,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Any]:
        with core_database.SessionLocal() as db:
            distance_expr = (
                CategoryDiagramRequirementEmbedding.embedding.cosine_distance(query_embedding)
                .label("distance")
            )
            rows = db.execute(
                select(CategoryDiagramRequirement, distance_expr)
                .join(
                    CategoryDiagramRequirementEmbedding,
                    CategoryDiagramRequirementEmbedding.diagram_requirement_id
                    == CategoryDiagramRequirement.id,
                )
                .where(
                    CategoryDiagramRequirement.category_id == category_id,
                    CategoryDiagramRequirement.ingestion_job_id == ingestion_job_id,
                    CategoryDiagramRequirementEmbedding.is_active == True,
                )
                .order_by("distance", CategoryDiagramRequirement.ordinal)
                .limit(top_k)
            ).all()
            return [row[0] for row in rows]
