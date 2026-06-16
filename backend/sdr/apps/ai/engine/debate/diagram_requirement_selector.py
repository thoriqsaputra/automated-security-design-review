from __future__ import annotations

import logging
from typing import Any, List

from sdr.apps.ai.client import get_embedding


class DiagramRequirementSelector:
    def __init__(self, *, config, workflow_repository) -> None:
        self.config = config
        self.workflow_repository = workflow_repository
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def select_for_diagram(
        self,
        *,
        diagram,
        tsd_document,
        category,
        ingestion_job,
        effective_asvs_level: int,
    ) -> List[Any]:
        top_k = self.config.vision_diagram_requirements_max_items
        query_text = self._build_query_text(diagram=diagram, tsd_document=tsd_document)
        if not query_text:
            return self._fallback_requirements(
                category_id=category.id,
                ingestion_job_id=ingestion_job.id,
                effective_asvs_level=effective_asvs_level,
                top_k=top_k,
            )

        try:
            query_vector = get_embedding(text=query_text, dimensions=1024)
        except Exception as exc:
            self.logger.warning(
                "DiagramRequirementSelector.select_for_diagram: embedding failed for diagram_id=%s: %s",
                getattr(diagram, "diagram_id", None),
                exc,
            )
            query_vector = []

        if query_vector:
            try:
                requirements = self.workflow_repository.search_diagram_requirements(
                    category_id=category.id,
                    ingestion_job_id=ingestion_job.id,
                    effective_asvs_level=effective_asvs_level,
                    query_embedding=query_vector,
                    top_k=top_k,
                )
            except Exception as exc:
                self.logger.warning(
                    "DiagramRequirementSelector.select_for_diagram: vector search failed for diagram_id=%s: %s",
                    getattr(diagram, "diagram_id", None),
                    exc,
                )
                requirements = []
            if requirements:
                return requirements

        return self._fallback_requirements(
            category_id=category.id,
            ingestion_job_id=ingestion_job.id,
            effective_asvs_level=effective_asvs_level,
            top_k=top_k,
        )

    def _build_query_text(self, *, diagram, tsd_document) -> str:
        parts = []
        caption = (getattr(diagram, "caption", "") or "").strip()
        surrounding = (getattr(diagram, "surrounding_text", "") or "").strip()
        if caption:
            parts.append(caption)
        if surrounding:
            parts.append(surrounding)

        if not parts:
            page_number = getattr(diagram, "page_number", None)
            pages = getattr(tsd_document, "pages", None) or []
            if isinstance(page_number, int) and 1 <= page_number <= len(pages):
                page_text = (getattr(pages[page_number - 1], "all_text", "") or "").strip()
                if page_text:
                    parts.append(page_text[:1200])

        return "\n\n".join(parts).strip()

    def _fallback_requirements(
        self,
        *,
        category_id: Any,
        ingestion_job_id: Any,
        effective_asvs_level: int,
        top_k: int,
    ) -> List[Any]:
        requirements = self.workflow_repository.list_diagram_requirements(
            category_id=category_id,
            ingestion_job_id=ingestion_job_id,
            effective_asvs_level=effective_asvs_level,
        )
        return list(requirements[:top_k])
