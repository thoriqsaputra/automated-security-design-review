from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

from sdr.apps.ai.agents.vision import DiagramInput

from sdr.apps.ai.engine.config import AnalysisPipelineConfig
from sdr.apps.ai.engine.debate.diagram_requirement_selector import DiagramRequirementSelector


class DiagramAnalysisCoordinator:
    def __init__(
        self,
        *,
        config: AnalysisPipelineConfig,
        workflow_repository,
        diagram_debate_service,
        persistence_service,
        requirement_selector=None,
    ) -> None:
        self.config = config
        self.workflow_repository = workflow_repository
        self.diagram_debate_service = diagram_debate_service
        self.persistence = persistence_service
        self.requirement_selector = requirement_selector or DiagramRequirementSelector(
            config=config,
            workflow_repository=workflow_repository,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run(
        self,
        *,
        review,
        tsd_document,
        category,
        ingestion_job,
        summary,
        cancel_check=None,
    ) -> None:
        if not self.config.vision_diagram_analysis_enabled:
            self.logger.info("DiagramAnalysisCoordinator.run: diagram analysis disabled — skipping")
            return
        if not self.config.vision_enabled:
            self.logger.info("DiagramAnalysisCoordinator.run: vision disabled — skipping")
            return

        all_diagram_blocks = getattr(tsd_document, "all_diagrams", []) or []
        if not all_diagram_blocks:
            self.logger.info("DiagramAnalysisCoordinator.run: no diagrams in TSD — skipping")
            return

        eligible_diagrams: List[DiagramInput] = []
        min_bytes = self.config.vision_min_diagram_bytes
        for dblock in all_diagram_blocks:
            if callable(cancel_check):
                try:
                    if cancel_check():
                        self.logger.warning(
                            "DiagramAnalysisCoordinator.run: cancellation detected before diagram filtering"
                        )
                        return
                except Exception:
                    pass
            dblock.ensure_image_loaded(min_bytes)
            if not dblock.is_valid():
                continue
            diagram_input = DiagramInput(
                diagram_id=dblock.diagram_id,
                image_b64=dblock.image_b64,
                page_number=dblock.page_number,
                caption=dblock.caption,
                surrounding_text=dblock.surrounding_text,
                image_format=dblock.image_format,
                bbox_x0=dblock.bbox_x0,
                bbox_y0=dblock.bbox_y0,
                bbox_x1=dblock.bbox_x1,
                bbox_y1=dblock.bbox_y1,
            )
            try:
                image_bytes = diagram_input.decode_image_bytes()
            except ValueError:
                continue
            if len(image_bytes) < min_bytes:
                continue
            eligible_diagrams.append(diagram_input)

        if not eligible_diagrams:
            self.logger.info(
                "DiagramAnalysisCoordinator.run: no eligible diagrams after filtering — skipping"
            )
            return

        diagram_reqs = self.workflow_repository.list_diagram_requirements(
            category_id=category.id,
            ingestion_job_id=ingestion_job.id,
        )
        if not diagram_reqs:
            self.logger.info(
                "DiagramAnalysisCoordinator.run: no diagram requirements for category=%s — skipping",
                getattr(category, "code", None),
            )
            return
        tsd_context = tsd_document.full_text[:3000] if hasattr(tsd_document, "full_text") else ""

        diagram_outputs = []
        with ThreadPoolExecutor(
            max_workers=self.config.vision_max_concurrency,
            thread_name_prefix="DiagramDebate",
        ) as executor:
            futures = {}
            for diagram in eligible_diagrams:
                requirements = self.requirement_selector.select_for_diagram(
                    diagram=diagram,
                    tsd_document=tsd_document,
                    category=category,
                    ingestion_job=ingestion_job,
                )
                self.logger.info(
                    "DiagramAnalysisCoordinator.run: diagram_id=%s requirements_selected=%d",
                    diagram.diagram_id,
                    len(requirements),
                )
                future = executor.submit(
                    self.diagram_debate_service.run_diagram_debate,
                    diagram=diagram,
                    requirements=requirements,
                    tsd_context=tsd_context,
                    cancel_check=cancel_check,
                )
                futures[future] = diagram
            for future in as_completed(futures):
                if callable(cancel_check):
                    try:
                        if cancel_check():
                            self.logger.warning(
                                "DiagramAnalysisCoordinator.run: cancellation detected during diagram debate"
                            )
                            return
                    except Exception:
                        pass
                diagram = futures[future]
                try:
                    diagram_outputs.append(future.result())
                except Exception as exc:
                    self.logger.exception(
                        "DiagramAnalysisCoordinator.run: debate failed for diagram_id=%s: %s",
                        diagram.diagram_id,
                        exc,
                    )

        persisted_count = 0
        for output in diagram_outputs:
            if output.error:
                summary.error_count += 1
                continue
            finding = self.persistence.persist_diagram_debate_finding(
                review=review,
                category=category,
                diagram_debate_output=output,
                summary=summary,
            )
            if finding is not None:
                persisted_count += 1

        self.logger.info(
            "DiagramAnalysisCoordinator.run: COMPLETE — %d diagram findings persisted",
            persisted_count,
        )
