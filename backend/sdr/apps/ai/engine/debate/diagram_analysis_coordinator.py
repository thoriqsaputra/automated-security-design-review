from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

from sdr.apps.ai.agents.vision import DiagramInput
from sdr.apps.ai.client.session import capture_current_context

from sdr.apps.ai.engine.config import AnalysisPipelineConfig
from sdr.apps.ai.engine.debate.diagram_requirement_selector import DiagramRequirementSelector
from sdr.apps.reviews.services.debate_events import build_debate_id, review_debate_event_store

# Coarse per-agent progress checkpoints — diagram debate has no token-level
# streaming (each vision agent call is a single blocking round-trip), so only
# start/complete boundaries per agent are meaningful here.
_DIAGRAM_AGENT_PROGRESS = {
    ("hunter", "started"): 10,
    ("hunter", "completed"): 40,
    ("critic", "started"): 45,
    ("critic", "completed"): 70,
    ("mediator", "started"): 75,
    ("mediator", "completed"): 100,
}


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
        category_code = getattr(category, "code", None) or "unknown"

        review_debate_event_store.seed_debates(
            review.id,
            review_status=getattr(review, "status", None) or "running",
            debates=[
                self._build_live_debate_descriptor(diagram=diagram, category_code=category_code)
                for diagram in eligible_diagrams
            ],
        )

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
                    capture_current_context(self.diagram_debate_service.run_diagram_debate),
                    diagram=diagram,
                    requirements=requirements,
                    tsd_context=tsd_context,
                    cancel_check=cancel_check,
                    agent_started_handler=lambda agent, diagram=diagram: self._publish_live_agent_start(
                        review=review, diagram=diagram, category_code=category_code, agent=agent,
                    ),
                    agent_completed_handler=lambda agent, content, diagram=diagram, critic_outcome=None, requires_rebuttal=None: self._publish_live_agent_complete(
                        review=review,
                        diagram=diagram,
                        agent=agent,
                        content=content,
                        critic_outcome=critic_outcome,
                        requires_rebuttal=requires_rebuttal,
                    ),
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
                    review_debate_event_store.fail_agent(
                        review.id,
                        debate_id=build_debate_id(None, diagram_id=diagram.diagram_id),
                        agent="mediator",
                        error_message=str(exc)[:500] or "Diagram debate failed unexpectedly.",
                    )

        persisted_count = 0
        for output in diagram_outputs:
            with summary.lock:
                summary.total_parameters += 1
                if output.error:
                    summary.error_count += 1
                    self.logger.error(
                        "DiagramAnalysisCoordinator.run: diagram_id=%s dropped — %s",
                        output.diagram.diagram_id,
                        output.error,
                    )
                    review_debate_event_store.fail_agent(
                        review.id,
                        debate_id=build_debate_id(None, diagram_id=output.diagram.diagram_id),
                        agent="mediator",
                        error_message=output.error[:500],
                    )
                    continue
            finding = self.persistence.persist_diagram_debate_finding(
                review=review,
                category=category,
                diagram_debate_output=output,
                summary=summary,
            )
            if finding is not None:
                persisted_count += 1
                review_debate_event_store.complete_debate(
                    review.id,
                    debate_id=build_debate_id(None, diagram_id=output.diagram.diagram_id),
                    finding_id=finding.id,
                    last_snippet=(output.mediator_result or {}).get("finding_description") or "",
                )

        self.logger.info(
            "DiagramAnalysisCoordinator.run: COMPLETE — %d diagram findings persisted",
            persisted_count,
        )

    def _build_live_debate_descriptor(self, *, diagram, category_code: str) -> dict:
        return {
            "debate_id": build_debate_id(None, diagram_id=diagram.diagram_id),
            "finding_type": "diagram",
            "diagram_id": diagram.diagram_id,
            "requirement_reference": None,
            "requirement_text": diagram.caption or f"Diagram (page {diagram.page_number})",
            "section_title": None,
            "category_code": category_code,
            "execution_mode": "single",
        }

    def _publish_live_agent_start(self, *, review, diagram, category_code: str, agent: str) -> None:
        review_debate_event_store.start_agent(
            review.id,
            debate=self._build_live_debate_descriptor(diagram=diagram, category_code=category_code),
            agent=agent,
            execution_mode="single",
            content=f"{agent.title()} is analyzing this diagram.",
            progress_percent=_DIAGRAM_AGENT_PROGRESS.get((agent, "started"), 0),
        )

    def _publish_live_agent_complete(
        self,
        *,
        review,
        diagram,
        agent: str,
        content: str,
        critic_outcome=None,
        requires_rebuttal=None,
    ) -> None:
        review_debate_event_store.complete_agent(
            review.id,
            debate_id=build_debate_id(None, diagram_id=diagram.diagram_id),
            agent=agent,
            content=content,
            progress_percent=_DIAGRAM_AGENT_PROGRESS.get((agent, "completed"), 100),
            execution_mode="single",
            critic_outcome=critic_outcome,
            requires_rebuttal=requires_rebuttal,
        )
