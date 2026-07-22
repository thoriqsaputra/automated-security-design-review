from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

from sdr.apps.ai.client.session import capture_current_context
from sdr.apps.ai.engine.persistence.review_run_state_service import AnalysisCancelledError


def _resolve_analysis_mode(review) -> str:
    mode = str(getattr(review, "analysis_mode", "default") or "default").strip().lower()
    if mode in {"default", "text_only", "diagram_only"}:
        return mode
    return "default"


class CategoryAnalysisCoordinator:
    def __init__(
        self,
        *,
        config,
        workflow_repository,
        progress_service,
        run_state_service,
        text_debate_coordinator,
        diagram_analysis_coordinator,
    ) -> None:
        self.config = config
        self.workflow_repository = workflow_repository
        self.progress_service = progress_service
        self.run_state = run_state_service
        self.text_debate = text_debate_coordinator
        self.diagram_analysis = diagram_analysis_coordinator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_category(
        self,
        *,
        review,
        category,
        indexes,
        tsd_document,
        summary,
        killed_assumptions_memory: deque,
    ) -> None:
        analysis_mode = _resolve_analysis_mode(review)
        if review.ingestion_job:
            ingestion_job = review.ingestion_job
        else:
            ingestion_job = self.workflow_repository.get_latest_active_ingestion_job(category.id)
        if not ingestion_job:
            self.logger.warning(
                "CategoryAnalysisCoordinator.run_category: no active ingestion job for category=%s",
                getattr(category, "code", None),
            )
            return

        parameters = self.workflow_repository.list_category_parameters(
            category_id=category.id,
            ingestion_job_id=ingestion_job.id,
        )
        if not parameters:
            return

        if analysis_mode != "diagram_only":
            with summary.lock:
                summary.total_parameters += len(parameters)

        category_code = getattr(category, "code", None) or "unknown"
        self.run_state.update_stage(review, summary, "4_parameter_resolution")
        self.logger.info(
            "CategoryAnalysisCoordinator.run_category: category=%s analysis_mode=%s parameters=%d",
            category_code,
            analysis_mode,
            len(parameters),
        )

        if analysis_mode == "diagram_only":
            self.run_state.update_stage(review, summary, "7_diagram_debate")
            self.diagram_analysis.run(
                review=review,
                tsd_document=tsd_document,
                category=category,
                ingestion_job=ingestion_job,
                summary=summary,
                cancel_check=lambda: self.run_state.is_cancelled(review),
            )
            return

        if analysis_mode == "text_only":
            self._run_raw_children_analysis(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameters=parameters,
                indexes=indexes,
                tsd_document=tsd_document,
                summary=summary,
                category_code=category_code,
                killed_assumptions_memory=killed_assumptions_memory,
                update_stage=True,
            )
            return

        self.run_state.update_stage(review, summary, "6_7_concurrent_debate")

        def _run_diagram_phase() -> None:
            self.diagram_analysis.run(
                review=review,
                tsd_document=tsd_document,
                category=category,
                ingestion_job=ingestion_job,
                summary=summary,
                cancel_check=lambda: self.run_state.is_cancelled(review),
            )

        def _run_text_phase() -> None:
            self._run_raw_children_analysis(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameters=parameters,
                indexes=indexes,
                tsd_document=tsd_document,
                summary=summary,
                category_code=category_code,
                killed_assumptions_memory=killed_assumptions_memory,
                update_stage=False,
            )

        branches = (("diagram", _run_diagram_phase), ("text", _run_text_phase))
        branch_errors: List[tuple[str, BaseException]] = []
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"CategoryAnalysis-{category_code}") as executor:
            future_map = {}
            for branch_name, branch_fn in branches:
                self.logger.info(
                    "CategoryAnalysisCoordinator.run_category: branch=%s [SUBMITTED] category=%s",
                    branch_name,
                    category_code,
                )
                future = executor.submit(capture_current_context(branch_fn))
                future_map[future] = branch_name

            for future in as_completed(future_map):
                branch_name = future_map[future]
                try:
                    future.result()
                    self.logger.info(
                        "CategoryAnalysisCoordinator.run_category: branch=%s [COMPLETE] category=%s",
                        branch_name,
                        category_code,
                    )
                except BaseException as exc:
                    branch_errors.append((branch_name, exc))
                    self.logger.exception(
                        "CategoryAnalysisCoordinator.run_category: branch=%s [FAILED] category=%s: %s",
                        branch_name,
                        category_code,
                        exc,
                    )

        if branch_errors:
            cancellation = next(
                (exc for _branch, exc in branch_errors if isinstance(exc, AnalysisCancelledError)),
                None,
            )
            raise cancellation or branch_errors[0][1]

    def _run_raw_children_analysis(
        self,
        *,
        review,
        category,
        ingestion_job,
        parameters: List[Any],
        indexes,
        tsd_document,
        summary,
        category_code: str,
        killed_assumptions_memory: deque,
        update_stage: bool = True,
    ) -> None:
        self.progress_service.prepare_category_stats(
            summary=summary,
            parameters=parameters,
            category_code=category_code,
        )
        if not parameters:
            return

        self.logger.info(
            "CategoryAnalysisCoordinator._run_raw_children_analysis: category=%s applicable=%d",
            category_code,
            len(parameters),
        )

        if update_stage:
            self.run_state.update_stage(review, summary, "6_text_debate")
        self.text_debate.run_single_analysis_for_category(
            review=review,
            category=category,
            ingestion_job=ingestion_job,
            parameters=parameters,
            indexes=indexes,
            tsd_document=tsd_document,
            summary=summary,
            killed_assumptions_memory=killed_assumptions_memory,
        )
