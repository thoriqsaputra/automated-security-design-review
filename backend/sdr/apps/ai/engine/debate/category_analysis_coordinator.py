from __future__ import annotations

import logging
from collections import deque
from typing import Any, List


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
        )

        if analysis_mode == "text_only":
            return

        self.run_state.update_stage(review, summary, "7_diagram_debate")
        self.diagram_analysis.run(
            review=review,
            tsd_document=tsd_document,
            category=category,
            ingestion_job=ingestion_job,
            summary=summary,
            cancel_check=lambda: self.run_state.is_cancelled(review),
        )

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
    ) -> None:
        self.progress_service.prepare_category_stats(
            summary=summary,
            parameters=parameters,
            category_code=category_code,
        )
        parent_skip_before = int(summary.applicability.get("children_marked_na_by_parent", 0) or 0)
        self.run_state.update_stage(review, summary, "5_parent_retrieval")
        applicable_parameters, parent_context_cache = self.text_debate.apply_parent_applicability_gate(
            review=review,
            category=category,
            ingestion_job=ingestion_job,
            parameters=parameters,
            indexes=indexes,
            tsd_document=tsd_document,
            summary=summary,
        )
        if not applicable_parameters:
            return

        summary.debate_total_parameters += len(applicable_parameters)
        summary.debate_remaining_parameters += len(applicable_parameters)
        summary.persistence_total_parameters += len(applicable_parameters)
        summary.persistence_remaining_parameters += len(applicable_parameters)
        self.progress_service.initialize_category_progress(
            summary=summary,
            category_code=category_code,
            total_count=len(applicable_parameters),
        )
        self.progress_service.sync_analysis_aliases(summary=summary, category_code=category_code)
        self.run_state.persist_summary_snapshot(review, summary)

        parent_skipped_for_category = max(
            int(summary.applicability.get("children_marked_na_by_parent", 0) or 0) - parent_skip_before,
            0,
        )
        self.logger.info(
            "CategoryAnalysisCoordinator._run_raw_children_analysis: category=%s applicable=%d skipped_by_parent=%d",
            category_code,
            len(applicable_parameters),
            parent_skipped_for_category,
        )

        self.run_state.update_stage(review, summary, "6_text_debate")
        if self.config.batch_debate_enabled:
            self.text_debate.run_batched_analysis_for_category(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameters=applicable_parameters,
                indexes=indexes,
                tsd_document=tsd_document,
                summary=summary,
                killed_assumptions_memory=killed_assumptions_memory,
                parent_context_cache=parent_context_cache,
            )
        else:
            self.text_debate.run_single_analysis_for_category(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameters=applicable_parameters,
                indexes=indexes,
                tsd_document=tsd_document,
                summary=summary,
                killed_assumptions_memory=killed_assumptions_memory,
            )
