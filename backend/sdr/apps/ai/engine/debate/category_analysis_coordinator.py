from __future__ import annotations

import logging
import threading
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

        diagram_thread = None
        diagram_errors: List[BaseException] = []
        if analysis_mode != "text_only":
            self.run_state.update_stage(review, summary, "6_7_concurrent_debate")
            # Diagram debate has no data dependency on text-debate output, so it's
            # started here on a background thread to run concurrently with the
            # text-debate phase below rather than waiting for it to finish first.
            # AnalysisSummary.lock (see dto.py) guards the now-concurrent writes
            # from both phases' driver threads.
            def _run_diagram_phase() -> None:
                try:
                    self.diagram_analysis.run(
                        review=review,
                        tsd_document=tsd_document,
                        category=category,
                        ingestion_job=ingestion_job,
                        summary=summary,
                        cancel_check=lambda: self.run_state.is_cancelled(review),
                    )
                except BaseException as exc:  # noqa: BLE001 - re-raised on join below
                    diagram_errors.append(exc)

            diagram_thread = threading.Thread(
                target=_run_diagram_phase,
                name=f"DiagramDebatePhase-{category_code}",
                daemon=True,
            )
            diagram_thread.start()

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

        if diagram_thread is not None:
            diagram_thread.join()
            if diagram_errors:
                raise diagram_errors[0]

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
        if not parameters:
            return

        with summary.lock:
            summary.debate_total_parameters += len(parameters)
            summary.debate_remaining_parameters += len(parameters)
            summary.persistence_total_parameters += len(parameters)
            summary.persistence_remaining_parameters += len(parameters)
        self.progress_service.initialize_category_progress(
            summary=summary,
            category_code=category_code,
            total_count=len(parameters),
        )
        self.progress_service.sync_analysis_aliases(summary=summary, category_code=category_code)
        self.run_state.persist_summary_snapshot(review, summary)

        self.logger.info(
            "CategoryAnalysisCoordinator._run_raw_children_analysis: category=%s applicable=%d",
            category_code,
            len(parameters),
        )

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
