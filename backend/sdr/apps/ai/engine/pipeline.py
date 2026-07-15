from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from sdr.apps.ai.client.session import build_tsd_analysis_session_id, job_session_context
from sdr.apps.ai.engine.config import AnalysisPipelineConfig
from sdr.apps.ai.engine.debate.category_analysis_coordinator import CategoryAnalysisCoordinator
from sdr.apps.ai.engine.debate.debate_input_factory import DebateInputFactory
from sdr.apps.ai.engine.debate.debate_service import DebateService
from sdr.apps.ai.engine.debate.diagram_analysis_coordinator import DiagramAnalysisCoordinator
from sdr.apps.ai.engine.debate.diagram_extract_reason_service import DiagramExtractReasonService
from sdr.apps.ai.engine.debate.text_debate_coordinator import TextDebateCoordinator
from sdr.apps.ai.engine.dto import AnalysisSummary
from sdr.apps.ai.engine.persistence.persistence_service import PersistenceService
from sdr.apps.ai.engine.persistence.progress_tracker import SummaryProgressService
from sdr.apps.ai.engine.persistence.review_run_state_service import (
    AnalysisCancelledError,
    ReviewRunStateService,
)
from sdr.apps.ai.engine.persistence.workflow_repository import (
    ReviewWorkflowRepository,
    SqlAlchemyReviewWorkflowRepository,
)
from sdr.apps.ai.engine.preparation.ingestion_service import IngestionService
from sdr.apps.ai.engine.preparation.retrieval_service import RetrievalService
from sdr.apps.ai.engine.reporting.overview_generator import OverviewGenerator
from sdr.apps.ai.engine.reporting.retrieval_snapshot_builder import RetrievalSnapshotBuilder
from sdr.apps.designs.preparation_store import (
    DesignPreparationStore,
    PreparationArtifactError,
    PreparationNotReadyError,
)
from sdr.apps.reviews.models import Review
from sdr.apps.reviews.models.choices import ReviewStatus
from sdr.core.database import SessionLocal

logger = logging.getLogger(__name__)


class TSDAnalysisPipeline:
    def __init__(
        self,
        ingestion_service: Optional[IngestionService] = None,
        retrieval_service: Optional[RetrievalService] = None,
        debate_service: Optional[DebateService] = None,
        persistence_service: Optional[PersistenceService] = None,
        diagram_debate_service: Optional[Any] = None,
        workflow_repository: Optional[ReviewWorkflowRepository] = None,
        overview_generator: Optional[OverviewGenerator] = None,
        debate_input_factory: Optional[DebateInputFactory] = None,
        progress_service: Optional[SummaryProgressService] = None,
        config: Optional[AnalysisPipelineConfig] = None,
        mediator_agent_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.ingestion = ingestion_service or IngestionService()
        self.retrieval = retrieval_service or RetrievalService()
        self.debate = debate_service or DebateService()
        self.persistence = persistence_service or PersistenceService(
            recommendation_generator=self._generate_not_met_recommendation,
        )
        self.diagram_debate_service = diagram_debate_service or DiagramExtractReasonService()
        self.workflow_repository = workflow_repository or SqlAlchemyReviewWorkflowRepository()
        self.overview_generator = overview_generator or OverviewGenerator()
        self.debate_input_factory = debate_input_factory or DebateInputFactory()
        self.progress_service = progress_service or SummaryProgressService()
        self.config = config or AnalysisPipelineConfig.from_settings()
        self.mediator_agent_factory = mediator_agent_factory or self._default_mediator_agent_factory
        self.run_state = ReviewRunStateService(workflow_repository=self.workflow_repository)
        self.snapshot_builder = RetrievalSnapshotBuilder(workflow_repository=self.workflow_repository)
        self.preparation_store = DesignPreparationStore(
            ingestion_service=self.ingestion,
            retrieval_service=self.retrieval,
        )
        self.diagram_analysis = DiagramAnalysisCoordinator(
            config=self.config,
            workflow_repository=self.workflow_repository,
            diagram_debate_service=self.diagram_debate_service,
            persistence_service=self.persistence,
            progress_service=self.progress_service,
            run_state_service=self.run_state,
        )
        self.text_debate = TextDebateCoordinator(
            config=self.config,
            retrieval_service=self.retrieval,
            debate_service=self.debate,
            persistence_service=self.persistence,
            debate_input_factory=self.debate_input_factory,
            progress_service=self.progress_service,
            run_state_service=self.run_state,
            mediator_agent_factory=self.mediator_agent_factory,
        )
        self.category_analysis = CategoryAnalysisCoordinator(
            config=self.config,
            workflow_repository=self.workflow_repository,
            progress_service=self.progress_service,
            run_state_service=self.run_state,
            text_debate_coordinator=self.text_debate,
            diagram_analysis_coordinator=self.diagram_analysis,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _default_mediator_agent_factory(self):
        from sdr.apps.ai.agents.mediator import MediatorAgent

        return MediatorAgent()

    def _generate_not_met_recommendation(self, **kwargs):
        mediator = self.mediator_agent_factory()
        generator = getattr(mediator, "generate_recommendation_for_not_met", None)
        if not callable(generator):
            return None
        return generator(**kwargs)

    def run(self, review: Review) -> AnalysisSummary:
        self.logger.info(
            "TSDAnalysisPipeline.run: [START] review_id=%s design='%s'",
            review.id,
            review.design.name,
        )
        summary = AnalysisSummary()
        latest = self.workflow_repository.get_latest_review(review.id)
        if latest and latest.status == ReviewStatus.CANCELLED.value:
            review.status = latest.status
            review.completed_at = getattr(latest, "completed_at", None)
            review.error_message = getattr(latest, "error_message", None)
            return summary

        self.run_state.mark_running(review)

        try:
            self.run_state.update_stage(review, summary, "4_parameter_resolution")
            tsd_document = None
            indexes = None
            if getattr(getattr(review, "design", None), "can_start_analysis", False):
                try:
                    with SessionLocal() as db:
                        _preparation, tsd_document, indexes = self.preparation_store.load_prepared_assets(
                            db,
                            review.design,
                        )
                    self.snapshot_builder.save(review, indexes)
                except (PreparationNotReadyError, PreparationArtifactError) as exc:
                    self.logger.error(
                        "TSDAnalysisPipeline.run: preparation unavailable for review_id=%s: %s",
                        review.id,
                        exc,
                    )
                    self.run_state.fail_review(review, str(exc))
                    return summary
            else:
                self.run_state.fail_review(review, "Design is not ready for analysis (can_start_analysis is False).")
                return summary

            if tsd_document is None or indexes is None:
                self.run_state.fail_review(review, "Failed to load TSD document or indexes from preparation store.")
                return summary

            category = review.category or (review.ingestion_job.category if review.ingestion_job else None)
            if not category:
                self.run_state.complete_review(review, summary)
                return summary

            killed_assumptions_memory = deque(maxlen=16)
            self.run_state.raise_if_cancelled(review, phase="run.before_category")
            self.category_analysis.run_category(
                review=review,
                category=category,
                indexes=indexes,
                tsd_document=tsd_document,
                summary=summary,
                killed_assumptions_memory=killed_assumptions_memory,
            )

            self.run_state.update_stage(review, summary, "8_overview")
            overview = self.overview_generator.generate(review, summary)
            if overview:
                self.run_state.save_overview(review, overview)

            self.run_state.complete_review(review, summary)
            return summary
        except Exception as exc:
            if self.run_state.is_cancelled(review):
                self.logger.warning(
                    "TSDAnalysisPipeline.run: [CANCELLED] review_id=%s stopping after cancellation signal: %s",
                    review.id,
                    exc,
                )
                return summary
            self.logger.exception("TSDAnalysisPipeline.run: [FATAL] review_id=%s: %s", review.id, exc)
            self.run_state.fail_review(review, str(exc))
            return summary
        finally:
            if "tsd_document" in locals():
                tsd_document.cleanup_temporary_artifacts()

    def _resolve_parameters(self, review: Review) -> tuple:
        category = review.category or (review.ingestion_job.category if review.ingestion_job else None)
        if not category:
            return None, None, []
        if review.ingestion_job:
            ingestion_job = review.ingestion_job
        else:
            ingestion_job = self.workflow_repository.get_latest_active_ingestion_job(category.id)
        if not ingestion_job:
            return category, None, []
        parameters = self.workflow_repository.list_category_parameters(
            category_id=category.id,
            ingestion_job_id=ingestion_job.id,
        )
        return category, ingestion_job, parameters

    def _prepare_category_stats(self, summary: AnalysisSummary, parameters: List[Any], category_code: str) -> None:
        self.progress_service.prepare_category_stats(summary=summary, parameters=parameters, category_code=category_code)

    def _initialize_category_progress(self, *, summary: AnalysisSummary, category_code: str, total_count: int) -> None:
        self.progress_service.initialize_category_progress(summary=summary, category_code=category_code, total_count=total_count)

    def _sync_analysis_aliases(self, summary: AnalysisSummary, category_code: Optional[str] = None) -> None:
        self.progress_service.sync_analysis_aliases(summary=summary, category_code=category_code)

    def _persist_summary_snapshot(self, review: Review, summary: AnalysisSummary) -> None:
        try:
            summary_dict = summary.to_dict()
            self.workflow_repository.save_summary_snapshot(review.id, summary=summary_dict)
            review.summary_json = summary_dict
        except Exception as exc:
            self.logger.exception(
                "TSDAnalysisPipeline._persist_summary_snapshot: failed for review_id=%s: %s",
                review.id,
                exc,
            )

    def _update_stage(self, review: Review, summary: AnalysisSummary, stage: str) -> None:
        summary.current_stage = stage
        self._persist_summary_snapshot(review, summary)

    def _persist_retrieval_snapshot(self, review: Review, indexes) -> None:
        self.snapshot_builder.save(review, indexes)

    def _build_retrieval_snapshot(self, indexes) -> Optional[Dict[str, Any]]:
        return self.snapshot_builder.build_snapshot(indexes)

    def _record_debate_progress(self, **kwargs) -> None:
        self.text_debate.record_debate_progress(**kwargs)

    def _record_persistence_progress(self, **kwargs) -> None:
        self.text_debate.record_persistence_progress(**kwargs)

    def _is_cancelled(self, review: Review) -> bool:
        latest = self.workflow_repository.get_latest_review(review.id)
        if not latest:
            return False
        if latest.status == ReviewStatus.CANCELLED.value:
            return True
        return (
            latest.status == (ReviewStatus.FAILED.value if hasattr(ReviewStatus, "value") else ReviewStatus.FAILED)
            and (latest.error_message or "").strip().lower().startswith("analysis was cancelled")
        )

    def _raise_if_cancelled(self, review: Review, *, phase: str) -> None:
        if not self._is_cancelled(review):
            return
        self.logger.warning(
            "TSDAnalysisPipeline.%s: cancellation detected review_id=%s",
            phase,
            review.id,
        )
        raise AnalysisCancelledError("Analysis was cancelled by user.")

    def _run_diagram_analysis(self, **kwargs) -> None:
        self.diagram_analysis.run(**kwargs)

    def _run_single_analysis_for_category(self, **kwargs) -> None:
        original_is_cancelled = self.run_state.is_cancelled
        original_persist_summary = self.run_state.persist_summary_snapshot
        original_analyze = self.text_debate.analyze_single_child_with_retrieval_result
        original_persist = self.text_debate.persist_debate_output
        try:
            self.run_state.is_cancelled = self._is_cancelled
            self.run_state.persist_summary_snapshot = self._persist_summary_snapshot
            self.text_debate.analyze_single_child_with_retrieval_result = self._analyze_single_child_with_retrieval_result
            self.text_debate.persist_debate_output = self._persist_debate_output
            self.text_debate.run_single_analysis_for_category(**kwargs)
        finally:
            self.run_state.is_cancelled = original_is_cancelled
            self.run_state.persist_summary_snapshot = original_persist_summary
            self.text_debate.analyze_single_child_with_retrieval_result = original_analyze
            self.text_debate.persist_debate_output = original_persist

    def _build_context_chunk_map(
        self,
        context_chunks: list,
        retrieval_metadata: Optional[dict] = None,
        tsd_document=None,
        source_block_ids: Optional[list] = None,
    ) -> dict:
        return self.debate_input_factory.build_context_chunk_map(
            context_chunks,
            retrieval_metadata=retrieval_metadata,
            tsd_document=tsd_document,
            source_block_ids=source_block_ids,
        )

    def _build_xml_context_chunks(
        self,
        context_chunks: list,
        retrieval_metadata: Optional[dict] = None,
        tsd_document=None,
        source_block_ids: Optional[list] = None,
    ) -> list:
        return self.debate_input_factory.build_xml_context_chunks(
            context_chunks,
            retrieval_metadata=retrieval_metadata,
            tsd_document=tsd_document,
            source_block_ids=source_block_ids,
        )

    def _apply_not_met_evidence_gate(self, **kwargs):
        return self.text_debate.apply_not_met_evidence_gate(**kwargs)

    def _persist_debate_output(self, **kwargs):
        return TextDebateCoordinator.persist_debate_output(self.text_debate, **kwargs)

    def _analyze_single_child_with_retrieval_result(self, **kwargs):
        return TextDebateCoordinator.analyze_single_child_with_retrieval_result(self.text_debate, **kwargs)


def run_tsd_analysis(review: Review) -> AnalysisSummary:
    session_id = build_tsd_analysis_session_id(review.id)
    logger.info("run_tsd_analysis: [ENTRY] review_id=%s session_id=%s", review.id, session_id)
    pipeline = TSDAnalysisPipeline()
    with job_session_context(session_id=session_id, job_type="tsd_analysis", job_id=review.id):
        summary = pipeline.run(review)
    logger.info("run_tsd_analysis: [EXIT] review_id=%s summary=%s", review.id, summary.to_dict())
    return summary
