# apps/ai/services/pipeline.py
"""
TSD Analysis Pipeline — Facade that orchestrates all services.
Single entry point: call this to run the full analysis.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from sdr.core.config import settings
from datetime import datetime, timezone

from sdr.apps.reviews.models import Review
from sdr.apps.reviews.models.choices import ReviewStatus
from sdr.apps.standards.utils import build_parameter_analysis_text
from sdr.apps.ai.utils.concurrency import ConcurrencyProbe
from .dto import AnalysisSummary, DebateInput, DebateOutput, PersistenceInput
from .ingestion_service import IngestionService
from .retrieval_service import RetrievalService
from .debate_service import DebateService
from .persistence_service import PersistenceService
from .contract_synthesizer import ContractSynthesizer
from .domain_classification import classify_requirement_domain, DOMAIN_KEYWORDS
from .asvs_level import classify_tsd_asvs_level, filter_parameters_for_asvs_level

logger = logging.getLogger(__name__)


class AnalysisCancelledError(RuntimeError):
    """Raised when the user has cancelled the review."""


class TSDAnalysisPipeline:
    """
    Orchestrates the full multi-agent TSD security review pipeline.
    Chains: Ingestion → Retrieval → Debate → Persistence
    """

    def __init__(
        self,
        ingestion_service: Optional[IngestionService] = None,
        retrieval_service: Optional[RetrievalService] = None,
        debate_service: Optional[DebateService] = None,
        persistence_service: Optional[PersistenceService] = None,
    ) -> None:
        """
        Args:
            All services are injected (or None for defaults).
            This allows easy mocking in tests.
        """
        self.ingestion = ingestion_service or IngestionService()
        self.retrieval = retrieval_service or RetrievalService()
        self.debate = debate_service or DebateService()
        self.persistence = persistence_service or PersistenceService()
        self.contract_synthesizer = ContractSynthesizer()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._last_batch_concurrency_stats: Dict[str, Dict[str, Any]] = {}

    def run(self, review: Review) -> AnalysisSummary:
        self.logger.info(
            "TSDAnalysisPipeline.run: [START] review_id=%s design='%s'",
            review.id,
            review.design.name,
        )

        summary = AnalysisSummary()
        from sdr.core.database import SessionLocal
        from sqlalchemy import select, update
        with SessionLocal() as db:
            latest = db.execute(select(Review).where(Review.id == review.id)).scalars().first()
            if latest and latest.status == ReviewStatus.CANCELLED.value:
                self.logger.warning(
                    "TSDAnalysisPipeline.run: review_id=%s already cancelled before start; aborting run",
                    review.id,
                )
                review.status = latest.status
                review.completed_at = getattr(latest, "completed_at", None)
                review.error_message = getattr(latest, "error_message", None)
                return summary
            new_status = ReviewStatus.RUNNING.value if hasattr(ReviewStatus, 'value') else ReviewStatus.RUNNING
            now = datetime.now(timezone.utc)
            db.execute(update(Review).where(Review.id == review.id).values(status=new_status, started_at=now))
            db.commit()
            review.status = new_status
            review.started_at = now

        try:
            # ---- STEP 1: Ingest ----
            self.logger.info("TSDAnalysisPipeline.run: [STEP 1] Ingestion")
            if True:
                ingestion_output = self.ingestion.ingest(review)

            if ingestion_output is None:
                self._fail_review(review, "Failed to ingest TSD document.")
                return summary

            tsd_document = ingestion_output.tsd_document

            # ---- STEP 2: Screen ----
            if not ingestion_output.is_valid_tsd:
                summary.screened_out = True
                self._fail_review(
                    review,
                    "Document failed TSD screening — does not appear to be a Technical Software Document.",
                )
                return summary

            # ---- STEP 3: Build Indexes ----
            self.logger.info("TSDAnalysisPipeline.run: [STEP 3] Building Retrieval Indexes")
            if True:
                indexes = self.retrieval.build_indexes(tsd_document)

            self.logger.info("TSDAnalysisPipeline.run: [STEP 4] Classifying ASVS Level")
            asvs_classification = self._classify_review_asvs_level(review, tsd_document)
            effective_asvs_level = self._resolve_effective_asvs_level(review, asvs_classification)
            summary.asvs = {
                "classified_level": asvs_classification.get("level"),
                "classification_confidence": asvs_classification.get("confidence"),
                "classification_reasoning": asvs_classification.get("reasoning"),
                "classification_evidence": asvs_classification.get("evidence", []),
                "classification_error": asvs_classification.get("error"),
                "definition_source": asvs_classification.get("definition_source"),
                "definition_count": asvs_classification.get("definition_count"),
                "override_level": getattr(review, "asvs_level_override", None),
                "effective_level": effective_asvs_level,
                "categories": {},
            }

            # ---- STEP 5: Resolve Parameters (ALL selected categories) ----
            self.logger.info("TSDAnalysisPipeline.run: [STEP 5] Resolving Parameters for all selected categories")

            # Build categories list in priority: selected_categories -> review.ingestion_job.category -> first standard.category
            selected_categories = list(review.selected_categories)
            categories = []
            if selected_categories:
                categories = selected_categories
            elif review.ingestion_job and review.ingestion_job.category:
                categories = [review.ingestion_job.category]


            if not categories:
                self.logger.warning(
                    "TSDAnalysisPipeline.run: no categories resolved for review_id=%s",
                    review.id,
                )
                self._complete_review(review, summary)
                return summary

            # Process each category sequentially so each parameter baseline is honoured
            # Local imports to avoid circular import at module scope
            from sdr.apps.standards.models import (
                StandardIngestionJob,
                CategoryParameterChild,
                CategoryParameterParent,
            )
            total_applicable = 0
            killed_assumptions_memory = deque(maxlen=16)
            for category in categories:
                # Resolve ingestion job for this category
                from sdr.core.database import SessionLocal
                from sqlalchemy import select

                with SessionLocal() as db:
                    if review.ingestion_job:
                        ingestion_job = review.ingestion_job
                    else:
                        ingestion_job = db.execute(
                            select(StandardIngestionJob)
                            .where(
                                StandardIngestionJob.category_id == category.id,
                                StandardIngestionJob.is_active == True
                            )
                            .order_by(StandardIngestionJob.created_at.desc())
                        ).scalars().first()

                if not ingestion_job:
                    self.logger.warning(
                        "TSDAnalysisPipeline.run: no active ingestion job for category=%s — skipping",
                        getattr(category, "code", None),
                    )
                    continue

                from sqlalchemy.orm import joinedload
                with SessionLocal() as db:
                    parameters = db.execute(
                        select(CategoryParameterChild)
                        .options(joinedload(CategoryParameterChild.parent).joinedload(CategoryParameterParent.category))
                        .join(CategoryParameterParent, CategoryParameterChild.parent_id == CategoryParameterParent.id)
                        .where(
                            CategoryParameterParent.category_id == category.id,
                            CategoryParameterParent.ingestion_job_id == ingestion_job.id,
                        )
                        .order_by(CategoryParameterParent.title, CategoryParameterChild.ordinal)
                    ).scalars().all()

                if not parameters:
                    self.logger.info(
                        "TSDAnalysisPipeline.run: no active parameters for category=%s — skipping",
                        getattr(category, "code", None),
                    )
                    continue

                summary.total_parameters += len(parameters)

                parameters, asvs_filter_stats = filter_parameters_for_asvs_level(parameters, effective_asvs_level)
                category_code = getattr(category, "code", None) or "unknown"
                summary.asvs["categories"][category_code] = asvs_filter_stats
                self.logger.info(
                    "TSDAnalysisPipeline.run: ASVS filter category=%s effective_level=L%s before=%d after=%d excluded=%d unknown_included=%d",
                    category_code,
                    effective_asvs_level,
                    asvs_filter_stats["before_count"],
                    asvs_filter_stats["after_count"],
                    asvs_filter_stats["excluded_by_level_count"],
                    asvs_filter_stats["unknown_level_included_count"],
                )

                if not parameters:
                    self.logger.info(
                        "TSDAnalysisPipeline.run: ASVS filter removed all parameters for category=%s — skipping",
                        getattr(category, "code", None),
                    )
                    continue

                applicable_parameters = parameters
                summary.analysis_total_parameters += len(applicable_parameters)
                summary.analysis_remaining_parameters += len(applicable_parameters)
                self._initialize_category_progress(
                    summary=summary,
                    category_code=category_code,
                    total_count=len(applicable_parameters),
                )

                # ---- STEP 6: Debate Loop for this category ----
                self.logger.info(
                    "TSDAnalysisPipeline.run: [STEP 6] Debate Loop — %d parameter(s) for category=%s",
                    len(applicable_parameters), getattr(category, "code", None),
                )

                total_applicable += len(applicable_parameters)
                if self._batch_analysis_enabled():
                    self._run_batched_analysis_for_category(
                        review=review,
                        category=category,
                        ingestion_job=ingestion_job,
                        parameters=applicable_parameters,
                        indexes=indexes,
                        tsd_document=tsd_document,
                        summary=summary,
                        killed_assumptions_memory=killed_assumptions_memory,
                    )
                else:
                    self._run_single_analysis_for_category(
                        review=review,
                        category=category,
                        ingestion_job=ingestion_job,
                        parameters=applicable_parameters,
                        indexes=indexes,
                        tsd_document=tsd_document,
                        summary=summary,
                        killed_assumptions_memory=killed_assumptions_memory,
                    )

            # ---- STEP 7: Generate Overview ----
            self.logger.info("TSDAnalysisPipeline.run: [STEP 7] Generating Overview")
            if True:
                overview = self._generate_overview(review, summary)
            if overview:
                from sdr.core.database import SessionLocal
                from sqlalchemy import update
                with SessionLocal() as db:
                    db.execute(update(Review).where(Review.id == review.id).values(overview=overview))
                    db.commit()
                    review.overview = overview

            # ---- STEP 8: Mark Completed ----
            self.logger.info("TSDAnalysisPipeline.run: [STEP 8] Marking Completed")
            self._complete_review(review, summary)

            self.logger.info(
                "TSDAnalysisPipeline.run: [SUCCESS] review_id=%s "
                "met=%d not_met=%d na=%d errors=%d analysis_total=%d analysis_processed=%d analysis_remaining=%d",
                review.id,
                summary.met_count,
                summary.not_met_count,
                summary.na_count,
                summary.error_count,
                summary.analysis_total_parameters,
                summary.analysis_processed_parameters,
                summary.analysis_remaining_parameters,
            )

            return summary

        except Exception as exc:
            if self._is_cancelled(review):
                self.logger.warning(
                    "TSDAnalysisPipeline.run: [CANCELLED] review_id=%s stopping after cancellation signal: %s",
                    review.id,
                    exc,
                )
                return summary
            self.logger.exception(
                "TSDAnalysisPipeline.run: [FATAL] review_id=%s: %s",
                review.id,
                exc,
            )
            self._fail_review(review, str(exc))
            return summary

        finally:
            if 'tsd_document' in locals():
                self.logger.info(
                    "TSDAnalysisPipeline.run: [CLEANUP] review_id=%s",
                    review.id,
                )
                tsd_document.cleanup_temporary_artifacts()

    # ---- Private Helpers ----

    def _classify_review_asvs_level(self, review: Review, tsd_document) -> Dict[str, Any]:
        try:
            from sdr.core.database import SessionLocal
            from sqlalchemy import select
            from sdr.apps.standards.models import ASVSLevel, ASVSLevelDefinition

            with SessionLocal() as db:
                levels = []
                definition_source = "missing"
                ingestion_job_id = getattr(review, "ingestion_job_id", None)
                if ingestion_job_id is None and getattr(review, "ingestion_job", None):
                    ingestion_job_id = getattr(review.ingestion_job, "id", None)
                if ingestion_job_id is not None:
                    levels = db.execute(
                        select(ASVSLevelDefinition)
                        .where(ASVSLevelDefinition.ingestion_job_id == ingestion_job_id)
                        .order_by(ASVSLevelDefinition.level)
                    ).scalars().all()
                    if levels:
                        definition_source = "standard_document"
                if not levels:
                    levels = db.execute(select(ASVSLevel).order_by(ASVSLevel.level)).scalars().all()
                    definition_source = "static_fallback" if levels else "missing"
            classification = classify_tsd_asvs_level(getattr(tsd_document, "full_text", "") or "", levels)
            result = classification.to_dict()
            result["definition_source"] = definition_source
            result["definition_count"] = len(levels)
            return result
        except Exception as exc:
            self.logger.exception(
                "TSDAnalysisPipeline._classify_review_asvs_level: failed for review_id=%s",
                getattr(review, "id", None),
            )
            return {
                "level": 1,
                "confidence": 0.0,
                "reasoning": "ASVS classification failed; defaulted to L1.",
                "evidence": [],
                "error": str(exc),
                "definition_source": "error",
                "definition_count": 0,
            }

    def _resolve_effective_asvs_level(self, review: Review, classification: Dict[str, Any]) -> int:
        override = getattr(review, "asvs_level_override", None)
        try:
            override_int = int(override) if override is not None else None
        except (TypeError, ValueError):
            override_int = None
        if override_int in (1, 2, 3):
            return override_int
        level = classification.get("level")
        try:
            level_int = int(level) if level is not None else None
        except (TypeError, ValueError):
            level_int = None
        if level_int in (1, 2, 3):
            return level_int
        return 1

    def _initialize_category_progress(
        self,
        *,
        summary: AnalysisSummary,
        category_code: str,
        total_count: int,
    ) -> None:
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
        category_stats["analysis_total_count"] = int(total_count)
        category_stats["analysis_processed_count"] = 0
        category_stats["analysis_remaining_count"] = int(total_count)

    def _record_parameter_progress(
        self,
        *,
        summary: AnalysisSummary,
        category_code: str,
        parameter_id: Optional[Any] = None,
        log_prefix: str = "TSDAnalysisPipeline.run",
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
        total_count = int(category_stats.get("analysis_total_count") or 0)
        processed_count = int(category_stats.get("analysis_processed_count") or 0) + 1
        remaining_count = max(total_count - processed_count, 0)
        category_stats["analysis_processed_count"] = processed_count
        category_stats["analysis_remaining_count"] = remaining_count

        summary.analysis_processed_parameters += 1
        summary.analysis_remaining_parameters = max(summary.analysis_remaining_parameters - 1, 0)

        extra_fields = dict(extra_fields or {})
        extra_suffix = ""
        if extra_fields:
            extra_suffix = " " + " ".join(f"{key}={value}" for key, value in extra_fields.items())

        self.logger.info(
            "%s: category=%s progress processed=%d remaining=%d total=%d parameter id=%s%s",
            log_prefix,
            category_code,
            processed_count,
            remaining_count,
            total_count,
            parameter_id,
            extra_suffix,
        )

    def _cancel_pending_futures(
        self,
        *,
        executor: ThreadPoolExecutor,
        future_map: Dict[Any, Any],
        review_id: Any,
        phase: str,
    ) -> None:
        cancelled_count = 0
        for future in future_map:
            if future.done():
                continue
            if future.cancel():
                cancelled_count += 1
        executor.shutdown(wait=False, cancel_futures=True)
        self.logger.warning(
            "TSDAnalysisPipeline.%s: cancellation detected review_id=%s pending_futures_cancelled=%d",
            phase,
            review_id,
            cancelled_count,
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

    def _resolve_parameters(self, review: Review) -> tuple:
        """Resolves category, ingestion_job, and parameters for the review."""
        from sdr.apps.standards.models import (
            StandardCategory,
            StandardIngestionJob,
            CategoryParameterChild,
            CategoryParameterParent,
        )

        category: Optional[StandardCategory] = None
        selected_categories = list(review.selected_categories)

        if selected_categories:
            category = selected_categories[0]
        elif review.ingestion_job and review.ingestion_job.category:
            category = review.ingestion_job.category


        if not category:
            return None, None, []

        from sdr.core.database import SessionLocal
        from sqlalchemy import select
        with SessionLocal() as db:
            if review.ingestion_job:
                ingestion_job = review.ingestion_job
            else:
                ingestion_job = db.execute(
                    select(StandardIngestionJob)
                    .where(
                        StandardIngestionJob.category_id == category.id,
                        StandardIngestionJob.is_active == True
                    )
                    .order_by(StandardIngestionJob.created_at.desc())
                ).scalars().first()

        if not ingestion_job:
            return category, None, []

        from sqlalchemy.orm import joinedload
        with SessionLocal() as db:
            parameters = db.execute(
                select(CategoryParameterChild)
                .options(joinedload(CategoryParameterChild.parent).joinedload(CategoryParameterParent.category))
                .join(CategoryParameterParent, CategoryParameterChild.parent_id == CategoryParameterParent.id)
                .where(
                    CategoryParameterParent.category_id == category.id,
                    CategoryParameterParent.ingestion_job_id == ingestion_job.id,
                )
                .order_by(CategoryParameterParent.title, CategoryParameterChild.ordinal)
            ).scalars().all()

        return category, ingestion_job, parameters

    def _batch_analysis_enabled(self) -> bool:
        return bool(getattr(settings, "AI_BATCH_DEBATE_ENABLED", True))

    def _batch_size(self) -> int:
        return max(1, int(getattr(settings, "AI_BATCH_DEBATE_BATCH_SIZE", 3)))

    def _batch_max_concurrency(self) -> int:
        return max(1, int(getattr(settings, "AI_BATCH_DEBATE_MAX_CONCURRENCY", 3)))

    def _group_parameters_by_parent(self, parameters: List[Any]) -> List[Tuple[Any, List[Any]]]:
        grouped: List[Tuple[Any, List[Any]]] = []
        index_by_key: Dict[Any, int] = {}
        for parameter in parameters:
            parent = getattr(parameter, "parent", None)
            key = getattr(parent, "id", None) if parent is not None else None
            if key not in index_by_key:
                index_by_key[key] = len(grouped)
                grouped.append((parent, []))
            grouped[index_by_key[key]][1].append(parameter)
        return grouped

    def _split_batches(self, parameters: List[Any], batch_size: int) -> List[List[Any]]:
        return [parameters[idx : idx + batch_size] for idx in range(0, len(parameters), batch_size)]

    def _run_single_analysis_for_category(
        self,
        *,
        review: Review,
        category,
        ingestion_job,
        parameters: List[Any],
        indexes,
        tsd_document,
        summary: AnalysisSummary,
        killed_assumptions_memory: deque,
    ) -> None:
        category_code = getattr(category, "code", None) or "unknown"
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
        if int(category_stats.get("analysis_total_count") or 0) == 0 and parameters:
            self._initialize_category_progress(
                summary=summary,
                category_code=category_code,
                total_count=len(parameters),
            )
            summary.analysis_total_parameters += len(parameters)
            summary.analysis_remaining_parameters += len(parameters)
        for idx, parameter in enumerate(parameters, start=1):
            self._raise_if_cancelled(review, phase="run.parameter_loop")
            self.logger.info(
                "TSDAnalysisPipeline.run: category=%s [%d/%d] parameter id=%s",
                getattr(category, "code", None),
                idx,
                len(parameters),
                parameter.id,
            )
            try:
                debate_output = self._analyze_single_child(
                    category=category,
                    ingestion_job=ingestion_job,
                    parameter=parameter,
                    indexes=indexes,
                    tsd_document=tsd_document,
                    killed_assumptions=list(killed_assumptions_memory),
                )
                self._raise_if_cancelled(review, phase="run.after_debate")
                debate_output = self._retry_if_needed(
                    category=category,
                    ingestion_job=ingestion_job,
                    parameter=parameter,
                    indexes=indexes,
                    tsd_document=tsd_document,
                    debate_output=debate_output,
                    killed_assumptions=list(killed_assumptions_memory),
                )
                killed_assumptions_memory.extend(
                    self._extract_killed_assumptions_from_output(debate_output, parameter)
                )
                debate_output.analysis_trace["killed_assumptions"] = list(killed_assumptions_memory)
                self._persist_debate_output(
                    review=review,
                    category=category,
                    ingestion_job=ingestion_job,
                    parameter=parameter,
                    indexes=indexes,
                    tsd_document=tsd_document,
                    debate_output=debate_output,
                    summary=summary,
                )
                self._record_parameter_progress(
                    summary=summary,
                    category_code=category_code,
                    parameter_id=parameter.id,
                )
            except Exception as exc:
                summary.error_count += 1
                self.logger.exception(
                    "TSDAnalysisPipeline.run: failed for parameter id=%s: %s",
                    parameter.id,
                    exc,
                )
                self._record_parameter_progress(
                    summary=summary,
                    category_code=category_code,
                    parameter_id=parameter.id,
                    extra_fields={"status": "error"},
                )

    def _run_batched_analysis_for_category(
        self,
        *,
        review: Review,
        category,
        ingestion_job,
        parameters: List[Any],
        indexes,
        tsd_document,
        summary: AnalysisSummary,
        killed_assumptions_memory: deque,
    ) -> None:
        start_ts = time.monotonic()
        category_code = getattr(category, "code", None) or "unknown"
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
        if int(category_stats.get("analysis_total_count") or 0) == 0 and parameters:
            self._initialize_category_progress(
                summary=summary,
                category_code=category_code,
                total_count=len(parameters),
            )
            summary.analysis_total_parameters += len(parameters)
            summary.analysis_remaining_parameters += len(parameters)
        batch_size = self._batch_size()
        max_concurrency = self._batch_max_concurrency()
        parent_groups = self._group_parameters_by_parent(parameters)
        batches: List[Tuple[Any, List[Any]]] = []
        for parent, children in parent_groups:
            for batch in self._split_batches(children, batch_size):
                batches.append((parent, batch))

        self.logger.info(
            "TSDAnalysisPipeline.batch: parent_groups=%d applicable_children=%d batches=%d batch_size=%d max_concurrency=%d category=%s",
            len(parent_groups),
            len(parameters),
            len(batches),
            batch_size,
            max_concurrency,
            category_code,
        )

        accepted_outputs: Dict[str, DebateOutput] = {}
        invalid_reasons: Dict[str, List[str]] = {}
        terminal_error_ids: set[str] = set()
        parent_context_cache: Dict[Tuple[Any, Any, Any], Any] = {}
        parent_context_by_key: Dict[Any, Any] = {}
        self._last_batch_concurrency_stats = {}
        parent_probe = ConcurrencyProbe(max_concurrency=max_concurrency)
        self.logger.info(
            "TSDAnalysisPipeline.batch.phase=parent_retrieval submitted=%d max_concurrency=%d category=%s",
            len(parent_groups),
            max_concurrency,
            category_code,
        )
        with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="ThreadPoolExecutor-7") as executor:
            parent_future_map = {}
            for parent, children in parent_groups:
                self._raise_if_cancelled(review, phase="batch.before_parent_retrieval")
                future = executor.submit(
                    parent_probe.wrap(self._get_parent_retrieval_result),
                    parent=parent,
                    child_parameters=children,
                    category=category,
                    ingestion_job=ingestion_job,
                    indexes=indexes,
                    tsd_document=tsd_document,
                    cache=parent_context_cache,
                )
                parent_future_map[future] = parent
            parent_probe.mark_submitted(len(parent_future_map))

            for future in as_completed(parent_future_map):
                if self._is_cancelled(review):
                    self._cancel_pending_futures(
                        executor=executor,
                        future_map=parent_future_map,
                        review_id=review.id,
                        phase="batch.parent_retrieval",
                    )
                    raise AnalysisCancelledError("Analysis was cancelled by user.")
                parent = parent_future_map[future]
                parent_key = getattr(parent, "id", None) or id(parent)
                parent_context_by_key[parent_key] = future.result()
        parent_stats = parent_probe.snapshot().to_dict()
        self._last_batch_concurrency_stats["parent_retrieval"] = parent_stats
        self.logger.info(
            "TSDAnalysisPipeline.batch.phase=parent_retrieval submitted=%d "
            "completed=%d failed=%d peak_in_flight=%d max_concurrency=%d "
            "elapsed_seconds=%.4f category=%s",
            parent_stats["submitted"],
            parent_stats["completed"],
            parent_stats["failed"],
            parent_stats["peak_in_flight"],
            parent_stats["max_concurrency"],
            parent_stats["elapsed_seconds"],
            category_code,
        )

        def _run_batch(parent, batch_parameters, killed_snapshot):
            if self._is_cancelled(review):
                return {}
            retrieval_result = parent_context_by_key[getattr(parent, "id", None) or id(parent)]
            debate_inputs = [
                self._build_debate_input_for_parameter(
                    parameter=parameter,
                    category=category,
                    retrieval_result=retrieval_result,
                    tsd_document=tsd_document,
                    killed_assumptions=killed_snapshot,
                )
                for parameter in batch_parameters
            ]
            return self.debate.run_batch_debate(
                debate_inputs=debate_inputs,
                retrieval_result=retrieval_result,
                tsd_document=tsd_document,
                enable_vision=bool(getattr(settings, "AI_VISION_ENABLED", True)),
            )

        batch_probe = ConcurrencyProbe(max_concurrency=max_concurrency)
        self.logger.info(
            "TSDAnalysisPipeline.batch.phase=batch_debate submitted=%d max_concurrency=%d category=%s",
            len(batches),
            max_concurrency,
            category_code,
        )
        with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="ThreadPoolExecutor-8") as executor:
            future_map = {}
            for parent, batch_parameters in batches:
                self._raise_if_cancelled(review, phase="batch.before_batch_submission")
                future = executor.submit(
                    batch_probe.wrap(_run_batch),
                    parent,
                    batch_parameters,
                    list(killed_assumptions_memory),
                )
                future_map[future] = batch_parameters
            batch_probe.mark_submitted(len(future_map))

            for future in as_completed(future_map):
                if self._is_cancelled(review):
                    self._cancel_pending_futures(
                        executor=executor,
                        future_map=future_map,
                        review_id=review.id,
                        phase="batch.batch_debate",
                    )
                    raise AnalysisCancelledError("Analysis was cancelled by user.")
                batch_parameters = future_map[future]
                try:
                    batch_outputs = future.result()
                except Exception as exc:
                    self.logger.exception(
                        "TSDAnalysisPipeline.batch: batch failed children=%s: %s",
                        [str(p.id) for p in batch_parameters],
                        exc,
                    )
                    for parameter in batch_parameters:
                        invalid_reasons[str(parameter.id)] = ["batch_exception"]
                    continue

                batch_valid, batch_invalid = self._validate_batch_outputs(
                    batch_parameters,
                    batch_outputs,
                )
                accepted_outputs.update(batch_valid)
                invalid_reasons.update(batch_invalid)
        batch_stats = batch_probe.snapshot().to_dict()
        self._last_batch_concurrency_stats["batch_debate"] = batch_stats
        self.logger.info(
            "TSDAnalysisPipeline.batch.phase=batch_debate submitted=%d "
            "completed=%d failed=%d peak_in_flight=%d max_concurrency=%d "
            "elapsed_seconds=%.4f category=%s",
            batch_stats["submitted"],
            batch_stats["completed"],
            batch_stats["failed"],
            batch_stats["peak_in_flight"],
            batch_stats["max_concurrency"],
            batch_stats["elapsed_seconds"],
            category_code,
        )

        fallback_count = 0
        fallback_enabled = bool(getattr(settings, "AI_BATCH_DEBATE_FALLBACK_ENABLED", True))
        final_outputs: Dict[str, DebateOutput] = {}
        fallback_parameters: List[Any] = []
        for parameter in parameters:
            child_id = str(parameter.id)
            output = accepted_outputs.get(child_id)
            if child_id in invalid_reasons:
                reasons = invalid_reasons[child_id]
                self.logger.warning(
                    "TSDAnalysisPipeline.batch: validation failed parameter=%s reasons=%s",
                    child_id,
                    reasons,
                )
                if (
                    self._ungrounded_not_met_policy() == "downgrade_na"
                    and set(reasons) == {"not_met_without_grounded_citations"}
                    and output is not None
                ):
                    final_outputs[child_id] = output
                    continue
                if fallback_enabled:
                    fallback_count += 1
                    fallback_parameters.append(parameter)
                    continue
                elif output is None:
                    summary.error_count += 1
                    terminal_error_ids.add(child_id)
                    continue
            if output is None:
                summary.error_count += 1
                terminal_error_ids.add(child_id)
                self.logger.warning(
                    "TSDAnalysisPipeline.batch: no final output for parameter=%s",
                    child_id,
                )
                continue
            final_outputs[child_id] = output

        if fallback_parameters:
            def _run_fallback(parameter, killed_snapshot):
                if self._is_cancelled(review):
                    return None
                parent = getattr(parameter, "parent", None)
                parent_key = getattr(parent, "id", None) or id(parent)
                retrieval_result = parent_context_by_key.get(parent_key)
                if retrieval_result is None:
                    return self._analyze_single_child(
                        category=category,
                        ingestion_job=ingestion_job,
                        parameter=parameter,
                        indexes=indexes,
                        tsd_document=tsd_document,
                        killed_assumptions=killed_snapshot,
                    )
                return self._analyze_single_child_with_retrieval_result(
                    category=category,
                    parameter=parameter,
                    retrieval_result=retrieval_result,
                    tsd_document=tsd_document,
                    killed_assumptions=killed_snapshot,
                )

            fallback_probe = ConcurrencyProbe(max_concurrency=max_concurrency)
            self.logger.info(
                "TSDAnalysisPipeline.batch.phase=fallback submitted=%d max_concurrency=%d category=%s",
                len(fallback_parameters),
                max_concurrency,
                category_code,
            )
            with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="ThreadPoolExecutor-9") as executor:
                future_map = {}
                for parameter in fallback_parameters:
                    self._raise_if_cancelled(review, phase="batch.before_fallback_submission")
                    future = executor.submit(
                        fallback_probe.wrap(_run_fallback),
                        parameter,
                        list(killed_assumptions_memory),
                    )
                    future_map[future] = parameter
                fallback_probe.mark_submitted(len(future_map))
                for future in as_completed(future_map):
                    if self._is_cancelled(review):
                        self._cancel_pending_futures(
                            executor=executor,
                            future_map=future_map,
                            review_id=review.id,
                            phase="batch.fallback",
                        )
                        raise AnalysisCancelledError("Analysis was cancelled by user.")
                    parameter = future_map[future]
                    child_id = str(parameter.id)
                    try:
                        output = future.result()
                        if output is not None:
                            final_outputs[child_id] = output
                    except Exception as exc:
                        summary.error_count += 1
                        terminal_error_ids.add(child_id)
                        self.logger.exception(
                            "TSDAnalysisPipeline.batch: fallback failed parameter=%s: %s",
                            child_id,
                            exc,
                        )
            fallback_stats = fallback_probe.snapshot().to_dict()
            self._last_batch_concurrency_stats["fallback"] = fallback_stats
            self.logger.info(
                "TSDAnalysisPipeline.batch.phase=fallback submitted=%d "
                "completed=%d failed=%d peak_in_flight=%d max_concurrency=%d "
                "elapsed_seconds=%.4f category=%s",
                fallback_stats["submitted"],
                fallback_stats["completed"],
                fallback_stats["failed"],
                fallback_stats["peak_in_flight"],
                fallback_stats["max_concurrency"],
                fallback_stats["elapsed_seconds"],
                category_code,
            )

        for parameter in parameters:
            self._raise_if_cancelled(review, phase="batch.before_persistence")
            output = final_outputs.get(str(parameter.id))
            if output is None:
                continue
            output = self._retry_if_needed(
                category=category,
                ingestion_job=ingestion_job,
                parameter=parameter,
                indexes=indexes,
                tsd_document=tsd_document,
                debate_output=output,
                killed_assumptions=list(killed_assumptions_memory),
            )
            killed_assumptions_memory.extend(
                self._extract_killed_assumptions_from_output(output, parameter)
            )
            output.analysis_trace["killed_assumptions"] = list(killed_assumptions_memory)
            self._persist_debate_output(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameter=parameter,
                indexes=indexes,
                tsd_document=tsd_document,
                debate_output=output,
                summary=summary,
            )
            self._record_parameter_progress(
                summary=summary,
                category_code=category_code,
                parameter_id=parameter.id,
                log_prefix="TSDAnalysisPipeline.batch",
                extra_fields={"final_child_output_count": len(final_outputs)},
            )

        for parameter in parameters:
            child_id = str(parameter.id)
            if child_id in final_outputs:
                continue
            if child_id not in terminal_error_ids:
                continue
            self._record_parameter_progress(
                summary=summary,
                category_code=category_code,
                parameter_id=parameter.id,
                log_prefix="TSDAnalysisPipeline.batch",
                extra_fields={"status": "terminal_error"},
            )

        self.logger.info(
            "TSDAnalysisPipeline.batch: fallback_count=%d final_child_output_count=%d elapsed_seconds=%.4f",
            fallback_count,
            len(final_outputs),
            time.monotonic() - start_ts,
        )

    def _analyze_single_child(
        self,
        *,
        category,
        ingestion_job,
        parameter,
        indexes,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
        enable_vision: bool = False,
    ) -> DebateOutput:
        if True:
            debate_input, retrieval_result = self._build_single_child_debate_input(
                parameter=parameter,
                category=category,
                ingestion_job=ingestion_job,
                indexes=indexes,
                tsd_document=tsd_document,
                killed_assumptions=killed_assumptions,
            )
        if True:
            return self.debate.run_debate(
                debate_input=debate_input,
                retrieval_result=retrieval_result,
                tsd_document=tsd_document,
                enable_vision=enable_vision,
            )

    def _analyze_single_child_with_retrieval_result(
        self,
        *,
        category,
        parameter,
        retrieval_result,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
        enable_vision: bool = False,
    ) -> DebateOutput:
        debate_input = self._build_debate_input_for_parameter(
            parameter=parameter,
            category=category,
            retrieval_result=retrieval_result,
            tsd_document=tsd_document,
            killed_assumptions=killed_assumptions,
        )
        if True:
            return self.debate.run_debate(
                debate_input=debate_input,
                retrieval_result=retrieval_result,
                tsd_document=tsd_document,
                enable_vision=enable_vision,
            )

    def _build_single_child_debate_input(
        self,
        *,
        parameter,
        category,
        ingestion_job,
        indexes,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
    ) -> Tuple[DebateInput, Any]:
        parameter_text = build_parameter_analysis_text(parameter).strip()
        parameter_section = parameter.parent.title if parameter.parent else "General"
        if True:
            contract = self._build_contract(
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                parent_description=(parameter.parent.description if parameter.parent else "") or "",
            )
        retrieval_query_details = self._build_retrieval_query_details(parameter, contract)
        retrieval_result = self.retrieval.retrieve_for_parameter(
            parameter=parameter,
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
            query_details=retrieval_query_details,
        )
        return (
            self._build_debate_input_for_parameter(
                parameter=parameter,
                category=category,
                retrieval_result=retrieval_result,
                tsd_document=tsd_document,
                killed_assumptions=killed_assumptions,
                contract=contract,
                retrieval_query_details=retrieval_query_details,
            ),
            retrieval_result,
        )

    def _build_debate_input_for_parameter(
        self,
        *,
        parameter,
        category,
        retrieval_result,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
        contract: Optional[dict] = None,
        retrieval_query_details: Optional[dict] = None,
    ) -> DebateInput:
        parameter_text = build_parameter_analysis_text(parameter).strip()
        parameter_section = parameter.parent.title if parameter.parent else "General"
        if contract is None:
            contract = self._build_contract(
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                parent_description=(parameter.parent.description if parameter.parent else "") or "",
            )
        if retrieval_query_details is None:
            retrieval_query_details = self._build_retrieval_query_details(parameter, contract)
        retrieval_metadata = dict(getattr(retrieval_result, "evidence_metadata", {}) or {})
        if retrieval_metadata:
            retrieval_query_details = {
                **retrieval_query_details,
                "retrieval_evidence_metadata": retrieval_metadata,
            }
        return DebateInput(
            parameter=parameter,
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
            hunter_plan={},
            retrieval_query_details=retrieval_query_details,
            killed_assumptions=list(killed_assumptions),
            context_chunks=self._build_xml_context_chunks(
                retrieval_result.context_chunks or [],
                retrieval_metadata=retrieval_metadata,
                tsd_document=tsd_document,
                source_block_ids=getattr(retrieval_result, "source_block_ids", []) or [],
            ),
            context_chunk_map=self._build_context_chunk_map(
                retrieval_result.context_chunks or [],
                retrieval_metadata=retrieval_metadata,
                tsd_document=tsd_document,
                source_block_ids=getattr(retrieval_result, "source_block_ids", []) or [],
            ),
            diagram_captions=self._resolve_diagram_captions(
                retrieval_result.get_diagram_block_ids() or [],
                tsd_document,
            ),
        )

    def _get_parent_retrieval_result(
        self,
        *,
        parent,
        child_parameters: List[Any],
        category,
        ingestion_job,
        indexes,
        tsd_document,
        cache: Dict[Tuple[Any, Any, Any], Any],
    ):
        cache_enabled = bool(
            getattr(settings, "AI_BATCH_DEBATE_PARENT_CONTEXT_CACHE_ENABLED", True)
        )
        key = (
            getattr(ingestion_job, "id", None),
            getattr(category, "id", None),
            getattr(parent, "id", None),
        )
        if cache_enabled and key in cache:
            self.logger.info(
                "TSDAnalysisPipeline.batch: parent_context_cache hit key=%s",
                key,
            )
            return cache[key]
        self.logger.info(
            "TSDAnalysisPipeline.batch: parent_context_cache miss key=%s",
            key,
        )
        query_details = self._build_parent_retrieval_query_details(parent, child_parameters)
        result = self.retrieval.retrieve_for_parent_group(
            parent=parent,
            child_parameters=child_parameters,
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
            query_details=query_details,
        )
        if cache_enabled:
            cache[key] = result
        return result

    def _build_parent_retrieval_query_details(self, parent, child_parameters: List[Any]) -> dict:
        domain_keywords = []
        child_requirements = []
        primary_domain = "general"
        secondary_domains: List[str] = []
        matched_domain_terms: Dict[str, List[str]] = {}
        classification_reason = "No child requirement text available."
        for parameter in child_parameters:
            text = build_parameter_analysis_text(parameter).strip()
            if text:
                child_requirements.append(text)
        if child_requirements:
            classification = classify_requirement_domain(
                child_requirement="\n".join(child_requirements),
                parent_title=(getattr(parent, "title", "") or "").strip(),
                parent_description=(getattr(parent, "description", "") or "").strip(),
            )
            primary_domain = classification.primary_domain
            secondary_domains = classification.secondary_domains
            matched_domain_terms = classification.matched_terms
            classification_reason = classification.reason
        domain_keywords.extend(DOMAIN_KEYWORDS.get(primary_domain, DOMAIN_KEYWORDS["general"]))
        return {
            "parent_title": (getattr(parent, "title", "") or "").strip(),
            "parent_description": (getattr(parent, "description", "") or "").strip(),
            "child_requirement": "\n".join(child_requirements),
            "domain_keywords": list(dict.fromkeys(domain_keywords)),
            "primary_domain": primary_domain,
            "secondary_domains": secondary_domains,
            "domain_classification_reason": classification_reason,
            "matched_domain_terms": matched_domain_terms,
            "generated_domain_keywords": list(dict.fromkeys(domain_keywords)),
            "retry_queries": [],
        }

    def _validate_batch_outputs(
        self,
        expected_parameters: List[Any],
        batch_outputs: Dict[str, DebateOutput],
    ) -> Tuple[Dict[str, DebateOutput], Dict[str, List[str]]]:
        expected_ids = [str(parameter.id) for parameter in expected_parameters]
        expected_set = set(expected_ids)
        accepted: Dict[str, DebateOutput] = {}
        invalid: Dict[str, List[str]] = {}
        threshold = float(
            getattr(settings, "AI_BATCH_DEBATE_CONFIDENCE_THRESHOLD", 0.75)
        )
        soft_threshold = float(
            getattr(settings, "AI_BATCH_DEBATE_SOFT_CONFIDENCE_THRESHOLD", 0.65)
        )

        for child_id in expected_ids:
            output = batch_outputs.get(child_id)
            reasons, soft_reject = self._validate_single_batch_output(
                child_id, output, threshold, soft_threshold
            )
            if reasons or soft_reject:
                invalid[child_id] = reasons
            elif output is not None:
                accepted[child_id] = output

        for child_id in batch_outputs:
            if child_id not in expected_set:
                invalid.setdefault(child_id, []).append("unknown_child_id")
        missing = [child_id for child_id in expected_ids if child_id not in batch_outputs]
        for child_id in missing:
            invalid.setdefault(child_id, []).append("missing_result")
        return accepted, invalid

    def _validate_single_batch_output(
        self,
        child_id: str,
        output: Optional[DebateOutput],
        threshold: float,
        soft_threshold: float,
    ) -> Tuple[List[str], bool]:
        if output is None:
            return ["missing_result"], True
        reasons: List[str] = []
        mediator = output.mediator_result
        hunter = output.hunter_result
        critic = output.critic_result
        verdict = getattr(mediator, "final_verdict", None)
        if verdict not in {"met", "not_met", "na"}:
            reasons.append("invalid_verdict")
        if float(getattr(mediator, "confidence", 0.0) or 0.0) < threshold:
            reasons.append("low_confidence")
        reasoning = (
            getattr(mediator, "logic_summary", None)
            or getattr(mediator, "reasoning", None)
            or ""
        ).strip()
        if self._is_weak_or_generic_reasoning(reasoning):
            reasons.append("weak_or_generic_reasoning")
        if verdict == "met" and not getattr(mediator, "final_citations", []):
            reasons.append("met_without_grounded_citations")
        if (
            verdict == "not_met"
            and bool(getattr(settings, "AI_BATCH_DEBATE_REQUIRE_CITATIONS_FOR_NOT_MET", True))
            and not self._has_grounded_citations(output)
        ):
            reasons.append("not_met_without_grounded_citations")
        if getattr(hunter, "verdict", None) == "met" and not getattr(hunter, "citations", []):
            reasons.append("weak_evidence")
        if getattr(critic, "weak_evidence", None) and verdict == "met":
            reasons.append("weak_evidence")
        allowed_ids = set(output.analysis_trace.get("retrieved_chunk_ids", []) or [])
        for citation in getattr(mediator, "final_citations", []) or []:
            if citation.block_id not in allowed_ids:
                reasons.append("invalid_citations")
                break
        if self._appears_to_cover_multiple_children(reasoning, child_id):
            reasons.append("generic_multi_child_result")
        deduped_reasons = list(dict.fromkeys(reasons))
        if (
            self._ungrounded_not_met_policy() == "downgrade_na"
            and set(deduped_reasons) == {"not_met_without_grounded_citations"}
        ):
            return [], False
        hard_invalid_reasons = {
            "invalid_verdict",
            "invalid_citations",
            "missing_result",
            "met_without_grounded_citations",
        }
        if self._ungrounded_not_met_policy() == "always_fallback":
            hard_invalid_reasons.add("not_met_without_grounded_citations")
        if any(reason in hard_invalid_reasons for reason in deduped_reasons):
            return deduped_reasons, True
        confidence = float(getattr(mediator, "confidence", 0.0) or 0.0)
        low_conf_only = deduped_reasons == ["low_confidence"]
        if low_conf_only and confidence >= soft_threshold and self._has_grounded_citations(output):
            return [], False
        return deduped_reasons, bool(deduped_reasons)

    def _has_grounded_citations(self, output: DebateOutput) -> bool:
        allowed_ids = set(output.analysis_trace.get("retrieved_chunk_ids", []) or [])
        # Check mediator's final citations first
        for citation in getattr(output.mediator_result, "final_citations", []) or []:
            if citation.block_id and citation.block_id in allowed_ids:
                return True
        # Also check Critic's validated citations — the Mediator's reconciliation
        # can legitimately collapse these, so don't penalise the finding for it
        for citation in getattr(output.critic_result, "valid_citations", []) or []:
            if citation.block_id and citation.block_id in allowed_ids:
                return True
        return False

    def _is_weak_or_generic_reasoning(self, reasoning: str) -> bool:
        lowered = reasoning.lower()
        if len(reasoning) < 60:
            return True
        generic_markers = {
            "as above",
            "same as previous",
            "all requirements",
            "the batch",
            "each child",
            "all children",
            "no reasoning provided",
        }
        return any(marker in lowered for marker in generic_markers)

    def _appears_to_cover_multiple_children(self, reasoning: str, child_id: str) -> bool:
        lowered = reasoning.lower()
        return (
            "all child" in lowered
            or "multiple child" in lowered
            or "the same evidence applies" in lowered
        )

    def _persist_debate_output(
        self,
        *,
        review: Review,
        category,
        ingestion_job,
        parameter,
        indexes,
        tsd_document,
        debate_output: DebateOutput,
        summary: AnalysisSummary,
    ) -> None:
        gated_output = self._apply_not_met_evidence_gate(
            category=category,
            ingestion_job=ingestion_job,
            parameter=parameter,
            indexes=indexes,
            tsd_document=tsd_document,
            debate_output=debate_output,
        )
        persistence_input = PersistenceInput(
            parameter=parameter,
            category=category,
            ingestion_job=ingestion_job,
            debate_output=gated_output,
        )
        self.persistence.persist_finding(review, persistence_input, summary)

    def _ungrounded_not_met_policy(self) -> str:
        raw = str(getattr(settings, "AI_BATCH_DEBATE_UNGROUNDED_NOT_MET_POLICY", "preserve_not_met") or "").strip().lower()
        if raw in {"downgrade_na", "selective_fallback", "always_fallback", "preserve_not_met"}:
            return raw
        return "preserve_not_met"

    def _apply_not_met_evidence_gate(
        self,
        *,
        category,
        ingestion_job,
        parameter,
        indexes,
        tsd_document,
        debate_output: DebateOutput,
    ) -> DebateOutput:
        debate_output.analysis_trace = dict(getattr(debate_output, "analysis_trace", {}) or {})
        mediator = debate_output.mediator_result
        verdict = getattr(mediator, "final_verdict", None)
        if verdict != "not_met":
            debate_output.analysis_trace["evidence_gate_attempted"] = False
            return debate_output
        if self._has_grounded_citations(debate_output):
            debate_output.analysis_trace["evidence_gate_attempted"] = False
            return debate_output

        evidence_quality = self._extract_retrieval_evidence_quality(debate_output.analysis_trace)
        implementation_count = int(evidence_quality.get("implementation_evidence_count") or 0)
        applicability_signal = bool(evidence_quality.get("applicability_signal"))
        if evidence_quality and implementation_count == 0 and not applicability_signal:
            reason = self._build_missing_evidence_reasoning(
                parameter,
                evidence_quality,
                applicability_established=False,
            )
            mediator.final_verdict = "na"
            mediator.raw_final_verdict = "na"
            mediator.severity = None
            mediator.recommendation = None
            mediator.reasoning = reason
            mediator.logic_summary = reason
            debate_output.analysis_trace["evidence_gate_attempted"] = True
            debate_output.analysis_trace["evidence_gate_outcome"] = "downgraded_to_na_no_applicability_signal"
            debate_output.analysis_trace["downgraded_due_to_missing_citations"] = True
            debate_output.analysis_trace["downgrade_reason"] = "not_met_without_applicability_or_implementation_evidence"
            return debate_output

        if evidence_quality and implementation_count == 0:
            reason = self._build_missing_evidence_reasoning(
                parameter,
                evidence_quality,
                applicability_established=True,
            )
            mediator.reasoning = reason
            mediator.logic_summary = reason
            debate_output.analysis_trace["evidence_gate_outcome"] = "missing_implementation_evidence_preserved_not_met"

        # With preserve_not_met policy, skip all downgrade logic entirely
        if self._ungrounded_not_met_policy() == "preserve_not_met":
            debate_output.analysis_trace["evidence_gate_attempted"] = bool(
                debate_output.analysis_trace.get("evidence_gate_outcome")
            )
            debate_output.analysis_trace.setdefault(
                "evidence_gate_outcome",
                "skipped_preserve_not_met_policy",
            )
            return debate_output

        debate_output.analysis_trace["evidence_gate_attempted"] = True
        retry_context_available = all(value is not None for value in [ingestion_job, indexes, tsd_document])
        debate_output.analysis_trace["evidence_gate_retry_context_available"] = retry_context_available
        if not retry_context_available:
            debate_output.analysis_trace["evidence_gate_outcome"] = "no_retry_context_preserved_not_met"
            debate_output.analysis_trace["downgraded_due_to_missing_citations"] = False
            debate_output.analysis_trace["downgrade_reason"] = None
            return debate_output

        downgrade_policy = self._ungrounded_not_met_policy()
        retry_output = debate_output
        try:
            retry_details = dict(
                self._build_retrieval_query_details(
                    parameter, debate_output.analysis_trace.get("contract") or {}
                )
            )
            retry_details["retry_queries"] = [
                {
                    "attempt": 1,
                    "reason": "evidence_gate_retry_for_ungrounded_not_met",
                    "primary_domain": retry_details.get("primary_domain"),
                    "keywords": retry_details.get("generated_domain_keywords", []),
                }
            ]
            retrieval_result = self.retrieval.retrieve_for_parameter(
                parameter=parameter,
                category=category,
                ingestion_job=ingestion_job,
                indexes=indexes,
                tsd_document=tsd_document,
                query_details=retry_details,
            )
            retry_input = self._build_debate_input_for_parameter(
                parameter=parameter,
                category=category,
                retrieval_result=retrieval_result,
                tsd_document=tsd_document,
                killed_assumptions=[],
                contract=debate_output.analysis_trace.get("contract") or {},
                retrieval_query_details=retry_details,
            )
            retry_output = self.debate.run_debate(
                debate_input=retry_input,
                retrieval_result=retrieval_result,
                tsd_document=tsd_document,
                enable_vision=bool(getattr(settings, "AI_VISION_ENABLED", True)),
            )
        except Exception as exc:
            self.logger.warning(
                "TSDAnalysisPipeline._apply_not_met_evidence_gate: retry failed parameter=%s: %s",
                parameter.id,
                exc,
            )
            retry_output.analysis_trace = dict(getattr(retry_output, "analysis_trace", {}) or {})
            retry_output.analysis_trace["evidence_gate_outcome"] = "retry_failed_preserved_not_met"
            retry_output.analysis_trace["downgraded_due_to_missing_citations"] = False
            retry_output.analysis_trace["downgrade_reason"] = None
            return retry_output
        if self._has_grounded_citations(retry_output):
            retry_output.analysis_trace = dict(getattr(retry_output, "analysis_trace", {}) or {})
            retry_output.analysis_trace["evidence_gate_outcome"] = "recovered_with_citations"
            retry_output.analysis_trace["downgraded_due_to_missing_citations"] = False
            return retry_output

        retry_trace = dict(getattr(retry_output, "analysis_trace", {}) or {})
        verdict_policy = retry_trace.get("verdict_policy") or {}
        applicability_established = bool(verdict_policy.get("applicability_established", True))
        retry_evidence_quality = self._extract_retrieval_evidence_quality(retry_trace)
        if (
            retry_evidence_quality
            and
            int(retry_evidence_quality.get("implementation_evidence_count") or 0) == 0
            and not bool(retry_evidence_quality.get("applicability_signal"))
        ):
            retry_output.analysis_trace = retry_trace
            reason = self._build_missing_evidence_reasoning(
                parameter,
                retry_evidence_quality,
                applicability_established=False,
            )
            retry_output.mediator_result.final_verdict = "na"
            retry_output.mediator_result.raw_final_verdict = "na"
            retry_output.mediator_result.severity = None
            retry_output.mediator_result.recommendation = None
            retry_output.mediator_result.reasoning = reason
            retry_output.mediator_result.logic_summary = reason
            retry_output.analysis_trace["evidence_gate_outcome"] = "downgraded_to_na_no_applicability_signal_after_retry"
            retry_output.analysis_trace["downgraded_due_to_missing_citations"] = True
            retry_output.analysis_trace["downgrade_reason"] = "not_met_without_applicability_or_implementation_evidence_after_retry"
            return retry_output
        if downgrade_policy == "selective_fallback" and not applicability_established:
            retry_output.analysis_trace = retry_trace
            retry_output.mediator_result.final_verdict = "na"
            retry_output.mediator_result.raw_final_verdict = "na"
            retry_output.mediator_result.severity = None
            retry_output.mediator_result.recommendation = None
            retry_output.mediator_result.reasoning = (
                (retry_output.mediator_result.reasoning or "").strip()
                + "\n\nInsufficient grounded evidence found, and applicability was not established."
            ).strip()
            retry_output.analysis_trace["evidence_gate_outcome"] = "downgraded_to_na_applicability_not_established"
            retry_output.analysis_trace["downgraded_due_to_missing_citations"] = True
            retry_output.analysis_trace["downgrade_reason"] = "not_met_without_grounded_citations_or_applicability_after_retry"
            return retry_output

        if downgrade_policy != "downgrade_na":
            retry_output.analysis_trace = dict(getattr(retry_output, "analysis_trace", {}) or {})
            retry_output.analysis_trace["evidence_gate_outcome"] = "retry_exhausted_preserved_not_met"
            retry_output.analysis_trace["downgraded_due_to_missing_citations"] = False
            retry_output.analysis_trace["downgrade_reason"] = None
            return retry_output

        retry_output.analysis_trace = dict(getattr(retry_output, "analysis_trace", {}) or {})
        retry_output.mediator_result.final_verdict = "na"
        retry_output.mediator_result.raw_final_verdict = "na"
        retry_output.mediator_result.severity = None
        retry_output.mediator_result.recommendation = None
        retry_output.mediator_result.reasoning = (
            (retry_output.mediator_result.reasoning or "").strip()
            + "\n\nInsufficient grounded evidence found to support a 'not_met' determination."
        ).strip()
        retry_output.analysis_trace["evidence_gate_outcome"] = "downgraded_to_na_missing_citations"
        retry_output.analysis_trace["downgraded_due_to_missing_citations"] = True
        retry_output.analysis_trace["downgrade_reason"] = "not_met_without_grounded_citations_after_retry"
        return retry_output

    def _extract_retrieval_evidence_quality(self, analysis_trace: dict) -> Dict[str, Any]:
        details = (analysis_trace or {}).get("retrieval_query_details") or {}
        retrieval_metadata = details.get("retrieval_evidence_metadata") or {}
        return dict(retrieval_metadata.get("evidence_quality") or {})

    def _build_missing_evidence_reasoning(
        self,
        parameter,
        evidence_quality: Dict[str, Any],
        *,
        applicability_established: bool,
    ) -> str:
        requirement = build_parameter_analysis_text(parameter).strip().splitlines()
        requirement_title = requirement[0] if requirement else "the requirement"
        if len(requirement_title) > 160:
            requirement_title = f"{requirement_title[:157]}..."
        terms = ", ".join(evidence_quality.get("applicability_terms") or [])
        if not terms:
            terms = "no direct scope terms"
        counts = evidence_quality.get("counts") or {}
        weak_parts = [
            f"{kind}={count}"
            for kind, count in sorted(counts.items())
            if count
        ]
        retrieved_summary = ", ".join(weak_parts) or "no usable retrieved chunks"
        if not applicability_established:
            return (
                f"The retrieved TSD context does not establish that '{requirement_title}' applies to this design. "
                f"The strongest retrieval signals were {terms}, and the returned material was {retrieved_summary}; "
                "there was no citation-grade implementation evidence showing the control is in scope. "
                "A security reviewer would mark this not assessable instead of treating absence of evidence as a control failure."
            )
        return (
            f"The requirement appears applicable based on retrieved TSD signals ({terms}), but no citation-grade "
            f"implementation evidence was found for '{requirement_title}'. The returned material was {retrieved_summary}; "
            "a security reviewer would expect explicit design evidence such as the configured control, validation behavior, "
            "enforcement point, or responsible component."
        )

    def _resolve_diagram_captions(
        self, diagram_block_ids: list, tsd_document
    ) -> list:
        """Resolves diagram captions from block_ids."""
        captions: list = []
        for block_id in diagram_block_ids:
            if "_d" not in block_id:
                continue
            diagram_block = tsd_document.get_diagram_by_id(block_id)
            if diagram_block and diagram_block.caption and diagram_block.caption.strip():
                captions.append(diagram_block.caption)
        return captions

    def _generate_overview(self, review: Review, summary: AnalysisSummary) -> Optional[str]:
        """Generates the executive overview."""
        from sdr.apps.ai.prompts.agent_prompt import (
            OVERVIEW_SYSTEM_PROMPT,
            build_overview_prompt,
        )
        from sdr.apps.ai.client import chat_completion

        try:
            category_name = "Unknown"
            selected_categories = list(review.selected_categories)

            if selected_categories:
                category_name = selected_categories[0].name
            elif review.ingestion_job and review.ingestion_job.category:
                category_name = review.ingestion_job.category.name

            prompt = build_overview_prompt(
                design_name=review.design.name,
                category_name=category_name,
                total_parameters=summary.total_parameters,
                met_count=summary.met_count,
                not_met_count=summary.not_met_count,
                na_count=summary.na_count,
                critical_findings=summary.critical_findings[:10],
                high_findings=summary.high_findings[:10],
            )

            response = chat_completion(
                messages=[
                    {"role": "system", "content": OVERVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                component="orchestrator",
                temperature=0.2,
                max_tokens=1024,
            )

            if response.error or not response.content:
                return None

            return (response.content or "").strip()

        except Exception as exc:
            self.logger.exception(
                "TSDAnalysisPipeline._generate_overview: failed: %s", exc
            )
            return None

    def _complete_review(self, review: Review, summary: AnalysisSummary) -> None:
        """Marks review as COMPLETED."""
        try:
            if self._is_cancelled(review):
                self.logger.warning(
                    "TSDAnalysisPipeline._complete_review: cancellation detected, skipping completion for review_id=%s",
                    review.id,
                )
                return
            from sdr.core.database import SessionLocal
            from sqlalchemy import update
            with SessionLocal() as db:
                if int(summary.not_met_count or 0) > 0:
                    new_status = ReviewStatus.COMPLETED_WITH_FINDINGS.value
                else:
                    new_status = ReviewStatus.COMPLETED_CLEAN.value
                now = datetime.now(timezone.utc)
                summary_dict = summary.to_dict()
                db.execute(update(Review).where(Review.id == review.id).values(
                    status=new_status,
                    completed_at=now,
                    summary_json=summary_dict
                ))
                db.commit()
                review.status = new_status
                review.completed_at = now
                review.summary_json = summary_dict
            self.logger.info(
                "TSDAnalysisPipeline._complete_review: [SUCCESS] review_id=%s",
                review.id,
            )
        except Exception as exc:
            self.logger.exception(
                "TSDAnalysisPipeline._complete_review: failed: %s", exc
            )

    def _fail_review(self, review: Review, error_message: str) -> None:
        """Marks review as FAILED."""
        try:
            if self._is_cancelled(review):
                self.logger.warning(
                    "TSDAnalysisPipeline._fail_review: skipping failed status because review_id=%s is cancelled",
                    review.id,
                )
                return
            from sdr.core.database import SessionLocal
            from sqlalchemy import update
            with SessionLocal() as db:
                new_status = ReviewStatus.FAILED.value if hasattr(ReviewStatus, 'value') else ReviewStatus.FAILED
                now = datetime.now(timezone.utc)
                db.execute(update(Review).where(Review.id == review.id).values(
                    status=new_status,
                    completed_at=now,
                    error_message=error_message
                ))
                db.commit()
                review.status = new_status
                review.completed_at = now
                review.error_message = error_message
            self.logger.error(
                "TSDAnalysisPipeline._fail_review: [FAILED] review_id=%s reason='%s'",
                review.id,
                error_message,
            )
        except Exception as exc:
            self.logger.exception(
                "TSDAnalysisPipeline._fail_review: could not mark failed: %s", exc
            )

    def _is_cancelled(self, review: Review) -> bool:
        """
        Detects user-initiated cancellation persisted on the Review row.
        """
        from sdr.core.database import SessionLocal
        from sqlalchemy import select
        with SessionLocal() as db:
            latest = db.execute(select(Review).where(Review.id == review.id)).scalars().first()
            if not latest:
                return False
            if latest.status == ReviewStatus.CANCELLED.value:
                return True
            return (
                latest.status == (ReviewStatus.FAILED.value if hasattr(ReviewStatus, 'value') else ReviewStatus.FAILED)
                and (latest.error_message or "").strip().lower().startswith("analysis was cancelled")
            )

    def _build_contract(
        self,
        parameter_text: str,
        parameter_section: str,
        parent_description: str = "",
    ) -> dict:
        return self.contract_synthesizer.synthesize(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            parent_description=parent_description,
        )

    def _build_retrieval_query_details(self, parameter, contract: Optional[dict] = None) -> dict:
        def _to_text(value) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (list, tuple)):
                parts = [str(item).strip() for item in value if str(item).strip()]
                return " ".join(parts).strip()
            return str(value).strip()

        def _to_text_list(value) -> list:
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return [str(item).strip() for item in value if str(item).strip()]
            text = _to_text(value)
            return [text] if text else []

        parent = getattr(parameter, "parent", None)
        parameter_text = build_parameter_analysis_text(parameter).strip()
        classification = classify_requirement_domain(
            child_requirement=parameter_text,
            parent_title=(getattr(parent, "title", "") or "").strip(),
            parent_description=(getattr(parent, "description", "") or "").strip(),
            extra_parts=[_to_text((contract or {}).get("then"))],
        )
        domain = _to_text((contract or {}).get("domain")) or classification.primary_domain or "general"
        domain_keywords = list(DOMAIN_KEYWORDS.get(domain, DOMAIN_KEYWORDS["general"]))
        return {
            "parent_title": (getattr(parent, "title", "") or "").strip(),
            "parent_description": (getattr(parent, "description", "") or "").strip(),
            "child_requirement": parameter_text,
            "contract_then": _to_text((contract or {}).get("then")),
            "contract_not_sufficient": _to_text_list((contract or {}).get("not_sufficient")),
            "domain_keywords": domain_keywords,
            "domain_signal": domain,
            "primary_domain": classification.primary_domain,
            "secondary_domains": classification.secondary_domains,
            "domain_classification_reason": classification.reason,
            "matched_domain_terms": classification.matched_terms,
            "generated_domain_keywords": domain_keywords,
            "retry_queries": [],
        }

    def _is_concurrency_domain(self, details: Dict[str, Any]) -> bool:
        return (details.get("primary_domain") or details.get("domain_signal")) in {
            "business_logic_concurrency",
            "transaction_integrity",
        }

    def _is_restatement_or_weak_not_met(self, output: DebateOutput) -> bool:
        if getattr(output.hunter_result, "verdict", None) != "not_met":
            return False
        reasoning_parts = [
            getattr(output.hunter_result, "reasoning", "") or "",
            getattr(output.critic_result, "reasoning", "") or "",
            getattr(output.mediator_result, "reasoning", "") or "",
        ]
        text = " ".join(reasoning_parts).lower()
        weak_markers = [
            "restatement",
            "generic",
            "no explicit evidence",
            "requirement text",
            "policy statement",
            "aspirational",
        ]
        return any(marker in text for marker in weak_markers) or not getattr(output.critic_result, "valid_citations", [])

    def _retry_if_needed(
        self,
        *,
        category,
        ingestion_job,
        parameter,
        indexes,
        tsd_document,
        debate_output: DebateOutput,
        killed_assumptions: List[Dict[str, Any]],
    ) -> DebateOutput:
        details = debate_output.analysis_trace.get("retrieval_query_details", {}) or {}
        if not self._is_concurrency_domain(details):
            return debate_output
        if not self._is_restatement_or_weak_not_met(debate_output):
            return debate_output
        if details.get("retry_queries"):
            return debate_output

        retry_details = dict(self._build_retrieval_query_details(parameter, debate_output.analysis_trace.get("contract") or {}))
        retry_details["retry_queries"] = [
            {
                "attempt": 1,
                "reason": "targeted_concurrency_retry_after_weak_not_met",
                "primary_domain": retry_details.get("primary_domain"),
                "keywords": retry_details.get("generated_domain_keywords", []),
            }
        ]
        retrieval_result = self.retrieval.retrieve_for_parameter(
            parameter=parameter,
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
            query_details=retry_details,
        )
        retry_input = self._build_debate_input_for_parameter(
            parameter=parameter,
            category=category,
            retrieval_result=retrieval_result,
            tsd_document=tsd_document,
            killed_assumptions=killed_assumptions,
            contract=debate_output.analysis_trace.get("contract") or {},
            retrieval_query_details=retry_details,
        )
        retry_output = self.debate.run_debate(
            debate_input=retry_input,
            retrieval_result=retrieval_result,
            tsd_document=tsd_document,
            enable_vision=bool(getattr(settings, "AI_VISION_ENABLED", True)),
        )
        retry_output.analysis_trace = dict(getattr(retry_output, "analysis_trace", {}) or {})
        retry_output.analysis_trace["retrieval_query_details"] = retry_details
        return retry_output

    def _extract_killed_assumptions_from_output(self, debate_output, parameter) -> list:
        killed = []
        invalid_ids = list(getattr(debate_output.critic_result, "invalid_citation_ids", []) or [])
        for citation_id in invalid_ids:
            killed.append(
                {
                    "parameter_id": str(parameter.id),
                    "assumption": f"Citation {citation_id} is invalid as supporting evidence.",
                    "reason": "critic_invalid_citation",
                }
            )
        if not getattr(debate_output.critic_result, "valid_citations", []):
            killed.append(
                {
                    "parameter_id": str(parameter.id),
                    "assumption": "Claims without critic-validated citations are insufficient.",
                    "reason": "mediator_evidence_policy",
                }
            )
        return killed

    def _extract_chunk_id(self, chunk_text: str, fallback_idx: int) -> str:
        match = re.search(r"\b(p\d+_[bd]\d+)\b", chunk_text or "")
        if match:
            return match.group(1)
        return f"chunk_{fallback_idx}"

    def _build_context_chunk_map(
        self,
        context_chunks: list,
        retrieval_metadata: Optional[dict] = None,
        tsd_document=None,
        source_block_ids: Optional[list] = None,
    ) -> dict:
        chunk_map = {}
        evidence_quality = (retrieval_metadata or {}).get("evidence_quality") or {}
        for idx, chunk in enumerate(context_chunks, start=1):
            chunk_id = self._extract_chunk_id(chunk, idx)
            evidence_kind = self._classify_context_chunk_text(chunk)
            if evidence_kind == "graph_summary" and chunk_id.startswith("chunk_"):
                chunk_id = f"graph_summary_{idx}"
            elif chunk_id.startswith("chunk_") and source_block_ids and idx <= len(source_block_ids):
                chunk_id = source_block_ids[idx - 1]
            source_location = self._resolve_chunk_source_location(chunk_id, tsd_document)
            chunk_map[chunk_id] = {
                "source": "retrieval_context",
                "section": source_location.get("section") or "unknown",
                "text": chunk,
                "evidence_kind": evidence_kind,
                "citation_grade": evidence_kind != "graph_summary" and not chunk_id.startswith("graph_summary_"),
                "evidence_quality": evidence_quality,
                **source_location,
            }
        for block_id in source_block_ids or []:
            if not block_id or block_id in chunk_map or "_d" in block_id:
                continue
            source_location = self._resolve_chunk_source_location(block_id, tsd_document)
            text = ""
            try:
                block = tsd_document.get_block_by_id(block_id) if tsd_document is not None else None
                text = getattr(block, "text", "") or ""
            except Exception:
                text = ""
            if not text:
                continue
            chunk_map[block_id] = {
                "source": "retrieval_grounded_source",
                "section": source_location.get("section") or "unknown",
                "text": text,
                "evidence_kind": "grounded_text_block",
                "citation_grade": True,
                "evidence_quality": evidence_quality,
                **source_location,
            }
        return chunk_map

    def _resolve_chunk_source_location(self, chunk_id: str, tsd_document) -> Dict[str, Any]:
        if not chunk_id or tsd_document is None:
            return {}
        block = None
        try:
            if "_d" in chunk_id and hasattr(tsd_document, "get_diagram_by_id"):
                block = tsd_document.get_diagram_by_id(chunk_id)
            elif hasattr(tsd_document, "get_block_by_id"):
                block = tsd_document.get_block_by_id(chunk_id)
        except Exception:
            return {}
        if block is None:
            return {}
        page_number = getattr(block, "page_number", None)
        bbox = {
            "x0": getattr(block, "bbox_x0", None),
            "y0": getattr(block, "bbox_y0", None),
            "x1": getattr(block, "bbox_x1", None),
            "y1": getattr(block, "bbox_y1", None),
        }
        return {
            "page": page_number,
            "page_number": page_number,
            "bbox": bbox,
            "bbox_x0": bbox["x0"],
            "bbox_y0": bbox["y0"],
            "bbox_x1": bbox["x1"],
            "bbox_y1": bbox["y1"],
            "section": getattr(block, "section_heading", None),
        }

    def _classify_context_chunk_text(self, chunk: str) -> str:
        text = (chunk or "").strip()
        lowered = text.lower()
        if not text:
            return "empty"
        if text.startswith("--- VECTOR RESULT"):
            return "baseline_requirement"
        if text.startswith("--- GRAPH RESULT") or text.startswith("--- GRAPH PATH") or lowered.startswith("graph node:"):
            return "graph_summary"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(text) < 120 and len(lines) <= 2:
            return "heading_only"
        if re.search(
            r"\b(use|uses|using|implemented|configured|enabled|enforced|validated|verified|required|requires|oauth|oidc|token|jwt|pkce|jwks|mfa|rbac|encrypt|encrypted)\b",
            lowered,
        ):
            return "implementation_or_scope_context"
        return "weak_context"

    def _build_xml_context_chunks(
        self,
        context_chunks: list,
        retrieval_metadata: Optional[dict] = None,
        tsd_document=None,
        source_block_ids: Optional[list] = None,
    ) -> list:
        chunk_map = self._build_context_chunk_map(
            context_chunks,
            retrieval_metadata=retrieval_metadata,
            tsd_document=tsd_document,
            source_block_ids=source_block_ids,
        )
        xml_chunks = []
        for chunk_id, payload in chunk_map.items():
            xml_chunks.append(
                "\n".join(
                    [
                        f'<CONTEXT_CHUNK id="{chunk_id}" source="{payload["source"]}" section="{payload["section"]}">',
                        payload["text"],
                        "</CONTEXT_CHUNK>",
                    ]
                )
            )
        return xml_chunks


def run_tsd_analysis(review: Review) -> AnalysisSummary:
    """
    Module-level convenience function — single entry point.
    Called by apps/reviews/tasks.py (Celery task).
    """
    logger.info("run_tsd_analysis: [ENTRY] review_id=%s", review.id)
    pipeline = TSDAnalysisPipeline()
    summary = pipeline.run(review)
    logger.info(
        "run_tsd_analysis: [EXIT] review_id=%s summary=%s",
        review.id,
        summary.to_dict(),
    )
    return summary
