from __future__ import annotations

import logging
import time
from copy import deepcopy
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from sdr.apps.ai.agents.base import CriticResult, HunterResult, MediatorResult
from sdr.apps.ai.client.session import capture_current_context
from sdr.apps.ai.engine.dto import AnalysisSummary, DebateInput, DebateOutput, PersistenceInput
from sdr.apps.ai.utils.concurrency import ConcurrencyProbe
from sdr.apps.reviews.models import Review
from sdr.apps.reviews.services.debate_events import build_debate_id, review_debate_event_store
from sdr.apps.standards.utils import build_parameter_analysis_text

from sdr.apps.ai.engine.persistence.review_run_state_service import AnalysisCancelledError


class TextDebateCoordinator:
    def __init__(
        self,
        *,
        config,
        retrieval_service,
        debate_service,
        persistence_service,
        debate_input_factory,
        progress_service,
        run_state_service,
        mediator_agent_factory,
    ) -> None:
        self.config = config
        self.retrieval = retrieval_service
        self.debate = debate_service
        self.persistence = persistence_service
        self.debate_input_factory = debate_input_factory
        self.progress_service = progress_service
        self.run_state = run_state_service
        self.mediator_agent_factory = mediator_agent_factory
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.last_batch_concurrency_stats: Dict[str, Dict[str, Any]] = {}

    def run_single_analysis_for_category(
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
        self.seed_live_debates(
            review=review,
            category_code=category_code,
            parameters=parameters,
            execution_mode="single",
        )
        self.progress_service.register_analysis_work(
            summary=summary,
            category_code=category_code,
            total_count=len(parameters),
        )
        self.run_state.persist_summary_snapshot(review, summary)
        max_concurrency = self.config.batch_debate_max_concurrency
        killed_snapshot = list(killed_assumptions_memory)

        def _run_single(parameter):
            if self.run_state.is_cancelled(review):
                return None
            self.publish_work_phase(
                review=review,
                parameter=parameter,
                debate_id=self.build_live_debate_id(parameter),
                work_phase="retrieval",
                last_snippet="Retrieving and ranking supporting evidence.",
                progress_percent=2,
            )
            return self.analyze_single_child(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameter=parameter,
                indexes=indexes,
                tsd_document=tsd_document,
                killed_assumptions=killed_snapshot,
                execution_mode="single",
            )

        single_probe = ConcurrencyProbe(max_concurrency=max_concurrency)
        with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="SingleDebate") as executor:
            future_map = {}
            for parameter in parameters:
                self.run_state.raise_if_cancelled(review, phase="single.before_submission")
                future = executor.submit(capture_current_context(single_probe.wrap(_run_single)), parameter)
                future_map[future] = parameter
            single_probe.mark_submitted(len(future_map))
            for idx, future in enumerate(as_completed(future_map), start=1):
                if self.run_state.is_cancelled(review):
                    self.cancel_pending_futures(
                        executor=executor,
                        future_map=future_map,
                        review_id=review.id,
                        phase="single.parameter_loop",
                    )
                    raise AnalysisCancelledError("Analysis was cancelled by user.")
                parameter = future_map[future]
                self.logger.info(
                    "TextDebateCoordinator.run_single_analysis_for_category: category=%s [%d/%d] parameter id=%s",
                    getattr(category, "code", None),
                    idx,
                    len(parameters),
                    parameter.id,
                )
                try:
                    debate_output = future.result()
                    if debate_output is None:
                        continue
                    killed_assumptions_memory.extend(self.extract_killed_assumptions_from_output(debate_output, parameter))
                    debate_output.analysis_trace["killed_assumptions"] = list(killed_assumptions_memory)
                    self.record_debate_progress(
                        review=review,
                        summary=summary,
                        category_code=category_code,
                        completed_count=1,
                        parameter_ids=[parameter.id],
                        log_prefix="TextDebateCoordinator.debate",
                        source="single",
                    )
                    self.publish_work_phase(
                        review=review,
                        parameter=parameter,
                        debate_id=self.build_live_debate_id(parameter),
                        work_phase="persistence",
                        last_snippet="Persisting the completed analysis.",
                        progress_percent=95,
                    )
                    self.persist_debate_output(
                        review=review,
                        category=category,
                        ingestion_job=ingestion_job,
                        parameter=parameter,
                        indexes=indexes,
                        tsd_document=tsd_document,
                        debate_output=debate_output,
                        summary=summary,
                    )
                    self.record_persistence_progress(
                        review=review,
                        summary=summary,
                        category_code=category_code,
                        parameter_id=parameter.id,
                    )
                except Exception as exc:
                    with summary.lock:
                        summary.error_count += 1
                    self.logger.exception(
                        "TextDebateCoordinator.run_single_analysis_for_category: failed for parameter id=%s: %s",
                        parameter.id,
                        exc,
                    )
                    self.record_debate_progress(
                        review=review,
                        summary=summary,
                        category_code=category_code,
                        completed_count=1,
                        parameter_ids=[parameter.id],
                        log_prefix="TextDebateCoordinator.debate",
                        source="single",
                        status="terminal_error",
                    )
                    self.record_persistence_progress(
                        review=review,
                        summary=summary,
                        category_code=category_code,
                        parameter_id=parameter.id,
                        status="terminal_error",
                    )
                    review_debate_event_store.fail_agent(
                        review.id,
                        debate_id=self.build_live_debate_id(parameter),
                        agent="mediator",
                        error_message=str(exc),
                    )
        self.last_batch_concurrency_stats["single"] = single_probe.snapshot().to_dict()

    def analyze_single_child(
        self,
        *,
        review,
        category,
        ingestion_job,
        parameter,
        indexes,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
        execution_mode: str = "single",
    ) -> DebateOutput:
        debate_input, retrieval_result = self.build_single_child_debate_input(
            parameter=parameter,
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
            killed_assumptions=killed_assumptions,
        )
        return self.debate.run_debate(
            debate_input=debate_input,
            retrieval_result=retrieval_result,
            tsd_document=tsd_document,
            cancel_check=lambda: self.run_state.is_cancelled(review),
            agent_chunk_handler=lambda agent, chunk: self.publish_live_agent_chunk(
                review=review,
                parameter=parameter,
                agent=agent,
                chunk=chunk,
            ),
            agent_started_handler=lambda agent, round_number=None: self.publish_live_agent_start(
                review=review,
                parameter=parameter,
                category_code=getattr(category, "code", None) or "unknown",
                agent=agent,
                execution_mode=execution_mode,
                progress_percent=self.agent_progress_percent(agent, "started"),
                content=f"{agent.title()} is analyzing this requirement.",
                round_number=round_number,
            ),
            agent_completed_handler=lambda agent, content, critic_outcome=None, requires_rebuttal=None, round_number=None: self.publish_live_agent_complete(
                review=review,
                parameter=parameter,
                agent=agent,
                content=content,
                progress_percent=self.agent_progress_percent(agent, "completed"),
                execution_mode=execution_mode,
                critic_outcome=critic_outcome,
                requires_rebuttal=requires_rebuttal,
                round_number=round_number,
            ),
        )

    def analyze_single_child_with_retrieval_result(
        self,
        *,
        review,
        category,
        parameter,
        retrieval_result,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
        execution_mode: str = "single",
    ) -> DebateOutput:
        debate_input = self.build_debate_input_for_parameter(
            parameter=parameter,
            category=category,
            retrieval_result=retrieval_result,
            tsd_document=tsd_document,
            killed_assumptions=killed_assumptions,
        )
        return self.debate.run_debate(
            debate_input=debate_input,
            retrieval_result=retrieval_result,
            tsd_document=tsd_document,
            cancel_check=lambda: self.run_state.is_cancelled(review),
            agent_chunk_handler=lambda agent, chunk: self.publish_live_agent_chunk(
                review=review,
                parameter=parameter,
                agent=agent,
                chunk=chunk,
            ),
            agent_started_handler=lambda agent, round_number=None: self.publish_live_agent_start(
                review=review,
                parameter=parameter,
                category_code=getattr(category, "code", None) or "unknown",
                agent=agent,
                execution_mode=execution_mode,
                progress_percent=self.agent_progress_percent(agent, "started"),
                content=f"{agent.title()} is analyzing this requirement.",
                round_number=round_number,
            ),
            agent_completed_handler=lambda agent, content, critic_outcome=None, requires_rebuttal=None, round_number=None: self.publish_live_agent_complete(
                review=review,
                parameter=parameter,
                agent=agent,
                content=content,
                progress_percent=self.agent_progress_percent(agent, "completed"),
                execution_mode=execution_mode,
                critic_outcome=critic_outcome,
                requires_rebuttal=requires_rebuttal,
                round_number=round_number,
            ),
        )

    def build_single_child_debate_input(
        self,
        *,
        parameter,
        category,
        ingestion_job,
        indexes,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
    ) -> Tuple[DebateInput, Any]:
        retrieval_query_details = self.debate_input_factory.build_retrieval_query_details(parameter)
        retrieval_result = self.retrieval.retrieve_for_parameter(
            parameter=parameter,
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
            query_details=retrieval_query_details,
        )
        return (
            self._build_debate_input_with_refresh(
                parameter=parameter,
                category=category,
                ingestion_job=ingestion_job,
                indexes=indexes,
                retrieval_result=retrieval_result,
                tsd_document=tsd_document,
                killed_assumptions=killed_assumptions,
                retrieval_query_details=retrieval_query_details,
            ),
            retrieval_result,
        )

    def _build_debate_input_with_refresh(
        self,
        *,
        parameter,
        category,
        ingestion_job,
        indexes,
        retrieval_result,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
        retrieval_query_details: Optional[dict] = None,
    ) -> DebateInput:
        debate_input = self.debate_input_factory.build_debate_input(
            parameter=parameter,
            category=category,
            retrieval_result=retrieval_result,
            tsd_document=tsd_document,
            killed_assumptions=killed_assumptions,
            retrieval_query_details=retrieval_query_details,
        )
        debate_input.retrieval_refresh_callback = self._build_retrieval_refresh_callback(
            parameter=parameter,
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
            killed_assumptions=killed_assumptions,
            retrieval_query_details=retrieval_query_details or debate_input.retrieval_query_details,
        )
        return debate_input

    def build_debate_input_for_parameter(
        self,
        *,
        parameter,
        category,
        retrieval_result,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
        retrieval_query_details: Optional[dict] = None,
    ) -> DebateInput:
        return self.debate_input_factory.build_debate_input(
            parameter=parameter,
            category=category,
            retrieval_result=retrieval_result,
            tsd_document=tsd_document,
            killed_assumptions=killed_assumptions,
            retrieval_query_details=retrieval_query_details,
        )

    def _build_retrieval_refresh_callback(
        self,
        *,
        parameter,
        category,
        ingestion_job,
        indexes,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
        retrieval_query_details: Dict[str, Any],
    ):
        if indexes is None:
            return None

        def _refresh(*, critic_result, hunter_result=None, round_number: int, refresh_reason: str = "missed_evidence") -> Dict[str, Any]:
            retry_queries: List[str] = []
            for values in (
                list(getattr(critic_result, "missed_evidence", []) or []),
                list(getattr(critic_result, "objections", []) or [])[:3],
                list(getattr(critic_result, "weak_evidence", []) or [])[:2],
            ):
                for value in values:
                    if isinstance(value, str) and value.strip():
                        retry_queries.append(value.strip())
            deduped_queries: List[str] = []
            seen = set()
            for value in retry_queries:
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped_queries.append(value)
            updated_query_details = deepcopy(retrieval_query_details or {})
            updated_query_details["retry_queries"] = deduped_queries[:6]
            refreshed_result = self.retrieval.retrieve_for_parameter(
                parameter=parameter,
                category=category,
                ingestion_job=ingestion_job,
                indexes=indexes,
                tsd_document=tsd_document,
                query_details=updated_query_details,
            )
            refreshed_input = self.debate_input_factory.build_debate_input(
                parameter=parameter,
                category=category,
                retrieval_result=refreshed_result,
                tsd_document=tsd_document,
                killed_assumptions=killed_assumptions,
                retrieval_query_details=updated_query_details,
            )
            refreshed_input.retrieval_refresh_callback = None
            return {
                "debate_input": refreshed_input,
                "retrieval_result": refreshed_result,
                "trace": {
                    "round": round_number,
                    "reason": refresh_reason,
                    "retry_queries": deduped_queries[:6],
                    "hunter_verdict": getattr(hunter_result, "verdict", None),
                    "critic_invalid_citation_ids": list(getattr(critic_result, "invalid_citation_ids", []) or []),
                    "refreshed_chunk_count": len(refreshed_input.context_chunks or []),
                    "refreshed_chunk_ids": list(refreshed_input.context_chunk_map.keys()),
                },
            }

        return _refresh

    def has_grounded_citations(self, output: DebateOutput) -> bool:
        allowed_ids = set(output.analysis_trace.get("retrieved_chunk_ids", []) or [])
        for citation in getattr(output.mediator_result, "final_citations", []) or []:
            if citation.block_id and citation.block_id in allowed_ids:
                return True
        for citation in getattr(output.critic_result, "valid_citations", []) or []:
            if citation.block_id and citation.block_id in allowed_ids:
                return True
        return False

    def persist_debate_output(
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
    ):
        persistence_input = PersistenceInput(
            parameter=parameter,
            category=category,
            ingestion_job=ingestion_job,
            debate_output=debate_output,
        )
        finding = self.persistence.persist_finding(review, persistence_input, summary)
        if finding is not None:
            review_debate_event_store.complete_debate(
                review.id,
                debate_id=self.build_live_debate_id(parameter),
                finding_id=getattr(finding, "id", None),
                last_snippet=getattr(debate_output.mediator_result, "logic_summary", None)
                or getattr(debate_output.mediator_result, "reasoning", None)
                or "",
            )
        else:
            review_debate_event_store.fail_agent(
                review.id,
                debate_id=self.build_live_debate_id(parameter),
                agent="mediator",
                error_message="Failed to persist debate result.",
            )
        return finding

    def extract_killed_assumptions_from_output(self, debate_output, parameter) -> list:
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

    def build_live_debate_id(self, parameter: Any) -> str:
        return build_debate_id(getattr(parameter, "id", None), getattr(parameter, "stable_key", None))

    def build_live_debate_descriptor(
        self,
        *,
        parameter: Any,
        category_code: str,
        execution_mode: str,
    ) -> Dict[str, Any]:
        requirement_text = build_parameter_analysis_text(parameter).strip()
        section_title = getattr(getattr(parameter, "parent", None), "title", None) or "General"
        return {
            "debate_id": self.build_live_debate_id(parameter),
            "parameter_id": getattr(parameter, "id", None),
            "requirement_reference": getattr(parameter, "stable_key", None),
            "requirement_text": requirement_text,
            "section_title": section_title,
            "category_code": category_code,
            "execution_mode": execution_mode,
            "work_phase": "queued",
        }

    def seed_live_debates(
        self,
        *,
        review: Review,
        category_code: str,
        parameters: List[Any],
        execution_mode: str,
    ) -> None:
        review_debate_event_store.seed_debates(
            review.id,
            review_status=getattr(review, "status", Review.STATUS_RUNNING),
            debates=[
                self.build_live_debate_descriptor(
                    parameter=parameter,
                    category_code=category_code,
                    execution_mode=execution_mode,
                )
                for parameter in parameters
            ],
        )

    def agent_progress_percent(self, agent: str, phase: str) -> int:
        agent_key = str(agent or "").strip().lower()
        if phase == "started":
            return {"hunter": 5, "critic": 40, "mediator": 72}.get(agent_key, 0)
        return {"hunter": 33, "critic": 66, "mediator": 90}.get(agent_key, 0)

    def publish_work_phase(
        self,
        *,
        review: Review,
        parameter: Any,
        debate_id: str,
        work_phase: str,
        last_snippet: str,
        progress_percent: int,
    ) -> None:
        review_debate_event_store.set_work_phase(
            review.id,
            debate_id=debate_id or self.build_live_debate_id(parameter),
            work_phase=work_phase,
            last_snippet=last_snippet,
            progress_percent=progress_percent,
        )

    def publish_live_agent_start(
        self,
        *,
        review: Review,
        parameter: Any,
        category_code: str,
        agent: str,
        execution_mode: str,
        progress_percent: int,
        content: str,
        round_number: Optional[int] = None,
    ) -> None:
        review_debate_event_store.start_agent(
            review.id,
            debate=self.build_live_debate_descriptor(
                parameter=parameter,
                category_code=category_code,
                execution_mode=execution_mode,
            ),
            agent=agent,
            execution_mode=execution_mode,
            content=content,
            progress_percent=progress_percent,
            round_number=round_number,
        )

    def publish_live_agent_chunk(
        self,
        *,
        review: Review,
        parameter: Any,
        agent: str,
        chunk: str,
    ) -> None:
        review_debate_event_store.append_agent_chunk(
            review.id,
            debate_id=self.build_live_debate_id(parameter),
            agent=agent,
            chunk=chunk,
        )

    def publish_live_agent_complete(
        self,
        *,
        review: Review,
        parameter: Any,
        agent: str,
        content: str,
        progress_percent: int,
        execution_mode: str,
        critic_outcome: Optional[str] = None,
        requires_rebuttal: Optional[bool] = None,
        round_number: Optional[int] = None,
    ) -> None:
        review_debate_event_store.complete_agent(
            review.id,
            debate_id=self.build_live_debate_id(parameter),
            agent=agent,
            content=content,
            progress_percent=progress_percent,
            execution_mode=execution_mode,
            critic_outcome=critic_outcome,
            requires_rebuttal=requires_rebuttal,
            round_number=round_number,
        )

    def record_debate_progress(
        self,
        *,
        review: Review,
        summary: AnalysisSummary,
        category_code: str,
        completed_count: int,
        parameter_ids: List[Any],
        log_prefix: str,
        source: str,
        status: str = "completed",
    ) -> None:
        snapshot = self.progress_service.record_debate_progress(
            summary=summary,
            category_code=category_code,
            completed_count=completed_count,
        )
        if snapshot is None:
            return
        self.run_state.persist_summary_snapshot(review, summary)
        self.logger.info(
            "%s: category=%s progress completed=%d remaining=%d total=%d parameter_ids=%s source=%s status=%s",
            log_prefix,
            category_code,
            snapshot["processed_count"],
            snapshot["remaining_count"],
            snapshot["total_count"],
            parameter_ids,
            source,
            status,
        )

    def record_persistence_progress(
        self,
        *,
        review: Review,
        summary: AnalysisSummary,
        category_code: str,
        parameter_id: Optional[Any] = None,
        log_prefix: str = "TextDebateCoordinator.persistence",
        status: str = "persisted",
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        snapshot = self.progress_service.record_persistence_progress(summary=summary, category_code=category_code)
        self.run_state.persist_summary_snapshot(review, summary)
        extra_fields = dict(extra_fields or {})
        extra_fields.setdefault("status", status)
        extra_suffix = ""
        if extra_fields:
            extra_suffix = " " + " ".join(f"{key}={value}" for key, value in extra_fields.items())
        self.logger.info(
            "%s: category=%s progress processed=%d remaining=%d total=%d parameter id=%s%s",
            log_prefix,
            category_code,
            snapshot["processed_count"],
            snapshot["remaining_count"],
            snapshot["total_count"],
            parameter_id,
            extra_suffix,
        )

    def cancel_pending_futures(
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
            "TextDebateCoordinator.%s: cancellation detected review_id=%s pending_futures_cancelled=%d",
            phase,
            review_id,
            cancelled_count,
        )
