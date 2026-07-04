from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from sdr.apps.ai.agents.base import CriticResult, HunterResult, MediatorResult
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
        category_stats = summary.category_stats.setdefault(category_code, {})
        if int(category_stats.get("analysis_total_count") or 0) == 0 and parameters:
            self.progress_service.initialize_category_progress(
                summary=summary,
                category_code=category_code,
                total_count=len(parameters),
            )
            with summary.lock:
                summary.debate_total_parameters += len(parameters)
                summary.debate_remaining_parameters += len(parameters)
                summary.persistence_total_parameters += len(parameters)
                summary.persistence_remaining_parameters += len(parameters)
            self.progress_service.sync_analysis_aliases(summary=summary, category_code=category_code)
            self.run_state.persist_summary_snapshot(review, summary)
        max_concurrency = self.config.batch_debate_max_concurrency
        killed_snapshot = list(killed_assumptions_memory)

        def _run_single(parameter):
            if self.run_state.is_cancelled(review):
                return None
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
                future = executor.submit(single_probe.wrap(_run_single), parameter)
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
                    debate_output = self.retry_if_needed(
                        category=category,
                        ingestion_job=ingestion_job,
                        parameter=parameter,
                        indexes=indexes,
                        tsd_document=tsd_document,
                        debate_output=debate_output,
                        killed_assumptions=list(killed_assumptions_memory),
                    )
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
            agent_started_handler=lambda agent: self.publish_live_agent_start(
                review=review,
                parameter=parameter,
                category_code=getattr(category, "code", None) or "unknown",
                agent=agent,
                execution_mode=execution_mode,
                progress_percent=self.agent_progress_percent(agent, "started"),
                content=f"{agent.title()} is analyzing this requirement.",
            ),
            agent_completed_handler=lambda agent, content: self.publish_live_agent_complete(
                review=review,
                parameter=parameter,
                agent=agent,
                content=content,
                progress_percent=self.agent_progress_percent(agent, "completed"),
                execution_mode=execution_mode,
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
            agent_started_handler=lambda agent: self.publish_live_agent_start(
                review=review,
                parameter=parameter,
                category_code=getattr(category, "code", None) or "unknown",
                agent=agent,
                execution_mode=execution_mode,
                progress_percent=self.agent_progress_percent(agent, "started"),
                content=f"{agent.title()} is analyzing this requirement.",
            ),
            agent_completed_handler=lambda agent, content: self.publish_live_agent_complete(
                review=review,
                parameter=parameter,
                agent=agent,
                content=content,
                progress_percent=self.agent_progress_percent(agent, "completed"),
                execution_mode=execution_mode,
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
        parameter_text = build_parameter_analysis_text(parameter).strip()
        parameter_section = parameter.parent.title if parameter.parent else "General"
        contract = self.debate_input_factory.build_contract(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            parent_description=(parameter.parent.description if parameter.parent else "") or "",
        )
        retrieval_query_details = self.debate_input_factory.build_retrieval_query_details(parameter, contract)
        retrieval_result = self.retrieval.retrieve_for_parameter(
            parameter=parameter,
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
            query_details=retrieval_query_details,
        )
        return (
            self.build_debate_input_for_parameter(
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

    def build_debate_input_for_parameter(
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
        return self.debate_input_factory.build_debate_input(
            parameter=parameter,
            category=category,
            retrieval_result=retrieval_result,
            tsd_document=tsd_document,
            killed_assumptions=killed_assumptions,
            contract=contract,
            retrieval_query_details=retrieval_query_details,
        )

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
        gated_output = self.apply_not_met_evidence_gate(
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
        finding = self.persistence.persist_finding(review, persistence_input, summary)
        if finding is not None:
            review_debate_event_store.complete_debate(
                review.id,
                debate_id=self.build_live_debate_id(parameter),
                finding_id=getattr(finding, "id", None),
                last_snippet=getattr(gated_output.mediator_result, "logic_summary", None)
                or getattr(gated_output.mediator_result, "reasoning", None)
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

    def ungrounded_not_met_policy(self) -> str:
        raw = self.config.batch_debate_ungrounded_not_met_policy
        if raw in {"downgrade_na", "selective_fallback", "always_fallback", "preserve_not_met"}:
            return raw
        return "preserve_not_met"

    def apply_not_met_evidence_gate(
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
        if self.has_grounded_citations(debate_output):
            debate_output.analysis_trace["evidence_gate_attempted"] = False
            return debate_output
        evidence_quality = self.extract_retrieval_evidence_quality(debate_output.analysis_trace)
        implementation_count = int(evidence_quality.get("implementation_evidence_count") or 0)
        applicability_signal = bool(evidence_quality.get("applicability_signal"))
        if evidence_quality and implementation_count == 0 and not applicability_signal:
            reason = self.build_missing_evidence_reasoning(parameter, evidence_quality, applicability_established=False)
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
            reason = self.build_missing_evidence_reasoning(parameter, evidence_quality, applicability_established=True)
            mediator.reasoning = reason
            mediator.logic_summary = reason
            debate_output.analysis_trace["evidence_gate_outcome"] = "missing_implementation_evidence_preserved_not_met"
        if self.ungrounded_not_met_policy() == "preserve_not_met":
            debate_output.analysis_trace["evidence_gate_attempted"] = bool(debate_output.analysis_trace.get("evidence_gate_outcome"))
            debate_output.analysis_trace.setdefault("evidence_gate_outcome", "skipped_preserve_not_met_policy")
            return debate_output
        debate_output.analysis_trace["evidence_gate_attempted"] = True
        debate_output.analysis_trace["evidence_gate_retry_context_available"] = False
        debate_output.analysis_trace["evidence_gate_retry_skipped"] = "disabled_no_new_evidence_source"
        downgrade_policy = self.ungrounded_not_met_policy()
        applicability_established = True
        verdict_policy = debate_output.analysis_trace.get("verdict_policy") or {}
        if isinstance(verdict_policy, dict):
            applicability_established = bool(verdict_policy.get("applicability_established", True))
        if downgrade_policy == "selective_fallback" and applicability_established:
            debate_output.analysis_trace["evidence_gate_outcome"] = "retry_disabled_preserved_not_met"
            debate_output.analysis_trace["downgraded_due_to_missing_citations"] = False
            debate_output.analysis_trace["downgrade_reason"] = None
            return debate_output
        if downgrade_policy == "selective_fallback" and not applicability_established:
            debate_output.mediator_result.final_verdict = "na"
            debate_output.mediator_result.raw_final_verdict = "na"
            debate_output.mediator_result.severity = None
            debate_output.mediator_result.recommendation = None
            debate_output.mediator_result.reasoning = (
                (debate_output.mediator_result.reasoning or "").strip()
                + "\n\nInsufficient grounded evidence found, and applicability was not established."
            ).strip()
            debate_output.mediator_result.logic_summary = debate_output.mediator_result.reasoning
            debate_output.analysis_trace["evidence_gate_outcome"] = "downgraded_to_na_applicability_not_established_without_retry"
            debate_output.analysis_trace["downgraded_due_to_missing_citations"] = True
            debate_output.analysis_trace["downgrade_reason"] = "not_met_without_grounded_citations_or_applicability_without_retry"
            return debate_output
        if downgrade_policy != "downgrade_na":
            debate_output.analysis_trace["evidence_gate_outcome"] = "retry_disabled_preserved_not_met"
            debate_output.analysis_trace["downgraded_due_to_missing_citations"] = False
            debate_output.analysis_trace["downgrade_reason"] = None
            return debate_output
        debate_output.mediator_result.final_verdict = "na"
        debate_output.mediator_result.raw_final_verdict = "na"
        debate_output.mediator_result.severity = None
        debate_output.mediator_result.recommendation = None
        debate_output.mediator_result.reasoning = (
            (debate_output.mediator_result.reasoning or "").strip()
            + "\n\nInsufficient grounded evidence found to support a 'not_met' determination."
        ).strip()
        debate_output.mediator_result.logic_summary = debate_output.mediator_result.reasoning
        debate_output.analysis_trace["evidence_gate_outcome"] = "downgraded_to_na_missing_citations_without_retry"
        debate_output.analysis_trace["downgraded_due_to_missing_citations"] = True
        debate_output.analysis_trace["downgrade_reason"] = "not_met_without_grounded_citations_without_retry"
        return debate_output

    def extract_retrieval_evidence_quality(self, analysis_trace: dict) -> Dict[str, Any]:
        details = (analysis_trace or {}).get("retrieval_query_details") or {}
        retrieval_metadata = details.get("retrieval_evidence_metadata") or {}
        return dict(retrieval_metadata.get("evidence_quality") or {})

    def build_missing_evidence_reasoning(
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
        terms = ", ".join(evidence_quality.get("applicability_terms") or []) or "no direct scope terms"
        counts = evidence_quality.get("counts") or {}
        weak_parts = [f"{kind}={count}" for kind, count in sorted(counts.items()) if count]
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

    def is_concurrency_domain(self, details: Dict[str, Any]) -> bool:
        return (details.get("primary_domain") or details.get("domain_signal")) in {
            "business_logic_concurrency",
            "transaction_integrity",
        }

    def is_restatement_or_weak_not_met(self, output: DebateOutput) -> bool:
        if getattr(output.hunter_result, "verdict", None) != "not_met":
            return False
        reasoning_parts = [
            getattr(output.hunter_result, "reasoning", "") or "",
            getattr(output.critic_result, "reasoning", "") or "",
            getattr(output.mediator_result, "reasoning", "") or "",
        ]
        text = " ".join(reasoning_parts).lower()
        weak_markers = ["restatement", "generic", "no explicit evidence", "requirement text", "policy statement", "aspirational"]
        return any(marker in text for marker in weak_markers) or not getattr(output.critic_result, "valid_citations", [])

    def retry_if_needed(
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
        if not self.is_concurrency_domain(details):
            return debate_output
        if not self.is_restatement_or_weak_not_met(debate_output):
            return debate_output
        if details.get("retry_queries"):
            return debate_output
        retry_details = dict(self.debate_input_factory.build_retrieval_query_details(parameter, debate_output.analysis_trace.get("contract") or {}))
        retry_details["retry_queries"] = [{
            "attempt": 1,
            "reason": "targeted_concurrency_retry_after_weak_not_met",
            "primary_domain": retry_details.get("primary_domain"),
            "keywords": retry_details.get("generated_domain_keywords", []),
        }]
        retrieval_result = self.retrieval.retrieve_for_parameter(
            parameter=parameter,
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
            query_details=retry_details,
        )
        retry_input = self.build_debate_input_for_parameter(
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
            cancel_check=lambda: self.run_state.is_cancelled(review),
        )
        retry_output.analysis_trace = dict(getattr(retry_output, "analysis_trace", {}) or {})
        retry_output.analysis_trace["retrieval_query_details"] = retry_details
        return retry_output

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
    ) -> None:
        review_debate_event_store.complete_agent(
            review.id,
            debate_id=self.build_live_debate_id(parameter),
            agent=agent,
            content=content,
            progress_percent=progress_percent,
            execution_mode=execution_mode,
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
