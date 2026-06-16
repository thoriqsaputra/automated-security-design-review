from __future__ import annotations

import logging
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from sdr.apps.ai.agents.base import CriticResult, HunterResult, MediatorResult
from sdr.apps.ai.engine.classification.domain_classification import DOMAIN_KEYWORDS, classify_requirement_domain
from sdr.apps.ai.engine.classification.parent_applicability import classify_parent_applicability
from sdr.apps.ai.engine.dto import AnalysisSummary, DebateInput, DebateOutput, PersistenceInput
from sdr.apps.ai.utils.concurrency import ConcurrencyProbe
from sdr.apps.reviews.models import Review
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

    def group_parameters_by_parent(self, parameters: List[Any]) -> List[Tuple[Any, List[Any]]]:
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

    def split_batches(self, parameters: List[Any], batch_size: int) -> List[List[Any]]:
        return [parameters[idx : idx + batch_size] for idx in range(0, len(parameters), batch_size)]

    def _should_assume_applicable_on_low_confidence(self, fallback_mode: str) -> bool:
        return (fallback_mode or "").strip().lower() == "assume_applicable"

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
        parent_context_cache: Optional[Dict[Tuple[Any, Any, Any], Any]] = None,
    ) -> None:
        category_code = getattr(category, "code", None) or "unknown"
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
        if int(category_stats.get("analysis_total_count") or 0) == 0 and parameters:
            self.progress_service.initialize_category_progress(
                summary=summary,
                category_code=category_code,
                total_count=len(parameters),
            )
            summary.debate_total_parameters += len(parameters)
            summary.debate_remaining_parameters += len(parameters)
            summary.persistence_total_parameters += len(parameters)
            summary.persistence_remaining_parameters += len(parameters)
            self.progress_service.sync_analysis_aliases(summary=summary, category_code=category_code)
            self.run_state.persist_summary_snapshot(review, summary)
        for idx, parameter in enumerate(parameters, start=1):
            self.run_state.raise_if_cancelled(review, phase="run.parameter_loop")
            self.logger.info(
                "TextDebateCoordinator.run_single_analysis_for_category: category=%s [%d/%d] parameter id=%s",
                getattr(category, "code", None),
                idx,
                len(parameters),
                parameter.id,
            )
            try:
                debate_output = self.analyze_single_child_with_parent_context(
                    category=category,
                    ingestion_job=ingestion_job,
                    parameter=parameter,
                    indexes=indexes,
                    tsd_document=tsd_document,
                    killed_assumptions=list(killed_assumptions_memory),
                    parent_context_cache=parent_context_cache or {},
                )
                self.run_state.raise_if_cancelled(review, phase="run.after_debate")
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

    def run_batched_analysis_for_category(
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
        parent_context_cache: Optional[Dict[Tuple[Any, Any, Any], Any]] = None,
    ) -> None:
        start_ts = time.monotonic()
        category_code = getattr(category, "code", None) or "unknown"
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
        if int(category_stats.get("analysis_total_count") or 0) == 0 and parameters:
            self.progress_service.initialize_category_progress(
                summary=summary,
                category_code=category_code,
                total_count=len(parameters),
            )
            summary.debate_total_parameters += len(parameters)
            summary.debate_remaining_parameters += len(parameters)
            summary.persistence_total_parameters += len(parameters)
            summary.persistence_remaining_parameters += len(parameters)
            self.progress_service.sync_analysis_aliases(summary=summary, category_code=category_code)
            self.run_state.persist_summary_snapshot(review, summary)
        batch_size = self.config.batch_debate_batch_size
        max_concurrency = self.config.batch_debate_max_concurrency
        parent_groups = self.group_parameters_by_parent(parameters)
        batches: List[Tuple[Any, List[Any]]] = []
        for parent, children in parent_groups:
            for batch in self.split_batches(children, batch_size):
                batches.append((parent, batch))

        accepted_outputs: Dict[str, DebateOutput] = {}
        invalid_reasons: Dict[str, List[str]] = {}
        terminal_error_ids: set[str] = set()
        debate_recorded_ids: set[str] = set()
        parent_context_cache = dict(parent_context_cache or {})
        parent_context_by_key: Dict[Any, Any] = {}
        self.last_batch_concurrency_stats = {}

        parent_probe = ConcurrencyProbe(max_concurrency=max_concurrency)
        with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="ParentRetrieval") as executor:
            parent_future_map = {}
            for parent, children in parent_groups:
                self.run_state.raise_if_cancelled(review, phase="batch.before_parent_retrieval")
                future = executor.submit(
                    parent_probe.wrap(self.get_parent_retrieval_result),
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
                if self.run_state.is_cancelled(review):
                    self.cancel_pending_futures(
                        executor=executor,
                        future_map=parent_future_map,
                        review_id=review.id,
                        phase="batch.parent_retrieval",
                    )
                    raise AnalysisCancelledError("Analysis was cancelled by user.")
                parent = parent_future_map[future]
                parent_key = getattr(parent, "id", None) or id(parent)
                parent_context_by_key[parent_key] = future.result()
        self.last_batch_concurrency_stats["parent_retrieval"] = parent_probe.snapshot().to_dict()

        def _run_batch(parent, batch_parameters, killed_snapshot):
            if self.run_state.is_cancelled(review):
                return {}
            retrieval_result = parent_context_by_key[getattr(parent, "id", None) or id(parent)]
            debate_inputs = [
                self.build_debate_input_for_parameter(
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
                cancel_check=lambda: self.run_state.is_cancelled(review),
            )

        batch_probe = ConcurrencyProbe(max_concurrency=max_concurrency)
        with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="BatchDebate") as executor:
            future_map = {}
            for parent, batch_parameters in batches:
                self.run_state.raise_if_cancelled(review, phase="batch.before_batch_submission")
                future = executor.submit(
                    batch_probe.wrap(_run_batch),
                    parent,
                    batch_parameters,
                    list(killed_assumptions_memory),
                )
                future_map[future] = batch_parameters
            batch_probe.mark_submitted(len(future_map))
            for future in as_completed(future_map):
                if self.run_state.is_cancelled(review):
                    self.cancel_pending_futures(
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
                        "TextDebateCoordinator.run_batched_analysis_for_category: batch failed children=%s: %s",
                        [str(p.id) for p in batch_parameters],
                        exc,
                    )
                    for parameter in batch_parameters:
                        invalid_reasons[str(parameter.id)] = ["batch_exception"]
                    continue
                batch_valid, batch_invalid = self.validate_batch_outputs(batch_parameters, batch_outputs)
                accepted_outputs.update(batch_valid)
                invalid_reasons.update(batch_invalid)
                completed_ids = [
                    int(child_id) if str(child_id).isdigit() else child_id
                    for child_id in batch_valid.keys()
                    if child_id not in debate_recorded_ids
                ]
                if completed_ids:
                    debate_recorded_ids.update(str(child_id) for child_id in completed_ids)
                    self.record_debate_progress(
                        review=review,
                        summary=summary,
                        category_code=category_code,
                        completed_count=len(completed_ids),
                        parameter_ids=completed_ids,
                        log_prefix="TextDebateCoordinator.debate",
                        source="batch",
                    )
        self.last_batch_concurrency_stats["batch_debate"] = batch_probe.snapshot().to_dict()

        fallback_count = 0
        fallback_enabled = self.config.batch_debate_fallback_enabled
        final_outputs: Dict[str, DebateOutput] = {}
        fallback_parameters: List[Any] = []
        for parameter in parameters:
            child_id = str(parameter.id)
            output = accepted_outputs.get(child_id)
            if child_id in invalid_reasons:
                reasons = invalid_reasons[child_id]
                if (
                    self.ungrounded_not_met_policy() == "downgrade_na"
                    and set(reasons) == {"not_met_without_grounded_citations"}
                    and output is not None
                ):
                    final_outputs[child_id] = output
                    if child_id not in debate_recorded_ids:
                        debate_recorded_ids.add(child_id)
                        self.record_debate_progress(
                            review=review,
                            summary=summary,
                            category_code=category_code,
                            completed_count=1,
                            parameter_ids=[parameter.id],
                            log_prefix="TextDebateCoordinator.debate",
                            source="batch",
                        )
                    continue
                if fallback_enabled:
                    fallback_count += 1
                    fallback_parameters.append(parameter)
                    continue
                if output is None:
                    summary.error_count += 1
                    terminal_error_ids.add(child_id)
                    if child_id not in debate_recorded_ids:
                        debate_recorded_ids.add(child_id)
                        self.record_debate_progress(
                            review=review,
                            summary=summary,
                            category_code=category_code,
                            completed_count=1,
                            parameter_ids=[parameter.id],
                            log_prefix="TextDebateCoordinator.debate",
                            source="batch_terminal",
                            status="terminal_error",
                        )
                    continue
            if output is None:
                summary.error_count += 1
                terminal_error_ids.add(child_id)
                if child_id not in debate_recorded_ids:
                    debate_recorded_ids.add(child_id)
                    self.record_debate_progress(
                        review=review,
                        summary=summary,
                        category_code=category_code,
                        completed_count=1,
                        parameter_ids=[parameter.id],
                        log_prefix="TextDebateCoordinator.debate",
                        source="batch_terminal",
                        status="terminal_error",
                    )
                continue
            final_outputs[child_id] = output
            if child_id not in debate_recorded_ids:
                debate_recorded_ids.add(child_id)
                self.record_debate_progress(
                    review=review,
                    summary=summary,
                    category_code=category_code,
                    completed_count=1,
                    parameter_ids=[parameter.id],
                    log_prefix="TextDebateCoordinator.debate",
                    source="batch",
                )

        if fallback_parameters:
            def _run_fallback(parameter, killed_snapshot):
                if self.run_state.is_cancelled(review):
                    return None
                parent = getattr(parameter, "parent", None)
                parent_key = getattr(parent, "id", None) or id(parent)
                retrieval_result = parent_context_by_key.get(parent_key)
                if retrieval_result is None:
                    return self.analyze_single_child(
                        category=category,
                        ingestion_job=ingestion_job,
                        parameter=parameter,
                        indexes=indexes,
                        tsd_document=tsd_document,
                        killed_assumptions=killed_snapshot,
                    )
                return self.analyze_single_child_with_retrieval_result(
                    category=category,
                    parameter=parameter,
                    retrieval_result=retrieval_result,
                    tsd_document=tsd_document,
                    killed_assumptions=killed_snapshot,
                )

            fallback_probe = ConcurrencyProbe(max_concurrency=max_concurrency)
            with ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="BatchFallback") as executor:
                future_map = {}
                for parameter in fallback_parameters:
                    self.run_state.raise_if_cancelled(review, phase="batch.before_fallback_submission")
                    future = executor.submit(
                        fallback_probe.wrap(_run_fallback),
                        parameter,
                        list(killed_assumptions_memory),
                    )
                    future_map[future] = parameter
                fallback_probe.mark_submitted(len(future_map))
                for future in as_completed(future_map):
                    if self.run_state.is_cancelled(review):
                        self.cancel_pending_futures(
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
                            if child_id not in debate_recorded_ids:
                                debate_recorded_ids.add(child_id)
                                self.record_debate_progress(
                                    review=review,
                                    summary=summary,
                                    category_code=category_code,
                                    completed_count=1,
                                    parameter_ids=[parameter.id],
                                    log_prefix="TextDebateCoordinator.debate",
                                    source="fallback",
                                )
                    except Exception as exc:
                        summary.error_count += 1
                        terminal_error_ids.add(child_id)
                        self.logger.exception(
                            "TextDebateCoordinator.run_batched_analysis_for_category: fallback failed parameter=%s: %s",
                            child_id,
                            exc,
                        )
                        if child_id not in debate_recorded_ids:
                            debate_recorded_ids.add(child_id)
                            self.record_debate_progress(
                                review=review,
                                summary=summary,
                                category_code=category_code,
                                completed_count=1,
                                parameter_ids=[parameter.id],
                                log_prefix="TextDebateCoordinator.debate",
                                source="fallback",
                                status="terminal_error",
                            )
            self.last_batch_concurrency_stats["fallback"] = fallback_probe.snapshot().to_dict()

        for parameter in parameters:
            self.run_state.raise_if_cancelled(review, phase="batch.before_persistence")
            output = final_outputs.get(str(parameter.id))
            if output is None:
                continue
            output = self.retry_if_needed(
                category=category,
                ingestion_job=ingestion_job,
                parameter=parameter,
                indexes=indexes,
                tsd_document=tsd_document,
                debate_output=output,
                killed_assumptions=list(killed_assumptions_memory),
            )
            killed_assumptions_memory.extend(self.extract_killed_assumptions_from_output(output, parameter))
            output.analysis_trace["killed_assumptions"] = list(killed_assumptions_memory)
            self.persist_debate_output(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameter=parameter,
                indexes=indexes,
                tsd_document=tsd_document,
                debate_output=output,
                summary=summary,
            )
            self.record_persistence_progress(
                review=review,
                summary=summary,
                category_code=category_code,
                parameter_id=parameter.id,
                log_prefix="TextDebateCoordinator.persistence",
                extra_fields={"final_child_output_count": len(final_outputs)},
            )

        for parameter in parameters:
            child_id = str(parameter.id)
            if child_id in final_outputs or child_id not in terminal_error_ids:
                continue
            self.record_persistence_progress(
                review=review,
                summary=summary,
                category_code=category_code,
                parameter_id=parameter.id,
                log_prefix="TextDebateCoordinator.persistence",
                status="terminal_error",
            )

        self.logger.info(
            "TextDebateCoordinator.run_batched_analysis_for_category: fallback_count=%d final_child_output_count=%d elapsed_seconds=%.4f",
            fallback_count,
            len(final_outputs),
            time.monotonic() - start_ts,
        )

    def analyze_single_child(
        self,
        *,
        category,
        ingestion_job,
        parameter,
        indexes,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
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
        )

    def analyze_single_child_with_retrieval_result(
        self,
        *,
        category,
        parameter,
        retrieval_result,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
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

    def get_parent_retrieval_result(
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
        cache_enabled = self.config.batch_debate_parent_context_cache_enabled
        key = (
            getattr(ingestion_job, "id", None),
            getattr(category, "id", None),
            getattr(parent, "id", None),
        )
        if cache_enabled and key in cache:
            return cache[key]
        query_details = self.build_parent_retrieval_query_details(parent, child_parameters)
        if hasattr(self.retrieval, "retrieve_for_parent_group"):
            result = self.retrieval.retrieve_for_parent_group(
                parent=parent,
                child_parameters=child_parameters,
                category=category,
                ingestion_job=ingestion_job,
                indexes=indexes,
                tsd_document=tsd_document,
                query_details=query_details,
            )
        else:
            result = self.retrieval.retrieve_for_parameter(
                parameter=child_parameters[0],
                category=category,
                ingestion_job=ingestion_job,
                indexes=indexes,
                tsd_document=tsd_document,
                query_details=query_details,
            )
        if cache_enabled:
            cache[key] = result
        return result

    def build_parent_retrieval_query_details(self, parent, child_parameters: List[Any]) -> dict:
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
        family_scope_terms = []
        raw_scope_parts = [
            (getattr(parent, "title", "") or "").strip(),
            (getattr(parent, "description", "") or "").strip(),
            "\n".join(child_requirements[:4]),
        ]
        for part in raw_scope_parts:
            family_scope_terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", part))
        family_scope_terms = [
            term.lower()
            for term in family_scope_terms
            if term and term.lower() not in {"the", "and", "for", "with", "control", "controls", "security", "requirement", "requirements"}
        ]
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
            "family_scope_terms": list(dict.fromkeys(family_scope_terms))[:12],
            "retry_queries": [],
        }

    def apply_parent_applicability_gate(
        self,
        *,
        review: Review,
        category,
        ingestion_job,
        parameters: List[Any],
        indexes,
        tsd_document,
        summary: AnalysisSummary,
    ) -> tuple[List[Any], Dict[Tuple[Any, Any, Any], Any]]:
        if not self.config.parent_applicability_enabled:
            return list(parameters), {}
        version_label = f"v{getattr(ingestion_job, 'version_no', 1)}"
        category_code = getattr(category, "code", None) or "unknown"
        confidence_threshold = self.config.parent_applicability_confidence_threshold
        fallback_mode = self.config.parent_applicability_fallback_mode
        parent_groups = self.group_parameters_by_parent(parameters)
        parent_context_cache: Dict[Tuple[Any, Any, Any], Any] = {}
        applicable_parameters: List[Any] = []

        summary.applicability["parents_total"] += len(parent_groups)
        for parent, child_parameters in parent_groups:
            self.run_state.raise_if_cancelled(review, phase="parent_applicability")
            retrieval_result = self.get_parent_retrieval_result(
                parent=parent,
                child_parameters=child_parameters,
                category=category,
                ingestion_job=ingestion_job,
                indexes=indexes,
                tsd_document=tsd_document,
                cache=parent_context_cache,
            )
            query_details = self.build_parent_retrieval_query_details(parent, child_parameters)
            child_requirement_texts = [
                build_parameter_analysis_text(parameter).strip()
                for parameter in child_parameters[: self.config.parent_applicability_max_child_texts]
                if build_parameter_analysis_text(parameter).strip()
            ]
            applicability = classify_parent_applicability(
                category_code=category_code,
                version_label=version_label,
                parent_title=(getattr(parent, "title", "") or "").strip(),
                parent_description=(getattr(parent, "description", "") or "").strip(),
                child_requirements=child_requirement_texts,
                retrieved_context="\n\n".join(retrieval_result.context_chunks or []),
                query_details=query_details,
            )
            parent_payload = {
                "parent_id": getattr(parent, "id", None),
                "parent_title": (getattr(parent, "title", "") or "").strip(),
                "applicable": applicability.applicable,
                "confidence": applicability.confidence,
                "reasoning": applicability.reasoning,
                "evidence": list(applicability.evidence or []),
                "decision_mode": applicability.decision_mode,
                "error": applicability.error,
                "child_count": len(child_parameters),
            }
            summary.applicability["parents"].append(parent_payload)

            if not applicability.applicable and applicability.confidence >= confidence_threshold:
                summary.applicability["parents_not_applicable"] += 1
                summary.applicability["children_marked_na_by_parent"] += len(child_parameters)
                self.persist_parent_not_applicable_children(
                    review=review,
                    category=category,
                    ingestion_job=ingestion_job,
                    parameters=child_parameters,
                    summary=summary,
                    applicability_payload=parent_payload,
                    retrieval_query_details=query_details,
                )
                continue

            if not applicability.applicable and applicability.confidence < confidence_threshold:
                parent_payload["fallback_mode"] = fallback_mode or "skip"
                if self._should_assume_applicable_on_low_confidence(fallback_mode):
                    summary.applicability["parents_applicable"] += 1
                    applicable_parameters.extend(child_parameters)
                    continue
                summary.applicability["parents_not_applicable"] += 1
                summary.applicability["children_marked_na_by_parent"] += len(child_parameters)
                self.persist_parent_not_applicable_children(
                    review=review,
                    category=category,
                    ingestion_job=ingestion_job,
                    parameters=child_parameters,
                    summary=summary,
                    applicability_payload=parent_payload,
                    retrieval_query_details=query_details,
                )
                continue

            summary.applicability["parents_applicable"] += 1
            applicable_parameters.extend(child_parameters)
        return applicable_parameters, parent_context_cache

    def persist_parent_not_applicable_children(
        self,
        *,
        review: Review,
        category,
        ingestion_job,
        parameters: List[Any],
        summary: AnalysisSummary,
        applicability_payload: Dict[str, Any],
        retrieval_query_details: Dict[str, Any],
    ) -> None:
        for parameter in parameters:
            debate_output = self.build_parent_not_applicable_debate_output(
                parameter=parameter,
                applicability_payload=applicability_payload,
                retrieval_query_details=retrieval_query_details,
            )
            self.persist_debate_output(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameter=parameter,
                indexes=None,
                tsd_document=None,
                debate_output=debate_output,
                summary=summary,
            )

    def build_parent_not_applicable_debate_output(
        self,
        *,
        parameter,
        applicability_payload: Dict[str, Any],
        retrieval_query_details: Dict[str, Any],
    ) -> DebateOutput:
        reasoning = applicability_payload.get("reasoning") or "This parent control family is out of scope for the documented design."
        analysis_trace = {
            "retrieved_chunk_ids": [],
            "contract": {
                "in_scope": False,
                "specific_enough": True,
                "domain": "general",
                "synth_mode": "parent_applicability_gate",
            },
            "retrieval_query_details": retrieval_query_details,
            "parent_applicability": dict(applicability_payload),
            "verdict_policy": {
                "source": "parent_applicability_gate",
                "raw_final_verdict": "na",
                "final_verdict": "na",
                "applicability_established": False,
                "evidence_sufficiency": "not_applicable",
                "not_assessable_reason": reasoning,
                "verified_control_evidence_ids": [],
            },
        }
        return DebateOutput.model_construct(
            parameter=parameter,
            hunter_result=HunterResult(
                verdict="na",
                confidence=applicability_payload.get("confidence") or 0.0,
                reasoning=reasoning,
                logic_summary=reasoning,
                evidence_found=False,
                citations=[],
            ),
            critic_result=CriticResult(
                revised_verdict="na",
                revised_confidence=applicability_payload.get("confidence") or 0.0,
                reasoning=reasoning,
                logic_summary=reasoning,
                valid_citations=[],
            ),
            mediator_result=MediatorResult(
                final_verdict="na",
                raw_final_verdict="na",
                confidence=applicability_payload.get("confidence") or 0.0,
                reasoning=reasoning,
                logic_summary=reasoning,
                finding_description=reasoning,
                final_citations=[],
                severity=None,
                recommendation=None,
            ),
            retrieval_result=None,
            debate_rounds=0,
            analysis_trace=analysis_trace,
        )

    def analyze_single_child_with_parent_context(
        self,
        *,
        category,
        ingestion_job,
        parameter,
        indexes,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
        parent_context_cache: Dict[Tuple[Any, Any, Any], Any],
    ) -> DebateOutput:
        parent = getattr(parameter, "parent", None)
        if parent is not None:
            parent_result = self.get_parent_retrieval_result(
                parent=parent,
                child_parameters=[parameter],
                category=category,
                ingestion_job=ingestion_job,
                indexes=indexes,
                tsd_document=tsd_document,
                cache=parent_context_cache,
            )
            if parent_result is not None and not getattr(parent_result, "error", None):
                return self.analyze_single_child_with_retrieval_result(
                    category=category,
                    parameter=parameter,
                    retrieval_result=parent_result,
                    tsd_document=tsd_document,
                    killed_assumptions=killed_assumptions,
                )
        return self.analyze_single_child(
            category=category,
            ingestion_job=ingestion_job,
            parameter=parameter,
            indexes=indexes,
            tsd_document=tsd_document,
            killed_assumptions=killed_assumptions,
        )

    def validate_batch_outputs(
        self,
        expected_parameters: List[Any],
        batch_outputs: Dict[str, DebateOutput],
    ) -> Tuple[Dict[str, DebateOutput], Dict[str, List[str]]]:
        expected_ids = [str(parameter.id) for parameter in expected_parameters]
        expected_set = set(expected_ids)
        accepted: Dict[str, DebateOutput] = {}
        invalid: Dict[str, List[str]] = {}
        threshold = self.config.batch_debate_confidence_threshold
        soft_threshold = self.config.batch_debate_soft_confidence_threshold
        for child_id in expected_ids:
            output = batch_outputs.get(child_id)
            reasons, soft_reject = self.validate_single_batch_output(child_id, output, threshold, soft_threshold)
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

    def validate_single_batch_output(
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
        confidence_value = float(getattr(mediator, "confidence", 0.0) or 0.0)
        if confidence_value < threshold:
            reasons.append("low_confidence")
        reasoning = (getattr(mediator, "logic_summary", None) or getattr(mediator, "reasoning", None) or "").strip()
        if confidence_value < 0.85 and self.is_weak_or_generic_reasoning(reasoning):
            reasons.append("weak_or_generic_reasoning")
        if verdict == "met" and not getattr(mediator, "final_citations", []):
            reasons.append("met_without_grounded_citations")
        if verdict == "not_met" and self.config.batch_debate_require_citations_for_not_met and not self.has_grounded_citations(output):
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
        if self.appears_to_cover_multiple_children(reasoning, child_id):
            reasons.append("generic_multi_child_result")
        deduped_reasons = list(dict.fromkeys(reasons))
        if self.ungrounded_not_met_policy() == "downgrade_na" and set(deduped_reasons) == {"not_met_without_grounded_citations"}:
            return [], False
        hard_invalid_reasons = {"invalid_verdict", "invalid_citations", "missing_result", "met_without_grounded_citations"}
        if self.ungrounded_not_met_policy() == "always_fallback":
            hard_invalid_reasons.add("not_met_without_grounded_citations")
        if any(reason in hard_invalid_reasons for reason in deduped_reasons):
            return deduped_reasons, True
        confidence = float(getattr(mediator, "confidence", 0.0) or 0.0)
        low_conf_only = deduped_reasons == ["low_confidence"]
        if low_conf_only and confidence >= soft_threshold and self.has_grounded_citations(output):
            return [], False
        return deduped_reasons, bool(deduped_reasons)

    def has_grounded_citations(self, output: DebateOutput) -> bool:
        allowed_ids = set(output.analysis_trace.get("retrieved_chunk_ids", []) or [])
        for citation in getattr(output.mediator_result, "final_citations", []) or []:
            if citation.block_id and citation.block_id in allowed_ids:
                return True
        for citation in getattr(output.critic_result, "valid_citations", []) or []:
            if citation.block_id and citation.block_id in allowed_ids:
                return True
        return False

    def is_weak_or_generic_reasoning(self, reasoning: str) -> bool:
        lowered = reasoning.lower()
        if len(reasoning) < 40:
            return True
        generic_markers = {"as above", "same as previous", "all requirements", "the batch", "each child", "all children", "no reasoning provided"}
        return any(marker in lowered for marker in generic_markers)

    def appears_to_cover_multiple_children(self, reasoning: str, child_id: str) -> bool:
        lowered = reasoning.lower()
        return (
            "all children" in lowered
            or "multiple child" in lowered
            or "the same evidence applies to all" in lowered
            or "applies to every child" in lowered
        )

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
    ) -> None:
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
        self.persistence.persist_finding(review, persistence_input, summary)

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
        retry_context_available = all(value is not None for value in [ingestion_job, indexes, tsd_document])
        debate_output.analysis_trace["evidence_gate_retry_context_available"] = retry_context_available
        if not retry_context_available:
            debate_output.analysis_trace["evidence_gate_outcome"] = "no_retry_context_preserved_not_met"
            debate_output.analysis_trace["downgraded_due_to_missing_citations"] = False
            debate_output.analysis_trace["downgrade_reason"] = None
            return debate_output
        downgrade_policy = self.ungrounded_not_met_policy()
        retry_output = debate_output
        try:
            retry_details = dict(self.debate_input_factory.build_retrieval_query_details(parameter, debate_output.analysis_trace.get("contract") or {}))
            retry_details["retry_queries"] = [{
                "attempt": 1,
                "reason": "evidence_gate_retry_for_ungrounded_not_met",
                "primary_domain": retry_details.get("primary_domain"),
                "keywords": retry_details.get("generated_domain_keywords", []),
            }]
            self.retrieval.retrieve_for_parameter(
                parameter=parameter,
                category=category,
                ingestion_job=ingestion_job,
                indexes=indexes,
                tsd_document=tsd_document,
                query_details=retry_details,
            )
            mediator_agent = self.mediator_agent_factory()
            hunter_result = debate_output.hunter_result
            critic_result = debate_output.critic_result
            contract = debate_output.analysis_trace.get("contract") or {}
            parameter_text = build_parameter_analysis_text(parameter).strip()
            parameter_section = (getattr(getattr(parameter, "parent", None), "title", "") or "").strip()
            mediator_result = mediator_agent.run(
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                contract=contract,
                hunter_result=hunter_result,
                critic_result=critic_result,
                debate_history=[],
            )
            if mediator_result:
                retry_output = DebateOutput(
                    parameter=debate_output.parameter,
                    hunter_result=hunter_result,
                    critic_result=critic_result,
                    mediator_result=mediator_result,
                    analysis_trace={**(debate_output.analysis_trace or {}), "evidence_gate_retry": "mediator_only"},
                )
        except Exception as exc:
            self.logger.warning(
                "TextDebateCoordinator.apply_not_met_evidence_gate: retry failed parameter=%s: %s",
                parameter.id,
                exc,
            )
            retry_output.analysis_trace = dict(getattr(retry_output, "analysis_trace", {}) or {})
            retry_output.analysis_trace["evidence_gate_outcome"] = "retry_failed_preserved_not_met"
            retry_output.analysis_trace["downgraded_due_to_missing_citations"] = False
            retry_output.analysis_trace["downgrade_reason"] = None
            return retry_output
        if self.has_grounded_citations(retry_output):
            retry_output.analysis_trace = dict(getattr(retry_output, "analysis_trace", {}) or {})
            retry_output.analysis_trace["evidence_gate_outcome"] = "recovered_with_citations"
            retry_output.analysis_trace["downgraded_due_to_missing_citations"] = False
            return retry_output
        retry_trace = dict(getattr(retry_output, "analysis_trace", {}) or {})
        verdict_policy = retry_trace.get("verdict_policy") or {}
        applicability_established = bool(verdict_policy.get("applicability_established", True))
        retry_evidence_quality = self.extract_retrieval_evidence_quality(retry_trace)
        if retry_evidence_quality and int(retry_evidence_quality.get("implementation_evidence_count") or 0) == 0 and not bool(retry_evidence_quality.get("applicability_signal")):
            retry_output.analysis_trace = retry_trace
            reason = self.build_missing_evidence_reasoning(parameter, retry_evidence_quality, applicability_established=False)
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
