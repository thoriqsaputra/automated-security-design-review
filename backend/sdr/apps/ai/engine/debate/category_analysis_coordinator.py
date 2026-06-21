from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from sdr.apps.ai.agents.base import Citation, CriticResult, HunterResult, MediatorResult
from sdr.apps.ai.engine.classification.asvs_level import filter_parameters_for_asvs_level
from sdr.apps.ai.engine.dto import DebateOutput, PersistenceInput
from sdr.apps.reviews.models import Finding
from sdr.core.database import SessionLocal


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
        effective_asvs_level: int,
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
        self.run_state.update_stage(review, summary, "3_asvs_classification")
        self.logger.info(
            "CategoryAnalysisCoordinator.run_category: category=%s effective_asvs_level=%s analysis_mode=%s",
            category_code,
            effective_asvs_level,
            analysis_mode,
        )
        parameters, asvs_filter_stats = filter_parameters_for_asvs_level(parameters, effective_asvs_level)
        summary.asvs["categories"][category_code] = asvs_filter_stats
        if not parameters:
            return
        if analysis_mode == "diagram_only":
            self.run_state.update_stage(review, summary, "7_diagram_debate")
            self.diagram_analysis.run(
                review=review,
                tsd_document=tsd_document,
                category=category,
                ingestion_job=ingestion_job,
                effective_asvs_level=effective_asvs_level,
                summary=summary,
                cancel_check=lambda: self.run_state.is_cancelled(review),
            )
            return
        self.run_state.update_stage(review, summary, "4_parameter_resolution")

        # Attempt CFSR path: debate against distilled family summaries if available.
        cfsrs = self.workflow_repository.list_control_summary_requirements(
            category_id=category.id,
            ingestion_job_id=ingestion_job.id,
            effective_asvs_level=effective_asvs_level,
        )

        multi_child_cfsrs = [
            c for c in cfsrs
            if len(getattr(c, "covered_child_keys", None) or []) > 1
        ]

        if multi_child_cfsrs:
            self.logger.info(
                "CategoryAnalysisCoordinator.run_category: CFSR path "
                "category=%s multi_child_cfsrs=%d single_child_cfsrs_bypassed=%d raw_children=%d",
                category_code,
                len(multi_child_cfsrs),
                len(cfsrs) - len(multi_child_cfsrs),
                len(parameters),
            )
            self._run_cfsr_analysis(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                cfsrs=multi_child_cfsrs,
                raw_parameters=parameters,
                indexes=indexes,
                tsd_document=tsd_document,
                summary=summary,
                effective_asvs_level=effective_asvs_level,
                category_code=category_code,
                killed_assumptions_memory=killed_assumptions_memory,
            )
        else:
            # Fallback: original raw-children path (no change to existing logic)
            self.logger.info(
                "CategoryAnalysisCoordinator.run_category: raw-children path "
                "category=%s parameters=%d (no CFSRs found)",
                category_code,
                len(parameters),
            )
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
            effective_asvs_level=effective_asvs_level,
            summary=summary,
            cancel_check=lambda: self.run_state.is_cancelled(review),
        )

    # ------------------------------------------------------------------
    # Raw-children path (original logic, extracted into helper)
    # ------------------------------------------------------------------

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
                parent_context_cache=parent_context_cache,
            )

    # ------------------------------------------------------------------
    # CFSR path
    # ------------------------------------------------------------------

    def _run_cfsr_analysis(
        self,
        *,
        review,
        category,
        ingestion_job,
        cfsrs: List[Any],
        raw_parameters: List[Any],
        indexes,
        tsd_document,
        summary,
        effective_asvs_level: int,
        category_code: str,
        killed_assumptions_memory: deque,
    ) -> None:
        self.progress_service.prepare_category_stats(
            summary=summary,
            parameters=cfsrs,
            category_code=category_code,
        )
        parent_skip_before = int(summary.applicability.get("children_marked_na_by_parent", 0) or 0)
        self.run_state.update_stage(review, summary, "5_parent_retrieval")
        # CFSRs are duck-type compatible with CategoryParameterChild for the applicability gate
        # (they have .parent, .requirement_text, .stable_key, .asvs_level, .details).
        applicable_cfsrs, parent_context_cache = self.text_debate.apply_parent_applicability_gate(
            review=review,
            category=category,
            ingestion_job=ingestion_job,
            parameters=cfsrs,
            indexes=indexes,
            tsd_document=tsd_document,
            summary=summary,
        )
        if not applicable_cfsrs:
            return
        summary.debate_total_parameters += len(applicable_cfsrs)
        summary.debate_remaining_parameters += len(applicable_cfsrs)
        summary.persistence_total_parameters += len(applicable_cfsrs)
        summary.persistence_remaining_parameters += len(applicable_cfsrs)
        self.progress_service.initialize_category_progress(
            summary=summary,
            category_code=category_code,
            total_count=len(applicable_cfsrs),
        )
        self.progress_service.sync_analysis_aliases(summary=summary, category_code=category_code)
        self.run_state.persist_summary_snapshot(review, summary)
        parent_skipped_for_category = max(
            int(summary.applicability.get("children_marked_na_by_parent", 0) or 0) - parent_skip_before,
            0,
        )
        self.logger.info(
            "CategoryAnalysisCoordinator._run_cfsr_analysis: category=%s applicable_cfsrs=%d skipped_by_parent=%d",
            category_code,
            len(applicable_cfsrs),
            parent_skipped_for_category,
        )
        self.run_state.update_stage(review, summary, "6_text_debate")

        # Run debate against CFSRs using the same batched/single path
        if self.config.batch_debate_enabled:
            self.text_debate.run_batched_analysis_for_category(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameters=applicable_cfsrs,
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
                parameters=applicable_cfsrs,
                indexes=indexes,
                tsd_document=tsd_document,
                summary=summary,
                killed_assumptions_memory=killed_assumptions_memory,
                parent_context_cache=parent_context_cache,
            )

        # Cascade not_met verdicts from CFSRs to their covered raw children
        self._cascade_not_met_to_children(
            review=review,
            category=category,
            ingestion_job=ingestion_job,
            applicable_cfsrs=applicable_cfsrs,
            raw_parameters=raw_parameters,
            summary=summary,
        )

        # RAG-gate + debate remaining children (met/na CFSR children + orphans)
        self._rag_gate_children(
            review=review,
            category=category,
            ingestion_job=ingestion_job,
            applicable_cfsrs=applicable_cfsrs,
            raw_parameters=raw_parameters,
            indexes=indexes,
            tsd_document=tsd_document,
            summary=summary,
            category_code=category_code,
            killed_assumptions_memory=killed_assumptions_memory,
            parent_context_cache=parent_context_cache,
        )

    def _rag_gate_children(
        self,
        *,
        review,
        category,
        ingestion_job,
        applicable_cfsrs: List[Any],
        raw_parameters: List[Any],
        indexes,
        tsd_document,
        summary,
        category_code: str,
        killed_assumptions_memory: deque,
        parent_context_cache: Optional[Dict],
    ) -> None:
        """
        Tier 1+2: For children NOT covered by a not_met CFSR cascade, run a cheap
        RAG retrieval gate. Children with no TSD evidence are fast-failed as not_met
        (rag_gate_no_evidence). Children with evidence are individually debated.
        """
        cfsr_stable_keys = [
            getattr(c, "stable_key", None)
            for c in applicable_cfsrs
            if getattr(c, "stable_key", None)
        ]
        children_by_key: Dict[str, Any] = {
            getattr(p, "stable_key", ""): p
            for p in raw_parameters
            if getattr(p, "stable_key", None)
        }

        not_met_keys: set = set()
        if cfsr_stable_keys:
            with SessionLocal() as db:
                not_met_findings = db.execute(
                    select(Finding).where(
                        Finding.review_id == review.id,
                        Finding.requirement_reference.in_(cfsr_stable_keys),
                        Finding.met_status == "not_met",
                    )
                ).scalars().all()
            not_met_keys = {f.requirement_reference for f in not_met_findings}

        all_covered: set = set()
        keys_for_gate: set = set()
        for cfsr in applicable_cfsrs:
            covered = list(getattr(cfsr, "covered_child_keys", None) or [])
            all_covered.update(covered)
            if getattr(cfsr, "stable_key", None) not in not_met_keys:
                keys_for_gate.update(covered)
        for key in children_by_key:
            if key not in all_covered:
                keys_for_gate.add(key)

        children_to_gate = [children_by_key[k] for k in keys_for_gate if k in children_by_key]
        if not children_to_gate:
            self.logger.info(
                "CategoryAnalysisCoordinator._rag_gate_children: category=%s no children to gate",
                category_code,
            )
            return

        self.logger.info(
            "CategoryAnalysisCoordinator._rag_gate_children: category=%s gating %d children",
            category_code,
            len(children_to_gate),
        )

        children_with_evidence: List[Any] = []
        children_no_evidence: List[Any] = []
        retrieval_results = self.text_debate.retrieval.retrieve_many_for_parameters(
            parameters=children_to_gate,
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
        )
        for child in children_to_gate:
            retrieval_result = retrieval_results.get(str(child.id))
            if retrieval_result is None:
                children_with_evidence.append(child)
                continue
            if retrieval_result.error:
                self.logger.warning(
                    "CategoryAnalysisCoordinator._rag_gate_children: retrieval failed for child=%s, routing to debate: %s",
                    getattr(child, "stable_key", None),
                    retrieval_result.error,
                )
                children_with_evidence.append(child)
                continue
            if retrieval_result.is_empty:
                children_no_evidence.append(child)
            else:
                children_with_evidence.append(child)

        self.logger.info(
            "CategoryAnalysisCoordinator._rag_gate_children: category=%s no_evidence=%d with_evidence=%d",
            category_code,
            len(children_no_evidence),
            len(children_with_evidence),
        )

        for child in children_no_evidence:
            try:
                self._persist_rag_gate_not_met(
                    review=review,
                    category=category,
                    ingestion_job=ingestion_job,
                    child=child,
                    summary=summary,
                )
            except Exception as exc:
                self.logger.warning(
                    "CategoryAnalysisCoordinator._rag_gate_children: failed to persist rag_gate not_met for child=%s: %s",
                    getattr(child, "stable_key", None),
                    exc,
                )

        if not children_with_evidence:
            return

        summary.debate_total_parameters += len(children_with_evidence)
        summary.debate_remaining_parameters += len(children_with_evidence)
        summary.persistence_total_parameters += len(children_with_evidence)
        summary.persistence_remaining_parameters += len(children_with_evidence)

        if self.config.batch_debate_enabled:
            self.text_debate.run_batched_analysis_for_category(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameters=children_with_evidence,
                indexes=indexes,
                tsd_document=tsd_document,
                summary=summary,
                killed_assumptions_memory=killed_assumptions_memory,
                parent_context_cache=parent_context_cache or {},
            )
        else:
            self.text_debate.run_single_analysis_for_category(
                review=review,
                category=category,
                ingestion_job=ingestion_job,
                parameters=children_with_evidence,
                indexes=indexes,
                tsd_document=tsd_document,
                summary=summary,
                killed_assumptions_memory=killed_assumptions_memory,
                parent_context_cache=parent_context_cache or {},
            )

    def _persist_rag_gate_not_met(self, *, review, category, ingestion_job, child, summary) -> None:
        analysis_trace = {
            "contract": {"synth_mode": "rag_gate_no_evidence"},
            "verdict_policy": {
                "source": "rag_gate",
                "raw_final_verdict": "not_met",
                "final_verdict": "not_met",
                "applicability_established": True,
                "evidence_sufficiency": "no_evidence",
                "not_assessable_reason": "No relevant evidence found in TSD for this control.",
                "verified_control_evidence_ids": [],
            },
        }
        reasoning = "No relevant evidence found in the TSD for this control requirement."
        hunter = HunterResult(
            verdict="not_met",
            confidence=0.85,
            evidence_found=False,
            citations=[],
            reasoning=reasoning,
            logic_summary=reasoning,
        )
        critic = CriticResult(
            outcome="uphold",
            revised_verdict="not_met",
            revised_confidence=0.85,
            valid_citations=[],
            reasoning=reasoning,
            logic_summary=reasoning,
        )
        mediator = MediatorResult(
            final_verdict="not_met",
            raw_final_verdict="not_met",
            confidence=0.85,
            finding_description=reasoning,
            final_citations=[],
            severity=None,
            recommendation=None,
            reasoning=reasoning,
            logic_summary=reasoning,
        )
        rag_gate_output = DebateOutput(
            parameter=child,
            hunter_result=hunter,
            critic_result=critic,
            mediator_result=mediator,
            retrieval_result=None,
            debate_rounds=0,
            analysis_trace=analysis_trace,
        )
        persistence_input = PersistenceInput(
            parameter=child,
            category=category,
            ingestion_job=ingestion_job,
            debate_output=rag_gate_output,
        )
        self.text_debate.persistence.persist_finding(review, persistence_input, summary)

    def _cascade_not_met_to_children(
        self,
        *,
        review,
        category,
        ingestion_job,
        applicable_cfsrs: List[Any],
        raw_parameters: List[Any],
        summary,
    ) -> None:
        """
        For each CFSR that was debated and found not_met, look up the covered raw
        children and persist cascade findings (no extra LLM calls).
        """
        cfsr_stable_keys = [
            getattr(c, "stable_key", None)
            for c in applicable_cfsrs
            if getattr(c, "stable_key", None)
        ]
        if not cfsr_stable_keys:
            return

        # Build cfsr index by stable_key
        cfsr_by_key: Dict[str, Any] = {
            getattr(c, "stable_key", ""): c
            for c in applicable_cfsrs
        }

        # Build child index by stable_key for O(1) lookup
        children_by_key: Dict[str, Any] = {
            getattr(p, "stable_key", ""): p
            for p in raw_parameters
            if getattr(p, "stable_key", None)
        }

        # Query Findings that were just created for this review and are CFSR-based not_met
        with SessionLocal() as db:
            not_met_cfsr_findings = db.execute(
                select(Finding)
                .options(selectinload(Finding.citations))
                .where(
                    Finding.review_id == review.id,
                    Finding.requirement_reference.in_(cfsr_stable_keys),
                    Finding.met_status == "not_met",
                )
            ).scalars().all()

        if not not_met_cfsr_findings:
            self.logger.info(
                "CategoryAnalysisCoordinator._cascade_not_met_to_children: no not_met CFSR findings, nothing to cascade."
            )
            return

        self.logger.info(
            "CategoryAnalysisCoordinator._cascade_not_met_to_children: cascading %d not_met CFSR findings to covered children",
            len(not_met_cfsr_findings),
        )

        cascade_count = 0
        for cfsr_finding in not_met_cfsr_findings:
            cfsr = cfsr_by_key.get(cfsr_finding.requirement_reference)
            if cfsr is None:
                continue
            covered_keys = list(getattr(cfsr, "covered_child_keys", None) or [])
            for child_key in covered_keys:
                child = children_by_key.get(child_key)
                if child is None:
                    continue
                cascade_output = self._build_cascade_debate_output(
                    child=child,
                    cfsr=cfsr,
                    cfsr_finding=cfsr_finding,
                )
                persistence_input = PersistenceInput(
                    parameter=child,
                    category=category,
                    ingestion_job=ingestion_job,
                    debate_output=cascade_output,
                )
                try:
                    self.text_debate.persistence.persist_finding(review, persistence_input, summary)
                    cascade_count += 1
                except Exception as exc:
                    self.logger.warning(
                        "CategoryAnalysisCoordinator._cascade_not_met_to_children: failed to persist cascade for child=%s: %s",
                        child_key,
                        exc,
                    )

        self.logger.info(
            "CategoryAnalysisCoordinator._cascade_not_met_to_children: persisted %d cascade findings",
            cascade_count,
        )

    @staticmethod
    def _build_cascade_debate_output(*, child, cfsr, cfsr_finding) -> DebateOutput:
        """
        Build a synthetic DebateOutput for a cascade child (no LLM calls).
        The child inherits the not_met verdict and confidence from the CFSR finding.
        """
        cfsr_ref = getattr(cfsr, "stable_key", "unknown_cfsr")
        cfsr_reason = getattr(cfsr_finding, "reason", "") or getattr(cfsr_finding, "description", "") or ""
        reasoning = (
            f"Control summary '{cfsr_ref}' was found not met. {cfsr_reason}".strip()
        )
        confidence = float(getattr(cfsr_finding, "confidence_score", 0.7) or 0.7)
        parent_citations = [
            Citation(
                block_id=anchor.block_id,
                page_number=anchor.page_number,
                quoted_text=anchor.quoted_text or "",
                bbox_x0=anchor.bbox_x0,
                bbox_y0=anchor.bbox_y0,
                bbox_x1=anchor.bbox_x1,
                bbox_y1=anchor.bbox_y1,
            )
            for anchor in (getattr(cfsr_finding, "citations", None) or [])
        ]
        analysis_trace = {
            "contract": {
                "synth_mode": "cfsr_cascade",
                "cfsr_stable_key": cfsr_ref,
            },
            "verdict_policy": {
                "source": "cfsr_cascade",
                "raw_final_verdict": "not_met",
                "final_verdict": "not_met",
                "applicability_established": True,
                "evidence_sufficiency": "cfsr_not_met",
                "not_assessable_reason": None,
                "verified_control_evidence_ids": [],
            },
        }
        hunter = HunterResult(
            verdict="not_met",
            confidence=confidence,
            evidence_found=True,
            citations=parent_citations,
            reasoning=reasoning,
            logic_summary=reasoning,
        )
        critic = CriticResult(
            outcome="uphold",
            revised_verdict="not_met",
            revised_confidence=confidence,
            valid_citations=parent_citations,
            reasoning=reasoning,
            logic_summary=reasoning,
        )
        mediator = MediatorResult(
            final_verdict="not_met",
            raw_final_verdict="not_met",
            confidence=confidence,
            finding_description=reasoning,
            final_citations=parent_citations,
            severity=getattr(cfsr_finding, "severity", None),
            recommendation=getattr(cfsr_finding, "recommendation", None),
            reasoning=reasoning,
            logic_summary=reasoning,
        )
        return DebateOutput(
            parameter=child,
            hunter_result=hunter,
            critic_result=critic,
            mediator_result=mediator,
            retrieval_result=None,
            debate_rounds=0,
            analysis_trace=analysis_trace,
        )
