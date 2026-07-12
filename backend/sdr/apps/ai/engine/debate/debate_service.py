from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from sdr.core.config import settings
from sdr.apps.ai.retrieval.postprocessing.quote_grounding import is_quote_grounded, normalize_quote_text

from sdr.apps.ai.agents.base import (
    OUTCOME_UPHOLD,
    OUTCOME_OVERTURN,
    OUTCOME_PARTIAL,
    VERDICT_MET,
    VERDICT_NOT_MET,
    CriticResult,
    HunterResult,
)
from sdr.apps.ai.agents.critic import CriticAgent
from sdr.apps.ai.agents.hunter import HunterAgent
from sdr.apps.ai.agents.mediator import MediatorAgent
from sdr.apps.ai.client import get_model_for_component
from sdr.apps.ai.retrieval.core import RetrievalResult
from sdr.apps.ai.tsd_processing.ingestor import TSDDocument

from sdr.apps.ai.engine.dto import DebateInput, DebateOutput

logger = logging.getLogger(__name__)

_MAX_AGENT_LOGIC_SUMMARY_CHARS = 2500
_MAX_AGENT_COT_TRACE_CHARS = 4000


class DebateService:
    """
    Orchestrates the multi-agent debate workflow for a single parameter.
    Pure logic — no database writes, dependency-injected agents.
    """

    def __init__(
        self,
        hunter: Optional[HunterAgent] = None,
        critic: Optional[CriticAgent] = None,
        mediator: Optional[MediatorAgent] = None,
    ) -> None:
        self.hunter = hunter or HunterAgent()
        self.critic = critic or CriticAgent()
        self.mediator = mediator or MediatorAgent()
        self.max_hunter_calls_per_parameter = int(
            getattr(settings, "AI_DEBATE_MAX_HUNTER_CALLS_PER_PARAMETER", 8)
        )
        self.max_debate_rounds = int(getattr(settings, "AI_DEBATE_MAX_DEBATE_ROUNDS", 2))
        self.max_cot_trace_chars_for_handoff = int(
            getattr(settings, "AI_DEBATE_MAX_COT_TRACE_CHARS_FOR_HANDOFF", _MAX_AGENT_COT_TRACE_CHARS)
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_debate(
        self,
        debate_input: DebateInput,
        retrieval_result: Optional[RetrievalResult],
        tsd_document: TSDDocument,
        cancel_check: Optional[Any] = None,
        agent_chunk_handler: Optional[Callable[[str, str], None]] = None,
        agent_started_handler: Optional[Callable[..., None]] = None,
        agent_completed_handler: Optional[Callable[..., None]] = None,
    ) -> DebateOutput:
        self._raise_if_cancelled(cancel_check, phase="run_debate.entry")
        self.logger.info(
            "DebateService.run_debate: [ENTRY] parameter id=%s",
            debate_input.parameter.id,
        )
        parameter = debate_input.parameter
        parameter_text = debate_input.parameter_text
        parameter_section = debate_input.parameter_section
        context_chunks = debate_input.context_chunks
        context_chunk_map = debate_input.context_chunk_map or {}
        killed_assumptions = list(debate_input.killed_assumptions or [])
        retrieved_chunk_ids = self._citation_grade_ids(context_chunk_map)
        model_routing = self._resolve_model_routing()
        start_ts = time.monotonic()
        timing: Dict[str, Any] = {}
        hunter_call_count = 0

        current_context_chunks = list(context_chunks)
        original_context_chunks = list(debate_input.original_context_chunks or context_chunks)
        warn_threshold = max(1, int(getattr(settings, "AI_DEBATE_WARN_CONTEXT_CHUNK_THRESHOLD", 40)))
        self.logger.info(
            "DebateService.run_debate: parameter id=%s prompt_context_chunks=%d citation_grade_ids=%d",
            debate_input.parameter.id,
            len(current_context_chunks),
            len(retrieved_chunk_ids),
        )
        if len(current_context_chunks) > warn_threshold:
            self.logger.warning(
                "DebateService.run_debate: parameter id=%s prompt_context_chunks=%d exceeded warn threshold=%d",
                debate_input.parameter.id,
                len(current_context_chunks),
                warn_threshold,
            )
        hunter_result: HunterResult = HunterResult()
        critic_result: CriticResult = CriticResult()
        debate_rounds = 0
        hunter_rejected = []
        critic_rejected = []
        debate_history: List[Dict[str, Any]] = []
        rebuttal_context: List[str] = []
        retrieval_refresh_trace: Dict[str, Any] = {"triggered": False, "attempts": 0, "events": []}
        # Once a block_id's underlying text has been confirmed grounded in an
        # earlier round, later rounds don't need to re-derive groundedness from
        # that round's own re-phrased quote — the source text hasn't changed,
        # only the agent's wording of it has.
        grounded_block_ids: set = set()

        round_number = 0
        escalation_round_granted = False
        previous_critic_result: Optional[CriticResult] = None
        while True:
            round_number += 1
            self._raise_if_cancelled(cancel_check, phase=f"run_debate.round_{round_number}.before_hunter")
            debate_rounds = round_number
            self.logger.info(
                "DebateService.run_debate: [HUNTER round=%d] parameter id=%s",
                round_number,
                parameter.id,
            )
            if agent_started_handler:
                agent_started_handler("hunter", round_number=round_number)
            hunter_started = time.monotonic()
            hunter_result, hunter_call_count = self._run_hunter_round(
                cancel_check=cancel_check,
                round_number=round_number,
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                current_context_chunks=current_context_chunks,
                killed_assumptions=killed_assumptions,
                hunter_call_count=hunter_call_count,
                available_block_ids=list(retrieved_chunk_ids),
                stream_handler=(lambda chunk: agent_chunk_handler("hunter", chunk)) if agent_chunk_handler else None,
            )
            timing[f"hunter_round_{round_number}_seconds"] = round(
                time.monotonic() - hunter_started, 4
            )
            hunter_result = self._normalize_reasoning_payload(hunter_result)
            hunter_result.citations, hunter_rejected = self._validate_citations(
                hunter_result.citations,
                retrieved_chunk_ids,
                "hunter",
                context_chunk_map=context_chunk_map,
                grounded_ids=grounded_block_ids,
            )
            grounded_block_ids.update(c.block_id for c in hunter_result.citations)
            # Capture pre-stabilize state: _stabilize_hunter_grounding (below) flips a
            # citation-less "met" to "not_met" and sets evidence_found=False, which
            # would otherwise silently defuse the retry-trigger check just below it
            # (evidence_found would already be False by the time it's checked) —
            # the one retry that exists specifically to rescue this case would then
            # never fire.
            pre_stabilize_verdict = hunter_result.verdict
            pre_stabilize_evidence_found = hunter_result.evidence_found
            hunter_result = self._stabilize_hunter_grounding(
                hunter_result,
                rejected_citations=hunter_rejected,
            )
            if (
                pre_stabilize_verdict in (VERDICT_MET, VERDICT_NOT_MET)
                and pre_stabilize_evidence_found
                and not hunter_result.citations
                and hunter_call_count < self.max_hunter_calls_per_parameter
            ):
                hunter_result, hunter_call_count = self._retry_hunter_for_citations(
                    parameter_text=parameter_text,
                    parameter_section=parameter_section,
                    current_context_chunks=current_context_chunks,
                    killed_assumptions=killed_assumptions,
                    prior_verdict=pre_stabilize_verdict,
                    rejected_ids=[c.block_id for c in hunter_rejected],
                    hunter_call_count=hunter_call_count,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    context_chunk_map=context_chunk_map,
                    grounded_ids=grounded_block_ids,
                    stream_handler=(lambda chunk: agent_chunk_handler("hunter", chunk)) if agent_chunk_handler else None,
                )
                grounded_block_ids.update(c.block_id for c in hunter_result.citations)
            if agent_completed_handler:
                agent_completed_handler(
                    "hunter",
                    hunter_result.cot_trace or hunter_result.logic_summary or hunter_result.reasoning,
                    round_number=round_number,
                )
            sanitized_hunter = self._sanitize_hunter_for_handoff(hunter_result)

            self.logger.info(
                "DebateService.run_debate: [CRITIC round=%d] parameter id=%s",
                round_number,
                parameter.id,
            )
            self._raise_if_cancelled(cancel_check, phase=f"run_debate.round_{round_number}.before_critic")
            if agent_started_handler:
                agent_started_handler("critic", round_number=round_number)
            critic_started = time.monotonic()
            prior_round = (
                {
                    "round": round_number - 1,
                    "objections": list(previous_critic_result.objections or []),
                    "weak_evidence": list(previous_critic_result.weak_evidence or []),
                    "missed_evidence": list(previous_critic_result.missed_evidence or []),
                }
                if previous_critic_result is not None
                else None
            )
            critic_result = self.critic.run(
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                context_chunks=current_context_chunks,
                hunter_result=sanitized_hunter,
                cited_blocks=self._build_cold_start_cited_blocks(
                    sanitized_hunter.citations, context_chunk_map
                ),
                stream_handler=(lambda chunk: agent_chunk_handler("critic", chunk)) if agent_chunk_handler else None,
                available_block_ids=list(retrieved_chunk_ids),
                prior_round=prior_round,
            )
            timing[f"critic_round_{round_number}_seconds"] = round(
                time.monotonic() - critic_started, 4
            )
            critic_result = self._normalize_reasoning_payload(critic_result)
            critic_result.valid_citations, critic_rejected = self._validate_citations(
                critic_result.valid_citations,
                retrieved_chunk_ids,
                "critic",
                context_chunk_map=context_chunk_map,
                grounded_ids=grounded_block_ids,
            )
            grounded_block_ids.update(c.block_id for c in critic_result.valid_citations)
            critic_result.invalid_citation_ids = self._merge_invalid_ids(
                critic_result.invalid_citation_ids,
                [citation.block_id for citation in critic_rejected],
            )
            # Symmetric to the hunter citation-retry above: a "met" critic verdict
            # with no valid_citations previously went straight to
            # _stabilize_critic_grounding's forced downgrade with zero chance to
            # re-ask the model for a citation, unlike hunter which gets one retry.
            if critic_result.revised_verdict == VERDICT_MET and not critic_result.valid_citations:
                pre_retry_revised_verdict = critic_result.revised_verdict
                critic_result = self._retry_critic_for_citations(
                    parameter_text=parameter_text,
                    parameter_section=parameter_section,
                    current_context_chunks=current_context_chunks,
                    hunter_result=sanitized_hunter,
                    cited_blocks=self._build_cold_start_cited_blocks(
                        sanitized_hunter.citations, context_chunk_map
                    ),
                    prior_round=prior_round,
                    prior_revised_verdict=pre_retry_revised_verdict,
                    rejected_ids=[c.block_id for c in critic_rejected],
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    context_chunk_map=context_chunk_map,
                    grounded_ids=grounded_block_ids,
                    stream_handler=(lambda chunk: agent_chunk_handler("critic", chunk)) if agent_chunk_handler else None,
                )
                grounded_block_ids.update(c.block_id for c in critic_result.valid_citations)
            critic_result = self._stabilize_critic_grounding(
                critic_result,
                rejected_citations=critic_rejected,
            )
            if agent_completed_handler:
                agent_completed_handler(
                    "critic",
                    critic_result.cot_trace or critic_result.logic_summary or critic_result.reasoning,
                    critic_outcome=critic_result.outcome,
                    requires_rebuttal=critic_result.requires_rebuttal,
                    round_number=round_number,
                )
            sanitized_critic = self._sanitize_critic_for_handoff(critic_result)

            debate_history.append(
                self._build_debate_history_entry(
                    round_number=round_number,
                    hunter_result=hunter_result,
                    critic_result=critic_result,
                    hunter_rejected=hunter_rejected,
                    critic_rejected=critic_rejected,
                    rebuttal_context=rebuttal_context,
                )
            )

            previous_critic_result = critic_result

            refreshed = self._maybe_refresh_retrieval_context(
                debate_input=debate_input,
                critic_result=critic_result,
                hunter_result=hunter_result,
                round_number=round_number,
            )
            if refreshed:
                refreshed_input = refreshed["debate_input"]
                debate_input.retrieval_refresh_callback = None
                retrieval_refresh_trace["triggered"] = True
                retrieval_refresh_trace["attempts"] += 1
                retrieval_refresh_trace["events"].append(refreshed.get("trace") or {})
                current_context_chunks = list(refreshed_input.context_chunks or current_context_chunks)
                original_context_chunks = list(
                    refreshed_input.original_context_chunks or original_context_chunks
                )
                context_chunk_map = dict(refreshed_input.context_chunk_map or context_chunk_map)
                debate_input.retrieval_query_details = dict(
                    refreshed_input.retrieval_query_details
                    or debate_input.retrieval_query_details
                    or {}
                )
                retrieved_chunk_ids = self._citation_grade_ids(context_chunk_map)

            should_continue, escalation_round_granted = self._should_continue_debate(
                parameter_text,
                hunter_result,
                critic_result,
                round_number,
                escalation_round_granted,
            )
            if refreshed and round_number < self.max_debate_rounds:
                should_continue = True
            if not should_continue:
                break

            self._raise_if_cancelled(cancel_check, phase=f"run_debate.round_{round_number}.before_rebuttal")
            current_context_chunks = self._build_rebuttal_context_chunks(
                round_number=round_number,
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                hunter_result=hunter_result,
                critic_result=critic_result,
                base_context_chunks=current_context_chunks,
            )
            rebuttal_context = current_context_chunks[:1]

        self.logger.info(
            "DebateService.run_debate: [MEDIATOR] parameter id=%s",
            parameter.id,
        )
        self._raise_if_cancelled(cancel_check, phase="run_debate.before_mediator")
        if agent_started_handler:
            agent_started_handler("mediator")
        mediator_started = time.monotonic()
        mediator_result = self.mediator.run(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            hunter_result=sanitized_hunter,
            critic_result=sanitized_critic,
            debate_history=debate_history,
            original_context_chunks=original_context_chunks,
            stream_handler=(lambda chunk: agent_chunk_handler("mediator", chunk)) if agent_chunk_handler else None,
        )
        timing["mediator_seconds"] = round(time.monotonic() - mediator_started, 4)
        mediator_result = self._normalize_reasoning_payload(mediator_result)
        if agent_completed_handler:
            agent_completed_handler("mediator", mediator_result.cot_trace or mediator_result.logic_summary or mediator_result.reasoning)

        timing["flow_total_seconds"] = round(time.monotonic() - start_ts, 4)

        output = DebateOutput(
            parameter=parameter,
            hunter_result=hunter_result,
            critic_result=critic_result,
            mediator_result=mediator_result,
            retrieval_result=retrieval_result,
            debate_rounds=debate_rounds,
            analysis_trace={
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "killed_assumptions": killed_assumptions[:],
                "retrieval_query_details": debate_input.retrieval_query_details or {},
                "context_chunk_map": context_chunk_map,
                "model_routing": model_routing,
                "timing": timing,
                "hunter_claim": {
                    "verdict": hunter_result.verdict,
                    "confidence": hunter_result.confidence,
                    "citation_ids": [c.block_id for c in hunter_result.citations],
                    "applicability_status": getattr(hunter_result, "applicability_status", None),
                    "applicability_reason": getattr(hunter_result, "applicability_reason", ""),
                    "missing_expected_evidence": list(getattr(hunter_result, "missing_expected_evidence", []) or []),
                },
                "hunter_diagnostics": {
                    "available_block_ids": list(retrieved_chunk_ids),
                    "examined_citation_ids": [c.block_id for c in hunter_result.citations],
                    "zero_citation_not_met": (
                        hunter_result.verdict == "not_met" and not hunter_result.citations
                    ),
                    "evidence_found": hunter_result.evidence_found,
                },
                "critic_verification": {
                    "valid_ids": [c.block_id for c in critic_result.valid_citations],
                    "invalid_ids": list(critic_result.invalid_citation_ids),
                    "decision": critic_result.decision,
                    "weak_evidence": list(critic_result.weak_evidence),
                    "missed_evidence": list(critic_result.missed_evidence),
                    "objections": list(critic_result.objections),
                    "requires_rebuttal": critic_result.requires_rebuttal,
                    "applicability_status": getattr(critic_result, "applicability_status", None),
                    "applicability_reason": getattr(critic_result, "applicability_reason", ""),
                    "missing_expected_evidence": list(getattr(critic_result, "missing_expected_evidence", []) or []),
                },
                "mediator_decision_basis": {
                    "final_verdict": mediator_result.final_verdict,
                    "raw_final_verdict": mediator_result.raw_final_verdict or mediator_result.final_verdict,
                    "critic_upheld_citations": [c.block_id for c in critic_result.valid_citations],
                    "mediator_final_citation_ids": [c.block_id for c in mediator_result.final_citations],
                    "verified_evidence": list(mediator_result.verified_evidence),
                    "rejected_evidence": list(mediator_result.rejected_evidence),
                    "debate_rounds_used": mediator_result.debate_rounds_used or debate_rounds,
                    "applicability_status": getattr(mediator_result, "applicability_status", None),
                    "applicability_reason": getattr(mediator_result, "applicability_reason", ""),
                    "missing_expected_evidence": list(getattr(mediator_result, "missing_expected_evidence", []) or []),
                },
                "rejected_evidence": {
                    "hunter": [c.block_id for c in hunter_rejected],
                    "critic": [c.block_id for c in critic_rejected],
                },
                "retrieval_refresh": retrieval_refresh_trace,
                "debate_history": debate_history,
            },
        )
        self.logger.info(
            "DebateService.run_debate: [SUCCESS] parameter id=%s",
            parameter.id,
        )
        return output

    def _raise_if_cancelled(self, cancel_check: Optional[Any], *, phase: str) -> None:
        if cancel_check is None:
            return
        try:
            cancelled = bool(cancel_check())
        except Exception:
            cancelled = False
        if not cancelled:
            return
        self.logger.warning("DebateService.%s: cancellation detected", phase)
        from sdr.apps.ai.engine.persistence.review_run_state_service import AnalysisCancelledError

        raise AnalysisCancelledError("Analysis was cancelled by user.")

    def _run_hunter_round(
        self,
        cancel_check: Optional[Any],
        round_number: int,
        parameter_text: str,
        parameter_section: str,
        current_context_chunks: List[str],
        killed_assumptions: List[Dict[str, Any]],
        hunter_call_count: int,
        available_block_ids: Optional[List[str]] = None,
        stream_handler: Optional[Callable[[str], None]] = None,
    ) -> tuple[HunterResult, int]:
        if hunter_call_count >= self.max_hunter_calls_per_parameter:
            self.logger.warning(
                "DebateService._run_hunter_round: max hunter calls reached (%d), using fallback not_met result",
                self.max_hunter_calls_per_parameter,
            )
            fallback = HunterResult(
                verdict=VERDICT_NOT_MET,
                confidence=0.2,
                reasoning="Hunter call cap reached; defaulting to conservative not_met.",
                evidence_found=False,
                citations=[],
            )
            return fallback, hunter_call_count

        self._raise_if_cancelled(cancel_check, phase="run_debate.hunter.before_llm")
        result = self.hunter.run(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            context_chunks=list(current_context_chunks),
            killed_assumptions=killed_assumptions,
            available_block_ids=available_block_ids,
            stream_handler=stream_handler,
        )
        hunter_call_count += 1
        result = self._normalize_reasoning_payload(result)
        return result, hunter_call_count

    def _resolve_model_routing(self) -> Dict[str, str]:
        return {
            "hunter": get_model_for_component("hunter"),
            "critic": get_model_for_component("critic"),
            "mediator": get_model_for_component("mediator"),
        }

    # Compatibility helper retained for Phase 2 tests/callers.
    def _run_hunters_with_aggregation(
        self,
        parameter_text: str,
        parameter_section: str,
        context_chunks: List[str],
        killed_assumptions: List[Dict[str, Any]],
    ) -> HunterResult:
        winner, _ = self._run_hunter_round(
            cancel_check=None,
            round_number=1,
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            current_context_chunks=context_chunks,
            killed_assumptions=killed_assumptions,
            hunter_call_count=0,
        )
        return winner

    def _normalize_reasoning_payload(self, result) -> Any:
        logic_summary = (getattr(result, "logic_summary", None) or "").strip()
        reasoning = (getattr(result, "reasoning", None) or "").strip()
        if logic_summary and len(logic_summary) > _MAX_AGENT_LOGIC_SUMMARY_CHARS:
            logic_summary = logic_summary[:_MAX_AGENT_LOGIC_SUMMARY_CHARS].rstrip()
        if not logic_summary:
            logic_summary = reasoning
        result.logic_summary = logic_summary
        result.reasoning = logic_summary
        cot_trace = (getattr(result, "cot_trace", None) or "").strip()
        if cot_trace and len(cot_trace) > self.max_cot_trace_chars_for_handoff:
            cot_trace = cot_trace[: self.max_cot_trace_chars_for_handoff].rstrip()
        result.cot_trace = cot_trace or None
        return result

    def _stabilize_hunter_grounding(
        self,
        result: HunterResult,
        *,
        rejected_citations: Optional[List[Any]] = None,
    ) -> HunterResult:
        if result.verdict != VERDICT_MET or result.citations:
            return result

        rejected_ids = [getattr(citation, "block_id", None) for citation in (rejected_citations or []) if getattr(citation, "block_id", None)]
        reason = "Hunter returned met without validated citations; downgraded to not_met."
        if rejected_ids:
            reason = f"{reason} Rejected citation ids: {', '.join(rejected_ids)}."

        result.verdict = VERDICT_NOT_MET
        result.evidence_found = False
        result.confidence = min(float(result.confidence or 0.0), 0.45)
        result.reasoning = reason
        result.logic_summary = reason
        result.evidence_assessment = reason
        return result

    def _stabilize_critic_grounding(
        self,
        result: CriticResult,
        *,
        rejected_citations: Optional[List[Any]] = None,
    ) -> CriticResult:
        if result.revised_verdict != VERDICT_MET or result.valid_citations:
            return result

        rejected_ids = [getattr(citation, "block_id", None) for citation in (rejected_citations or []) if getattr(citation, "block_id", None)]
        reason = "Critic returned met without validated citations; downgraded to not_met."
        if rejected_ids:
            reason = f"{reason} Rejected citation ids: {', '.join(rejected_ids)}."

        result.revised_verdict = VERDICT_NOT_MET
        result.revised_confidence = min(float(result.revised_confidence or 0.0), 0.45)
        result.outcome = OUTCOME_OVERTURN
        result.decision = "reject"
        result.reasoning = reason
        result.logic_summary = reason
        return result

    def _retry_hunter_for_citations(
        self,
        *,
        parameter_text: str,
        parameter_section: str,
        current_context_chunks: List[str],
        killed_assumptions: List[Dict[str, Any]],
        prior_verdict: str,
        rejected_ids: List[str],
        hunter_call_count: int,
        retrieved_chunk_ids: set,
        context_chunk_map: Optional[Dict[str, Any]] = None,
        grounded_ids: Optional[set] = None,
        stream_handler: Optional[Callable[[str], None]] = None,
    ) -> tuple[HunterResult, int]:
        valid_ids_str = ", ".join(sorted(retrieved_chunk_ids)) or "none"
        if rejected_ids:
            citation_note = (
                f"Your cited block_ids {rejected_ids} do not exist in the provided context. "
                f"Valid block_ids are: {valid_ids_str}."
            )
        else:
            citation_note = (
                f"Your response included no citations. "
                f"Valid block_ids available are: {valid_ids_str}."
            )
        retry_header = (
            f"--- CITATION RETRY ---\n"
            f"Your previous verdict='{prior_verdict}' with evidence_found=true requires citations. "
            f"{citation_note} "
            f"Re-examine the context and cite the block_ids you relied on. "
            f"If the requirement remains applicable but no citable support exists, keep or switch to 'not_met' and set evidence_found=false. "
            f"Use 'na' only when the governed capability is clearly absent from the design."
        )
        retry_chunks = [retry_header] + list(current_context_chunks)
        retry_result = self.hunter.run(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            context_chunks=retry_chunks,
            killed_assumptions=killed_assumptions,
            available_block_ids=list(retrieved_chunk_ids),
            stream_handler=stream_handler,
        )
        hunter_call_count += 1
        retry_result = self._normalize_reasoning_payload(retry_result)
        retry_result.citations, _ = self._validate_citations(
            retry_result.citations,
            retrieved_chunk_ids,
            "hunter_citation_retry",
            context_chunk_map=context_chunk_map,
            grounded_ids=grounded_ids,
        )
        retry_result = self._stabilize_hunter_grounding(retry_result)
        self.logger.info(
            "DebateService._retry_hunter_for_citations: prior_verdict=%s retry_verdict=%s citations=%d",
            prior_verdict,
            retry_result.verdict,
            len(retry_result.citations),
        )
        return retry_result, hunter_call_count

    def _retry_critic_for_citations(
        self,
        *,
        parameter_text: str,
        parameter_section: str,
        current_context_chunks: List[str],
        hunter_result,
        cited_blocks: List[Dict[str, Any]],
        prior_round: Optional[Dict[str, Any]],
        prior_revised_verdict: str,
        rejected_ids: List[str],
        retrieved_chunk_ids: set,
        context_chunk_map: Optional[Dict[str, Any]] = None,
        grounded_ids: Optional[set] = None,
        stream_handler: Optional[Callable[[str], None]] = None,
    ) -> CriticResult:
        valid_ids_str = ", ".join(sorted(retrieved_chunk_ids)) or "none"
        if rejected_ids:
            citation_note = (
                f"Your cited block_ids {rejected_ids} do not exist in the provided context "
                f"or were not grounded. Valid block_ids are: {valid_ids_str}."
            )
        else:
            citation_note = (
                f"Your response upheld/overturned to 'met' with no valid_citations. "
                f"Valid block_ids available are: {valid_ids_str}."
            )
        retry_header = (
            f"--- CITATION RETRY ---\n"
            f"Your previous revised_verdict='{prior_revised_verdict}' requires at least one "
            f"verified citation. {citation_note} "
            f"Re-examine the context and cite the block_id(s) you personally verified. "
            f"If you cannot locate a verified citation, you MUST change revised_verdict to "
            f"'not_met' (or 'na' only if the governed capability is architecturally absent)."
        )
        retry_chunks = [retry_header] + list(current_context_chunks)
        retry_result = self.critic.run(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            context_chunks=retry_chunks,
            hunter_result=hunter_result,
            cited_blocks=cited_blocks,
            stream_handler=stream_handler,
            available_block_ids=list(retrieved_chunk_ids),
            prior_round=prior_round,
        )
        retry_result = self._normalize_reasoning_payload(retry_result)
        retry_result.valid_citations, retry_rejected = self._validate_citations(
            retry_result.valid_citations,
            retrieved_chunk_ids,
            "critic_citation_retry",
            context_chunk_map=context_chunk_map,
            grounded_ids=grounded_ids,
        )
        retry_result.invalid_citation_ids = self._merge_invalid_ids(
            retry_result.invalid_citation_ids,
            [citation.block_id for citation in retry_rejected],
        )
        self.logger.info(
            "DebateService._retry_critic_for_citations: prior_verdict=%s retry_verdict=%s citations=%d",
            prior_revised_verdict,
            retry_result.revised_verdict,
            len(retry_result.valid_citations),
        )
        return retry_result

    def _sanitize_hunter_for_handoff(self, result: HunterResult) -> HunterResult:
        return HunterResult(
            verdict=result.verdict,
            confidence=result.confidence,
            applicability_status=result.applicability_status,
            applicability_reason=result.applicability_reason,
            missing_expected_evidence=list(result.missing_expected_evidence),
            reasoning=result.logic_summary or result.reasoning,
            assumptions=list(result.assumptions),
            logic_summary=result.logic_summary or result.reasoning,
            cot_trace=result.cot_trace,
            evidence_found=result.evidence_found,
            citations=list(result.citations),
            checked_context=result.checked_context,
            evidence_quotes=list(result.evidence_quotes),
            evidence_assessment=result.evidence_assessment,
            raw_response=result.raw_response,
            error=result.error,
        )

    def _sanitize_critic_for_handoff(self, result: CriticResult) -> CriticResult:
        return CriticResult(
            outcome=result.outcome,
            revised_verdict=result.revised_verdict,
            revised_confidence=result.revised_confidence,
            applicability_status=result.applicability_status,
            applicability_reason=result.applicability_reason,
            missing_expected_evidence=list(result.missing_expected_evidence),
            reasoning=result.logic_summary or result.reasoning,
            assumptions=list(result.assumptions),
            logic_summary=result.logic_summary or result.reasoning,
            cot_trace=result.cot_trace,
            valid_citations=list(result.valid_citations),
            invalid_citation_ids=list(result.invalid_citation_ids),
            decision=result.decision,
            weak_evidence=list(result.weak_evidence),
            missed_evidence=list(result.missed_evidence),
            objections=list(result.objections),
            requires_rebuttal=result.requires_rebuttal,
            raw_response=result.raw_response,
            error=result.error,
        )

    def _should_continue_debate(
        self,
        parameter_text: str,
        hunter_result: HunterResult,
        critic_result: CriticResult,
        round_number: int,
        escalation_round_granted: bool,
    ) -> tuple[bool, bool]:
        malformed_same_verdict_overturn = (
            critic_result.outcome == OUTCOME_OVERTURN
            and getattr(hunter_result, "verdict", None)
            and getattr(hunter_result, "verdict", None) == getattr(critic_result, "revised_verdict", None)
        )
        if (
            malformed_same_verdict_overturn
            and not list(getattr(critic_result, "missed_evidence", []) or [])
            and not bool(getattr(critic_result, "requires_rebuttal", False))
        ):
            return False, escalation_round_granted
        if round_number >= self.max_debate_rounds:
            # Allow ONE escalation round when the Critic fundamentally disagrees
            # with the Hunter (overturn) AND the Hunter has low confidence.
            # This prevents the Mediator from receiving a stale/contested result
            # that it cannot resolve without additional evidence.
            if not escalation_round_granted and critic_result.outcome == OUTCOME_OVERTURN:
                # Skip escalation when the Critic "overturns" to the same verdict as
                # the Hunter with no cited evidence — this is a malformed OVERTURN
                # (should have been UPHOLD). Granting a rebuttal here lets the Hunter
                # switch verdicts under Critic pressure without real evidence, creating
                # false positives. A genuine OVERTURN changes the verdict direction.
                hunter_ver = getattr(hunter_result, "verdict", None)
                critic_rev = getattr(critic_result, "revised_verdict", None)
                if hunter_ver and critic_rev and hunter_ver == critic_rev:
                    return False, escalation_round_granted
                hunter_conf = float(getattr(hunter_result, "confidence", 0.5) or 0.5)
                if hunter_conf < 0.80:
                    return True, True
            return False, escalation_round_granted
        if critic_result.outcome in {OUTCOME_OVERTURN, OUTCOME_PARTIAL}:
            return True, escalation_round_granted
        if critic_result.requires_rebuttal:
            return True, escalation_round_granted
        if hunter_result.error and not critic_result.error:
            return True, escalation_round_granted
        return False, escalation_round_granted

    def _build_rebuttal_context_chunks(
        self,
        round_number: int,
        parameter_text: str,
        parameter_section: str,
        hunter_result: HunterResult,
        critic_result: CriticResult,
        base_context_chunks: List[str],
    ) -> List[str]:
        invalid_citations = ", ".join(critic_result.invalid_citation_ids) or "none"
        valid_citations = ", ".join(c.block_id for c in critic_result.valid_citations) or "none"
        objections = "\n".join(f"  - {o}" for o in critic_result.objections) or "  (none)"
        weak_evidence = "\n".join(f"  - {w}" for w in critic_result.weak_evidence) or "  (none)"
        missed_evidence = "\n".join(f"  - {m}" for m in critic_result.missed_evidence) or "  (none)"
        rebuttal_lines = [
            f"--- DEBATE REBUTTAL ROUND {round_number} ---",
            f"Section: {parameter_section}",
            f"Requirement: {parameter_text}",
            f"Hunter verdict: {hunter_result.verdict} (confidence={hunter_result.confidence:.2f})",
            f"Critic outcome: {critic_result.outcome}",
            f"Critic revised verdict: {critic_result.revised_verdict} (confidence={critic_result.revised_confidence:.2f})",
            f"Critic reasoning: {critic_result.logic_summary or critic_result.reasoning}",
            "Critic objections you must respond to directly:",
            objections,
            "Critic-flagged weak evidence:",
            weak_evidence,
            "Critic-flagged evidence you may have missed:",
            missed_evidence,
            f"Valid citations: {valid_citations}",
            f"Invalid citations: {invalid_citations}",
            "Instruction: Re-check the original TSD context and respond directly to each Critic objection above.",
            "Defend valid evidence when Critic objections are unsupported; concede only when criticism disproves that the requirement is satisfied.",
            "Only cite evidence that is explicitly present in the supplied TSD context.",
        ]
        rebuttal_chunk = "\n".join(rebuttal_lines)
        return [rebuttal_chunk, *base_context_chunks]

    def _build_debate_history_entry(
        self,
        round_number: int,
        hunter_result: HunterResult,
        critic_result: CriticResult,
        hunter_rejected: List[Any],
        critic_rejected: List[Any],
        rebuttal_context: List[str],
    ) -> Dict[str, Any]:
        return {
            "round": round_number,
            "hunter": {
                "verdict": hunter_result.verdict,
                "confidence": hunter_result.confidence,
                "reasoning": hunter_result.logic_summary or hunter_result.reasoning,
                "checked_context": hunter_result.checked_context,
                "evidence_quotes": list(hunter_result.evidence_quotes),
                "evidence_assessment": hunter_result.evidence_assessment,
                "citation_ids": [c.block_id for c in hunter_result.citations],
                "rejected_citation_ids": [c.block_id for c in hunter_rejected],
            },
            "critic": {
                "outcome": critic_result.outcome,
                "decision": critic_result.decision,
                "revised_verdict": critic_result.revised_verdict,
                "revised_confidence": critic_result.revised_confidence,
                "reasoning": critic_result.logic_summary or critic_result.reasoning,
                "valid_citation_ids": [c.block_id for c in critic_result.valid_citations],
                "invalid_citation_ids": list(critic_result.invalid_citation_ids),
                "rejected_citation_ids": [c.block_id for c in critic_rejected],
                "weak_evidence": list(critic_result.weak_evidence),
                "missed_evidence": list(critic_result.missed_evidence),
                "objections": list(critic_result.objections),
                "requires_rebuttal": critic_result.requires_rebuttal,
            },
            "rebuttal_context": list(rebuttal_context),
        }

    def _build_cold_start_cited_blocks(self, citations, context_chunk_map):
        payload = []
        for citation in citations:
            chunk = context_chunk_map.get(citation.block_id)
            if not chunk:
                continue
            payload.append(
                {
                    "block_id": citation.block_id,
                    "source": chunk.get("source", ""),
                    "section": chunk.get("section", ""),
                    "text": chunk.get("text", ""),
                }
            )
        return payload

    def _is_quote_grounded(self, quoted_text: str, block_text: str) -> bool:
        return is_quote_grounded(quoted_text, block_text)

    def _is_quote_softly_grounded(self, quoted_text: str, block_text: str) -> bool:
        quote = normalize_quote_text(quoted_text)
        block = normalize_quote_text(block_text)
        if not quote or not block:
            return False
        quote_words = {w for w in quote.split() if len(w) > 3}
        if not quote_words:
            return False
        block_words = set(block.split())
        overlap = len(quote_words & block_words) / len(quote_words)
        return overlap >= 0.6

    def _validate_citations(self, citations, allowed_ids, agent_name, context_chunk_map=None, grounded_ids=None):
        allowed = set(allowed_ids)
        grounded = grounded_ids or set()
        valid = []
        rejected = []
        unknown_id_count = 0
        ungrounded_quote_count = 0
        for citation in citations:
            if citation.block_id not in allowed:
                rejected.append(citation)
                unknown_id_count += 1
                continue
            if citation.block_id in grounded:
                valid.append(citation)
                continue
            if context_chunk_map is not None:
                block_text = (context_chunk_map.get(citation.block_id) or {}).get("text", "")
                quoted_text = getattr(citation, "quoted_text", "")
                if not self._is_quote_grounded(quoted_text, block_text) and not self._is_quote_softly_grounded(
                    quoted_text, block_text
                ):
                    rejected.append(citation)
                    ungrounded_quote_count += 1
                    continue
            valid.append(citation)
        if unknown_id_count:
            self.logger.warning(
                "DebateService._validate_citations: agent=%s rejected=%d unknown ids",
                agent_name,
                unknown_id_count,
            )
        if ungrounded_quote_count:
            self.logger.warning(
                "DebateService._validate_citations: agent=%s rejected=%d quote not grounded in source text",
                agent_name,
                ungrounded_quote_count,
            )
        return valid, rejected

    def _citation_grade_ids(self, context_chunk_map):
        ids = []
        for block_id, payload in (context_chunk_map or {}).items():
            if payload.get("citation_grade", True) is False:
                continue
            evidence_kind = str(payload.get("evidence_kind") or "").lower()
            if evidence_kind in {"baseline_requirement"}:
                continue
            ids.append(block_id)
        return ids

    def _merge_invalid_ids(self, existing_ids, additional_ids):
        merged = list(existing_ids or [])
        for citation_id in additional_ids:
            if citation_id not in merged:
                merged.append(citation_id)
        return merged

    def _maybe_refresh_retrieval_context(
        self,
        *,
        debate_input: DebateInput,
        critic_result: CriticResult,
        hunter_result: Optional[HunterResult],
        round_number: int,
    ) -> Optional[Dict[str, Any]]:
        callback = getattr(debate_input, "retrieval_refresh_callback", None)
        if callback is None:
            return None
        missed_evidence = list(getattr(critic_result, "missed_evidence", []) or [])
        invalidated_supported_met = bool(
            hunter_result is not None
            and getattr(hunter_result, "verdict", None) == VERDICT_MET
            and getattr(critic_result, "outcome", None) == OUTCOME_OVERTURN
            and list(getattr(critic_result, "invalid_citation_ids", []) or [])
            and not list(getattr(critic_result, "valid_citations", []) or [])
        )
        if not missed_evidence and not invalidated_supported_met:
            return None
        refresh_reason = (
            "invalidated_met_evidence"
            if invalidated_supported_met and not missed_evidence
            else "missed_evidence"
        )
        try:
            return callback(
                critic_result=critic_result,
                hunter_result=hunter_result,
                round_number=round_number,
                refresh_reason=refresh_reason,
            )
        except Exception:
            self.logger.exception(
                "DebateService._maybe_refresh_retrieval_context: refresh callback failed for parameter id=%s",
                getattr(debate_input.parameter, "id", None),
            )
            return None
