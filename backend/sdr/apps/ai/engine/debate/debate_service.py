from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from sdr.core.config import settings
from sdr.apps.ai.retrieval.postprocessing.quote_grounding import is_quote_grounded

from sdr.apps.ai.agents.base import (
    OUTCOME_OVERTURN,
    OUTCOME_PARTIAL,
    VERDICT_MET,
    VERDICT_NA,
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
_MAX_AGENT_COT_TRACE_CHARS = 12000
_PERSONAS = [
    "architecture_network",
    "iam_access_control",
    "data_crypto_privacy",
]


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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_debate(
        self,
        debate_input: DebateInput,
        retrieval_result: Optional[RetrievalResult],
        tsd_document: TSDDocument,
        cancel_check: Optional[Any] = None,
        agent_chunk_handler: Optional[Callable[[str, str], None]] = None,
        agent_started_handler: Optional[Callable[[str], None]] = None,
        agent_completed_handler: Optional[Callable[[str, str], None]] = None,
    ) -> DebateOutput:
        self._raise_if_cancelled(cancel_check, phase="run_debate.entry")
        self.logger.info(
            "DebateService.run_debate: [ENTRY] parameter id=%s",
            debate_input.parameter.id,
        )
        parameter = debate_input.parameter
        parameter_text = debate_input.parameter_text
        parameter_section = debate_input.parameter_section
        contract = debate_input.contract or {}
        context_chunks = debate_input.context_chunks
        context_chunk_map = debate_input.context_chunk_map or {}
        killed_assumptions = list(debate_input.killed_assumptions or [])
        hunter_plan = debate_input.hunter_plan or {}
        retrieved_chunk_ids = self._citation_grade_ids(context_chunk_map)
        model_routing = self._resolve_model_routing()
        start_ts = time.monotonic()
        timing: Dict[str, Any] = {}
        hunter_call_count = 0

        current_context_chunks = list(context_chunks)
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
        hunter_results_by_persona: Dict[str, Any] = {}
        merged_evidence: Dict[str, Any] = {"deduped_ids": [], "provenance": {}}
        persona_plan: Dict[str, Any] = {}
        hunter_rejected = []
        critic_rejected = []
        debate_history: List[Dict[str, Any]] = []
        rebuttal_context: List[str] = []

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
                agent_started_handler("hunter")
            hunter_started = time.monotonic()
            (
                hunter_result,
                hunter_results_by_persona,
                merged_evidence,
                persona_plan,
                hunter_call_count,
            ) = self._run_hunter_round(
                cancel_check=cancel_check,
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                contract=contract,
                current_context_chunks=current_context_chunks,
                killed_assumptions=killed_assumptions,
                hunter_plan=hunter_plan,
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
            )
            hunter_result = self._stabilize_hunter_grounding(
                hunter_result,
                rejected_citations=hunter_rejected,
            )
            if (
                hunter_result.verdict in (VERDICT_MET, VERDICT_NOT_MET)
                and hunter_result.evidence_found
                and not hunter_result.citations
                and hunter_call_count < self.max_hunter_calls_per_parameter
            ):
                hunter_result, hunter_call_count = self._retry_hunter_for_citations(
                    parameter_text=parameter_text,
                    parameter_section=parameter_section,
                    contract=contract,
                    current_context_chunks=current_context_chunks,
                    killed_assumptions=killed_assumptions,
                    prior_verdict=hunter_result.verdict,
                    rejected_ids=[c.block_id for c in hunter_rejected],
                    hunter_call_count=hunter_call_count,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    context_chunk_map=context_chunk_map,
                    stream_handler=(lambda chunk: agent_chunk_handler("hunter", chunk)) if agent_chunk_handler else None,
                )
            if agent_completed_handler:
                agent_completed_handler("hunter", hunter_result.logic_summary or hunter_result.reasoning)
            sanitized_hunter = self._sanitize_hunter_for_handoff(hunter_result)

            self.logger.info(
                "DebateService.run_debate: [CRITIC round=%d] parameter id=%s",
                round_number,
                parameter.id,
            )
            self._raise_if_cancelled(cancel_check, phase=f"run_debate.round_{round_number}.before_critic")
            if agent_started_handler:
                agent_started_handler("critic")
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
                contract=contract,
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
            )
            critic_result.invalid_citation_ids = self._merge_invalid_ids(
                critic_result.invalid_citation_ids,
                [citation.block_id for citation in critic_rejected],
            )
            critic_result = self._stabilize_critic_grounding(
                critic_result,
                rejected_citations=critic_rejected,
            )
            if agent_completed_handler:
                agent_completed_handler("critic", critic_result.logic_summary or critic_result.reasoning)
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

            should_continue, escalation_round_granted = self._should_continue_debate(
                hunter_result, critic_result, round_number, escalation_round_granted
            )
            if not should_continue:
                break

            self._raise_if_cancelled(cancel_check, phase=f"run_debate.round_{round_number}.before_rebuttal")
            current_context_chunks = self._build_rebuttal_context_chunks(
                round_number=round_number,
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                hunter_result=hunter_result,
                critic_result=critic_result,
                base_context_chunks=context_chunks,
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
            contract=contract,
            hunter_result=sanitized_hunter,
            critic_result=sanitized_critic,
            debate_history=debate_history,
            stream_handler=(lambda chunk: agent_chunk_handler("mediator", chunk)) if agent_chunk_handler else None,
        )
        timing["mediator_seconds"] = round(time.monotonic() - mediator_started, 4)
        mediator_result = self._normalize_reasoning_payload(mediator_result)
        mediator_result = self._apply_mediator_evidence_policy(
            mediator_result=mediator_result,
            critic_result=critic_result,
            contract=contract,
            hunter_result=hunter_result,
        )
        mediator_result = self._calibrate_confidence(
            mediator_result=mediator_result,
            hunter_result=hunter_result,
            critic_result=critic_result,
        )
        if agent_completed_handler:
            agent_completed_handler("mediator", mediator_result.logic_summary or mediator_result.reasoning)

        timing["flow_total_seconds"] = round(time.monotonic() - start_ts, 4)

        output = DebateOutput(
            parameter=parameter,
            hunter_result=hunter_result,
            critic_result=critic_result,
            mediator_result=mediator_result,
            retrieval_result=retrieval_result,
            debate_rounds=debate_rounds,
            analysis_trace={
                "contract": contract,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "hunter_plan": persona_plan,
                "hunter_results": hunter_results_by_persona,
                "merged_evidence": merged_evidence,
                "killed_assumptions": killed_assumptions[:],
                "retrieval_query_details": debate_input.retrieval_query_details or {},
                "context_chunk_map": context_chunk_map,
                "model_routing": model_routing,
                "timing": timing,
                "hunter_claim": {
                    "verdict": hunter_result.verdict,
                    "confidence": hunter_result.confidence,
                    "citation_ids": [c.block_id for c in hunter_result.citations],
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
                },
                "mediator_decision_basis": {
                    "final_verdict": mediator_result.final_verdict,
                    "raw_final_verdict": mediator_result.raw_final_verdict or mediator_result.final_verdict,
                    "critic_upheld_citations": [c.block_id for c in critic_result.valid_citations],
                    "verified_evidence": list(mediator_result.verified_evidence),
                    "rejected_evidence": list(mediator_result.rejected_evidence),
                    "debate_rounds_used": mediator_result.debate_rounds_used or debate_rounds,
                },
                "verdict_policy": {
                    "source": getattr(mediator_result, "verdict_policy_source", "mediator"),
                    "raw_final_verdict": mediator_result.raw_final_verdict or mediator_result.final_verdict,
                    "final_verdict": mediator_result.final_verdict,
                    "applicability_established": bool(getattr(mediator_result, "applicability_established", True)),
                    "evidence_sufficiency": getattr(mediator_result, "evidence_sufficiency", None),
                    "not_assessable_reason": getattr(mediator_result, "not_assessable_reason", None),
                    "verified_control_evidence_ids": [c.block_id for c in mediator_result.final_citations],
                },
                "rejected_evidence": {
                    "hunter": [c.block_id for c in hunter_rejected],
                    "critic": [c.block_id for c in critic_rejected],
                },
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
        parameter_text: str,
        parameter_section: str,
        contract: Dict[str, Any],
        current_context_chunks: List[str],
        killed_assumptions: List[Dict[str, Any]],
        hunter_plan: Dict[str, Any],
        hunter_call_count: int,
        available_block_ids: Optional[List[str]] = None,
        stream_handler: Optional[Callable[[str], None]] = None,
    ) -> tuple[HunterResult, Dict[str, Any], Dict[str, Any], Dict[str, Any], int]:
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
            return fallback, {}, {"deduped_ids": [], "provenance": {}}, {"mode": "guardrail", "personas": []}, hunter_call_count

        persona_plan = self._build_hunter_plan(
            contract=contract,
            parameter_text=parameter_text,
            context_chunk_count=len(current_context_chunks),
            incoming_plan=hunter_plan,
        )
        personas = list(persona_plan.get("personas") or [contract.get("domain") or "general"])
        persona = personas[0]

        self._raise_if_cancelled(cancel_check, phase="run_debate.hunter.before_llm")
        result = self.hunter.run(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
            context_chunks=list(current_context_chunks),
            persona_focus=persona,
            killed_assumptions=killed_assumptions,
            available_block_ids=available_block_ids,
            stream_handler=stream_handler,
        )
        hunter_call_count += 1
        result = self._normalize_reasoning_payload(result)

        merged_citations = {
            c.block_id: {"citation": c, "personas": {persona}}
            for c in result.citations
        }
        persona_payload = {
            persona: {
                "verdict": result.verdict,
                "confidence": result.confidence,
                "reasoning": result.logic_summary or result.reasoning,
                "citation_ids": [c.block_id for c in result.citations],
            }
        }
        merged_evidence = {
            "deduped_ids": list(merged_citations.keys()),
            "provenance": {
                block_id: sorted(list(entry["personas"]))
                for block_id, entry in merged_citations.items()
            },
        }
        return result, persona_payload, merged_evidence, persona_plan, hunter_call_count

    def _resolve_model_routing(self) -> Dict[str, str]:
        return {
            "contract_synthesizer": get_model_for_component("contract_synthesizer"),
            "hunter": get_model_for_component("hunter"),
            "critic": get_model_for_component("critic"),
            "mediator": get_model_for_component("mediator"),
        }

    def _build_hunter_plan(
        self,
        contract: Dict[str, Any],
        parameter_text: str,
        context_chunk_count: int,
        incoming_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        if incoming_plan.get("personas"):
            persona = str(incoming_plan["personas"][0] or "").strip()
            if not persona:
                persona = str(contract.get("domain") or "general").strip()
            return {"mode": "single", "personas": [persona if persona in _PERSONAS else "general"]}

        domain = str(contract.get("domain") or "").strip()
        if domain in _PERSONAS:
            return {"mode": "single", "personas": [domain]}
        return {"mode": "single", "personas": ["general"]}

    # Compatibility helper retained for Phase 2 tests/callers.
    def _run_hunters_with_aggregation(
        self,
        parameter_text: str,
        parameter_section: str,
        contract: Dict[str, Any],
        context_chunks: List[str],
        killed_assumptions: List[Dict[str, Any]],
        persona_plan: Dict[str, Any],
    ) -> tuple[HunterResult, Dict[str, Any], Dict[str, Any]]:
        winner, payload, merged, _, _ = self._run_hunter_round(
            cancel_check=None,
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
            current_context_chunks=context_chunks,
            killed_assumptions=killed_assumptions,
            hunter_plan=persona_plan,
            hunter_call_count=0,
        )
        return winner, payload, merged

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
        if cot_trace and len(cot_trace) > _MAX_AGENT_COT_TRACE_CHARS:
            cot_trace = cot_trace[:_MAX_AGENT_COT_TRACE_CHARS].rstrip()
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
        contract: Dict[str, Any],
        current_context_chunks: List[str],
        killed_assumptions: List[Dict[str, Any]],
        prior_verdict: str,
        rejected_ids: List[str],
        hunter_call_count: int,
        retrieved_chunk_ids: set,
        context_chunk_map: Optional[Dict[str, Any]] = None,
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
            f"If no relevant block_id exists, change verdict to 'na' and set evidence_found=false."
        )
        retry_chunks = [retry_header] + list(current_context_chunks)
        retry_result = self.hunter.run(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
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
        )
        retry_result = self._stabilize_hunter_grounding(retry_result)
        self.logger.info(
            "DebateService._retry_hunter_for_citations: prior_verdict=%s retry_verdict=%s citations=%d",
            prior_verdict,
            retry_result.verdict,
            len(retry_result.citations),
        )
        return retry_result, hunter_call_count

    def _sanitize_hunter_for_handoff(self, result: HunterResult) -> HunterResult:
        return HunterResult(
            verdict=result.verdict,
            confidence=result.confidence,
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
        hunter_result: HunterResult,
        critic_result: CriticResult,
        round_number: int,
        escalation_round_granted: bool,
    ) -> tuple[bool, bool]:
        if round_number >= self.max_debate_rounds:
            # Allow ONE escalation round when the Critic fundamentally disagrees
            # with the Hunter (overturn) AND the Hunter has low confidence.
            # This prevents the Mediator from receiving a stale/contested result
            # that it cannot resolve without additional evidence.
            if not escalation_round_granted and critic_result.outcome == OUTCOME_OVERTURN:
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
        rebuttal_chunk = "\n".join(
            [
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
                "Defend valid evidence when Critic objections are unsupported; concede only when criticism disproves contract satisfaction.",
                "Only cite evidence that is explicitly present in the supplied TSD context.",
            ]
        )
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

    def _validate_citations(self, citations, allowed_ids, agent_name, context_chunk_map=None):
        allowed = set(allowed_ids)
        valid = []
        rejected = []
        unknown_id_count = 0
        ungrounded_quote_count = 0
        for citation in citations:
            if citation.block_id not in allowed:
                rejected.append(citation)
                unknown_id_count += 1
                continue
            if context_chunk_map is not None:
                block_text = (context_chunk_map.get(citation.block_id) or {}).get("text", "")
                if not self._is_quote_grounded(getattr(citation, "quoted_text", ""), block_text):
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

    def _apply_mediator_evidence_policy(self, mediator_result, critic_result, contract, hunter_result=None):
        valid_citations = list(critic_result.valid_citations or [])
        in_scope = bool(contract.get("in_scope", True))
        specific = bool(contract.get("specific_enough", True))
        raw_verdict = mediator_result.raw_final_verdict or mediator_result.final_verdict
        mediator_result.raw_final_verdict = raw_verdict
        mediator_result.verdict_policy_source = "mediator"
        mediator_result.applicability_established = bool(in_scope and specific)
        mediator_result.evidence_sufficiency = "unverified"
        mediator_result.not_assessable_reason = None
        if not in_scope or not specific:
            mediator_result.final_verdict = VERDICT_NA
            mediator_result.final_citations = []
            mediator_result.severity = None
            mediator_result.recommendation = None
            mediator_result.verdict_policy_source = "contract_not_applicable"
            mediator_result.evidence_sufficiency = "not_applicable"
            mediator_result.not_assessable_reason = "contract_not_in_scope_or_not_specific"
            return mediator_result

        if raw_verdict == VERDICT_NA or self._reasoning_indicates_not_assessable(mediator_result):
            mediator_result.final_verdict = VERDICT_NA
            mediator_result.final_citations = []
            mediator_result.severity = None
            mediator_result.recommendation = None
            mediator_result.verdict_policy_source = "not_assessable"
            mediator_result.evidence_sufficiency = "not_assessable"
            mediator_result.not_assessable_reason = "mediator_or_critic_indicated_na_or_not_assessable"
            return mediator_result

        accepted_met = (
            raw_verdict == VERDICT_MET
            and valid_citations
            and critic_result.revised_verdict == VERDICT_MET
        )
        if accepted_met:
            mediator_result.final_verdict = VERDICT_MET
            mediator_result.final_citations = valid_citations
            mediator_result.severity = None
            mediator_result.recommendation = None
            mediator_result.evidence_sufficiency = "verified_met"
            return mediator_result

        if raw_verdict == VERDICT_MET:
            mediator_result.final_verdict = VERDICT_NA
            mediator_result.final_citations = []
            mediator_result.severity = None
            mediator_result.recommendation = None
            mediator_result.verdict_policy_source = "met_without_verified_evidence"
            mediator_result.evidence_sufficiency = "insufficient_for_met"
            mediator_result.not_assessable_reason = "met_claim_lacked_critic_verified_citations"
            return mediator_result

        if raw_verdict == "partial":
            mediator_result.final_verdict = VERDICT_NOT_MET
            if "partial" not in (mediator_result.reasoning or "").lower():
                mediator_result.reasoning = f"Partial evidence only: {mediator_result.reasoning}"
                mediator_result.logic_summary = mediator_result.reasoning
            mediator_result.verdict_policy_source = "partial_evidence_not_met"
            mediator_result.evidence_sufficiency = "partial"
            mediator_result.final_citations = valid_citations
            return mediator_result

        mediator_result.final_verdict = VERDICT_NOT_MET
        mediator_result.final_citations = valid_citations if raw_verdict == VERDICT_NOT_MET else []
        mediator_result.verdict_policy_source = "applicable_missing_or_contradicted_evidence"
        mediator_result.evidence_sufficiency = "missing_or_contradicted"

        policy = str(
            getattr(settings, "AI_BATCH_DEBATE_UNGROUNDED_NOT_MET_POLICY", "preserve_not_met") or ""
        ).strip().lower()
        if (
            not valid_citations
            and raw_verdict == VERDICT_NOT_MET
            and policy in {"downgrade_na", "selective_fallback"}
        ):
            mediator_result.final_verdict = VERDICT_NA
            mediator_result.final_citations = []
            mediator_result.severity = None
            mediator_result.recommendation = None
            mediator_result.verdict_policy_source = "not_met_no_grounded_citations"
            mediator_result.evidence_sufficiency = "no_grounded_evidence"
            mediator_result.not_assessable_reason = "not_met_without_any_critic_verified_citations"

        return mediator_result

    def _calibrate_confidence(self, *, mediator_result, hunter_result, critic_result):
        """Post-hoc confidence calibration based on objective signals.

        Adjusts the LLM-generated confidence score using deterministic heuristics:
        - Boost when Hunter and Critic agree on verdict (+0.05 to +0.10).
        - Reduce when not_met has zero grounded citations (-0.05 to -0.10).
        - Cap na confidence adaptively (0.60 default, 0.70 when contract says out-of-scope).
        All adjustments are clamped to [0.0, 1.0].
        """
        raw_confidence = float(getattr(mediator_result, "confidence", 0.5) or 0.5)
        verdict = getattr(mediator_result, "final_verdict", None)
        hunter_verdict = getattr(hunter_result, "verdict", None)
        critic_verdict = getattr(critic_result, "revised_verdict", None)
        citation_count = len(getattr(mediator_result, "final_citations", []) or [])
        adjustment = 0.0

        # Agreement boost: Hunter and Critic both agree with the final verdict
        if hunter_verdict == critic_verdict == verdict:
            adjustment += 0.10
        elif hunter_verdict == critic_verdict:
            adjustment += 0.05

        # Citation signal
        if verdict == VERDICT_NOT_MET and citation_count == 0:
            adjustment -= 0.10
        elif verdict == VERDICT_MET and citation_count >= 2:
            adjustment += 0.05

        adjusted = max(0.0, min(raw_confidence + adjustment, 1.0))

        # Cap na confidence — it's inherently an uncertain verdict, but allow
        # slightly higher confidence when the contract explicitly says the
        # requirement is out of scope (a well-founded na).
        if verdict == VERDICT_NA:
            policy_source = getattr(mediator_result, "verdict_policy_source", "")
            if policy_source == "contract_not_applicable":
                adjusted = min(adjusted, 0.70)
            else:
                adjusted = min(adjusted, 0.60)

        mediator_result.confidence = round(adjusted, 2)
        return mediator_result

    def _reasoning_indicates_not_assessable(self, mediator_result) -> bool:
        text = " ".join(
            [
                str(getattr(mediator_result, "reasoning", "") or ""),
                str(getattr(mediator_result, "logic_summary", "") or ""),
            ]
        ).lower()
        if not text:
            return False
        markers = [
            "not applicable",
            "not assessable",
            "cannot be assessed",
            "cannot assess",
            "insufficient evidence to assess",
            "trigger condition is not established",
            "trigger conditions are not met",
            "applicability is not established",
            "applicability requires",
            "without applicability established",
            "na is appropriate",
            "n/a is appropriate",
        ]
        return any(marker in text for marker in markers)
