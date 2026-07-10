from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from sdr.core.config import settings
from sdr.apps.ai.retrieval.postprocessing.quote_grounding import is_quote_grounded, normalize_quote_text

from sdr.apps.ai.agents.base import (
    APPLICABILITY_ESTABLISHED,
    APPLICABILITY_NOT_ESTABLISHED,
    OUTCOME_UPHOLD,
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
_MAX_AGENT_COT_TRACE_CHARS = 4000
_PERSONAS = [
    "architecture_network",
    "iam_access_control",
    "data_crypto_privacy",
]
_NEGATIVE_ABSENCE_MARKERS = (
    "does not use",
    "does not include",
    "unsupported",
    "deprecated",
    "password hints",
    "knowledge-based",
    "secret question",
    "clear text",
    "weaker authenticator",
    "weak authenticators",
    "no weaker",
    "avoid",
    "limit the use of",
)
_EXPLICIT_ABSENCE_EVIDENCE_MARKERS = (
    "only",
    "solely",
    "exclusively",
    "must not",
    "does not use",
    "does not include",
    "without using",
    "no ",
    "none ",
    "all ",
    "every ",
)
_REQUIREMENT_STOPWORDS = {
    "the",
    "that",
    "this",
    "with",
    "from",
    "into",
    "using",
    "use",
    "used",
    "verify",
    "application",
    "system",
    "shall",
    "have",
    "has",
    "are",
    "all",
    "any",
    "for",
    "and",
    "or",
    "its",
    "their",
    "them",
    "than",
    "when",
    "where",
    "which",
}


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
        hunter_results_by_persona: Dict[str, Any] = {}
        merged_evidence: Dict[str, Any] = {"deduped_ids": [], "provenance": {}}
        persona_plan: Dict[str, Any] = {}
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
            (
                hunter_result,
                hunter_results_by_persona,
                merged_evidence,
                persona_plan,
                hunter_call_count,
            ) = self._run_hunter_round(
                cancel_check=cancel_check,
                round_number=round_number,
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
                    contract=contract,
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
                    contract=contract,
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
            contract=contract,
            hunter_result=sanitized_hunter,
            critic_result=sanitized_critic,
            debate_history=debate_history,
            original_context_chunks=original_context_chunks,
            stream_handler=(lambda chunk: agent_chunk_handler("mediator", chunk)) if agent_chunk_handler else None,
        )
        timing["mediator_seconds"] = round(time.monotonic() - mediator_started, 4)
        mediator_result = self._normalize_reasoning_payload(mediator_result)
        mediator_result = self._apply_mediator_evidence_policy(
            mediator_result=mediator_result,
            critic_result=critic_result,
            contract=contract,
            hunter_result=hunter_result,
            parameter_text=parameter_text,
        )
        mediator_result = self._calibrate_confidence(
            mediator_result=mediator_result,
            hunter_result=hunter_result,
            critic_result=critic_result,
        )
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
                "contract": contract,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "hunter_plan": persona_plan,
                "hunter_results": hunter_results_by_persona,
                "hunter_persona_outputs": hunter_results_by_persona,
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
                    "verified_evidence": list(mediator_result.verified_evidence),
                    "rejected_evidence": list(mediator_result.rejected_evidence),
                    "debate_rounds_used": mediator_result.debate_rounds_used or debate_rounds,
                    "applicability_status": getattr(mediator_result, "applicability_status", None),
                    "applicability_reason": getattr(mediator_result, "applicability_reason", ""),
                    "missing_expected_evidence": list(getattr(mediator_result, "missing_expected_evidence", []) or []),
                },
                "verdict_policy": {
                    "source": getattr(mediator_result, "verdict_policy_source", "mediator"),
                    "raw_final_verdict": mediator_result.raw_final_verdict or mediator_result.final_verdict,
                    "final_verdict": mediator_result.final_verdict,
                    "applicability_established": bool(getattr(mediator_result, "applicability_established", True)),
                    "structured_applicability_present": bool(getattr(mediator_result, "applicability_status", "")),
                    "applicability_status": getattr(mediator_result, "applicability_status", None),
                    "applicability_reason": getattr(mediator_result, "applicability_reason", ""),
                    "missing_expected_evidence": list(getattr(mediator_result, "missing_expected_evidence", []) or []),
                    "evidence_sufficiency": getattr(mediator_result, "evidence_sufficiency", None),
                    "not_assessable_reason": getattr(mediator_result, "not_assessable_reason", None),
                    "verified_control_evidence_ids": [c.block_id for c in mediator_result.final_citations],
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
            round_number=round_number,
        )
        personas = list(persona_plan.get("personas") or [contract.get("domain") or "general"])
        persona_results: Dict[str, HunterResult] = {}
        merged_citations: Dict[str, Any] = {}
        persona_payload: Dict[str, Any] = {}
        for persona in personas:
            if hunter_call_count >= self.max_hunter_calls_per_parameter:
                break
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
            persona_results[persona] = result
            persona_payload[persona] = {
                "verdict": result.verdict,
                "confidence": result.confidence,
                "reasoning": result.logic_summary or result.reasoning,
                "citation_ids": [c.block_id for c in result.citations],
                "evidence_found": result.evidence_found,
                "applicability_status": getattr(result, "applicability_status", None),
                "missing_expected_evidence": list(getattr(result, "missing_expected_evidence", []) or []),
            }
            for citation in result.citations:
                entry = merged_citations.setdefault(
                    citation.block_id,
                    {"citation": citation, "personas": set()},
                )
                entry["personas"].add(persona)
        result = self._select_hunter_winner(persona_results, personas)
        result.citations = [entry["citation"] for entry in merged_citations.values()]
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
        round_number: int,
    ) -> Dict[str, Any]:
        # Always a single Hunter call per parameter — no multi-persona fan-out.
        # Multi-persona used to re-send the full retrieved context up to 3x per
        # parameter (once per persona) regardless of difficulty; this picks the
        # single best-matched persona instead (round 1 and rebuttal rounds alike).
        domain = str(contract.get("domain") or "").strip()
        mode = "rebuttal_single" if round_number > 1 else "single_persona"
        if incoming_plan.get("personas"):
            persona = str(incoming_plan["personas"][0] or "").strip()
            if persona in _PERSONAS:
                return {"mode": mode, "personas": [persona]}
        if domain in _PERSONAS:
            return {"mode": mode, "personas": [domain]}
        return {"mode": mode, "personas": ["general"]}

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
            round_number=1,
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
        contract: Dict[str, Any],
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
        contract: Dict[str, Any],
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
            contract=contract,
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

    def _select_hunter_winner(
        self,
        persona_results: Dict[str, HunterResult],
        personas: List[str],
    ) -> HunterResult:
        if not persona_results:
            return HunterResult(
                verdict=VERDICT_NOT_MET,
                confidence=0.2,
                reasoning="No Hunter persona completed successfully.",
                logic_summary="No Hunter persona completed successfully.",
                evidence_found=False,
                citations=[],
            )

        def _score(result: HunterResult) -> tuple:
            verdict_rank = {
                VERDICT_MET: 3,
                VERDICT_NOT_MET: 2,
                VERDICT_NA: 1,
            }.get(result.verdict, 0)
            return (
                verdict_rank,
                len(result.citations or []),
                1 if result.evidence_found else 0,
                float(result.confidence or 0.0),
            )

        selected_persona = max(
            [persona for persona in personas if persona in persona_results],
            key=lambda persona: _score(persona_results[persona]),
        )
        return deepcopy(persona_results[selected_persona])

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
        if self._should_force_evidence_rescue_round(
            parameter_text=parameter_text,
            hunter_result=hunter_result,
            critic_result=critic_result,
            round_number=round_number,
        ):
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
            "Defend valid evidence when Critic objections are unsupported; concede only when criticism disproves contract satisfaction.",
            "Only cite evidence that is explicitly present in the supplied TSD context.",
        ]
        if self._should_force_evidence_rescue_round(
            parameter_text=parameter_text,
            hunter_result=hunter_result,
            critic_result=critic_result,
            round_number=round_number - 1,
        ):
            rebuttal_lines.extend(
                [
                    "RECOVERY FOCUS: The Critic verified real evidence in the cited blocks.",
                    "Do not repeat a generic missing-evidence argument unless you can name the exact essential property still missing from those verified citations.",
                    "If the verified citations satisfy the requirement's core security property at the TSD level, return 'met' and anchor the verdict only to those verified blocks.",
                ]
            )
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
        # is_quote_grounded requires the quote to be a near-verbatim contiguous
        # excerpt (exact substring or >=0.85 contiguous-match coverage). The
        # cost-optimized models this pipeline now routes to (see
        # model_routing) paraphrase citations rather than transcribing them
        # verbatim far more often than the models it was tuned against,
        # causing that strict check to reject a large share of citations that
        # do genuinely reference the cited block's content — which then
        # cascades into forced verdict downgrades below. This is a secondary,
        # looser fallback: it still requires most of the quote's substantive
        # words to actually appear in the block (guarding against citing an
        # unrelated block), just without requiring them to be contiguous or
        # verbatim.
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

    def _result_applicability_status(self, result: Optional[object]) -> Optional[str]:
        if result is None:
            return None
        status = str(getattr(result, "applicability_status", "") or "").strip().lower()
        if status in {APPLICABILITY_ESTABLISHED, APPLICABILITY_NOT_ESTABLISHED}:
            return status
        return None

    def _applicability_is_established(self, result: Optional[object]) -> Optional[bool]:
        status = self._result_applicability_status(result)
        if status == APPLICABILITY_ESTABLISHED:
            return True
        if status == APPLICABILITY_NOT_ESTABLISHED:
            return False
        return None

    def _apply_mediator_evidence_policy(
        self,
        mediator_result,
        critic_result,
        contract,
        hunter_result=None,
        parameter_text: str = "",
    ):
        valid_citations = list(critic_result.valid_citations or [])
        in_scope = bool(contract.get("in_scope", True))
        specific = bool(contract.get("specific_enough", True))
        raw_verdict = mediator_result.raw_final_verdict or mediator_result.final_verdict
        requirement_shape = self._classify_requirement_shape(parameter_text)
        mediator_applicability = self._applicability_is_established(mediator_result)
        critic_applicability = self._applicability_is_established(critic_result)
        hunter_applicability = self._applicability_is_established(hunter_result)
        applicability_not_established = False in {
            value for value in (mediator_applicability, critic_applicability, hunter_applicability) if value is not None
        }
        applicability_established_by_agents = True in {
            value for value in (mediator_applicability, critic_applicability, hunter_applicability) if value is not None
        }
        mediator_result.raw_final_verdict = raw_verdict
        mediator_result.verdict_policy_source = "mediator"
        mediator_result.applicability_established = (
            False
            if applicability_not_established
            else (
                True
                if applicability_established_by_agents
                else bool(in_scope and specific)
            )
        )
        mediator_result.evidence_sufficiency = "unverified"
        mediator_result.not_assessable_reason = None
        explicit_absent_capability = any(
            self._has_explicit_absent_capability_signal(result)
            for result in (mediator_result, critic_result, hunter_result)
            if result is not None
        )
        # A confident "not_met" from the actual debate outranks a pre-debate
        # contract guess about scope — contract synthesis runs before any TSD
        # evidence has been reviewed, while the debate has actually looked at
        # retrieved context and reached a definitive conclusion that the control
        # is missing. This is deliberately asymmetric: a "met" verdict still gets
        # the contract-gate scrutiny below (a wrong-scope guess pairing with a
        # coincidental citation is a real false-"met" risk), but silently
        # discarding a "not_met" the debate already reached is the worse failure
        # mode for a security review — it hides a real, well-reasoned finding.
        # This also covers the case where the Mediator's own raw verdict
        # diverges to "na" even though Hunter and Critic already reached an
        # explicit, unanimous "not_met" — the Mediator re-derives applicability
        # independently and can disagree with an already-established consensus;
        # that lone dissent shouldn't outrank the debate's own agreement.
        critic_reached_not_met = critic_result.revised_verdict == VERDICT_NOT_MET
        hunter_reached_not_met = hunter_result is None or hunter_result.verdict == VERDICT_NOT_MET
        debate_reached_definitive_not_met = raw_verdict == VERDICT_NOT_MET or (
            critic_reached_not_met and hunter_reached_not_met
        )
        # Symmetric carve-out for a genuine "met" consensus: unlike the mediator's
        # own raw verdict (which is enough on its own for the not_met carve-out
        # above, since a missed control is the worse failure mode), a "met"
        # claim carries more false-positive risk, so this requires actual
        # hunter+critic agreement, not just the mediator's own guess. Protecting
        # it here only means it isn't blunt-force downgraded by one dissenting
        # applicability flag below — it still has to clear the existing
        # evidence/citation sufficiency checks further down before it can
        # actually be accepted as "met".
        hunter_reached_met = hunter_result is not None and hunter_result.verdict == VERDICT_MET
        critic_reached_met = critic_result.revised_verdict == VERDICT_MET
        debate_reached_definitive_met = hunter_reached_met and critic_reached_met
        if (not in_scope or not specific) and explicit_absent_capability and not debate_reached_definitive_not_met:
            mediator_result.final_verdict = VERDICT_NA
            mediator_result.final_citations = []
            mediator_result.severity = None
            mediator_result.recommendation = None
            mediator_result.verdict_policy_source = "contract_not_applicable"
            mediator_result.applicability_status = APPLICABILITY_NOT_ESTABLISHED
            mediator_result.applicability_reason = (
                mediator_result.applicability_reason
                or "Contract synthesis marked the requirement out of scope or not specific enough."
            )
            mediator_result.missing_expected_evidence = []
            mediator_result.evidence_sufficiency = "not_applicable"
            mediator_result.not_assessable_reason = "contract_not_in_scope_or_not_specific"
            return mediator_result

        # Same principle for the free-text reasoning sniffer: it exists to catch
        # cases where the mediator's structured final_verdict field is missing or
        # ambiguous and its prose reveals the real intent. It must never override
        # an explicit, structured "not_met" the mediator already committed to.
        free_text_na = (
            not debate_reached_definitive_not_met
            and mediator_applicability is None
            and critic_applicability is None
            and hunter_applicability is None
            and self._reasoning_indicates_not_assessable(mediator_result)
        )
        if raw_verdict == VERDICT_NA and not applicability_not_established and applicability_established_by_agents:
            raw_verdict = VERDICT_NOT_MET
            mediator_result.raw_final_verdict = raw_verdict
        if raw_verdict == VERDICT_NA and self._should_demote_na_to_not_met(
            parameter_text=parameter_text,
            requirement_shape=requirement_shape,
            explicit_absent_capability=explicit_absent_capability,
        ):
            raw_verdict = VERDICT_NOT_MET
            mediator_result.raw_final_verdict = raw_verdict
        # A "na" conclusion that nonetheless comes with Hunter or Critic
        # citations is close to self-contradictory: "na" is meant to describe
        # "nothing relevant to find," but citing block content shows an agent
        # found and quoted *something* — typically a different-but-related
        # mechanism that then got misjudged as proving inapplicability rather
        # than an unaddressed property (the "named tech absent -> na"
        # over-triggering pattern). Prefer "not_met" over "na" in this case,
        # unless the contract gate above already established a genuine
        # structural absence (handled separately; this only runs when we're
        # still in-scope/specific per the contract).
        raw_has_citations = bool(
            (hunter_result.citations if hunter_result is not None else [])
            or (critic_result.valid_citations or [])
        )
        if raw_verdict == VERDICT_NA and raw_has_citations and in_scope and specific:
            raw_verdict = VERDICT_NOT_MET
            mediator_result.raw_final_verdict = raw_verdict
            applicability_not_established = False
            debate_reached_definitive_not_met = True
        # Same asymmetric carve-out as the contract-scope gate above: a
        # confident "not_met" the debate actually reached must survive even if
        # one agent separately flagged applicability as not established —
        # discarding it here is the same failure mode the contract-gate
        # carve-out exists to prevent.
        if explicit_absent_capability and not debate_reached_definitive_met and (
            (applicability_not_established and not debate_reached_definitive_not_met)
            or (raw_verdict == VERDICT_NA and not applicability_established_by_agents)
            or free_text_na
        ):
            mediator_result.final_verdict = VERDICT_NA
            mediator_result.final_citations = []
            mediator_result.severity = None
            mediator_result.recommendation = None
            mediator_result.verdict_policy_source = (
                "structured_not_applicable" if applicability_not_established else "not_assessable"
            )
            mediator_result.applicability_status = APPLICABILITY_NOT_ESTABLISHED
            mediator_result.missing_expected_evidence = []
            mediator_result.evidence_sufficiency = "not_assessable"
            mediator_result.not_assessable_reason = (
                "agent_marked_applicability_not_established"
                if applicability_not_established
                else "mediator_or_critic_indicated_na_or_not_assessable"
            )
            return mediator_result

        if self._should_accept_verified_met_consensus(
            parameter_text=parameter_text,
            raw_verdict=raw_verdict,
            hunter_result=hunter_result,
            critic_result=critic_result,
            valid_citations=valid_citations,
        ):
            sufficiency = self._assess_verified_control_sufficiency(
                parameter_text=parameter_text,
                raw_verdict=raw_verdict,
                critic_outcome=critic_result.outcome,
                hunter_result=hunter_result,
                critic_result=critic_result,
                valid_citations=valid_citations,
                applicability_established=not applicability_not_established,
                explicit_absent_capability=explicit_absent_capability,
            )
            if sufficiency["force_na"]:
                mediator_result.final_verdict = VERDICT_NA
                mediator_result.final_citations = []
                mediator_result.severity = None
                mediator_result.recommendation = None
                mediator_result.verdict_policy_source = str(sufficiency["policy_reason"])
                mediator_result.applicability_status = APPLICABILITY_NOT_ESTABLISHED
                mediator_result.missing_expected_evidence = []
                mediator_result.evidence_sufficiency = "not_assessable"
                return mediator_result
            if sufficiency["allow_met"]:
                mediator_result.final_verdict = VERDICT_MET
                mediator_result.final_citations = valid_citations
                mediator_result.severity = None
                mediator_result.recommendation = None
                mediator_result.verdict_policy_source = str(sufficiency["policy_reason"])
                mediator_result.applicability_status = APPLICABILITY_ESTABLISHED
                mediator_result.missing_expected_evidence = []
                mediator_result.evidence_sufficiency = "verified_met"
                return mediator_result

        if (
            hunter_result is not None
            and hunter_result.verdict == VERDICT_MET
            and critic_result.revised_verdict == VERDICT_MET
            and valid_citations
        ):
            sufficiency = self._assess_verified_control_sufficiency(
                parameter_text=parameter_text,
                raw_verdict=raw_verdict,
                critic_outcome=critic_result.outcome,
                hunter_result=hunter_result,
                critic_result=critic_result,
                valid_citations=valid_citations,
                applicability_established=not applicability_not_established,
                explicit_absent_capability=explicit_absent_capability,
            )
            if sufficiency["allow_met"]:
                mediator_result.final_verdict = VERDICT_MET
                mediator_result.final_citations = valid_citations
                mediator_result.severity = None
                mediator_result.recommendation = None
                mediator_result.verdict_policy_source = str(sufficiency["policy_reason"])
                mediator_result.applicability_status = APPLICABILITY_ESTABLISHED
                mediator_result.missing_expected_evidence = []
                mediator_result.evidence_sufficiency = "verified_met"
                return mediator_result

        accepted_met = (
            raw_verdict == VERDICT_MET
            and valid_citations
            and critic_result.revised_verdict == VERDICT_MET
            and (hunter_result is None or hunter_result.verdict == VERDICT_MET)
        )
        accepted_met_sufficiency = None
        if accepted_met:
            accepted_met_sufficiency = self._assess_verified_control_sufficiency(
                parameter_text=parameter_text,
                raw_verdict=raw_verdict,
                critic_outcome=critic_result.outcome,
                hunter_result=hunter_result,
                critic_result=critic_result,
                valid_citations=valid_citations,
                applicability_established=not applicability_not_established,
                explicit_absent_capability=explicit_absent_capability,
            )
            if accepted_met_sufficiency["allow_met"]:
                mediator_result.final_verdict = VERDICT_MET
                mediator_result.final_citations = valid_citations
                mediator_result.severity = None
                mediator_result.recommendation = None
                mediator_result.verdict_policy_source = str(accepted_met_sufficiency["policy_reason"])
                mediator_result.applicability_status = APPLICABILITY_ESTABLISHED
                mediator_result.evidence_sufficiency = "verified_met"
                mediator_result.missing_expected_evidence = []
                return mediator_result

        if (
            raw_verdict == VERDICT_NOT_MET
            and critic_result.outcome == OUTCOME_UPHOLD
            and critic_result.revised_verdict == VERDICT_NOT_MET
            and self._should_promote_not_met_with_verified_core_evidence(
                parameter_text=parameter_text,
                critic_result=critic_result,
                valid_citations=valid_citations,
            )
        ):
            sufficiency = self._assess_verified_control_sufficiency(
                parameter_text=parameter_text,
                raw_verdict=raw_verdict,
                critic_outcome=critic_result.outcome,
                hunter_result=hunter_result,
                critic_result=critic_result,
                valid_citations=valid_citations,
                applicability_established=not applicability_not_established,
                explicit_absent_capability=explicit_absent_capability,
            )
            if sufficiency["allow_met"]:
                mediator_result.final_verdict = VERDICT_MET
                mediator_result.final_citations = valid_citations
                mediator_result.severity = None
                mediator_result.recommendation = None
                mediator_result.verdict_policy_source = str(sufficiency["policy_reason"])
                mediator_result.applicability_status = APPLICABILITY_ESTABLISHED
                mediator_result.missing_expected_evidence = []
                mediator_result.evidence_sufficiency = "verified_met"
                return mediator_result

        if raw_verdict == VERDICT_MET:
            hunter_said_not_met = hunter_result is not None and hunter_result.verdict == VERDICT_NOT_MET
            if not hunter_said_not_met:
                mediator_result.final_verdict = VERDICT_NOT_MET
                mediator_result.final_citations = []
                # Preserve the real rejection reason from the sufficiency check
                # that was already run above (empty citations vs. failed
                # core-claim vs. essential gap) instead of collapsing every
                # rejection into one generic label — makes ablation/debugging
                # able to tell these apart. Only fall back to the generic label
                # when no sufficiency check actually ran (e.g. accepted_met was
                # never true because valid_citations was empty).
                mediator_result.verdict_policy_source = (
                    str(accepted_met_sufficiency["policy_reason"])
                    if accepted_met_sufficiency is not None
                    else "met_without_verified_evidence"
                )
                mediator_result.applicability_status = APPLICABILITY_ESTABLISHED
                mediator_result.missing_expected_evidence = list(
                    getattr(critic_result, "missing_expected_evidence", []) or []
                )
                mediator_result.evidence_sufficiency = "insufficient_for_met"
                mediator_result.not_assessable_reason = None
                return mediator_result
            # Hunter=not_met + Mediator=met but accepted_met=False → fall through to not_met

        if raw_verdict == "partial":
            sufficiency = self._assess_verified_control_sufficiency(
                parameter_text=parameter_text,
                raw_verdict=raw_verdict,
                critic_outcome=critic_result.outcome,
                hunter_result=hunter_result,
                critic_result=critic_result,
                valid_citations=valid_citations,
                applicability_established=not applicability_not_established,
                explicit_absent_capability=explicit_absent_capability,
            )
            if sufficiency["force_na"]:
                mediator_result.final_verdict = VERDICT_NA
                mediator_result.final_citations = []
                mediator_result.severity = None
                mediator_result.recommendation = None
                mediator_result.verdict_policy_source = str(sufficiency["policy_reason"])
                mediator_result.applicability_status = APPLICABILITY_NOT_ESTABLISHED
                mediator_result.missing_expected_evidence = []
                mediator_result.evidence_sufficiency = "not_assessable"
            elif sufficiency["allow_met"]:
                mediator_result.final_verdict = VERDICT_MET
                mediator_result.final_citations = valid_citations
                mediator_result.severity = None
                mediator_result.recommendation = None
                mediator_result.verdict_policy_source = str(sufficiency["policy_reason"])
                mediator_result.applicability_status = APPLICABILITY_ESTABLISHED
                mediator_result.evidence_sufficiency = "partial_met"
            else:
                mediator_result.final_verdict = VERDICT_NOT_MET
                if "partial" not in (mediator_result.reasoning or "").lower():
                    mediator_result.reasoning = f"Partial evidence only: {mediator_result.reasoning}"
                    mediator_result.logic_summary = mediator_result.reasoning
                mediator_result.verdict_policy_source = str(sufficiency["policy_reason"])
                mediator_result.applicability_status = APPLICABILITY_ESTABLISHED
                mediator_result.evidence_sufficiency = "partial"
                mediator_result.final_citations = []
            return mediator_result

        fallback_sufficiency = self._assess_verified_control_sufficiency(
            parameter_text=parameter_text,
            raw_verdict=raw_verdict,
            critic_outcome=critic_result.outcome,
            hunter_result=hunter_result,
            critic_result=critic_result,
            valid_citations=valid_citations,
            applicability_established=not applicability_not_established,
            explicit_absent_capability=explicit_absent_capability,
        )
        if (
            hunter_result is not None
            and hunter_result.verdict == VERDICT_MET
            and valid_citations
            and not fallback_sufficiency["essential_gap_present"]
            and fallback_sufficiency["allow_met"]
        ):
            mediator_result.final_verdict = VERDICT_MET
            mediator_result.final_citations = valid_citations
            mediator_result.severity = None
            mediator_result.recommendation = None
            mediator_result.verdict_policy_source = str(fallback_sufficiency["policy_reason"])
            mediator_result.applicability_status = APPLICABILITY_ESTABLISHED
            mediator_result.missing_expected_evidence = []
            mediator_result.evidence_sufficiency = "verified_met"
            return mediator_result

        mediator_result.final_verdict = VERDICT_NOT_MET
        mediator_result.applicability_status = APPLICABILITY_ESTABLISHED
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
            mediator_result.applicability_status = APPLICABILITY_NOT_ESTABLISHED
            mediator_result.missing_expected_evidence = []
            mediator_result.evidence_sufficiency = "no_grounded_evidence"
            mediator_result.not_assessable_reason = "not_met_without_any_critic_verified_citations"

        return mediator_result

    def _has_explicit_absent_capability_signal(self, result: Optional[object]) -> bool:
        if self._applicability_is_established(result) is not False:
            return False
        text = " ".join(
            [
                str(getattr(result, "applicability_reason", "") or ""),
                str(getattr(result, "reasoning", "") or ""),
                str(getattr(result, "logic_summary", "") or ""),
            ]
        ).lower()
        if not text:
            return False
        markers = [
            "clearly does not have",
            "does not have",
            "does not use",
            "no mobile client",
            "no mobile app",
            "server-only",
            "not part of the design",
            "not present in the design",
            "absent from the design",
            "capability is absent",
            "prerequisite is absent",
            "technology is absent",
        ]
        return any(marker in text for marker in markers)

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

    def _classify_requirement_shape(self, parameter_text: str) -> str:
        text = (parameter_text or "").lower()
        if any(marker in text for marker in _NEGATIVE_ABSENCE_MARKERS):
            return "negative_absence"
        if (
            " or other " in text
            or " such as " in text
            or "or equivalent" in text
            or "or comparable" in text
        ):
            return "technology_specific_but_family_applicable"
        if any(marker in text for marker in ("health data", "financial data", "personal data", "sensitive data")):
            return "data_family_requirement"
        if any(marker in text for marker in ("document", "define", "documentation", "justify", "classif")):
            return "documentation_completeness"
        return "positive_presence"

    def _classify_requirement_family(self, parameter_text: str) -> str:
        text = (parameter_text or "").lower()
        if (
            ("high-level architecture" in text or "high level architecture" in text)
            and ("connected remote services" in text or "remote services" in text)
        ):
            return "high_level_architecture_remote_services"
        if "out of band" in text and "independent channel" in text:
            return "oob_secure_independent_channel"
        if any(
            marker in text
            for marker in (
                "weak authenticators",
                "sms and email",
                "secondary verification",
                "transaction approval",
            )
        ):
            return "weak_authenticator_restriction"
        if any(
            marker in text
            for marker in (
                "no weaker",
                "weaker authentication",
                "authentication pathways",
                "identity management apis",
                "consistent authentication security control strength",
            )
        ):
            return "uniform_auth_strength"
        if "communications between application components" in text or (
            "least necessary privileges" in text and "component" in text
        ):
            return "component_auth_least_privilege"
        if any(
            marker in text
            for marker in (
                "segregation of duties",
                "step up",
                "step-up",
                "adaptive authentication",
            )
        ):
            return "alternative_control_satisfaction"
        if "input and output requirements" in text or (
            "handle and process data" in text and any(marker in text for marker in ("laws", "regulations", "policy compliance"))
        ):
            return "input_output_compliance"
        if (
            ("protection level" in text or "protection levels" in text)
            and "protection requirements" in text
        ):
            return "protection_level_mapping"
        if "sensitive data" in text and any(marker in text for marker in ("audit", "audited", "logging")):
            return "sensitive_data_access_audit"
        if any(marker in text for marker in ("trust boundaries", "significant data flows", "definition and documentation of all application components")):
            return "documentation_completeness"
        if "abnormal numbers of requests" in text or "detect and alert" in text:
            return "abnormal_request_alerting"
        if any(
            marker in text
            for marker in (
                "data retention classification",
                "retention classification",
                "deleted automatically",
                "out of date data is deleted",
            )
        ):
            return "retention_lifecycle"
        return "generic_positive_presence"

    def _citations_joined_text(self, citations: List[Any]) -> str:
        return " ".join(str(getattr(citation, "quoted_text", "") or "").lower() for citation in citations)

    def _has_uniform_strong_auth_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        has_scope = any(marker in text for marker in ("all system roles", "every role", "all roles", "every endpoint"))
        has_auth_strength = any(
            marker in text
            for marker in (
                "mandatory multi factor authentication",
                "mandatory mfa",
                "multi factor authentication",
                "fully authenticated",
            )
        )
        return has_scope and has_auth_strength

    def _has_weak_authenticator_restriction_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        mentions_weak_or_restricted_factor = any(
            marker in text
            for marker in (
                "sms",
                "email",
                "one time password",
                "otp",
                "secondary verification",
                "transaction approval",
                "weak authenticators",
            )
        )
        has_explicit_restriction = any(
            marker in text
            for marker in (
                "limited to",
                "only",
                "not as a replacement",
                "not used as",
                "must not",
                "secondary verification",
                "transaction approval",
            )
        )
        return mentions_weak_or_restricted_factor and has_explicit_restriction

    def _has_explicit_absence_evidence(self, parameter_text: str, citations: List[Any]) -> bool:
        requirement_text = (parameter_text or "").lower()
        citation_text = self._citations_joined_text(citations)
        if not citation_text:
            return False

        if any(term in requirement_text for term in ("password hints", "knowledge-based", "secret question")):
            mentions_target = any(
                term in citation_text
                for term in (
                    "password hint",
                    "password hints",
                    "knowledge-based",
                    "secret question",
                    "secret questions",
                )
            )
            has_prohibition = any(marker in citation_text for marker in ("no ", "not ", "without", "must not", "does not"))
            return mentions_target and has_prohibition

        if any(term in requirement_text for term in ("unsupported", "deprecated")):
            mentions_target = any(
                term in citation_text for term in ("unsupported", "deprecated", "obsolete", "end-of-life", "must not")
            )
            has_prohibition = any(marker in citation_text for marker in ("must not", "does not", "only", "exclusively", "solely"))
            if mentions_target or has_prohibition:
                return True
            # A citation that names a concrete, specific technology inventory for
            # the relevant tier (e.g. explicitly-named frameworks/libraries making
            # up "the presentation tier"/"the client-side stack") is itself
            # closed-world evidence for this requirement class — a TSD that
            # documents exactly which frontend libraries are used doesn't leave
            # room for an undocumented Flash/ActiveX component, even without an
            # explicit "we don't use X" line.
            return self._has_named_technology_inventory_evidence(citation_text)

        if any(term in requirement_text for term in ("weak authenticators", "sms and email", "secondary verification")):
            return self._has_weak_authenticator_restriction_evidence(citations)

        if any(term in requirement_text for term in ("weaker authenticator", "weaker authentication", "no weaker")):
            return self._has_uniform_strong_auth_evidence(citations)

        return any(marker in citation_text for marker in _EXPLICIT_ABSENCE_EVIDENCE_MARKERS)

    def _has_named_technology_inventory_evidence(self, citation_text: str) -> bool:
        tier_markers = (
            "presentation tier",
            "client-side stack",
            "client side stack",
            "front end",
            "frontend",
            "technology stack",
            "client-side technolog",
            "client side technolog",
        )
        if not any(marker in citation_text for marker in tier_markers):
            return False
        named_tech_markers = (
            "jquery",
            "jsp",
            "java server pages",
            "bootstrap",
            "react",
            "angular",
            "vue",
            "javascript",
            "typescript",
            "css",
            "html",
            "ajax",
            "dom",
        )
        hits = sum(1 for marker in named_tech_markers if marker in citation_text)
        return hits >= 2

    def _requirement_keywords(self, parameter_text: str) -> List[str]:
        words = []
        for raw in (parameter_text or "").lower().replace("/", " ").replace("-", " ").split():
            token = "".join(ch for ch in raw if ch.isalnum())
            if len(token) < 4 or token in _REQUIREMENT_STOPWORDS:
                continue
            words.append(token)
        return words

    def _has_documentation_core_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "architecture",
                "component",
                "data flow",
                "api endpoint",
                "layer",
                "role based access control",
                "security framework",
                "trust boundary",
                "service",
            )
        )

    def _has_high_level_architecture_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "high-level architecture",
                "system architecture",
                "architecture diagram",
                "application layer",
                "service layer",
                "trust boundary",
            )
        )

    def _has_remote_service_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "external",
                "third party",
                "remote service",
                "gateway",
                "dispatch",
                "api",
                "integration",
            )
        )

    def _has_secure_channel_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "tls 1.2",
                "tls 1.3",
                "mtls",
                "mutual tls",
                "https",
                "encrypted channel",
                "cryptographically signed",
            )
        )

    def _has_independent_channel_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "out of band",
                "sms gateway",
                "sms",
                "email",
                "separate channel",
                "independent channel",
                "separate provider",
            )
        )

    def _has_at_rest_encryption_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "encrypted at rest",
                "stored encrypted",
                "encryption at rest",
                "data at rest",
                "database encryption",
            )
        )

    def _has_financial_data_scope_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "financial",
                "credit history",
                "tax record",
                "beneficiar",
                "pay history",
                "bank account",
                "transaction",
                "market or research",
                "credit",
            )
        )

    def _has_rp_csp_reauth_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "re-authenticat",
                "reauthenticat",
                "relying party",
                "credential service provider",
            )
        )

    def _has_detection_and_alerting_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        has_detection = any(marker in text for marker in ("detect", "detection", "threshold", "maximum number", "rate_limit", "per minute"))
        has_alert = any(
            marker in text
            for marker in (
                "alert",
                "alerts",
                "alarm",
                "notification",
                "administrative review",
                "locks the account",
                "fail closed",
            )
        )
        return has_detection and has_alert

    def _has_centralized_control_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "common library",
                "centralized",
                "shared service",
                "shared component",
                "gateway",
                "vault",
                "role based access control",
            )
        )

    def _has_step_up_or_adaptive_auth_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "step up authentication",
                "step-up authentication",
                "adaptive authentication",
                "risk based authentication",
                "risk-based authentication",
            )
        )

    def _has_segregation_of_duties_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        return any(
            marker in text
            for marker in (
                "segregation of duties",
                "separation of duties",
                "maker checker",
                "four eyes",
                "role based access control",
                "privileged actions require a different role",
                "admin actions are separated",
            )
        )

    def _has_component_auth_and_least_privilege_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        has_component_auth = any(
            marker in text
            for marker in (
                "mutual tls",
                "mtls",
                "fully authenticated",
                "cryptographically signed",
                "session tokens",
                "oauth 2.0",
                "audience validation",
            )
        )
        has_least_privilege = any(
            marker in text
            for marker in (
                "least privilege",
                "role based access control",
                "attribute based access control",
                "rbac",
                "abac",
                "privileges",
                "permission",
            )
        )
        return has_component_auth and has_least_privilege

    def _has_protection_level_mapping_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        mentions_level = any(
            marker in text
            for marker in (
                "protection level",
                "security level",
                "data classification",
                "classification tier",
                "zone",
                "trust zone",
            )
        )
        mentions_mapping = any(
            marker in text
            for marker in (
                "mapped to",
                "associated with",
                "for each level",
                "per level",
                "each classification requires",
                "control set",
                "requirement set",
            )
        )
        return mentions_level and mentions_mapping

    def _has_input_output_compliance_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        has_input_handling = any(
            marker in text
            for marker in (
                "allow listing",
                "allow-list",
                "input validation",
                "sanitizes the input",
                "data sanitization",
                "incoming data payloads",
            )
        )
        has_output_handling = any(
            marker in text
            for marker in (
                "redaction proxies",
                "scrub",
                "scrubs",
                "personally identifiable information",
                "pii",
            )
        )
        has_policy_or_law = any(
            marker in text
            for marker in (
                "general data protection regulation",
                "gdpr",
                "regulatory compliance",
                "data privacy laws",
                "policy compliance",
            )
        )
        return has_input_handling and has_output_handling and has_policy_or_law

    def _has_retention_lifecycle_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        has_retention = any(
            marker in text
            for marker in (
                "retention period",
                "lifecycle management",
                "data categorized",
                "active lifetime",
                "archival states",
            )
        )
        has_deletion = any(
            marker in text
            for marker in (
                "deleted automatically",
                "permanently destroys",
                "cryptographic erasure",
                "digital shredding",
                "sweeps the core database",
            )
        )
        return has_retention and has_deletion

    def _has_sensitive_data_access_audit_evidence(self, citations: List[Any]) -> bool:
        text = self._citations_joined_text(citations)
        has_access_audit = any(
            marker in text
            for marker in (
                "access is logged",
                "every attempt to access",
                "access to a restricted resource is logged",
                "audit log",
                "audited",
                "generates an alert",
            )
        )
        avoids_sensitive_logging = any(
            marker in text
            for marker in (
                "redact",
                "redaction",
                "scrub",
                "scrubs",
                "without logging the sensitive data",
                "does not log sensitive data",
                "pii masking",
            )
        )
        return has_access_audit and avoids_sensitive_logging

    def _verified_core_claim_satisfied(self, parameter_text: str, citations: List[Any]) -> bool:
        requirement_shape = self._classify_requirement_shape(parameter_text)
        requirement_family = self._classify_requirement_family(parameter_text)

        if requirement_family == "high_level_architecture_remote_services":
            return self._has_high_level_architecture_evidence(citations) and self._has_remote_service_evidence(citations)
        if requirement_family == "oob_secure_independent_channel":
            return self._has_independent_channel_evidence(citations) and self._has_secure_channel_evidence(citations)
        if requirement_family == "weak_authenticator_restriction":
            return self._has_weak_authenticator_restriction_evidence(citations)
        if requirement_family == "uniform_auth_strength":
            return self._has_uniform_strong_auth_evidence(citations)
        if requirement_family == "component_auth_least_privilege":
            return self._has_component_auth_and_least_privilege_evidence(citations)
        if requirement_family == "alternative_control_satisfaction":
            return self._has_step_up_or_adaptive_auth_evidence(citations) or self._has_segregation_of_duties_evidence(citations)
        if requirement_family == "input_output_compliance":
            return self._has_input_output_compliance_evidence(citations)
        if requirement_family == "protection_level_mapping":
            return self._has_protection_level_mapping_evidence(citations)
        if requirement_family == "sensitive_data_access_audit":
            return self._has_sensitive_data_access_audit_evidence(citations)
        if requirement_family == "documentation_completeness":
            return self._has_documentation_core_evidence(citations)
        if requirement_family == "abnormal_request_alerting":
            return self._has_detection_and_alerting_evidence(citations)
        if requirement_family == "retention_lifecycle":
            return self._has_retention_lifecycle_evidence(citations)
        if requirement_shape == "negative_absence":
            return self._has_explicit_absence_evidence(parameter_text, citations)
        return self._has_core_mechanism_match(parameter_text, citations)

    def _has_core_mechanism_match(self, parameter_text: str, citations: List[Any]) -> bool:
        requirement_text = (parameter_text or "").lower()
        citation_text = self._citations_joined_text(citations)
        keyword_hits = sum(1 for token in self._requirement_keywords(parameter_text) if token in citation_text)
        mechanism_hits = 0

        if any(term in requirement_text for term in ("detect and alert", "alert on abnormal", "abnormal numbers of requests")):
            return self._has_detection_and_alerting_evidence(citations)
        if any(term in requirement_text for term in ("encrypted while at rest", "encrypted at rest", "stored encrypted")):
            if any(term in requirement_text for term in ("financial", "regulated financial data")):
                # Generic "data at rest is encrypted" evidence isn't enough here —
                # the requirement names specific regulated categories (financial
                # accounts, credit history, tax records, beneficiaries, etc.), so
                # the cited evidence must actually cover one of those, not just
                # unrelated PII fields that happen to also be encrypted.
                return self._has_at_rest_encryption_evidence(citations) and self._has_financial_data_scope_evidence(
                    citations
                )
            return self._has_at_rest_encryption_evidence(citations)
        if any(term in requirement_text for term in ("common library", "centralized", "shared service", "intra-service secret", "unchanging")):
            return self._has_centralized_control_evidence(citations)
        if any(term in requirement_text for term in ("connected remote services", "remote service")):
            return self._has_documentation_core_evidence(citations) and self._has_remote_service_evidence(citations)
        if "relying part" in requirement_text and "credential service provider" in requirement_text:
            # A generic idle-session timeout that merely destroys a token isn't
            # the same claim as "the RP specifies a maximum authentication time
            # to the CSP and the CSP re-authenticates" — that's a specific
            # RP/CSP protocol relationship, not just any session expiry
            # mechanism. Require the evidence to actually name that
            # relationship or an explicit re-authentication trigger.
            return self._has_rp_csp_reauth_evidence(citations)

        if any(term in requirement_text for term in ("communicat", "transit", "payload", "header")):
            mechanism_hits += int(any(term in citation_text for term in ("tls", "https", "cryptographically signed", "authenticated")))
        if any(term in requirement_text for term in ("auth", "credential", "session", "password", "login")):
            mechanism_hits += int(any(term in citation_text for term in ("multi factor authentication", "authenticated", "session token", "email verification")))
        if any(term in requirement_text for term in ("business logic", "limit", "rate", "threshold", "abnormal")):
            mechanism_hits += int(any(term in citation_text for term in ("maximum number", "per minute", "rate_limit", "threshold", "prevent", "alert", "detect")))
        if any(term in requirement_text for term in ("encrypt", "crypto", "signed")):
            mechanism_hits += int(any(term in citation_text for term in ("tls", "cryptographic", "signed", "encrypted")))

        # Two independent signals (a requirement keyword present, and a matching
        # technical-mechanism term present) are as strong a sign as two of the same
        # kind — without this, wording that splits across categories (e.g. the
        # requirement's "authenticated" keyword vs. a citation reading "authentication
        # tokens", a different word form) fails the single-category >=2 bar even
        # though both a domain keyword and a concrete mechanism term are genuinely
        # present in the citation.
        return keyword_hits >= 2 or mechanism_hits >= 2 or (keyword_hits >= 1 and mechanism_hits >= 1)

    def _has_essential_gap(
        self,
        *,
        parameter_text: str,
        critic_result: CriticResult,
        valid_citations: List[Any],
    ) -> bool:
        requirement_text = (parameter_text or "").lower()
        requirement_family = self._classify_requirement_family(parameter_text)
        objections = " ".join(str(item or "") for item in list(getattr(critic_result, "objections", []) or [])).lower()
        weak_evidence = " ".join(str(item or "") for item in list(getattr(critic_result, "weak_evidence", []) or [])).lower()
        missing_expected = " ".join(str(item or "") for item in list(getattr(critic_result, "missing_expected_evidence", []) or [])).lower()
        combined = " ".join(filter(None, [objections, weak_evidence, missing_expected]))
        if not combined:
            return False

        essential_markers = (
            "missing mechanism",
            "does not show",
            "not explicit",
            "not documented",
            "not stated",
            "not named",
            "unrelated",
            "insufficient",
            "no evidence",
            "missing alert",
            "missing encryption",
        )
        if any(marker in combined for marker in essential_markers):
            consensus_met = (
                getattr(critic_result, "outcome", None) == OUTCOME_UPHOLD
                and getattr(critic_result, "revised_verdict", None) == VERDICT_MET
            )
            softened_markers = {"not explicit", "not documented", "not stated", "not named"}
            if not (
                consensus_met
                and requirement_family in {
                    "high_level_architecture_remote_services",
                    "oob_secure_independent_channel",
                    "uniform_auth_strength",
                    "component_auth_least_privilege",
                    "alternative_control_satisfaction",
                    "input_output_compliance",
                    "sensitive_data_access_audit",
                    "documentation_completeness",
                    "abnormal_request_alerting",
                    "retention_lifecycle",
                }
                and any(marker in combined for marker in softened_markers)
                and not any(
                    marker in combined
                    for marker in (
                        "missing mechanism",
                        "does not show",
                        "unrelated",
                        "insufficient",
                        "no evidence",
                        "missing alert",
                        "missing encryption",
                    )
                )
            ):
                return True

        if requirement_family == "weak_authenticator_restriction":
            return not self._has_weak_authenticator_restriction_evidence(valid_citations)
        if requirement_family == "high_level_architecture_remote_services":
            return not (
                self._has_high_level_architecture_evidence(valid_citations)
                and self._has_remote_service_evidence(valid_citations)
            )
        if requirement_family == "oob_secure_independent_channel":
            return not (
                self._has_independent_channel_evidence(valid_citations)
                and self._has_secure_channel_evidence(valid_citations)
            )
        if requirement_family == "uniform_auth_strength":
            return not self._has_uniform_strong_auth_evidence(valid_citations)
        if requirement_family == "component_auth_least_privilege":
            return not self._has_component_auth_and_least_privilege_evidence(valid_citations)
        if requirement_family == "alternative_control_satisfaction":
            return not (
                self._has_step_up_or_adaptive_auth_evidence(valid_citations)
                or self._has_segregation_of_duties_evidence(valid_citations)
            )
        if requirement_family == "input_output_compliance":
            return not self._has_input_output_compliance_evidence(valid_citations)
        if requirement_family == "protection_level_mapping":
            return not self._has_protection_level_mapping_evidence(valid_citations)
        if requirement_family == "sensitive_data_access_audit":
            return not self._has_sensitive_data_access_audit_evidence(valid_citations)
        if requirement_family == "documentation_completeness":
            return not self._has_documentation_core_evidence(valid_citations)
        if requirement_family == "abnormal_request_alerting":
            return not self._has_detection_and_alerting_evidence(valid_citations)
        if requirement_family == "retention_lifecycle":
            return not self._has_retention_lifecycle_evidence(valid_citations)

        if any(term in requirement_text for term in ("detect and alert", "abnormal numbers of requests")):
            return not self._has_detection_and_alerting_evidence(valid_citations)
        if any(term in requirement_text for term in ("encrypted while at rest", "encrypted at rest", "stored encrypted")):
            return not self._has_at_rest_encryption_evidence(valid_citations)
        if any(term in requirement_text for term in ("connected remote services", "remote service")):
            return not self._has_remote_service_evidence(valid_citations)
        return False

    def _assess_verified_control_sufficiency(
        self,
        *,
        parameter_text: str,
        raw_verdict: str,
        critic_outcome: str,
        hunter_result: Optional[HunterResult],
        critic_result: CriticResult,
        valid_citations: List[Any],
        applicability_established: bool,
        explicit_absent_capability: bool,
    ) -> Dict[str, Any]:
        if not applicability_established or explicit_absent_capability:
            return {
                "allow_met": False,
                "force_not_met": False,
                "force_na": True,
                "policy_reason": "applicability_not_established_for_met",
                "core_claim_satisfied": False,
                "essential_gap_present": True,
            }
        if not valid_citations:
            return {
                "allow_met": False,
                "force_not_met": True,
                "force_na": False,
                "policy_reason": "verified_met_insufficient_evidence",
                "core_claim_satisfied": False,
                "essential_gap_present": True,
            }

        core_claim_satisfied = self._verified_core_claim_satisfied(parameter_text, valid_citations)

        # The *named-family* core-claim checks (uniform_auth_strength,
        # oob_secure_independent_channel, high_level_architecture_remote_services,
        # etc.) are literal keyword/phrase matchers over narrow citation excerpts,
        # and they miss legitimate paraphrases (e.g. "Short Message Service"
        # instead of "sms", "Architecture Type: ..." instead of "system
        # architecture", "all system access" instead of "all system roles"). When
        # Hunter and Critic reach a fully clean, unanimous "met" — the Critic
        # raised zero objections, zero weak_evidence, and zero
        # missing_expected_evidence — that already is the verification;
        # re-deriving "was this really satisfied" via brittle keyword matching on
        # top of an unchallenged two-agent consensus only produces false
        # rejections. This bypass is deliberately scoped to the named families
        # only: the default/generic family's own core-mechanism check
        # (_has_core_mechanism_match) is a separate, less literal heuristic that
        # has its own real false-positive risk (e.g. accepting evidence that talks
        # about the topic but doesn't cover the requirement's specific scope) —
        # bypassing it here as well would undo that check's precision for exactly
        # the same "clean uphold" cases where the two agents were simply wrong.
        requirement_family = self._classify_requirement_family(parameter_text)
        critic_clean_uphold = (
            not core_claim_satisfied
            and requirement_family != "generic_positive_presence"
            and hunter_result is not None
            and hunter_result.verdict == VERDICT_MET
            and critic_outcome == OUTCOME_UPHOLD
            and critic_result.revised_verdict == VERDICT_MET
            and not list(getattr(critic_result, "objections", []) or [])
            and not list(getattr(critic_result, "weak_evidence", []) or [])
            and not list(getattr(critic_result, "missing_expected_evidence", []) or [])
        )
        if critic_clean_uphold:
            core_claim_satisfied = True

        essential_gap_present = self._has_essential_gap(
            parameter_text=parameter_text,
            critic_result=critic_result,
            valid_citations=valid_citations,
        )
        if critic_outcome == OUTCOME_PARTIAL and list(getattr(critic_result, "missing_expected_evidence", []) or []):
            essential_gap_present = essential_gap_present or True
        if (
            hunter_result is not None
            and hunter_result.verdict == VERDICT_MET
            and critic_outcome == OUTCOME_UPHOLD
            and critic_result.revised_verdict == VERDICT_MET
            and (
                list(getattr(critic_result, "objections", []) or [])
                or list(getattr(critic_result, "missing_expected_evidence", []) or [])
            )
            and not core_claim_satisfied
        ):
            essential_gap_present = True
        peripheral_only = bool(valid_citations) and core_claim_satisfied and not essential_gap_present

        if core_claim_satisfied and peripheral_only:
            policy_reason = "verified_met_core_sufficient"
            if critic_outcome == OUTCOME_PARTIAL:
                policy_reason = "partial_core_sufficient"
            if raw_verdict == VERDICT_NOT_MET:
                policy_reason = "verified_core_evidence_rescue"
            return {
                "allow_met": True,
                "force_not_met": False,
                "force_na": False,
                "policy_reason": policy_reason,
                "core_claim_satisfied": True,
                "essential_gap_present": False,
            }

        return {
            "allow_met": False,
            "force_not_met": True,
            "force_na": False,
            "policy_reason": "partial_essential_gap" if critic_outcome == OUTCOME_PARTIAL else "verified_met_adjacent_only",
            "core_claim_satisfied": core_claim_satisfied,
            "essential_gap_present": True,
        }

    def _should_promote_not_met_with_verified_core_evidence(
        self,
        *,
        parameter_text: str,
        critic_result: CriticResult,
        valid_citations: List[Any],
    ) -> bool:
        if not valid_citations:
            return False
        if list(getattr(critic_result, "objections", []) or []):
            return False
        if list(getattr(critic_result, "missing_expected_evidence", []) or []):
            return False
        requirement_shape = self._classify_requirement_shape(parameter_text)
        if requirement_shape == "negative_absence":
            return False
        if requirement_shape == "documentation_completeness":
            return self._has_documentation_core_evidence(valid_citations)
        return self._has_core_mechanism_match(parameter_text, valid_citations)

    def _should_demote_na_to_not_met(
        self,
        *,
        parameter_text: str,
        requirement_shape: str,
        explicit_absent_capability: bool,
    ) -> bool:
        if explicit_absent_capability:
            return False
        text = (parameter_text or "").lower()
        if requirement_shape == "technology_specific_but_family_applicable":
            return True
        if requirement_shape == "data_family_requirement" and any(
            marker in text for marker in ("stored encrypted", "encrypted at rest", "protected", "retention", "classification")
        ):
            return True
        return False

    def _should_force_evidence_rescue_round(
        self,
        *,
        parameter_text: str,
        hunter_result: HunterResult,
        critic_result: CriticResult,
        round_number: int,
    ) -> bool:
        if round_number >= self.max_debate_rounds:
            return False
        if hunter_result.verdict != VERDICT_NOT_MET:
            return False
        if critic_result.outcome != OUTCOME_UPHOLD or critic_result.revised_verdict != VERDICT_NOT_MET:
            return False
        if not list(critic_result.valid_citations or []):
            return False
        requirement_shape = self._classify_requirement_shape(parameter_text)
        return requirement_shape in {"positive_presence", "documentation_completeness"}

    def _should_accept_verified_met_consensus(
        self,
        *,
        parameter_text: str,
        raw_verdict: str,
        hunter_result: Optional[HunterResult],
        critic_result: CriticResult,
        valid_citations: List[Any],
    ) -> bool:
        if hunter_result is None:
            return False
        if hunter_result.verdict != VERDICT_MET:
            return False
        if critic_result.outcome != OUTCOME_UPHOLD or critic_result.revised_verdict != VERDICT_MET:
            return False
        if not valid_citations:
            return False
        requirement_shape = self._classify_requirement_shape(parameter_text)
        if requirement_shape == "negative_absence":
            return self._has_explicit_absence_evidence(parameter_text, valid_citations)
        return raw_verdict in {VERDICT_MET, VERDICT_NOT_MET}

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
