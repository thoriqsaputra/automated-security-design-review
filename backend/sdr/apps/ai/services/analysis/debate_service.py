"""
Debate Service — Orchestrates Hunter → Critic → Mediator debate.
Responsibility: Pure reasoning workflow. NO database writes.
Output is structured, ready for persistence service to write.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any, Dict, List, Optional

from sdr.core.config import settings

from sdr.apps.ai.agents.base import (
    OUTCOME_OVERTURN,
    OUTCOME_PARTIAL,
    VERDICT_MET,
    VERDICT_NA,
    VERDICT_NOT_MET,
    CriticResult,
    HunterResult,
)
from sdr.apps.ai.telemetry import capture_ai_usage_context, run_with_ai_usage_context
from sdr.apps.ai.agents.critic import CriticAgent
from sdr.apps.ai.agents.hunter import HunterAgent
from sdr.apps.ai.agents.mediator import MediatorAgent
from sdr.apps.ai.agents.vision import VisionAgent
from sdr.apps.ai.client import get_model_for_component
from sdr.apps.ai.retrieval.router import RetrievalResult
from sdr.apps.ai.tsd_processing.ingestor import TSDDocument

from .dto import DebateInput, DebateOutput

logger = logging.getLogger(__name__)

_MAX_AGENT_LOGIC_SUMMARY_CHARS = 2500
_MAX_AGENT_COT_TRACE_CHARS = 12000
_MAX_HUNTER_FANOUT = 3
_DEFAULT_PARALLEL_TIMEOUT_SECONDS = 180
_MULTI_HUNTER_CHUNK_THRESHOLD = 8
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
        vision: Optional[VisionAgent] = None,
    ) -> None:
        self.hunter = hunter or HunterAgent()
        self.critic = critic or CriticAgent()
        self.mediator = mediator or MediatorAgent()
        self.vision = vision or VisionAgent()
        self.scope_stratified_enabled = bool(
            getattr(settings, "AI_DEBATE_SCOPE_STRATIFIED_HUNTING_ENABLED", False)
        )
        self.scope_chunk_threshold = int(
            getattr(settings, "AI_DEBATE_SCOPE_CHUNK_THRESHOLD", 15)
        )
        self.scope_token_threshold = int(
            getattr(settings, "AI_DEBATE_SCOPE_TOKEN_THRESHOLD", 7000)
        )
        self.scope_max_groups = int(getattr(settings, "AI_DEBATE_SCOPE_MAX_GROUPS", 4))
        self.max_hunter_calls_per_parameter = int(
            getattr(settings, "AI_DEBATE_MAX_HUNTER_CALLS_PER_PARAMETER", 8)
        )
        self.max_debate_rounds = int(getattr(settings, "AI_DEBATE_MAX_DEBATE_ROUNDS", 2))
        self.parallel_timeout_seconds = int(
            getattr(
                settings,
                "AI_DEBATE_PARALLEL_TIMEOUT_SECONDS",
                _DEFAULT_PARALLEL_TIMEOUT_SECONDS,
            )
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_debate(
        self,
        debate_input: DebateInput,
        retrieval_result: Optional[RetrievalResult],
        tsd_document: TSDDocument,
        enable_vision: bool = False,
    ) -> DebateOutput:
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
        diagram_captions = debate_input.diagram_captions
        killed_assumptions = list(debate_input.killed_assumptions or [])
        hunter_plan = debate_input.hunter_plan or {}
        retrieved_chunk_ids = self._citation_grade_ids(context_chunk_map)
        model_routing = self._resolve_model_routing()
        start_ts = time.monotonic()
        timing: Dict[str, Any] = {}
        hunter_call_count = 0

        scope_meta = self._build_scope_groups(
            context_chunks=context_chunks,
            context_chunk_map=context_chunk_map,
            contract=contract,
        )
        scope_groups = scope_meta["groups"]
        if not scope_meta.get("triggered"):
            self.logger.info(
                "DebateService.run_debate: scope-stratified hunting skipped reason=%s",
                scope_meta.get("reason", "disabled"),
            )

        current_context_chunks = list(context_chunks)
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

        for round_number in range(1, self.max_debate_rounds + 1):
            debate_rounds = round_number
            self.logger.info(
                "DebateService.run_debate: [HUNTER round=%d] parameter id=%s",
                round_number,
                parameter.id,
            )
            hunter_started = time.monotonic()
            (
                hunter_result,
                hunter_results_by_persona,
                merged_evidence,
                persona_plan,
                hunter_call_count,
            ) = self._run_hunter_round(
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                contract=contract,
                current_context_chunks=current_context_chunks,
                diagram_captions=diagram_captions,
                killed_assumptions=killed_assumptions,
                hunter_plan=hunter_plan,
                scope_groups=scope_groups,
                hunter_call_count=hunter_call_count,
            )
            timing[f"hunter_round_{round_number}_seconds"] = round(
                time.monotonic() - hunter_started, 4
            )
            hunter_result = self._normalize_reasoning_payload(hunter_result)
            hunter_result.citations, hunter_rejected = self._validate_citations(
                hunter_result.citations,
                retrieved_chunk_ids,
                "hunter",
            )
            sanitized_hunter = self._sanitize_hunter_for_handoff(hunter_result)

            self.logger.info(
                "DebateService.run_debate: [CRITIC round=%d] parameter id=%s",
                round_number,
                parameter.id,
            )
            critic_started = time.monotonic()
            critic_result = self.critic.run(
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                contract=contract,
                context_chunks=current_context_chunks,
                hunter_result=sanitized_hunter,
                cited_blocks=self._build_cold_start_cited_blocks(
                    sanitized_hunter.citations, context_chunk_map
                ),
            )
            timing[f"critic_round_{round_number}_seconds"] = round(
                time.monotonic() - critic_started, 4
            )
            critic_result = self._normalize_reasoning_payload(critic_result)
            critic_result.valid_citations, critic_rejected = self._validate_citations(
                critic_result.valid_citations,
                retrieved_chunk_ids,
                "critic",
            )
            critic_result.invalid_citation_ids = self._merge_invalid_ids(
                critic_result.invalid_citation_ids,
                [citation.block_id for citation in critic_rejected],
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

            if not self._should_continue_debate(hunter_result, critic_result, round_number):
                break

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
        mediator_started = time.monotonic()
        mediator_result = self.mediator.run(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
            hunter_result=sanitized_hunter,
            critic_result=sanitized_critic,
            debate_history=debate_history,
        )
        timing["mediator_seconds"] = round(time.monotonic() - mediator_started, 4)
        mediator_result = self._normalize_reasoning_payload(mediator_result)
        mediator_result = self._apply_mediator_evidence_policy(
            mediator_result=mediator_result,
            critic_result=critic_result,
            contract=contract,
            hunter_result=hunter_result,
        )

        vision_results: List[tuple] = []
        vision_enabled = bool(getattr(settings, "AI_VISION_ENABLED", True)) and enable_vision
        max_diagrams = max(0, int(getattr(settings, "AI_VISION_MAX_DIAGRAMS_PER_PARAMETER", 1)))
        if vision_enabled and retrieval_result and retrieval_result.get_diagram_block_ids() and max_diagrams > 0:
            vision_started = time.monotonic()
            diagram_block_ids = retrieval_result.get_diagram_block_ids()[:max_diagrams]
            vision_results = self._run_vision_audit(
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                diagram_block_ids=diagram_block_ids,
                tsd_document=tsd_document,
            )
            timing["vision_seconds"] = round(time.monotonic() - vision_started, 4)
        timing["flow_total_seconds"] = round(time.monotonic() - start_ts, 4)

        output = DebateOutput(
            parameter=parameter,
            hunter_result=hunter_result,
            critic_result=critic_result,
            mediator_result=mediator_result,
            vision_results=vision_results,
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
                "scope_stratified": {
                    "triggered": bool(scope_meta.get("triggered")),
                    "reason": scope_meta.get("reason"),
                    "group_count": len(scope_groups),
                    "group_keys": [group.get("key") for group in scope_groups],
                },
                "timing": timing,
                "cost_estimate": {
                    "available": False,
                    "reason": "provider usage not surfaced at agent result level",
                },
                "hunter_claim": {
                    "verdict": hunter_result.verdict,
                    "confidence": hunter_result.confidence,
                    "citation_ids": [c.block_id for c in hunter_result.citations],
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

    def run_batch_debate(
        self,
        debate_inputs: List[DebateInput],
        retrieval_result: Optional[RetrievalResult],
        tsd_document: TSDDocument,
        enable_vision: bool = False,
    ) -> Dict[str, DebateOutput]:
        """
        Runs one Hunter -> Critic -> Mediator flow for a small set of children.

        This method performs no database writes. It returns normal DebateOutput
        objects keyed by child parameter id so callers can validate and persist
        findings exactly like the single-child path.
        """
        if not debate_inputs:
            return {}

        first = debate_inputs[0]
        context_chunks = list(first.context_chunks or [])
        context_chunk_map = dict(first.context_chunk_map or {})
        retrieved_chunk_ids = self._citation_grade_ids(context_chunk_map)
        child_inputs = [
            {
                "id": str(item.parameter.id),
                "requirement": item.parameter_text,
                "contract": item.contract or {},
            }
            for item in debate_inputs
        ]
        start_ts = time.monotonic()

        self.logger.info(
            "DebateService.run_batch_debate: [ENTRY] children=%d parent_section=%s",
            len(debate_inputs),
            first.parameter_section,
        )
        hunter_results = self.hunter.run_batch(
            child_inputs=child_inputs,
            parameter_section=first.parameter_section,
            context_chunks=context_chunks,
            diagram_captions=first.diagram_captions,
            killed_assumptions=first.killed_assumptions,
        )
        sanitized_hunters: Dict[str, HunterResult] = {}
        cited_blocks_by_child: Dict[str, List[dict]] = {}
        hunter_rejected_by_child: Dict[str, List[Any]] = {}
        for child_id, hunter_result in list(hunter_results.items()):
            hunter_result = self._normalize_reasoning_payload(hunter_result)
            hunter_result.citations, rejected = self._validate_citations(
                hunter_result.citations,
                retrieved_chunk_ids,
                "hunter_batch",
            )
            hunter_results[child_id] = hunter_result
            hunter_rejected_by_child[child_id] = rejected
            sanitized_hunters[child_id] = self._sanitize_hunter_for_handoff(hunter_result)
            cited_blocks_by_child[child_id] = self._build_cold_start_cited_blocks(
                sanitized_hunters[child_id].citations,
                context_chunk_map,
            )

        critic_results = self.critic.run_batch(
            child_inputs=child_inputs,
            parameter_section=first.parameter_section,
            context_chunks=context_chunks,
            hunter_results=sanitized_hunters,
            cited_blocks_by_child=cited_blocks_by_child,
        )
        sanitized_critics: Dict[str, CriticResult] = {}
        critic_rejected_by_child: Dict[str, List[Any]] = {}
        debate_history_by_child: Dict[str, List[dict]] = {}
        for debate_input in debate_inputs:
            child_id = str(debate_input.parameter.id)
            critic_result = critic_results.get(child_id)
            hunter_result = hunter_results.get(child_id)
            if not critic_result or not hunter_result:
                continue
            critic_result = self._normalize_reasoning_payload(critic_result)
            critic_result.valid_citations, rejected = self._validate_citations(
                critic_result.valid_citations,
                retrieved_chunk_ids,
                "critic_batch",
            )
            critic_result.invalid_citation_ids = self._merge_invalid_ids(
                critic_result.invalid_citation_ids,
                [citation.block_id for citation in rejected],
            )
            critic_results[child_id] = critic_result
            critic_rejected_by_child[child_id] = rejected
            sanitized_critics[child_id] = self._sanitize_critic_for_handoff(critic_result)
            debate_history_by_child[child_id] = [
                self._build_debate_history_entry(
                    round_number=1,
                    hunter_result=hunter_result,
                    critic_result=critic_result,
                    hunter_rejected=hunter_rejected_by_child.get(child_id, []),
                    critic_rejected=rejected,
                    rebuttal_context=[],
                )
            ]

        mediator_results = self.mediator.run_batch(
            child_inputs=child_inputs,
            parameter_section=first.parameter_section,
            hunter_results=sanitized_hunters,
            critic_results=sanitized_critics,
            debate_history_by_child=debate_history_by_child,
        )

        outputs: Dict[str, DebateOutput] = {}
        for debate_input in debate_inputs:
            child_id = str(debate_input.parameter.id)
            hunter_result = hunter_results.get(child_id)
            critic_result = critic_results.get(child_id)
            mediator_result = mediator_results.get(child_id)
            if not hunter_result or not critic_result or not mediator_result:
                continue
            mediator_result = self._normalize_reasoning_payload(mediator_result)
            mediator_result = self._apply_mediator_evidence_policy(
                mediator_result=mediator_result,
                critic_result=critic_result,
                contract=debate_input.contract or {},
                hunter_result=hunter_result,
            )
            vision_results: List[tuple] = []
            vision_enabled = bool(getattr(settings, "AI_VISION_ENABLED", True)) and enable_vision
            max_diagrams = max(0, int(getattr(settings, "AI_VISION_MAX_DIAGRAMS_PER_PARAMETER", 1)))
            if (
                vision_enabled
                and retrieval_result
                and retrieval_result.get_diagram_block_ids()
                and max_diagrams > 0
            ):
                vision_results = self._run_vision_audit(
                    parameter_text=debate_input.parameter_text,
                    parameter_section=debate_input.parameter_section,
                    diagram_block_ids=retrieval_result.get_diagram_block_ids()[:max_diagrams],
                    tsd_document=tsd_document,
                )
            outputs[child_id] = DebateOutput(
                parameter=debate_input.parameter,
                hunter_result=hunter_result,
                critic_result=critic_result,
                mediator_result=mediator_result,
                vision_results=vision_results,
                retrieval_result=retrieval_result,
                debate_rounds=1,
                analysis_trace={
                    "contract": debate_input.contract or {},
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "killed_assumptions": list(debate_input.killed_assumptions or []),
                    "retrieval_query_details": debate_input.retrieval_query_details or {},
                    "batch": {
                        "enabled": True,
                        "child_count": len(debate_inputs),
                        "child_ids": [str(item.parameter.id) for item in debate_inputs],
                        "elapsed_seconds": round(time.monotonic() - start_ts, 4),
                    },
                    "hunter_claim": {
                        "verdict": hunter_result.verdict,
                        "confidence": hunter_result.confidence,
                        "citation_ids": [c.block_id for c in hunter_result.citations],
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
                        "debate_rounds_used": mediator_result.debate_rounds_used or 1,
                    },
                    "rejected_evidence": {
                        "hunter": [c.block_id for c in hunter_rejected_by_child.get(child_id, [])],
                        "critic": [c.block_id for c in critic_rejected_by_child.get(child_id, [])],
                    },
                    "debate_history": debate_history_by_child.get(child_id, []),
                },
            )
        self.logger.info(
            "DebateService.run_batch_debate: [SUCCESS] children=%d outputs=%d elapsed=%.4f",
            len(debate_inputs),
            len(outputs),
            time.monotonic() - start_ts,
        )
        return outputs

    def _run_hunter_round(
        self,
        parameter_text: str,
        parameter_section: str,
        contract: Dict[str, Any],
        current_context_chunks: List[str],
        diagram_captions: List[str],
        killed_assumptions: List[Dict[str, Any]],
        hunter_plan: Dict[str, Any],
        scope_groups: List[Dict[str, Any]],
        hunter_call_count: int,
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

        if not scope_groups:
            scope_groups = [{"key": "all_context", "chunks": list(current_context_chunks)}]

        all_results: Dict[str, HunterResult] = {}
        merged_citations: Dict[str, Any] = {}
        persona_payload: Dict[str, Any] = {}
        winner: Optional[HunterResult] = None
        persona_plan = self._build_hunter_plan(
            contract=contract,
            parameter_text=parameter_text,
            context_chunk_count=len(current_context_chunks),
            incoming_plan=hunter_plan,
        )
        personas = list(persona_plan.get("personas") or [contract.get("domain") or "general"])

        for group in scope_groups:
            group_chunks = list(group.get("chunks") or [])
            if not group_chunks:
                continue
            group_key = str(group.get("key") or "group")
            available_budget = self.max_hunter_calls_per_parameter - hunter_call_count
            group_personas = personas[: max(0, available_budget)]
            if not group_personas:
                break

            def _run_single(persona_value: str):
                return persona_value, self.hunter.run(
                    parameter_text=parameter_text,
                    parameter_section=parameter_section,
                    contract=contract,
                    context_chunks=group_chunks,
                    diagram_captions=diagram_captions,
                    persona_focus=persona_value,
                    killed_assumptions=killed_assumptions,
                )

            if len(group_personas) == 1:
                results = [_run_single(group_personas[0])]
            else:
                timeout = (
                    self.parallel_timeout_seconds
                    if self.parallel_timeout_seconds > 0
                    else _DEFAULT_PARALLEL_TIMEOUT_SECONDS
                )
                results = []
                pool = ThreadPoolExecutor(max_workers=min(_MAX_HUNTER_FANOUT, len(group_personas)), thread_name_prefix="ThreadPoolExecutor-10")
                try:
                    context_snapshot = capture_ai_usage_context()
                    futures = [
                        pool.submit(
                            run_with_ai_usage_context,
                            context_snapshot,
                            _run_single,
                            persona,
                        )
                        for persona in group_personas
                    ]
                    try:
                        for future in as_completed(futures, timeout=timeout):
                            results.append(future.result())
                    except TimeoutError:
                        self.logger.warning(
                            "DebateService._run_hunter_round: hunter parallel timeout reached (%ss)",
                            timeout,
                        )
                        for future in futures:
                            if future.done():
                                results.append(future.result())
                            else:
                                future.cancel()
                    finally:
                        pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise

            hunter_call_count += len(results)
            for persona, result in results:
                result = self._normalize_reasoning_payload(result)
                result_key = f"{persona}@{group_key}" if len(scope_groups) > 1 else persona
                all_results[result_key] = result
                persona_payload[result_key] = {
                    "verdict": result.verdict,
                    "confidence": result.confidence,
                    "reasoning": result.logic_summary or result.reasoning,
                    "citation_ids": [c.block_id for c in result.citations],
                }
                for citation in result.citations:
                    if citation.block_id not in merged_citations:
                        merged_citations[citation.block_id] = {"citation": citation, "personas": set()}
                    merged_citations[citation.block_id]["personas"].add(result_key)
                if winner is None or result.confidence > winner.confidence:
                    winner = result

        if winner is None:
            winner = HunterResult(
                verdict=VERDICT_NOT_MET,
                confidence=0.2,
                reasoning="No hunter evidence produced in current round.",
                evidence_found=False,
                citations=[],
            )
        winner.citations = [entry["citation"] for entry in merged_citations.values()]
        merged_evidence = {
            "deduped_ids": list(merged_citations.keys()),
            "provenance": {
                block_id: sorted(list(entry["personas"]))
                for block_id, entry in merged_citations.items()
            },
        }
        if len(scope_groups) > 1:
            persona_plan["scope_stratified"] = True
        return winner, persona_payload, merged_evidence, persona_plan, hunter_call_count

    def _build_scope_groups(
        self,
        context_chunks: List[str],
        context_chunk_map: Dict[str, Dict[str, Any]],
        contract: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            if not self.scope_stratified_enabled:
                return {"triggered": False, "reason": "disabled", "groups": [{"key": "all_context", "chunks": list(context_chunks)}]}
            token_estimate = sum(max(1, len(chunk) // 4) for chunk in context_chunks)
            sections = set((payload.get("section") or "unknown") for payload in context_chunk_map.values())
            should_trigger = (
                len(context_chunks) > self.scope_chunk_threshold
                or token_estimate > self.scope_token_threshold
                or len(sections) > 3
            )
            if not should_trigger:
                return {"triggered": False, "reason": "below_threshold", "groups": [{"key": "all_context", "chunks": list(context_chunks)}]}

            groups: Dict[str, List[str]] = {}
            domain = (contract.get("domain") or "general").strip()
            for block_id, payload in context_chunk_map.items():
                section = (payload.get("section") or "unknown").strip()
                source = (payload.get("source") or "retrieval_context").strip()
                page_match = re.match(r"p(\d+)_", block_id or "")
                page = f"p{page_match.group(1)}" if page_match else "p?"
                key = f"{domain}|{section}|{source}|{page}"
                groups.setdefault(key, []).append(payload.get("text") or "")

            compact_groups = []
            for key, chunks in groups.items():
                if chunks:
                    compact_groups.append({"key": key, "chunks": chunks})
            compact_groups.sort(key=lambda g: len(g["chunks"]), reverse=True)
            compact_groups = compact_groups[: self.scope_max_groups] or [{"key": "all_context", "chunks": list(context_chunks)}]
            return {"triggered": True, "reason": "threshold_exceeded", "groups": compact_groups}
        except Exception as exc:
            self.logger.warning(
                "DebateService._build_scope_groups: fallback to normal flow due to error: %s",
                exc,
            )
            return {"triggered": False, "reason": "grouping_failed", "groups": [{"key": "all_context", "chunks": list(context_chunks)}]}

    def _resolve_model_routing(self) -> Dict[str, str]:
        return {
            "contract_synthesizer": get_model_for_component("contract_synthesizer"),
            "hunter": get_model_for_component("hunter"),
            "critic": get_model_for_component("critic"),
            "mediator": get_model_for_component("mediator"),
            "vision": get_model_for_component("vision"),
        }

    def _build_hunter_plan(
        self,
        contract: Dict[str, Any],
        parameter_text: str,
        context_chunk_count: int,
        incoming_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        if incoming_plan.get("personas"):
            return incoming_plan
        
        domain = str(contract.get("domain") or "").strip()
        
        fanout_enabled = str(getattr(settings, "AI_DEBATE_ENABLE_FANOUT", "False")).lower() in {"true", "1", "yes"}
        if not fanout_enabled:
            return {"mode": "single", "personas": [domain if domain in _PERSONAS else "general"]}
            
        confident = float(contract.get("confidence") or 0.0) >= 0.7
        broad = len((parameter_text or "").split()) > 22
        if (
            domain in _PERSONAS
            and confident
            and not broad
            and context_chunk_count < _MULTI_HUNTER_CHUNK_THRESHOLD
        ):
            personas = [domain]
            mode = "single"
        else:
            personas = list(_PERSONAS)
            mode = "fanout"
        return {"mode": mode, "personas": personas}

    # Compatibility helper retained for Phase 2 tests/callers.
    def _run_hunters_with_aggregation(
        self,
        parameter_text: str,
        parameter_section: str,
        contract: Dict[str, Any],
        context_chunks: List[str],
        diagram_captions: List[str],
        killed_assumptions: List[Dict[str, Any]],
        persona_plan: Dict[str, Any],
    ) -> tuple[HunterResult, Dict[str, Any], Dict[str, Any]]:
        winner, payload, merged, _, _ = self._run_hunter_round(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
            current_context_chunks=context_chunks,
            diagram_captions=diagram_captions,
            killed_assumptions=killed_assumptions,
            hunter_plan=persona_plan,
            scope_groups=[{"key": "all_context", "chunks": list(context_chunks)}],
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

    def _sanitize_hunter_for_handoff(self, result: HunterResult) -> HunterResult:
        return HunterResult(
            verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.logic_summary or result.reasoning,
            assumptions=list(result.assumptions),
            logic_summary=result.logic_summary or result.reasoning,
            cot_trace=None,
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
            cot_trace=None,
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
    ) -> bool:
        if round_number >= self.max_debate_rounds:
            return False
        if critic_result.outcome in {OUTCOME_OVERTURN, OUTCOME_PARTIAL}:
            return True
        if critic_result.requires_rebuttal:
            return True
        if hunter_result.error and not critic_result.error:
            return True
        if hunter_result.verdict == "met" and not hunter_result.citations:
            return True
        return False

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
        rebuttal_chunk = "\n".join(
            [
                f"--- DEBATE REBUTTAL ROUND {round_number} ---",
                f"Section: {parameter_section}",
                f"Requirement: {parameter_text}",
                f"Hunter verdict: {hunter_result.verdict} (confidence={hunter_result.confidence:.2f})",
                f"Critic outcome: {critic_result.outcome}",
                f"Critic revised verdict: {critic_result.revised_verdict} (confidence={critic_result.revised_confidence:.2f})",
                f"Critic reasoning: {critic_result.logic_summary or critic_result.reasoning}",
                f"Valid citations: {valid_citations}",
                f"Invalid citations: {invalid_citations}",
                "Instruction: Re-check the original TSD context and respond directly to the Critic's objections.",
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

    def _validate_citations(self, citations, allowed_ids, agent_name):
        allowed = set(allowed_ids)
        valid = []
        rejected = []
        for citation in citations:
            if citation.block_id in allowed:
                valid.append(citation)
            else:
                rejected.append(citation)
        if rejected:
            self.logger.warning(
                "DebateService._validate_citations: agent=%s rejected=%d unknown ids",
                agent_name,
                len(rejected),
            )
        return valid, rejected

    def _citation_grade_ids(self, context_chunk_map):
        ids = []
        for block_id, payload in (context_chunk_map or {}).items():
            if payload.get("citation_grade", True) is False:
                continue
            evidence_kind = str(payload.get("evidence_kind") or "").lower()
            if evidence_kind in {"graph_summary", "baseline_requirement"}:
                continue
            if str(block_id).startswith("graph_summary_"):
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

    def _run_vision_audit(
        self,
        parameter_text: str,
        parameter_section: str,
        diagram_block_ids: List[str],
        tsd_document: TSDDocument,
    ) -> List[tuple]:
        from sdr.apps.ai.agents.vision import DiagramInput, audit_diagrams_for_parameter

        diagram_inputs = []
        for block_id in diagram_block_ids:
            if "_d" not in block_id:
                continue
            diagram_block = tsd_document.get_diagram_by_id(block_id)
            if not diagram_block:
                continue
            diagram_input = DiagramInput(
                diagram_id=diagram_block.diagram_id,
                image_b64=diagram_block.image_b64,
                page_number=diagram_block.page_number,
                caption=diagram_block.caption,
                surrounding_text=diagram_block.surrounding_text,
                image_format=diagram_block.image_format,
                bbox_x0=diagram_block.bbox_x0,
                bbox_y0=diagram_block.bbox_y0,
                bbox_x1=diagram_block.bbox_x1,
                bbox_y1=diagram_block.bbox_y1,
            )
            if diagram_input.is_valid():
                diagram_inputs.append(diagram_input)
        if not diagram_inputs:
            return []
        max_diagrams = max(0, int(getattr(settings, "AI_VISION_MAX_DIAGRAMS_PER_PARAMETER", 1)))
        if max_diagrams < 1:
            return []
        raw_vision_results = audit_diagrams_for_parameter(
            agent=self.vision,
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            diagrams=diagram_inputs[:max_diagrams],
        )
        return [
            (diagram_input, self._normalize_reasoning_payload(vision_result))
            for diagram_input, vision_result in raw_vision_results
        ]
