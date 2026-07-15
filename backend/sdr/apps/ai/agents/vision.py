from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from sdr.apps.ai.prompts.agents import (
    VISION_CRITIC_DEBATE_SYSTEM_PROMPT,
    VISION_HUNTER_SYSTEM_PROMPT,
    VISION_MEDIATOR_DEBATE_SYSTEM_PROMPT,
    build_vision_critic_debate_prompt,
    build_vision_hunter_prompt,
    build_vision_mediator_debate_prompt,
)
from .base import BaseAgent, VERDICT_MET, VERDICT_NA, VERDICT_NOT_MET
from sdr.core.config import settings

logger = logging.getLogger(__name__)
_DIAGRAM_SCOPE_VALUES = {"architecture_relevant", "non_architecture", "uncertain"}


@dataclass
class DiagramInput:
    diagram_id: str
    image_b64: str
    page_number: int
    caption: Optional[str] = None
    surrounding_text: Optional[str] = None
    image_format: str = "png"
    bbox_x0: Optional[float] = None
    bbox_y0: Optional[float] = None
    bbox_x1: Optional[float] = None
    bbox_y1: Optional[float] = None

    def is_valid(self) -> bool:
        return bool(self.diagram_id) and bool(self.image_b64) and self.page_number > 0

    def decode_image_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.image_b64)
        except Exception as exc:
            raise ValueError(
                f"DiagramInput.decode_image_bytes: failed to decode "
                f"base64 for diagram_id='{self.diagram_id}': {exc}"
            ) from exc


@dataclass
class DiagramDebateOutput:
    """Output of a single diagram debate cycle."""

    diagram: DiagramInput
    hunter_result: Dict[str, Any] = field(default_factory=dict)
    hunter_rebuttal_result: Dict[str, Any] = field(default_factory=dict)
    critic_result: Dict[str, Any] = field(default_factory=dict)
    mediator_result: Dict[str, Any] = field(default_factory=dict)
    requirements: List[Any] = field(default_factory=list)
    debate_rounds: int = 1
    error: Optional[str] = None
    pipeline_mode: str = "debate"


class VisionAgent(BaseAgent):
    """Thin wrapper around BaseAgent for diagram debate multimodal calls."""
    model_component: str = "vision"
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 0.9
    seed: int = 42

    def run_multimodal(
        self,
        *,
        user_prompt: str,
        image_bytes: Optional[bytes] = None,
        image_format: str = "png",
        image_payloads: Optional[List[Dict[str, Any]]] = None,
        system_prompt: str = "",
        log_context: str = "",
    ) -> Optional[Dict[str, Any]]:
        original_prompt = self.system_prompt
        if system_prompt:
            self.system_prompt = system_prompt
        try:
            response = self._call_llm_with_truncation_retry(
                user_prompt,
                image_b64=image_bytes,
                image_format=image_format,
                image_payloads=image_payloads,
                top_p=self.top_p,
                log_context=log_context,
            )
        finally:
            self.system_prompt = original_prompt

        if response.error:
            self.logger.error("VisionDebateAgent[%s] LLM error: %s", log_context, response.error)
            return None

        parsed = self._parse_json_response(response, log_context=log_context)
        if parsed is None:
            self.logger.error("VisionDebateAgent[%s]: failed to parse JSON response", log_context)
        return parsed

    def run_text(
        self,
        *,
        user_prompt: str,
        system_prompt: str = "",
        log_context: str = "",
    ) -> Optional[Dict[str, Any]]:
        original_prompt = self.system_prompt
        if system_prompt:
            self.system_prompt = system_prompt
        try:
            response = self._call_llm_with_truncation_retry(user_prompt, log_context=log_context)
        finally:
            self.system_prompt = original_prompt

        if response.error:
            self.logger.error("VisionDebateAgent[%s] text LLM error: %s", log_context, response.error)
            return None

        parsed = self._parse_json_response(response, log_context=log_context)
        if parsed is None:
            self.logger.error("VisionDebateAgent[%s]: failed to parse JSON text response", log_context)
        return parsed


class VisionHunterAgent(VisionAgent):
    """VisionAgent pinned to the Hunter's own model (AI_MODEL_VISION_HUNTER) —
    deliberately a fast, shallow first pass (low token budget, minimal reasoning)
    so it behaves like a realistic single-agent baseline, not one already
    hardened with the Critic's anti-hallucination discipline."""
    model_component: str = "vision_hunter"
    # NOTE: max_tokens must stay large enough to cover a full assessment object
    # for every requirement in the largest diagrams (18-23 requirements) — the
    # Hunter must assess EVERY requirement (completeness rule), so shrinking
    # this below ~6144 causes outright JSON-truncation failures (not "dumber"
    # reasoning, just broken output) rather than weaker per-item reasoning.
    # The "dumbing down" instead comes from the simplified prompt/guardrails.
    max_tokens: int = 6144
    reasoning_effort: str = "low"


class VisionCriticAgent(VisionAgent):
    """VisionAgent pinned to the Critic's own model (AI_MODEL_VISION_CRITIC) —
    a different underlying model than the Hunter so the Critic's re-examination
    of the same image is a genuinely independent second opinion."""
    model_component: str = "vision_critic"
    max_tokens: int = 12288
    reasoning_effort: str = "high"


class VisionMediatorAgent(VisionAgent):
    """VisionAgent pinned to the Mediator's own model (AI_MODEL_VISION_MEDIATOR)."""
    model_component: str = "vision_mediator"
    max_tokens: int = 12288
    reasoning_effort: str = "high"


class VisionExtractorAgent(VisionAgent):
    """VisionAgent pinned to the Extractor's own model (AI_MODEL_VISION_EXTRACTOR) —
    Stage 1 of the extract-then-reason pipeline. This is a pure perception pass:
    describe what's visibly in the diagram, no requirement judgment at all, so a
    low token budget / low reasoning effort is appropriate (and cheap, since it
    may run multiple independent times for self-consistency voting)."""
    model_component: str = "vision_extractor"
    max_tokens: int = 4096
    reasoning_effort: str = "low"


class VisionReasonerAgent(VisionAgent):
    """VisionAgent pinned to the Reasoner's own model (AI_MODEL_VISION_REASONER) —
    Stage 2 of the extract-then-reason pipeline. Deliberately uses run_text(),
    never run_multimodal() — no image is attached here, only the structured
    extraction produced by the Extractor. This is where the reasoning budget
    goes now that grounding is decoupled from perception."""
    model_component: str = "vision_reasoner"
    max_tokens: int = 8192
    reasoning_effort: str = "high"


@dataclass
class MergedDiagramExtraction:
    """Self-consistency-merged output of N independent Stage-1 extraction
    passes over the same diagram image. Each element (component/trust_boundary/
    flow) carries a `confirmed` flag: True if it appeared in >= merge_threshold
    fraction of the N passes, False if it only appeared in a minority (kept,
    not dropped, so Stage 2 can still cite it at reduced trust)."""

    components: List[Dict[str, Any]] = field(default_factory=list)
    trust_boundaries: List[Dict[str, Any]] = field(default_factory=list)
    flows: List[Dict[str, Any]] = field(default_factory=list)
    other_visible_text: List[str] = field(default_factory=list)
    diagram_scope_verdict: str = "uncertain"
    diagram_scope_reasoning: str = ""
    diagram_style: str = "other"
    votes_total: int = 1
    raw_passes: List[Dict[str, Any]] = field(default_factory=list)
    merge_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def confirmed_element_ids(self) -> set:
        return {
            str(element["id"])
            for group in (self.components, self.trust_boundaries, self.flows)
            for element in group
            if element.get("confirmed")
        }

    def all_element_ids(self) -> set:
        return {
            str(element["id"])
            for group in (self.components, self.trust_boundaries, self.flows)
            for element in group
        }

    def _component_label(self, component_id: Optional[str]) -> str:
        for component in self.components:
            if str(component.get("id")) == str(component_id):
                return str(component.get("name") or component_id)
        return str(component_id or "unknown")

    def _confirmation_tag(self, element: Dict[str, Any]) -> str:
        return "[confirmed]" if element.get("confirmed") else "[unconfirmed]"

    def to_reasoner_text(self) -> str:
        """
        Serializes this extraction for the Stage-2 (text-only) reasoning prompt.
        Renders TWO blocks so the reasoner never has to reconstruct structural
        relationships (containment, adjacency) by mentally joining ids across
        separate arrays: (1) tagged raw JSON — needed so the reasoner can emit
        valid cited_element_ids — and (2) a deterministically-generated
        relationship narrative, one explicit sentence per resolved relationship.
        Unconfirmed elements are tagged inline in both blocks.
        """
        import json as _json

        raw_json = _json.dumps(
            {
                "diagram_scope_verdict": self.diagram_scope_verdict,
                "diagram_scope_reasoning": self.diagram_scope_reasoning,
                "diagram_style": self.diagram_style,
                "components": self.components,
                "trust_boundaries": self.trust_boundaries,
                "flows": self.flows,
                "other_visible_text": self.other_visible_text,
            },
            ensure_ascii=True,
            indent=2,
        )

        narrative_lines: List[str] = []
        for boundary in self.trust_boundaries:
            enclosed = boundary.get("encloses_component_ids") or []
            enclosed_desc = ", ".join(
                f"{cid} ({self._component_label(cid)})" for cid in enclosed
            ) or "nothing visibly enclosed"
            narrative_lines.append(
                f"Trust boundary {boundary.get('id')} ('{boundary.get('label', '')}') "
                f"{self._confirmation_tag(boundary)} encloses: {enclosed_desc}."
            )
        for flow_item in self.flows:
            source_id = flow_item.get("source_component_id")
            target_id = flow_item.get("target_component_id")
            narrative_lines.append(
                f"Flow {flow_item.get('id')} {self._confirmation_tag(flow_item)}: "
                f"{source_id} ({self._component_label(source_id)}) -> "
                f"{target_id} ({self._component_label(target_id)}), "
                f"protocol: {flow_item.get('protocol') or 'unlabeled'}, "
                f"direction: {flow_item.get('direction', 'unclear')}, "
                f"label: '{flow_item.get('label', '')}'."
            )
        for component in self.components:
            narrative_lines.append(
                f"Component {component.get('id')} {self._confirmation_tag(component)}: "
                f"'{component.get('name', '')}' (type: {component.get('type', 'other')})."
            )

        narrative_block = "\n".join(narrative_lines) if narrative_lines else "(no elements extracted)"

        return (
            "### Structured extraction (JSON)\n"
            f"{raw_json}\n\n"
            "### Relationship narrative (derived from the JSON above; "
            "[confirmed] = seen in a majority of independent extraction passes, "
            "[unconfirmed] = seen in a minority only)\n"
            f"{narrative_block}"
        )


def _format_requirements_compact(requirements: List[Any]) -> str:
    lines = []
    for req in requirements:
        req_id = getattr(req, "stable_key", f"D-{getattr(req, 'ordinal', 0)}")
        text = getattr(req, "requirement_text", "")
        lines.append(f"{req.ordinal}. [{req_id}] {text}")
    return "\n".join(lines)


def _format_requirements_with_hints(requirements: List[Any]) -> str:
    lines = []
    for req in requirements:
        req_id = getattr(req, "stable_key", f"D-{getattr(req, 'ordinal', 0)}")
        text = getattr(req, "requirement_text", "")
        hint = getattr(req, "verification_hint", "")
        lines.append(f"{req.ordinal}. [{req_id}] {text}")
        if hint:
            lines.append(f"   VERIFY: {hint}")
    return "\n".join(lines)


def _apply_diagram_evidence_policy(
    mediator_result: Dict[str, Any],
    critic_result: Dict[str, Any],
    hunter_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hunter_result = dict(hunter_result or {})
    mediator_result = dict(mediator_result or {})
    verdict = str(mediator_result.get("final_verdict", VERDICT_NA)).strip().lower()
    assessments = _merge_assessed_requirements(
        hunter_result.get("requirement_assessments") or [],
        mediator_result.get("assessed_requirements") or [],
    )

    hunter_scope = _normalize_diagram_scope_verdict(hunter_result.get("diagram_scope_verdict"))
    critic_scope = _normalize_diagram_scope_verdict(critic_result.get("diagram_scope_verdict"))
    mediator_scope = _normalize_diagram_scope_verdict(mediator_result.get("diagram_scope_verdict"))
    scope_reasoning = (
        critic_result.get("diagram_scope_reasoning")
        if critic_scope == "non_architecture"
        else hunter_result.get("diagram_scope_reasoning")
        if hunter_scope == "non_architecture"
        else mediator_result.get("diagram_scope_reasoning")
        or critic_result.get("diagram_scope_reasoning")
        or hunter_result.get("diagram_scope_reasoning")
    )

    if "diagram_scope_verdict" not in mediator_result:
        mediator_result["diagram_scope_verdict"] = mediator_scope
    if scope_reasoning:
        mediator_result["diagram_scope_reasoning"] = scope_reasoning

    should_force_non_architecture = (
        hunter_scope == "non_architecture" and critic_scope == "non_architecture"
    ) or (
        critic_scope == "non_architecture"
        and not (_critic_has_requirement_reviews(critic_result) or critic_result.get("validated_requirements"))
        and bool((critic_result.get("diagram_scope_reasoning") or "").strip())
    )
    if should_force_non_architecture:
        mediator_result["diagram_scope_verdict"] = "non_architecture"
        if scope_reasoning:
            mediator_result["diagram_scope_reasoning"] = scope_reasoning
        assessments = _force_assessment_verdicts_na(assessments, default_summary="Image is not an architecture/security-relevant diagram.")
        mediator_result["assessed_requirements"] = assessments
        mediator_result["final_verdict"] = VERDICT_NA
        mediator_result["verdict_policy_source"] = "diagram_non_architecture_image"
        return mediator_result

    if assessments:
        assessments, corrected = _ground_assessed_requirements_against_critic(assessments, critic_result)
        mediator_result["assessed_requirements"] = assessments
        # The topline verdict is never trusted from the Mediator's own
        # free-standing claim — it's always the deterministic worst-case of
        # the (citation-corrected) individual assessments: not_met > na > met.
        # This is exactly what the mediator prompt itself instructs, but
        # previously nothing enforced it outside the narrow "a citation got
        # downgraded" branch, so a diagram whose top-level claim disagreed
        # with its own assessments (e.g. claimed "met" while one assessment
        # was "not_met") would silently keep the wrong topline verdict.
        aggregated = _worst_case_diagram_verdict(assessments)
        all_na = all(
            str(assessment.get("verdict", "")).strip().lower() == VERDICT_NA
            for assessment in assessments
        )
        if corrected:
            mediator_result["verdict_policy_source"] = "diagram_requirement_not_corroborated"
        elif all_na and aggregated != verdict:
            mediator_result["verdict_policy_source"] = "diagram_all_requirements_not_applicable"
        elif aggregated != verdict:
            mediator_result["verdict_policy_source"] = "diagram_verdict_aggregated_from_assessments"
        verdict = aggregated
        mediator_result["final_verdict"] = verdict

    # Evidence-quality safety net: a not_met verdict (whether from the
    # aggregate above or the raw claim, if there were no assessments at all)
    # can still be suppressed to "na" when the Critic thinks the underlying
    # claim is a hallucination artifact — this is about trustworthiness of
    # the evidence, not about counting requirements, so it stays an
    # independent layer on top of the aggregation rather than being folded
    # into it.
    hallucinated = critic_result.get("hallucinated_claims") or []
    if hallucinated and verdict == VERDICT_NOT_MET:
        requirement_reviews = _critic_requirement_reviews_by_id(critic_result)
        not_met_in_invalidated = any(
            str(review.get("critic_verdict", "")).strip().lower() == VERDICT_NOT_MET
            for review in requirement_reviews.values()
        )
        if not_met_in_invalidated or len(hallucinated) >= 2:
            mediator_result["final_verdict"] = VERDICT_NA
            mediator_result["verdict_policy_source"] = "diagram_hallucinated_evidence"
            verdict = VERDICT_NA

    # The remaining special cases only apply when there were no individual
    # assessments to aggregate over at all (the aggregation above already
    # covers every case where `assessments` is non-empty, including "all na"
    # and "not_met claimed but nothing applicable" — both resolve to "na" via
    # _worst_case_diagram_verdict without needing their own branches here).
    validated = critic_result.get("validated_requirements") or []
    if not assessments and verdict == VERDICT_MET and not validated:
        mediator_result["final_verdict"] = VERDICT_NA
        mediator_result["verdict_policy_source"] = "diagram_met_without_validated_evidence"
        verdict = VERDICT_NA

    if not assessments and verdict == VERDICT_NOT_MET:
        mediator_result["final_verdict"] = VERDICT_NA
        mediator_result["verdict_policy_source"] = "diagram_no_applicable_security_scope"

    return mediator_result


def _calibrate_diagram_confidence(
    mediator_result: Dict[str, Any],
    hunter_result: Dict[str, Any],
    critic_result: Dict[str, Any],
) -> Dict[str, Any]:
    raw_confidence = float(mediator_result.get("confidence", 0.5) or 0.5)
    verdict = str(mediator_result.get("final_verdict", VERDICT_NA)).strip().lower()

    hunter_verdict = str(hunter_result.get("overall_verdict", "")).strip().lower()
    critic_outcome = str(critic_result.get("outcome", "")).strip().lower()

    adjustment = 0.0

    if hunter_verdict == verdict and critic_outcome == "uphold":
        adjustment += 0.10
    elif hunter_verdict == verdict:
        adjustment += 0.05

    if critic_outcome == "overturn":
        adjustment -= 0.10

    assessments = hunter_result.get("requirement_assessments") or []
    not_met_count = sum(
        1 for assessment in assessments
        if str(assessment.get("verdict", "")).lower() == VERDICT_NOT_MET
    )
    visual_evidence_count = len(hunter_result.get("visual_elements_cited") or [])

    if verdict == VERDICT_NOT_MET and visual_evidence_count == 0 and not_met_count > 0:
        adjustment -= 0.10
    elif verdict == VERDICT_MET and visual_evidence_count >= 2 and not critic_result.get("hallucinated_claims"):
        adjustment += 0.05

    adjusted = max(0.0, min(raw_confidence + adjustment, 1.0))
    if verdict == VERDICT_NA:
        adjusted = min(adjusted, 0.65)

    mediator_result["confidence"] = round(adjusted, 2)
    return mediator_result


def _has_scope_evidence(review: Dict[str, Any]) -> bool:
    return bool(str(review.get("scope_evidence") or "").strip())


def _has_absence_evidence(review: Dict[str, Any]) -> bool:
    return bool(str(review.get("absence_evidence") or "").strip())


def _normalized_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    return None


def _normalized_failure_mode(review: Dict[str, Any]) -> str:
    return str(review.get("failure_mode") or "none").strip().lower()


def _normalized_check_answer(answer: Any) -> str:
    normalized = str(answer or "").strip().lower()
    if normalized in {"present", "pass", "passed", "satisfied", "yes", "true"}:
        return "present"
    if normalized in {"absent", "fail", "failed", "missing", "no", "false"}:
        return "absent"
    if normalized in {"unclear", "unknown", "partial", "ambiguous"}:
        return "unclear"
    return ""


def _verification_checks(review: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for check in review.get("verification_checks") or []:
        if isinstance(check, dict):
            checks.append(check)
    return checks


def _review_has_absent_check(review: Dict[str, Any]) -> bool:
    """An "unclear" check alone is not evidence of failure — only an outright
    "absent" answer blocks "met" by itself. A lone "unclear" is already caught
    by the supports_met=false branch in _critic_blocks_met when the Critic
    also explicitly says the evidence doesn't support "met"."""
    for check in _verification_checks(review):
        if _normalized_check_answer(check.get("answer")) == "absent":
            return True
    return False


def _review_supports_met(review: Dict[str, Any]) -> Optional[bool]:
    return _normalized_bool(review.get("supports_met"))


def _review_supports_not_met(review: Dict[str, Any]) -> Optional[bool]:
    return _normalized_bool(review.get("supports_not_met"))


def _review_has_absent_compound_subelement(review: Dict[str, Any]) -> bool:
    for item in review.get("compound_subelements_checked") or []:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if ": absent" in text or text.endswith("absent"):
            return True
        if ": unclear" in text or text.endswith("unclear"):
            return True
    return False


def _normalized_compound_status(review: Dict[str, Any]) -> str:
    value = str(review.get("compound_status") or "").strip().lower()
    if value in {"single", "compound_core_control", "compound_independent_controls"}:
        return value
    return ""


def _critic_blocks_met(review: Dict[str, Any]) -> bool:
    failure_mode = _normalized_failure_mode(review)
    supports_met = _review_supports_met(review)
    if supports_met is False:
        return True
    # "weak_positive_evidence" is a soft/fuzzy signal that over-fires on a
    # more talkative Critic — only concrete findings (a genuinely partial
    # compound requirement or an affirmative contradiction) block "met" here.
    if failure_mode in {"partial_compound", "contradiction"}:
        return True
    if _review_has_absent_check(review):
        return True
    if _review_has_absent_compound_subelement(review):
        return True
    return False


def _critic_prefers_not_met(review: Dict[str, Any]) -> bool:
    critic_verdict = str(review.get("critic_verdict", "")).strip().lower()
    supports_not_met = _review_supports_not_met(review)
    if critic_verdict == VERDICT_NOT_MET and supports_not_met is not False:
        return True
    return bool(supports_not_met)


def _critic_not_met_is_grounded(review: Dict[str, Any]) -> bool:
    critic_verdict = _normalized_requirement_verdict(review.get("critic_verdict"))
    supports_not_met = _review_supports_not_met(review)
    if critic_verdict != VERDICT_NOT_MET and supports_not_met is not True:
        return False
    if not _has_scope_evidence(review) or not _has_absence_evidence(review):
        return False
    failure_mode = _normalized_failure_mode(review)
    if failure_mode not in {"none", "partial_compound", "contradiction"}:
        return False
    if _normalized_compound_status(review) == "compound_independent_controls":
        return _review_has_absent_compound_subelement(review)
    return True


def _mediator_has_judged(item: Dict[str, Any]) -> bool:
    return bool(
        str(item.get("resolution_basis") or "").strip()
        or str(item.get("winning_side") or "").strip()
        or str(item.get("judge_reason") or "").strip()
    )


def _normalized_requirement_verdict(value: Any) -> str:
    verdict = str(value or "").strip().lower()
    if verdict in {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}:
        return verdict
    return ""


def _is_policy_na_downgrade(item: Dict[str, Any], final_verdict: str, hunter_verdict: str) -> bool:
    if final_verdict != VERDICT_NA or hunter_verdict == final_verdict:
        return False
    return str(item.get("verdict_policy_source") or "").strip().lower() in {
        "diagram_not_met_without_scope_evidence",
        "diagram_not_met_without_absence_evidence",
        "diagram_mediator_not_met_without_scope_evidence",
        "diagram_mediator_not_met_without_absence_evidence",
        "diagram_met_not_supported_by_critic_verifier",
    }


def _determine_final_decision_source(
    item: Dict[str, Any],
    *,
    hunter_verdict: str,
    critic_verdict: str,
    mediator_has_judged: bool,
) -> str:
    final_verdict = _normalized_requirement_verdict(item.get("verdict"))
    winning_side = str(item.get("winning_side") or "").strip().lower()

    if _is_policy_na_downgrade(item, final_verdict, hunter_verdict):
        return "policy_downgraded_to_na"

    if mediator_has_judged:
        if winning_side == "critic" and final_verdict and final_verdict != hunter_verdict:
            return "mediator_tiebreak_to_critic"
        if winning_side == "hunter" and critic_verdict and final_verdict and final_verdict != critic_verdict:
            return "mediator_tiebreak_to_hunter"

    if critic_verdict and final_verdict and final_verdict != hunter_verdict and final_verdict == critic_verdict:
        return "critic_corrected"

    if final_verdict and final_verdict == hunter_verdict:
        return "hunter_preserved"

    if final_verdict == VERDICT_NA and hunter_verdict != final_verdict:
        return "policy_downgraded_to_na"

    return "grounded_fallback"


_DUPLICATE_EVIDENCE_SIMILARITY_THRESHOLD = 0.82


def _normalized_evidence_text(review: Dict[str, Any]) -> str:
    return " ".join(
        str(review.get(field) or "").strip().lower()
        for field in ("scope_evidence", "absence_evidence")
    ).strip()


def _find_contaminated_requirement_ids(critic_result: Dict[str, Any]) -> set[str]:
    """Detect boilerplate evidence bleeding across sibling requirements in the
    same batched Critic response: if two or more DIFFERENT requirements both
    got a "not_met" verdict justified by near-identical scope/absence evidence
    text, that evidence almost certainly isn't independently grounded for each
    one — the model likely reused one requirement's finding as a template. Keep
    the first (order of appearance) as the trusted exemplar and flag the rest
    as contaminated so callers fall back to the Hunter's verdict for them."""
    reviews = _critic_requirement_reviews_by_id(critic_result)
    candidates = [
        (req_id, _normalized_evidence_text(review))
        for req_id, review in reviews.items()
        if _normalized_requirement_verdict(review.get("critic_verdict")) == VERDICT_NOT_MET
        and _has_scope_evidence(review)
        and _has_absence_evidence(review)
    ]
    contaminated: set[str] = set()
    for i in range(len(candidates)):
        req_id_i, text_i = candidates[i]
        if req_id_i in contaminated or not text_i:
            continue
        for j in range(i + 1, len(candidates)):
            req_id_j, text_j = candidates[j]
            if req_id_j in contaminated or not text_j:
                continue
            if SequenceMatcher(None, text_i, text_j).ratio() >= _DUPLICATE_EVIDENCE_SIMILARITY_THRESHOLD:
                contaminated.add(req_id_j)
    return contaminated


def _find_met_rejection_contaminated_ids(critic_result: Dict[str, Any]) -> set[str]:
    """Same boilerplate-bleeding check as _find_contaminated_requirement_ids,
    but for the met-rejection path: if 2+ different requirements both got
    blocked from "met" (_critic_blocks_met) with near-identical `reason` text,
    that's a blanket/boilerplate objection, not a per-row grounded one."""
    reviews = _critic_requirement_reviews_by_id(critic_result)
    candidates = [
        (req_id, str(review.get("reason") or "").strip().lower())
        for req_id, review in reviews.items()
        if _critic_blocks_met(review)
    ]
    contaminated: set[str] = set()
    for i in range(len(candidates)):
        req_id_i, text_i = candidates[i]
        if req_id_i in contaminated or not text_i:
            continue
        for j in range(i + 1, len(candidates)):
            req_id_j, text_j = candidates[j]
            if req_id_j in contaminated or not text_j:
                continue
            if SequenceMatcher(None, text_i, text_j).ratio() >= _DUPLICATE_EVIDENCE_SIMILARITY_THRESHOLD:
                contaminated.add(req_id_j)
    return contaminated


def _ground_assessed_requirements_against_critic(
    assessments: List[Dict[str, Any]],
    critic_result: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], bool]:
    requirement_reviews = _critic_requirement_reviews_by_id(critic_result)
    contaminated_ids = _find_contaminated_requirement_ids(critic_result)
    met_rejection_contaminated_ids = _find_met_rejection_contaminated_ids(critic_result)
    grounded: List[Dict[str, Any]] = []
    corrected = False
    for assessment in assessments:
        item = dict(assessment)
        req_id = str(item.get("requirement_id", "")).strip().lower()
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}:
            item["verdict"] = VERDICT_NA
            item.setdefault("summary", "Mediator omitted a stable verdict; preserving conservative applicability-only fallback.")
            item.setdefault("reasoning", item["summary"])
            verdict = VERDICT_NA
        review = requirement_reviews.get(req_id)
        hunter_verdict = _normalized_requirement_verdict(item.get("hunter_verdict"))
        critic_verdict = ""
        mediator_has_judged = _mediator_has_judged(item)
        if review:
            hunter_verdict = hunter_verdict or _normalized_requirement_verdict(review.get("hunter_verdict"))
            critic_verdict = _normalized_requirement_verdict(review.get("critic_verdict"))
            if critic_verdict:
                item["critic_verdict"] = critic_verdict
            if req_id in contaminated_ids and critic_verdict == VERDICT_NOT_MET:
                # This row's "not_met" evidence duplicated another requirement's
                # finding in the same batched Critic response — a sign the model
                # reused one requirement's absence finding as a template rather
                # than independently grounding this one. Don't trust it.
                item["verdict"] = hunter_verdict or VERDICT_NA
                item["summary"] = (
                    "Critic's 'not_met' evidence for this requirement duplicated "
                    "another requirement's finding in the same batch; treating as "
                    "unsupported and reverting to the Hunter's opening verdict."
                )
                item["reasoning"] = item["summary"]
                item["verdict_policy_source"] = "diagram_not_met_contaminated_evidence"
                corrected = True
            elif critic_verdict == VERDICT_NOT_MET and not _has_scope_evidence(review):
                # A "not_met" claim still needs explicit scope-establishing
                # evidence from the raw image; otherwise it is not assessable.
                item["verdict"] = VERDICT_NA
                item["summary"] = (
                    "Critic's 'not_met' verdict named no scope-establishing "
                    "evidence; downgraded to 'na'."
                )
                item["reasoning"] = item["summary"]
                item["verdict_policy_source"] = "diagram_not_met_without_scope_evidence"
                corrected = True
            elif critic_verdict == VERDICT_NOT_MET and not _has_absence_evidence(review):
                item["verdict"] = VERDICT_NA
                item["summary"] = (
                    "Critic's 'not_met' verdict named no concrete absence or "
                    "contradiction evidence; downgraded to 'na'."
                )
                item["reasoning"] = item["summary"]
                item["verdict_policy_source"] = "diagram_not_met_without_absence_evidence"
                corrected = True
            elif req_id in met_rejection_contaminated_ids and verdict == VERDICT_MET and _critic_blocks_met(review):
                # This row's met-rejection reasoning duplicated another
                # requirement's objection in the same batch — a blanket
                # boilerplate objection, not one grounded in this row's own
                # evidence. Don't trust it; keep the Hunter's "met".
                item["summary"] = (
                    "Critic's objection to this 'met' verdict duplicated another "
                    "requirement's reasoning in the same batch; treating as "
                    "unsupported and preserving the Hunter's opening verdict."
                )
                item["reasoning"] = item["summary"]
                item["verdict_policy_source"] = "diagram_met_rejection_contaminated_evidence"
            elif not mediator_has_judged and verdict == VERDICT_MET and _critic_blocks_met(review):
                if _critic_prefers_not_met(review) and _critic_not_met_is_grounded(review):
                    item["verdict"] = VERDICT_NOT_MET
                    explanation = str(review.get("reason") or "").strip() or (
                        "Critic verification checks showed the governed scope but "
                        "did not support a 'met' verdict."
                    )
                    item["summary"] = explanation
                    item["reasoning"] = explanation
                    item["verdict_policy_source"] = "diagram_met_rejected_by_critic_verifier"
                else:
                    item["verdict"] = VERDICT_NA
                    explanation = str(review.get("reason") or "").strip() or (
                        "Critic verification checks did not support a 'met' verdict; "
                        "downgraded to 'na'."
                    )
                    item["summary"] = explanation
                    item["reasoning"] = explanation
                    item["verdict_policy_source"] = "diagram_met_not_supported_by_critic_verifier"
                corrected = True
            elif not mediator_has_judged and critic_verdict in {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA} and critic_verdict != verdict:
                item["verdict"] = critic_verdict
                explanation = str(review.get("reason") or review.get("summary") or "").strip()
                if not explanation:
                    explanation = f"Corrected to '{critic_verdict}' based on critic re-examination."
                item["summary"] = explanation
                item["reasoning"] = explanation
                corrected = True
            elif mediator_has_judged and verdict == VERDICT_NOT_MET and not _has_scope_evidence(review):
                item["verdict"] = VERDICT_NA
                item["summary"] = (
                    "Mediator selected 'not_met' but the Critic named no "
                    "scope-establishing evidence; downgraded to 'na'."
                )
                item["reasoning"] = item["summary"]
                item["verdict_policy_source"] = "diagram_mediator_not_met_without_scope_evidence"
                corrected = True
            elif mediator_has_judged and verdict == VERDICT_NOT_MET and not _has_absence_evidence(review):
                item["verdict"] = VERDICT_NA
                item["summary"] = (
                    "Mediator selected 'not_met' but the Critic named no "
                    "concrete absence or contradiction evidence; downgraded to 'na'."
                )
                item["reasoning"] = item["summary"]
                item["verdict_policy_source"] = "diagram_mediator_not_met_without_absence_evidence"
                corrected = True
            elif review.get("reason") and not item.get("summary"):
                item["summary"] = str(review["reason"]).strip()
                item["reasoning"] = item["summary"]
        if hunter_verdict:
            item["hunter_verdict"] = hunter_verdict
        item["final_decision_source"] = _determine_final_decision_source(
            item,
            hunter_verdict=hunter_verdict,
            critic_verdict=critic_verdict,
            mediator_has_judged=mediator_has_judged,
        )
        grounded.append(item)
    return grounded, corrected


def _critic_has_requirement_reviews(critic_result: Dict[str, Any]) -> bool:
    return bool(_critic_requirement_reviews_by_id(critic_result))


def _critic_requirement_reviews_by_id(critic_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    reviews: Dict[str, Dict[str, Any]] = {}
    for review in critic_result.get("requirement_reviews") or []:
        if not isinstance(review, dict):
            continue
        req_id = str(review.get("requirement_id", "")).strip().lower()
        if not req_id:
            continue
        reviews[req_id] = dict(review)

    for review in critic_result.get("validated_requirements") or []:
        if not isinstance(review, dict):
            continue
        req_id = str(review.get("requirement_id", "")).strip().lower()
        if not req_id or req_id in reviews:
            continue
        reviews[req_id] = {
            "requirement_id": review.get("requirement_id"),
            "critic_verdict": review.get("verdict"),
            "disposition": "uphold",
            "reason": review.get("reason", ""),
        }

    for review in critic_result.get("invalidated_requirements") or []:
        if not isinstance(review, dict):
            continue
        req_id = str(review.get("requirement_id", "")).strip().lower()
        if not req_id:
            continue
        if req_id in reviews:
            continue
        legacy_verdict = str(review.get("verdict", "")).strip().lower()
        reviews[req_id] = {
            "requirement_id": review.get("requirement_id"),
            "critic_verdict": review.get("corrected_verdict")
            or (VERDICT_NA if legacy_verdict == VERDICT_MET else review.get("verdict")),
            "disposition": "overturn",
            "reason": review.get("reason", ""),
        }
    return reviews


def _merge_assessed_requirements(
    hunter_assessments: List[Dict[str, Any]],
    mediator_assessments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []
    for source in (hunter_assessments or []):
        if not isinstance(source, dict):
            continue
        req_id = str(source.get("requirement_id", "")).strip()
        if not req_id:
            continue
        ordered_ids.append(req_id)
        merged_item = dict(source)
        hunter_verdict = _normalized_requirement_verdict(source.get("verdict"))
        if hunter_verdict:
            merged_item.setdefault("hunter_verdict", hunter_verdict)
        merged[req_id] = merged_item
    for source in (mediator_assessments or []):
        if not isinstance(source, dict):
            continue
        req_id = str(source.get("requirement_id", "")).strip()
        if not req_id:
            continue
        if req_id not in merged:
            ordered_ids.append(req_id)
            merged[req_id] = {}
        updated = dict(merged[req_id])
        for key, value in source.items():
            if value in (None, "", []):
                continue
            updated[key] = value
        merged[req_id] = updated
    return [merged[req_id] for req_id in ordered_ids if req_id in merged]


def _worst_case_diagram_verdict(assessments: List[Dict[str, Any]]) -> str:
    verdicts = [
        str(assessment.get("verdict", "")).strip().lower()
        for assessment in assessments
        if str(assessment.get("verdict", "")).strip().lower() in {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}
    ]
    if not verdicts:
        return VERDICT_NA
    if VERDICT_NOT_MET in verdicts:
        return VERDICT_NOT_MET
    if VERDICT_MET in verdicts:
        return VERDICT_MET
    return VERDICT_NA


def _normalize_diagram_scope_verdict(value: Any) -> str:
    normalized = str(value or "uncertain").strip().lower()
    return normalized if normalized in _DIAGRAM_SCOPE_VALUES else "uncertain"


def _force_assessment_verdicts_na(
    assessments: List[Dict[str, Any]],
    *,
    default_summary: str,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for assessment in assessments:
        item = dict(assessment or {})
        item["verdict"] = VERDICT_NA
        if not item.get("summary") and item.get("reasoning"):
            item["summary"] = item.get("reasoning")
        if not item.get("summary") and item.get("visual_evidence"):
            item["summary"] = item.get("visual_evidence")
        item.setdefault("summary", default_summary)
        item.setdefault("reasoning", item["summary"])
        normalized.append(item)
    return normalized

__all__ = [
    "DiagramInput",
    "DiagramDebateOutput",
    "DiagramDebateService",
]


def __getattr__(name: str) -> Any:
    if name == "DiagramDebateService":
        from sdr.apps.ai.engine.debate.diagram_debate_service import DiagramDebateService
        return DiagramDebateService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

