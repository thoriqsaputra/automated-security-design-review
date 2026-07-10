from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from sdr.core.config import settings
from sdr.apps.ai.agents.vision import (
    DiagramInput,
    DiagramDebateOutput,
    VisionHunterAgent,
    VisionCriticAgent,
    VisionMediatorAgent,
    _format_requirements_compact,
    _format_requirements_with_hints,
    _apply_diagram_evidence_policy,
    _calibrate_diagram_confidence,
    VERDICT_MET,
    VERDICT_NA,
    VERDICT_NOT_MET,
)
from sdr.apps.ai.prompts.agents import (
    VISION_HUNTER_SYSTEM_PROMPT,
    VISION_HUNTER_REBUTTAL_SYSTEM_PROMPT,
    VISION_CRITIC_BLIND_SYSTEM_PROMPT,
    VISION_CRITIC_DEBATE_SYSTEM_PROMPT,
    VISION_MEDIATOR_DEBATE_SYSTEM_PROMPT,
    build_vision_hunter_prompt,
    build_vision_hunter_rebuttal_prompt,
    build_vision_critic_blind_prompt,
    build_vision_critic_debate_prompt,
    build_vision_mediator_debate_prompt,
)
class DiagramDebateService:
    """
    Orchestrates Hunter→Critic→Mediator debate for a single diagram.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._hunter_agent = VisionHunterAgent()
        self._critic_agent = VisionCriticAgent()
        self._mediator_agent = VisionMediatorAgent()
        self._skip_mediator_on_uphold = bool(
            getattr(settings, "AI_VISION_SKIP_MEDIATOR_ON_UPHOLD", True)
        )
        self._debate_batch_size = max(
            1, int(getattr(settings, "AI_VISION_DEBATE_REQUIREMENT_BATCH_SIZE", 10))
        )
        self._batch_retry_limit = max(
            0, int(getattr(settings, "AI_VISION_DEBATE_BATCH_RETRY_LIMIT", 1))
        )
        self._debate_batch_max_concurrency = max(
            1, int(getattr(settings, "AI_VISION_DEBATE_BATCH_MAX_CONCURRENCY", 6))
        )
        self._rebuttal_batch_max_concurrency = max(
            1, int(getattr(settings, "AI_VISION_DEBATE_REBUTTAL_MAX_CONCURRENCY", 6))
        )

    @staticmethod
    def _requirement_id(requirement: Any) -> str:
        return str(getattr(requirement, "stable_key", "") or "").strip()

    @classmethod
    def _batch_requirements(cls, requirements: List[Any], batch_size: int) -> List[List[Any]]:
        if batch_size <= 0 or len(requirements) <= batch_size:
            return [list(requirements)]
        return [requirements[i:i + batch_size] for i in range(0, len(requirements), batch_size)]

    @classmethod
    def _subset_hunter_result(cls, hunter_result: Dict[str, Any], requirements: List[Any]) -> Dict[str, Any]:
        expected_ids = {cls._requirement_id(req) for req in requirements if cls._requirement_id(req)}
        subset = dict(hunter_result or {})
        subset["requirement_assessments"] = [
            dict(item)
            for item in hunter_result.get("requirement_assessments") or []
            if isinstance(item, dict) and str(item.get("requirement_id", "")).strip() in expected_ids
        ]
        return subset

    def _run_parallel_batches(
        self,
        *,
        items: List[Any],
        max_workers: int,
        work_fn: Callable[[int, Any], Dict[str, Any]],
        cancel_check=None,
    ) -> List[Dict[str, Any]]:
        if not items:
            return []
        if callable(cancel_check):
            try:
                if cancel_check():
                    return []
            except Exception:
                self.logger.warning("DiagramDebateService: cancel_check() raised before batch scheduling", exc_info=True)
        if len(items) == 1 or max_workers <= 1:
            return [work_fn(index, item) for index, item in enumerate(items, start=1)]

        ordered: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
            future_to_index = {
                executor.submit(work_fn, index, item): index
                for index, item in enumerate(items, start=1)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                if callable(cancel_check):
                    try:
                        if cancel_check():
                            continue
                    except Exception:
                        self.logger.warning(
                            "DiagramDebateService: cancel_check() raised while collecting parallel batches",
                            exc_info=True,
                        )
                ordered[index] = future.result()
        return [ordered[index] for index in sorted(ordered)]

    @classmethod
    def _review_state(cls, review: Dict[str, Any]) -> str:
        raw = str(review.get("review_state", "")).strip().lower()
        if raw:
            return raw
        hunter_verdict = str(review.get("hunter_verdict", "")).strip().lower()
        critic_verdict = str(review.get("critic_verdict", "")).strip().lower()
        if critic_verdict and hunter_verdict and critic_verdict != hunter_verdict:
            return "critic_changes_verdict"
        if critic_verdict and hunter_verdict and critic_verdict == hunter_verdict:
            return "uphold_same_verdict"
        return "critic_insufficient_evidence"

    @classmethod
    def _normalized_critic_review(
        cls,
        *,
        requirement: Any,
        hunter_item: Dict[str, Any] | None,
        review: Dict[str, Any] | None,
        fallback_reason: str = "",
    ) -> Dict[str, Any]:
        requirement_id = cls._requirement_id(requirement)
        hunter_item = dict(hunter_item or {})
        review = dict(review or {})
        hunter_verdict = str(
            review.get("hunter_verdict")
            or hunter_item.get("verdict")
            or VERDICT_NA
        ).strip().lower()
        critic_verdict = str(review.get("critic_verdict") or "").strip().lower() or VERDICT_NA
        scope_evidence = review.get("scope_evidence") or ""
        if not scope_evidence and critic_verdict == VERDICT_NOT_MET:
            scope_evidence = (
                hunter_item.get("strongest_scope_evidence")
                or hunter_item.get("visual_evidence")
                or ""
            )
        normalized = {
            "requirement_id": requirement_id,
            "hunter_verdict": hunter_verdict,
            "critic_verdict": critic_verdict,
            "disposition": review.get("disposition") or (
                "overturn" if critic_verdict != hunter_verdict else "uphold"
            ),
            "reason": review.get("reason") or fallback_reason,
            "supports_met": review.get("supports_met"),
            "supports_not_met": review.get("supports_not_met"),
            "failure_mode": review.get("failure_mode") or "none",
            "verification_checks": list(review.get("verification_checks") or []),
            "scope_evidence": scope_evidence,
            "absence_evidence": review.get("absence_evidence") or "",
            "compound_status": review.get("compound_status") or "single",
            "compound_subelements_checked": list(review.get("compound_subelements_checked") or []),
            "prosecution_case": review.get("prosecution_case") or "",
            "admitted_evidence_for_met": list(review.get("admitted_evidence_for_met") or []),
            "admitted_evidence_for_not_met": list(review.get("admitted_evidence_for_not_met") or []),
            "rejected_hunter_claims": list(review.get("rejected_hunter_claims") or []),
            "cross_examination_questions": list(review.get("cross_examination_questions") or []),
            "evidence_quality": review.get("evidence_quality") or "missing",
            "review_state": cls._review_state(review),
        }
        return normalized

    @classmethod
    def _fallback_critic_review(
        cls,
        *,
        requirement: Any,
        hunter_item: Dict[str, Any] | None,
        reason: str,
    ) -> Dict[str, Any]:
        review = cls._normalized_critic_review(
            requirement=requirement,
            hunter_item=hunter_item,
            review={
                "critic_verdict": VERDICT_NA,
                "disposition": "incomplete",
                "reason": reason,
                "review_state": "fallback_incomplete",
                "failure_mode": "missing_scope",
            },
            fallback_reason=reason,
        )
        review["supports_met"] = False
        review["supports_not_met"] = False
        return review

    @classmethod
    def _normalize_critic_batch_result(
        cls,
        *,
        raw_result: Dict[str, Any] | None,
        requirements: List[Any],
        hunter_batch: Dict[str, Any],
        fallback_reason: str,
    ) -> tuple[Dict[str, Any], bool, int]:
        raw_result = dict(raw_result or {})
        expected_ids = [cls._requirement_id(req) for req in requirements if cls._requirement_id(req)]
        hunter_items = {
            str(item.get("requirement_id", "")).strip(): dict(item)
            for item in hunter_batch.get("requirement_assessments") or []
            if isinstance(item, dict) and str(item.get("requirement_id", "")).strip()
        }
        parsed_reviews: Dict[str, Dict[str, Any]] = {}
        for review in raw_result.get("requirement_reviews") or []:
            if not isinstance(review, dict):
                continue
            requirement_id = str(review.get("requirement_id", "")).strip()
            if not requirement_id or requirement_id not in expected_ids:
                continue
            matching_req = next(req for req in requirements if cls._requirement_id(req) == requirement_id)
            parsed_reviews[requirement_id] = cls._normalized_critic_review(
                requirement=matching_req,
                hunter_item=hunter_items.get(requirement_id),
                review=review,
            )
        for review in raw_result.get("validated_requirements") or []:
            if not isinstance(review, dict):
                continue
            requirement_id = str(review.get("requirement_id", "")).strip()
            if (
                not requirement_id
                or requirement_id not in expected_ids
                or requirement_id in parsed_reviews
            ):
                continue
            matching_req = next(req for req in requirements if cls._requirement_id(req) == requirement_id)
            parsed_reviews[requirement_id] = cls._normalized_critic_review(
                requirement=matching_req,
                hunter_item=hunter_items.get(requirement_id),
                review={
                    "requirement_id": requirement_id,
                    "hunter_verdict": hunter_items.get(requirement_id, {}).get("verdict"),
                    "critic_verdict": review.get("verdict"),
                    "disposition": "uphold",
                    "reason": review.get("reason") or "Critic upheld the Hunter verdict.",
                    "review_state": "uphold_same_verdict",
                },
            )
        for review in raw_result.get("invalidated_requirements") or []:
            if not isinstance(review, dict):
                continue
            requirement_id = str(review.get("requirement_id", "")).strip()
            if (
                not requirement_id
                or requirement_id not in expected_ids
                or requirement_id in parsed_reviews
            ):
                continue
            matching_req = next(req for req in requirements if cls._requirement_id(req) == requirement_id)
            corrected_verdict = review.get("corrected_verdict")
            legacy_verdict = str(review.get("verdict", "")).strip().lower()
            if not corrected_verdict:
                corrected_verdict = VERDICT_NA if legacy_verdict == VERDICT_MET else review.get("verdict")
            parsed_reviews[requirement_id] = cls._normalized_critic_review(
                requirement=matching_req,
                hunter_item=hunter_items.get(requirement_id),
                review={
                    "requirement_id": requirement_id,
                    "hunter_verdict": hunter_items.get(requirement_id, {}).get("verdict"),
                    "critic_verdict": corrected_verdict,
                    "disposition": "overturn",
                    "reason": review.get("reason") or "Critic invalidated the Hunter verdict.",
                    "review_state": "critic_changes_verdict",
                },
            )

        missing_ids = [req_id for req_id in expected_ids if req_id not in parsed_reviews]
        complete = not missing_ids
        reviews = list(parsed_reviews.values())
        normalized = {
            "diagram_scope_verdict": cls._normalize_scope(raw_result.get("diagram_scope_verdict")),
            "diagram_scope_reasoning": raw_result.get("diagram_scope_reasoning")
            or hunter_batch.get("diagram_scope_reasoning")
            or "",
            "outcome": str(raw_result.get("outcome", "uphold")).strip().lower() or "uphold",
            "requirement_reviews": reviews,
            "validated_requirements": [
                {
                    "requirement_id": review["requirement_id"],
                    "verdict": review["critic_verdict"],
                }
                for review in reviews
                if review["critic_verdict"] == review["hunter_verdict"]
            ],
            "invalidated_requirements": [
                {
                    "requirement_id": review["requirement_id"],
                    "verdict": review["hunter_verdict"],
                    "corrected_verdict": review["critic_verdict"],
                    "reason": review.get("reason", ""),
                }
                for review in reviews
                if review["critic_verdict"] != review["hunter_verdict"]
            ],
            "hallucinated_claims": list(raw_result.get("hallucinated_claims") or []),
            "reasoning": raw_result.get("reasoning") or "",
        }
        if not complete:
            normalized["missing_requirement_ids"] = missing_ids
        return normalized, complete, len(missing_ids)

    @classmethod
    def _ensure_complete_critic_batch_result(
        cls,
        *,
        normalized_result: Dict[str, Any],
        requirements: List[Any],
        hunter_batch: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        by_id = {
            str(review.get("requirement_id", "")).strip(): dict(review)
            for review in normalized_result.get("requirement_reviews") or []
            if isinstance(review, dict) and str(review.get("requirement_id", "")).strip()
        }
        hunter_items = {
            str(item.get("requirement_id", "")).strip(): dict(item)
            for item in hunter_batch.get("requirement_assessments") or []
            if isinstance(item, dict) and str(item.get("requirement_id", "")).strip()
        }
        fallback_count = 0
        for requirement in requirements:
            requirement_id = cls._requirement_id(requirement)
            if requirement_id and requirement_id not in by_id:
                by_id[requirement_id] = cls._fallback_critic_review(
                    requirement=requirement,
                    hunter_item=hunter_items.get(requirement_id),
                    reason=reason,
                )
                fallback_count += 1
        ordered_reviews = [
            by_id[cls._requirement_id(requirement)]
            for requirement in requirements
            if cls._requirement_id(requirement) in by_id
        ]
        normalized_result = dict(normalized_result)
        normalized_result["requirement_reviews"] = ordered_reviews
        normalized_result["validated_requirements"] = [
            {"requirement_id": review["requirement_id"], "verdict": review["critic_verdict"]}
            for review in ordered_reviews
            if review["critic_verdict"] == review["hunter_verdict"]
        ]
        normalized_result["invalidated_requirements"] = [
            {
                "requirement_id": review["requirement_id"],
                "verdict": review["hunter_verdict"],
                "corrected_verdict": review["critic_verdict"],
                "reason": review.get("reason", ""),
            }
            for review in ordered_reviews
            if review["critic_verdict"] != review["hunter_verdict"]
        ]
        normalized_result.setdefault("batch_diagnostics", {})
        normalized_result["batch_diagnostics"]["fallback_rows"] = fallback_count
        return normalized_result

    @classmethod
    def _merge_critic_batches(cls, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "diagram_scope_verdict": "uncertain",
            "diagram_scope_reasoning": "",
            "outcome": "uphold",
            "requirement_reviews": [],
            "validated_requirements": [],
            "invalidated_requirements": [],
            "hallucinated_claims": [],
            "reasoning": "",
            "batch_diagnostics": {
                "batches": len(batches),
                "retry_batches": 0,
                "fallback_rows": 0,
            },
        }
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            scope = cls._normalize_scope(batch.get("diagram_scope_verdict"))
            if merged["diagram_scope_verdict"] == "uncertain" and scope != "uncertain":
                merged["diagram_scope_verdict"] = scope
            if not merged["diagram_scope_reasoning"] and batch.get("diagram_scope_reasoning"):
                merged["diagram_scope_reasoning"] = batch.get("diagram_scope_reasoning")
            merged["requirement_reviews"].extend(batch.get("requirement_reviews") or [])
            merged["validated_requirements"].extend(batch.get("validated_requirements") or [])
            merged["invalidated_requirements"].extend(batch.get("invalidated_requirements") or [])
            merged["hallucinated_claims"].extend(batch.get("hallucinated_claims") or [])
            if batch.get("reasoning"):
                merged["reasoning"] = (merged["reasoning"] + "\n" + str(batch.get("reasoning"))).strip()
            diagnostics = batch.get("batch_diagnostics") or {}
            merged["batch_diagnostics"]["retry_batches"] += int(diagnostics.get("retry_used", 0))
            merged["batch_diagnostics"]["fallback_rows"] += int(diagnostics.get("fallback_rows", 0))
        if any(
            cls._review_state(review) in {"critic_changes_verdict", "fallback_incomplete", "critic_insufficient_evidence"}
            or str(review.get("critic_verdict", "")).strip().lower() != str(review.get("hunter_verdict", "")).strip().lower()
            for review in merged["requirement_reviews"]
            if isinstance(review, dict)
        ):
            merged["outcome"] = "overturn"
        return merged

    @classmethod
    def _subset_critic_result(cls, critic_result: Dict[str, Any], requirements: List[Any]) -> Dict[str, Any]:
        expected_ids = {cls._requirement_id(req) for req in requirements if cls._requirement_id(req)}
        subset = {
            key: value
            for key, value in dict(critic_result or {}).items()
            if key not in {"requirement_reviews", "validated_requirements", "invalidated_requirements"}
        }
        subset["requirement_reviews"] = [
            dict(review)
            for review in critic_result.get("requirement_reviews") or []
            if isinstance(review, dict) and str(review.get("requirement_id", "")).strip() in expected_ids
        ]
        subset["validated_requirements"] = [
            dict(review)
            for review in critic_result.get("validated_requirements") or []
            if isinstance(review, dict) and str(review.get("requirement_id", "")).strip() in expected_ids
        ]
        subset["invalidated_requirements"] = [
            dict(review)
            for review in critic_result.get("invalidated_requirements") or []
            if isinstance(review, dict) and str(review.get("requirement_id", "")).strip() in expected_ids
        ]
        if any(
            cls._review_state(review) in {"critic_changes_verdict", "fallback_incomplete", "critic_insufficient_evidence"}
            or str(review.get("critic_verdict", "")).strip().lower() != str(review.get("hunter_verdict", "")).strip().lower()
            for review in subset["requirement_reviews"]
        ):
            subset["outcome"] = "overturn"
        else:
            subset["outcome"] = "uphold"
        return subset

    @classmethod
    def _subset_hunter_rebuttal_result(cls, rebuttal_result: Dict[str, Any], requirements: List[Any]) -> Dict[str, Any]:
        expected_ids = {cls._requirement_id(req) for req in requirements if cls._requirement_id(req)}
        subset = dict(rebuttal_result or {})
        subset["rebuttal_requirements"] = [
            dict(item)
            for item in rebuttal_result.get("rebuttal_requirements") or []
            if isinstance(item, dict) and str(item.get("requirement_id", "")).strip() in expected_ids
        ]
        return subset

    @classmethod
    def _ensure_complete_mediator_batch_result(
        cls,
        *,
        mediator_result: Dict[str, Any],
        requirements: List[Any],
        hunter_batch: Dict[str, Any],
        critic_batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        mediator_result = dict(mediator_result or {})
        by_id = {
            str(item.get("requirement_id", "")).strip(): dict(item)
            for item in mediator_result.get("assessed_requirements") or []
            if isinstance(item, dict) and str(item.get("requirement_id", "")).strip()
        }
        hunter_items = {
            str(item.get("requirement_id", "")).strip(): dict(item)
            for item in hunter_batch.get("requirement_assessments") or []
            if isinstance(item, dict) and str(item.get("requirement_id", "")).strip()
        }
        critic_by_id = {
            str(review.get("requirement_id", "")).strip(): dict(review)
            for review in critic_batch.get("requirement_reviews") or []
            if isinstance(review, dict) and str(review.get("requirement_id", "")).strip()
        }
        independent_rows = 0
        for requirement in requirements:
            requirement_id = cls._requirement_id(requirement)
            if requirement_id and requirement_id not in by_id:
                hunter_item = hunter_items.get(requirement_id, {})
                critic_review = critic_by_id.get(requirement_id, {})
                fallback_verdict = str(
                    hunter_item.get("verdict")
                    if cls._review_state(critic_review) != "fallback_incomplete"
                    else VERDICT_NA
                ).strip().lower() or VERDICT_NA
                by_id[requirement_id] = {
                    "requirement_id": requirement_id,
                    "verdict": fallback_verdict,
                    "resolution_basis": "same_verdict_after_cross_exam"
                    if fallback_verdict == str(hunter_item.get("verdict", "")).strip().lower()
                    else "mediator_tiebreak",
                    "winning_side": "hunter" if fallback_verdict == str(hunter_item.get("verdict", "")).strip().lower() else "split",
                    "judge_reason": (
                        "Mediator batch omitted this requirement; synthesized a conservative fallback row."
                    ),
                    "summary": (
                        "Mediator batch omitted this requirement; preserved the opening claim."
                        if fallback_verdict == str(hunter_item.get("verdict", "")).strip().lower()
                        else "Mediator batch omitted this requirement and critic coverage was incomplete; downgraded to na."
                    ),
                }
                independent_rows += 1
        mediator_result["assessed_requirements"] = [
            by_id[cls._requirement_id(requirement)]
            for requirement in requirements
            if cls._requirement_id(requirement) in by_id
        ]
        mediator_result.setdefault("batch_diagnostics", {})
        mediator_result["batch_diagnostics"]["independent_rows"] = independent_rows
        return mediator_result

    @classmethod
    def _merge_mediator_batches(
        cls,
        batches: List[Dict[str, Any]],
        *,
        hunter_result: Dict[str, Any],
        critic_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "diagram_scope_verdict": "uncertain",
            "diagram_scope_reasoning": "",
            "final_verdict": VERDICT_NA,
            "confidence": 0.0,
            "finding_description": "",
            "recommendation": None,
            "assessed_requirements": [],
            "reasoning": "",
            "batch_diagnostics": {
                "batches": len(batches),
                "independent_rows": 0,
            },
        }
        confidences: List[float] = []
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            scope = cls._normalize_scope(batch.get("diagram_scope_verdict"))
            if merged["diagram_scope_verdict"] == "uncertain" and scope != "uncertain":
                merged["diagram_scope_verdict"] = scope
            if not merged["diagram_scope_reasoning"] and batch.get("diagram_scope_reasoning"):
                merged["diagram_scope_reasoning"] = batch.get("diagram_scope_reasoning")
            merged["assessed_requirements"].extend(batch.get("assessed_requirements") or [])
            if batch.get("finding_description"):
                merged["finding_description"] = (merged["finding_description"] + "\n" + str(batch.get("finding_description"))).strip()
            if batch.get("reasoning"):
                merged["reasoning"] = (merged["reasoning"] + "\n" + str(batch.get("reasoning"))).strip()
            if batch.get("recommendation"):
                merged["recommendation"] = batch.get("recommendation")
            confidences.append(float(batch.get("confidence", 0.0) or 0.0))
            diagnostics = batch.get("batch_diagnostics") or {}
            merged["batch_diagnostics"]["independent_rows"] += int(diagnostics.get("independent_rows", 0))
        if confidences:
            merged["confidence"] = round(sum(confidences) / len(confidences), 2)
        merged["final_verdict"] = VERDICT_NA
        merged = _apply_diagram_evidence_policy(merged, critic_result, hunter_result)
        return merged

    def run_diagram_debate(
        self,
        *,
        diagram: DiagramInput,
        requirements: List[Any],
        tsd_context: str = "",
        cancel_check=None,
        skip_mediator_on_uphold: Optional[bool] = None,
        agent_started_handler: Optional[Callable[[str], None]] = None,
        agent_completed_handler: Optional[Callable[..., None]] = None,
    ) -> DiagramDebateOutput:
        if skip_mediator_on_uphold is None:
            skip_mediator_on_uphold = self._skip_mediator_on_uphold
        output = DiagramDebateOutput(
            diagram=diagram,
            requirements=requirements,
        )

        if callable(cancel_check):
            try:
                if cancel_check():
                    output.error = "Analysis was cancelled by user."
                    self.logger.warning(
                        "DiagramDebateService: cancellation detected diagram_id=%s before start",
                        diagram.diagram_id,
                    )
                    return output
            except Exception:
                self.logger.warning(
                    "DiagramDebateService: cancel_check() raised for diagram_id=%s — treating as not cancelled",
                    diagram.diagram_id,
                    exc_info=True,
                )

        if not requirements:
            output.error = "No diagram requirements provided — cannot ground debate."
            self.logger.warning(
                "DiagramDebateService: skipping diagram_id=%s — no requirements",
                diagram.diagram_id,
            )
            return output

        try:
            image_bytes = diagram.decode_image_bytes()
        except ValueError as exc:
            output.error = str(exc)
            return output

        min_bytes = int(getattr(settings, "AI_VISION_MIN_DIAGRAM_BYTES", 512))
        if len(image_bytes) < min_bytes:
            output.error = (
                f"Image too small ({len(image_bytes)} bytes < {min_bytes}) — "
                f"likely icon/logo, not architectural diagram."
            )
            return output

        image_payloads: Optional[List[Dict[str, Any]]] = [
            {
                "label": "raw",
                "image_bytes": image_bytes,
                "image_format": diagram.image_format or "png",
            }
        ]

        compact_reqs = _format_requirements_compact(requirements)
        caption = diagram.caption or ""
        surrounding = diagram.surrounding_text or ""
        image_format = diagram.image_format or "png"
        requirement_batches = self._batch_requirements(requirements, self._debate_batch_size)

        self.logger.info(
            "DiagramDebateService: VisionHunter diagram_id=%s requirements=%d",
            diagram.diagram_id,
            len(requirements),
        )
        if agent_started_handler:
            agent_started_handler("hunter")
        hunter_prompt = build_vision_hunter_prompt(
            requirements_text=compact_reqs,
            diagram_caption=caption,
            surrounding_text=surrounding,
            tsd_context=tsd_context[:2000],
        )
        hunter_result = self._hunter_agent.run_multimodal(
            user_prompt=hunter_prompt,
            image_bytes=None,
            image_payloads=image_payloads,
            image_format=image_format,
            system_prompt=VISION_HUNTER_SYSTEM_PROMPT,
            log_context=f"diagram_id={diagram.diagram_id} agent=hunter",
        )
        if callable(cancel_check):
            try:
                if cancel_check():
                    output.error = "Analysis was cancelled by user."
                    self.logger.warning(
                        "DiagramDebateService: cancellation detected diagram_id=%s after hunter",
                        diagram.diagram_id,
                    )
                    return output
            except Exception:
                self.logger.warning(
                    "DiagramDebateService: cancel_check() raised for diagram_id=%s — treating as not cancelled",
                    diagram.diagram_id,
                    exc_info=True,
                )
        if hunter_result is None:
            self.logger.error(
                "DiagramDebateService: VisionHunter failed for diagram_id=%s — "
                "falling back to inconclusive result",
                diagram.diagram_id,
            )
            hunter_result = {
                "overall_verdict": VERDICT_NA,
                "confidence": 0.0,
                "reasoning": "Hunter analysis failed to produce a parseable result; treating as inconclusive.",
                "requirement_assessments": [],
                "diagram_scope_verdict": "uncertain",
                "diagram_scope_reasoning": "Hunter call failed before scope could be assessed.",
            }

        hunter_verdict = str(
            hunter_result.get("overall_verdict", VERDICT_NA)
        ).strip().lower()
        if hunter_verdict not in {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}:
            hunter_result["overall_verdict"] = VERDICT_NA
        hunter_result["diagram_scope_verdict"] = self._normalize_scope(
            hunter_result.get("diagram_scope_verdict")
        )
        output.hunter_result = hunter_result
        if agent_completed_handler:
            agent_completed_handler("hunter", self._reasoning_content(hunter_result))

        self.logger.info(
            "DiagramDebateService: VisionCritic diagram_id=%s batches=%d",
            diagram.diagram_id,
            len(requirement_batches),
        )
        if agent_started_handler:
            agent_started_handler("critic")
        def run_critic_batch(batch_index: int, requirement_batch: List[Any]) -> Dict[str, Any]:
            hunter_batch = self._subset_hunter_result(hunter_result, requirement_batch)
            batch_requirements_text = _format_requirements_with_hints(requirement_batch)

            # Blind independent pass FIRST, before the Critic ever sees the
            # Hunter's claim — a verifier that reasons about a claim already
            # stated to it tends to anchor on it and reproduce the generator's
            # errors, even when explicitly instructed to be skeptical. Forming
            # an independent verdict first, then comparing, removes that
            # anchoring structurally instead of relying on a prompt caveat.
            blind_prompt = build_vision_critic_blind_prompt(
                requirements_with_hints=batch_requirements_text,
                diagram_caption=caption,
                surrounding_text=surrounding,
            )
            raw_blind_result = self._critic_agent.run_multimodal(
                user_prompt=blind_prompt,
                image_bytes=None,
                image_payloads=image_payloads,
                image_format=image_format,
                system_prompt=VISION_CRITIC_BLIND_SYSTEM_PROMPT,
                log_context=(
                    f"diagram_id={diagram.diagram_id} agent=critic_blind "
                    f"batch={batch_index}/{len(requirement_batches)}"
                ),
            )
            blind_result = raw_blind_result or {}
            if not raw_blind_result:
                self.logger.warning(
                    "DiagramDebateService: critic blind pass failed diagram_id=%s batch=%d/%d — "
                    "reconciliation will proceed without an independent anchor",
                    diagram.diagram_id,
                    batch_index,
                    len(requirement_batches),
                )

            raw_critic_result = None
            normalized_critic_batch = None
            complete = False
            retry_used = 0
            for attempt in range(self._batch_retry_limit + 1):
                critic_prompt = build_vision_critic_debate_prompt(
                    requirements_with_hints=batch_requirements_text,
                    hunter_result=hunter_batch,
                    blind_result=blind_result,
                    diagram_caption=caption,
                    surrounding_text=surrounding,
                    completeness_retry=attempt > 0,
                )
                raw_critic_result = self._critic_agent.run_multimodal(
                    user_prompt=critic_prompt,
                    image_bytes=None,
                    image_payloads=image_payloads,
                    image_format=image_format,
                    system_prompt=VISION_CRITIC_DEBATE_SYSTEM_PROMPT,
                    log_context=(
                        f"diagram_id={diagram.diagram_id} agent=critic "
                        f"batch={batch_index}/{len(requirement_batches)} attempt={attempt + 1}"
                    ),
                )
                normalized_critic_batch, complete, _ = self._normalize_critic_batch_result(
                    raw_result=raw_critic_result,
                    requirements=requirement_batch,
                    hunter_batch=hunter_batch,
                    fallback_reason="Critic batch could not be parsed; synthesized conservative fallback review.",
                )
                if complete:
                    break
                retry_used = 1
                self.logger.warning(
                    "DiagramDebateService: critic batch incomplete diagram_id=%s batch=%d/%d attempt=%d",
                    diagram.diagram_id,
                    batch_index,
                    len(requirement_batches),
                    attempt + 1,
                )
            assert normalized_critic_batch is not None
            normalized_critic_batch = self._ensure_complete_critic_batch_result(
                normalized_result=normalized_critic_batch,
                requirements=requirement_batch,
                hunter_batch=hunter_batch,
                reason=(
                    "Critic omitted this requirement after completeness retry; "
                    "synthesized fallback review so mediation can still proceed."
                ),
            )
            normalized_critic_batch.setdefault("batch_diagnostics", {})
            normalized_critic_batch["batch_diagnostics"]["retry_used"] = retry_used
            return normalized_critic_batch
        critic_batches = self._run_parallel_batches(
            items=requirement_batches,
            max_workers=self._debate_batch_max_concurrency,
            work_fn=run_critic_batch,
            cancel_check=cancel_check,
        )
        critic_result = self._merge_critic_batches(critic_batches)

        if callable(cancel_check):
            try:
                if cancel_check():
                    output.error = "Analysis was cancelled by user."
                    self.logger.warning(
                        "DiagramDebateService: cancellation detected diagram_id=%s after critic",
                        diagram.diagram_id,
                    )
                    return output
            except Exception:
                self.logger.warning(
                    "DiagramDebateService: cancel_check() raised for diagram_id=%s — treating as not cancelled",
                    diagram.diagram_id,
                    exc_info=True,
                )

        critic_outcome = str(critic_result.get("outcome", "uphold")).strip().lower()
        if self._critic_has_material_disagreement(hunter_result, critic_result):
            critic_outcome = "overturn"
            critic_result["outcome"] = "overturn"
        elif critic_outcome not in {"uphold", "overturn"}:
            critic_outcome = "uphold"
            critic_result["outcome"] = "uphold"
        critic_result["diagram_scope_verdict"] = self._normalize_scope(
            critic_result.get("diagram_scope_verdict")
        )
        output.critic_result = critic_result
        if agent_completed_handler:
            # Diagrams only have uphold/overturn (no partial) — normalize to the
            # same uppercase vocabulary text findings use so a single frontend
            # filter covers both finding types.
            normalized_outcome = str(critic_result.get("outcome") or "uphold").strip().upper()
            agent_completed_handler(
                "critic",
                self._reasoning_content(critic_result),
                critic_outcome=normalized_outcome,
                requires_rebuttal=normalized_outcome == "OVERTURN",
            )

        hunter_rebuttal_result: Dict[str, Any] = {"rebuttal_requirements": [], "reasoning": ""}
        disputed_requirement_ids = self._disputed_requirement_ids(hunter_result, critic_result)
        if disputed_requirement_ids and self._normalize_scope(critic_result.get("diagram_scope_verdict")) != "non_architecture":
            self.logger.info(
                "DiagramDebateService: VisionHunter rebuttal diagram_id=%s disputed_requirements=%d",
                diagram.diagram_id,
                len(disputed_requirement_ids),
            )
            disputed_batches = [
                requirement_batch
                for requirement_batch in requirement_batches
                if {self._requirement_id(req) for req in requirement_batch}.intersection(disputed_requirement_ids)
            ]

            def run_rebuttal_batch(batch_index: int, requirement_batch: List[Any]) -> Dict[str, Any]:
                batch_ids = {self._requirement_id(req) for req in requirement_batch}
                if not batch_ids.intersection(disputed_requirement_ids):
                    return {"rebuttal_requirements": [], "reasoning": ""}
                hunter_rebuttal_prompt = build_vision_hunter_rebuttal_prompt(
                    hunter_result=self._subset_hunter_result(hunter_result, requirement_batch),
                    critic_result=self._subset_critic_result(critic_result, requirement_batch),
                )
                hunter_rebuttal = self._hunter_agent.run_multimodal(
                    user_prompt=hunter_rebuttal_prompt,
                    image_bytes=None,
                    image_payloads=image_payloads,
                    image_format=image_format,
                    system_prompt=VISION_HUNTER_REBUTTAL_SYSTEM_PROMPT,
                    log_context=(
                        f"diagram_id={diagram.diagram_id} agent=hunter_rebuttal "
                        f"batch={batch_index}/{len(requirement_batches)}"
                    ),
                )
                if hunter_rebuttal is None:
                    return {"rebuttal_requirements": [], "reasoning": ""}
                return {
                    "rebuttal_requirements": hunter_rebuttal.get("rebuttal_requirements") or [],
                    "reasoning": str(hunter_rebuttal.get("reasoning") or "").strip(),
                }
            rebuttal_batches = self._run_parallel_batches(
                items=disputed_batches,
                max_workers=self._rebuttal_batch_max_concurrency,
                work_fn=run_rebuttal_batch,
                cancel_check=cancel_check,
            )
            hunter_rebuttal_result = {
                "rebuttal_requirements": [
                    item
                    for batch in rebuttal_batches
                    for item in (batch.get("rebuttal_requirements") or [])
                ],
                "reasoning": "\n".join(
                    str(batch.get("reasoning") or "").strip()
                    for batch in rebuttal_batches
                    if str(batch.get("reasoning") or "").strip()
                ).strip(),
            }
        output.hunter_rebuttal_result = hunter_rebuttal_result

        self.logger.info(
            "DiagramDebateService: VisionMediator diagram_id=%s critic_outcome=%s",
            diagram.diagram_id,
            critic_outcome,
        )
        if agent_started_handler:
            agent_started_handler("mediator")
        def run_mediator_batch(batch_index: int, requirement_batch: List[Any]) -> Dict[str, Any]:
            hunter_batch = self._subset_hunter_result(hunter_result, requirement_batch)
            critic_batch = self._subset_critic_result(critic_result, requirement_batch)
            rebuttal_batch = self._subset_hunter_rebuttal_result(hunter_rebuttal_result, requirement_batch)
            batch_requirements_text = _format_requirements_with_hints(requirement_batch)
            batch_outcome = str(critic_batch.get("outcome", "uphold")).strip().lower()
            batch_has_fallback = any(
                self._review_state(review) == "fallback_incomplete"
                for review in critic_batch.get("requirement_reviews") or []
                if isinstance(review, dict)
            )
            if skip_mediator_on_uphold and batch_outcome != "overturn" and not batch_has_fallback:
                mediator_batch = {
                    "final_verdict": hunter_batch.get("overall_verdict", VERDICT_NA),
                    "confidence": hunter_batch.get("confidence", 0.5),
                    "finding_description": hunter_batch.get("reasoning", ""),
                    "recommendation": None,
                    "assessed_requirements": hunter_batch.get("requirement_assessments", []),
                    "diagram_scope_verdict": critic_batch.get(
                        "diagram_scope_verdict",
                        hunter_batch.get("diagram_scope_verdict", "uncertain"),
                    ),
                    "diagram_scope_reasoning": critic_batch.get("diagram_scope_reasoning")
                    or hunter_batch.get("diagram_scope_reasoning"),
                    "verdict_policy_source": "critic_upheld_skip_mediator",
                    "hunter_rebuttal_result": rebuttal_batch,
                    "batch_diagnostics": {"independent_rows": 0},
                }
            else:
                mediator_batch = None
                for attempt in range(self._batch_retry_limit + 1):
                    mediator_prompt = build_vision_mediator_debate_prompt(
                        requirements_with_hints=batch_requirements_text,
                        hunter_result=hunter_batch,
                        critic_result=critic_batch,
                        hunter_rebuttal_result=rebuttal_batch,
                        completeness_retry=attempt > 0,
                    )
                    self.logger.info(
                        "DiagramDebateService: Mediator will see the diagram image diagram_id=%s critic_outcome=%s batch=%d/%d",
                        diagram.diagram_id,
                        batch_outcome,
                        batch_index,
                        len(requirement_batches),
                    )
                    mediator_batch = self._mediator_agent.run_multimodal(
                        user_prompt=mediator_prompt,
                        image_bytes=None,
                        image_payloads=image_payloads,
                        image_format=image_format,
                        system_prompt=VISION_MEDIATOR_DEBATE_SYSTEM_PROMPT,
                        log_context=(
                            f"diagram_id={diagram.diagram_id} agent=mediator "
                            f"batch={batch_index}/{len(requirement_batches)} attempt={attempt + 1}"
                        ),
                    )
                    mediator_batch = self._ensure_complete_mediator_batch_result(
                        mediator_result=mediator_batch or {},
                        requirements=requirement_batch,
                        hunter_batch=hunter_batch,
                        critic_batch=critic_batch,
                    )
                    if len(mediator_batch.get("assessed_requirements") or []) == len(requirement_batch):
                        break
                if mediator_batch is None:
                    mediator_batch = {
                        "final_verdict": hunter_batch.get("overall_verdict", VERDICT_NA),
                        "confidence": hunter_batch.get("confidence", 0.5),
                        "finding_description": hunter_batch.get("reasoning", ""),
                        "recommendation": None,
                        "assessed_requirements": hunter_batch.get("requirement_assessments", []),
                        "diagram_scope_verdict": critic_batch.get(
                            "diagram_scope_verdict",
                            hunter_batch.get("diagram_scope_verdict", "uncertain"),
                        ),
                        "diagram_scope_reasoning": critic_batch.get("diagram_scope_reasoning")
                        or hunter_batch.get("diagram_scope_reasoning"),
                    }
            mediator_batch = self._ensure_complete_mediator_batch_result(
                mediator_result=mediator_batch,
                requirements=requirement_batch,
                hunter_batch=hunter_batch,
                critic_batch=critic_batch,
            )
            return mediator_batch
        mediator_batches = self._run_parallel_batches(
            items=requirement_batches,
            max_workers=self._debate_batch_max_concurrency,
            work_fn=run_mediator_batch,
            cancel_check=cancel_check,
        )
        if callable(cancel_check):
            try:
                if cancel_check():
                    output.error = "Analysis was cancelled by user."
                    self.logger.warning(
                        "DiagramDebateService: cancellation detected diagram_id=%s after mediator",
                        diagram.diagram_id,
                    )
                    return output
            except Exception:
                self.logger.warning(
                    "DiagramDebateService: cancel_check() raised for diagram_id=%s — treating as not cancelled",
                    diagram.diagram_id,
                    exc_info=True,
                )
        mediator_result = self._merge_mediator_batches(
            mediator_batches,
            hunter_result=hunter_result,
            critic_result=critic_result,
        )
        mediator_result["hunter_rebuttal_result"] = hunter_rebuttal_result

        final_verdict = str(
            mediator_result.get("final_verdict", VERDICT_NA)
        ).strip().lower()
        if final_verdict not in {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}:
            mediator_result["final_verdict"] = VERDICT_NA
        mediator_result["diagram_scope_verdict"] = self._normalize_scope(
            mediator_result.get("diagram_scope_verdict")
        )

        mediator_result = _calibrate_diagram_confidence(
            mediator_result, hunter_result, critic_result
        )

        output.mediator_result = mediator_result
        if agent_completed_handler and not (skip_mediator_on_uphold and critic_outcome != "overturn"):
            agent_completed_handler("mediator", self._reasoning_content(mediator_result))
        self.logger.info(
            "DiagramDebateService: COMPLETE diagram_id=%s verdict=%s confidence=%.2f",
            diagram.diagram_id,
            mediator_result.get("final_verdict"),
            mediator_result.get("confidence", 0.0),
        )
        return output

    def run_diagram_debate_voted(
        self,
        *,
        diagram: DiagramInput,
        requirements: List[Any],
        tsd_context: str = "",
        votes: int = 3,
        **kwargs: Any,
    ) -> DiagramDebateOutput:
        """
        Runs `run_diagram_debate` `votes` times and returns the run whose
        final_verdict matches the majority — a self-consistency pass to
        suppress per-call LLM non-determinism (confirmed present even at
        temperature=0 with a fixed seed, most likely from OpenRouter's
        automatic provider routing). Ties break toward the run with the
        highest mediator confidence.
        """
        if votes <= 1:
            return self.run_diagram_debate(
                diagram=diagram, requirements=requirements, tsd_context=tsd_context, **kwargs
            )

        outputs: List[DiagramDebateOutput] = []
        for i in range(votes):
            outputs.append(
                self.run_diagram_debate(
                    diagram=diagram, requirements=requirements, tsd_context=tsd_context, **kwargs
                )
            )

        tally: Dict[str, int] = {}
        for output in outputs:
            verdict = str((output.mediator_result or {}).get("final_verdict", VERDICT_NA)).strip().lower()
            tally[verdict] = tally.get(verdict, 0) + 1

        majority_verdict = max(tally.items(), key=lambda kv: kv[1])[0]
        candidates = [
            output for output in outputs
            if str((output.mediator_result or {}).get("final_verdict", VERDICT_NA)).strip().lower() == majority_verdict
        ]
        winner = max(candidates, key=lambda output: float((output.mediator_result or {}).get("confidence", 0.0) or 0.0))

        winner.mediator_result["self_consistency"] = {
            "votes": votes,
            "tally": tally,
            "agreement_rate": round(tally[majority_verdict] / votes, 4),
            # Secondary per-run evidence diagnostics are about evidence quality,
            # not the final verdict — reporting them
            # only from the single winning run would understate their sample
            # size under voting, so callers can aggregate over every run here.
            "all_hunter_results": [output.hunter_result for output in outputs],
            "all_hunter_rebuttals": [output.hunter_rebuttal_result for output in outputs],
            "all_critic_results": [output.critic_result for output in outputs],
        }
        self.logger.info(
            "DiagramDebateService: self-consistency vote diagram_id=%s tally=%s winner=%s",
            diagram.diagram_id,
            tally,
            majority_verdict,
        )
        return winner

    @staticmethod
    def _normalize_scope(value: Any) -> str:
        normalized = str(value or "uncertain").strip().lower()
        if normalized in {"architecture_relevant", "non_architecture", "uncertain"}:
            return normalized
        return "uncertain"

    @staticmethod
    def _reasoning_content(result: Any) -> str:
        """Chain-of-thought first: prefer cot_trace over the final-answer
        logic_summary/reasoning, matching the text debate pipeline's preference."""
        if not isinstance(result, dict):
            return ""
        return str(
            result.get("cot_trace") or result.get("logic_summary") or result.get("reasoning") or ""
        ).strip()

    @staticmethod
    def _disputed_requirement_ids(hunter_result: Dict[str, Any], critic_result: Dict[str, Any]) -> List[str]:
        hunter_by_id: Dict[str, str] = {}
        for item in hunter_result.get("requirement_assessments") or []:
            if not isinstance(item, dict):
                continue
            req_id = str(item.get("requirement_id", "")).strip()
            verdict = str(item.get("verdict", "")).strip().lower()
            if req_id:
                hunter_by_id[req_id] = verdict

        disputed: List[str] = []
        for review in critic_result.get("requirement_reviews") or []:
            if not isinstance(review, dict):
                continue
            req_id = str(review.get("requirement_id", "")).strip()
            if not req_id:
                continue
            hunter_verdict = str(review.get("hunter_verdict") or hunter_by_id.get(req_id, "")).strip().lower()
            critic_verdict = str(review.get("critic_verdict", "")).strip().lower()
            failure_mode = str(review.get("failure_mode", "")).strip().lower()
            supports_met = str(review.get("supports_met", "")).strip().lower()
            review_state = str(review.get("review_state", "")).strip().lower()
            if (
                (hunter_verdict and critic_verdict and hunter_verdict != critic_verdict)
                or failure_mode not in {"", "none"}
                or supports_met == "false"
                or review_state in {"fallback_incomplete", "critic_insufficient_evidence"}
            ):
                disputed.append(req_id)
        return disputed

    @classmethod
    def _critic_has_material_disagreement(cls, hunter_result: Dict[str, Any], critic_result: Dict[str, Any]) -> bool:
        return bool(cls._disputed_requirement_ids(hunter_result, critic_result))

__all__ = ["DiagramDebateService"]
