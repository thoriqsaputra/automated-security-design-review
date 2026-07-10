from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

from sdr.core.config import settings
from sdr.apps.ai.agents.vision import (
    DiagramInput,
    DiagramDebateOutput,
    MergedDiagramExtraction,
    VisionExtractorAgent,
    VisionReasonerAgent,
    _format_requirements_with_hints,
    _worst_case_diagram_verdict,
    VERDICT_MET,
    VERDICT_NA,
    VERDICT_NOT_MET,
)
from sdr.apps.ai.prompts.agents import (
    VISION_EXTRACTOR_SYSTEM_PROMPT,
    VISION_REASONER_SYSTEM_PROMPT,
    build_vision_extractor_prompt,
    build_vision_reasoner_prompt,
)

logger = logging.getLogger(__name__)

_COMPONENT_ALIASES = {
    "db": "database",
    "gw": "gateway",
    "lb": "load balancer",
    "svc": "service",
    "auth": "authentication",
    "fe": "frontend",
    "be": "backend",
    "api gw": "api gateway",
}
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_label(text: Any) -> str:
    normalized = str(text or "").strip().lower()
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = " ".join(normalized.split())
    words = [_COMPONENT_ALIASES.get(word, word) for word in normalized.split(" ")]
    return " ".join(words)


def _similarity(a: Any, b: Any) -> float:
    normalized_a, normalized_b = _normalize_label(a), _normalize_label(b)
    char_ratio = SequenceMatcher(None, normalized_a, normalized_b).ratio()
    # Word-order-insensitive component: SequenceMatcher alone scores
    # "API Gateway" vs "Gateway (API)" low because it's character-LCS-based
    # and the words are reordered — token-set overlap catches that case.
    tokens_a, tokens_b = normalized_a.split(), normalized_b.split()
    token_ratio = _jaccard(tokens_a, tokens_b) if (tokens_a or tokens_b) else 0.0
    return max(char_ratio, token_ratio)


def _jaccard(a: Any, b: Any) -> float:
    set_a, set_b = set(a or []), set(b or [])
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class DiagramExtractReasonService:
    """
    Two-stage alternative to DiagramDebateService: Stage 1 extracts a
    structured, self-consistency-merged description of the diagram (multiple
    independent vision passes, deterministically merged — no LLM judgment of
    requirements happens here); Stage 2 reasons over that structured
    extraction as TEXT ONLY (no image) to produce per-requirement verdicts,
    with citations validated deterministically against the extraction rather
    than by a second LLM's opinion.

    This exists to combat "visual blindness" without relying on multi-agent
    debate over raw pixels, which prior evaluation showed gives ~0 net F1/FPR
    improvement over a single-agent baseline (the bottleneck was citation
    grounding against the image, not debate logic). See
    DiagramDebateService for the debate-based approach this runs alongside.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._extractor_agent = VisionExtractorAgent()
        self._reasoner_agent = VisionReasonerAgent()
        self._extraction_votes = max(1, int(getattr(settings, "AI_VISION_EXTRACTION_VOTES", 3)))
        self._merge_threshold = float(getattr(settings, "AI_VISION_EXTRACTION_MERGE_THRESHOLD", 0.5))
        self._fuzzy_match_threshold = float(
            getattr(settings, "AI_VISION_EXTRACTION_FUZZY_MATCH_THRESHOLD", 0.75)
        )
        self._reasoner_batch_size = max(1, int(getattr(settings, "AI_VISION_REASONER_BATCH_SIZE", 10)))
        self._citation_retry_limit = max(0, int(getattr(settings, "AI_VISION_REASONER_CITATION_RETRY_LIMIT", 1)))
        self._full_failure_retry_limit = max(
            0, int(getattr(settings, "AI_VISION_REASONER_FULL_FAILURE_RETRY_LIMIT", 2))
        )
        self._extraction_max_concurrency = max(
            1, int(getattr(settings, "AI_VISION_EXTRACTION_MAX_CONCURRENCY", 3))
        )
        self._reasoner_batch_max_concurrency = max(
            1, int(getattr(settings, "AI_VISION_REASONER_BATCH_MAX_CONCURRENCY", 6))
        )

    @staticmethod
    def _requirement_id(requirement: Any) -> str:
        return str(getattr(requirement, "stable_key", "") or "").strip()

    @classmethod
    def _batch_requirements(cls, requirements: List[Any], batch_size: int) -> List[List[Any]]:
        if batch_size <= 0 or len(requirements) <= batch_size:
            return [list(requirements)]
        return [requirements[i:i + batch_size] for i in range(0, len(requirements), batch_size)]

    def _run_parallel(
        self,
        *,
        items: List[Any],
        max_workers: int,
        work_fn: Callable[[int, Any], Any],
        cancel_check=None,
    ) -> List[Any]:
        if not items:
            return []
        if callable(cancel_check):
            try:
                if cancel_check():
                    return []
            except Exception:
                self.logger.warning(
                    "DiagramExtractReasonService: cancel_check() raised before scheduling", exc_info=True
                )
        if len(items) == 1 or max_workers <= 1:
            return [work_fn(index, item) for index, item in enumerate(items, start=1)]

        ordered: Dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
            future_to_index = {
                executor.submit(work_fn, index, item): index
                for index, item in enumerate(items, start=1)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                ordered[index] = future.result()
        return [ordered[index] for index in sorted(ordered)]

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_diagram_extract_reason(
        self,
        *,
        diagram: DiagramInput,
        requirements: List[Any],
        tsd_context: str = "",
        cancel_check=None,
        agent_started_handler: Optional[Callable[[str], None]] = None,
        agent_completed_handler: Optional[Callable[..., None]] = None,
    ) -> DiagramDebateOutput:
        output = DiagramDebateOutput(diagram=diagram, requirements=requirements)

        if callable(cancel_check):
            try:
                if cancel_check():
                    output.error = "Analysis was cancelled by user."
                    return output
            except Exception:
                self.logger.warning(
                    "DiagramExtractReasonService: cancel_check() raised for diagram_id=%s — treating as not cancelled",
                    diagram.diagram_id,
                    exc_info=True,
                )

        if not requirements:
            output.error = "No diagram requirements provided — cannot ground extraction/reasoning."
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

        image_payloads: List[Dict[str, Any]] = [
            {"label": "raw", "image_bytes": image_bytes, "image_format": diagram.image_format or "png"}
        ]
        image_format = diagram.image_format or "png"
        caption = diagram.caption or ""
        surrounding = diagram.surrounding_text or ""

        # --- Stage 1: structured extraction (vision-in-the-loop, N votes) ---
        if agent_started_handler:
            agent_started_handler("extractor")
        raw_passes = self._run_extraction_passes(
            diagram_id=diagram.diagram_id,
            image_payloads=image_payloads,
            image_format=image_format,
            caption=caption,
            surrounding=surrounding,
            cancel_check=cancel_check,
        )
        merged = self._merge_extraction_passes(raw_passes)
        if agent_completed_handler:
            agent_completed_handler("extractor", merged.diagram_scope_reasoning)

        if callable(cancel_check):
            try:
                if cancel_check():
                    output.error = "Analysis was cancelled by user."
                    return output
            except Exception:
                self.logger.warning(
                    "DiagramExtractReasonService: cancel_check() raised after extraction diagram_id=%s",
                    diagram.diagram_id,
                    exc_info=True,
                )

        # --- Stage 2: text-only reasoning over the merged extraction ---
        if agent_started_handler:
            agent_started_handler("reasoner")
        assessments, reasoning_diagnostics = self._run_reasoning_batches(
            merged=merged,
            requirements=requirements,
            caption=caption,
            surrounding=surrounding,
            cancel_check=cancel_check,
        )

        aggregated_verdict = _worst_case_diagram_verdict(assessments)
        confidence = merged.merge_diagnostics.get("scope_agreement_ratio", 0.5)

        extraction_diagnostics = {
            "extraction_votes": self._extraction_votes,
            "extraction_passes_succeeded": len(raw_passes),
            "merge_threshold": self._merge_threshold,
            "confirmed_element_count": len(merged.confirmed_element_ids()),
            "total_element_count": len(merged.all_element_ids()),
            **merged.merge_diagnostics,
            **reasoning_diagnostics,
        }

        result_payload: Dict[str, Any] = {
            "overall_verdict": aggregated_verdict,
            "final_verdict": aggregated_verdict,
            "requirement_assessments": assessments,
            "assessed_requirements": assessments,
            "diagram_scope_verdict": merged.diagram_scope_verdict,
            "diagram_scope_reasoning": merged.diagram_scope_reasoning,
            "confidence": round(float(confidence), 2),
            "extraction_diagnostics": extraction_diagnostics,
        }

        # hunter_result / mediator_result carry the SAME assessments (there is
        # no separate "first pass vs final" split in this design — Stage 1
        # produces no verdicts at all) so the eval harness's existing
        # `_per_requirement_verdicts()` helper reads either interchangeably:
        # it looks for "requirement_assessments" on hunter_result and
        # "assessed_requirements" on mediator_result, both present here.
        output.hunter_result = result_payload
        output.mediator_result = result_payload
        output.critic_result = {}
        output.debate_rounds = 1

        if agent_completed_handler:
            agent_completed_handler("reasoner", result_payload.get("diagram_scope_reasoning", ""))

        return output

    def run_diagram_extract_reason_voted(
        self,
        *,
        diagram: DiagramInput,
        requirements: List[Any],
        tsd_context: str = "",
        votes: int = 1,
        cancel_check=None,
        agent_started_handler: Optional[Callable[[str], None]] = None,
        agent_completed_handler: Optional[Callable[..., None]] = None,
        **_ignored: Any,
    ) -> DiagramDebateOutput:
        """
        Thin pass-through kept ONLY so callers (notably the eval harness) can
        invoke this service symmetrically with
        DiagramDebateService.run_diagram_debate_voted. Self-consistency
        already happens at the Stage-1 extraction layer
        (AI_VISION_EXTRACTION_VOTES) — there is no separate run-level voting
        step here, so a legacy `votes > 1` (debate's run-level voting
        semantics) is a no-op, logged rather than silently doubling cost.
        """
        if votes and votes > 1:
            self.logger.warning(
                "DiagramExtractReasonService.run_diagram_extract_reason_voted: "
                "votes=%d requested but this pipeline votes at the extraction "
                "layer (AI_VISION_EXTRACTION_VOTES=%d), not the run layer — ignoring.",
                votes,
                self._extraction_votes,
            )
        return self.run_diagram_extract_reason(
            diagram=diagram,
            requirements=requirements,
            tsd_context=tsd_context,
            cancel_check=cancel_check,
            agent_started_handler=agent_started_handler,
            agent_completed_handler=agent_completed_handler,
        )

    # ------------------------------------------------------------------
    # Stage 1: extraction + merge
    # ------------------------------------------------------------------

    def _run_extraction_passes(
        self,
        *,
        diagram_id: str,
        image_payloads: List[Dict[str, Any]],
        image_format: str,
        caption: str,
        surrounding: str,
        cancel_check=None,
    ) -> List[Dict[str, Any]]:
        extractor_prompt = build_vision_extractor_prompt(
            diagram_caption=caption,
            surrounding_text=surrounding,
        )

        def run_pass(pass_index: int, _: int) -> Optional[Dict[str, Any]]:
            return self._extractor_agent.run_multimodal(
                user_prompt=extractor_prompt,
                image_bytes=None,
                image_payloads=image_payloads,
                image_format=image_format,
                system_prompt=VISION_EXTRACTOR_SYSTEM_PROMPT,
                log_context=f"diagram_id={diagram_id} agent=extractor pass={pass_index}/{self._extraction_votes}",
            )

        raw_results = self._run_parallel(
            items=list(range(self._extraction_votes)),
            max_workers=self._extraction_max_concurrency,
            work_fn=run_pass,
            cancel_check=cancel_check,
        )
        passes = [result for result in raw_results if result]
        if not passes:
            self.logger.error(
                "DiagramExtractReasonService: all %d extraction passes failed for diagram_id=%s",
                self._extraction_votes,
                diagram_id,
            )
        return passes

    def _merge_extraction_passes(self, passes: List[Dict[str, Any]]) -> MergedDiagramExtraction:
        n = len(passes)
        if n == 0:
            return MergedDiagramExtraction(
                diagram_scope_verdict="uncertain",
                diagram_scope_reasoning="All extraction passes failed to produce a parseable result.",
                votes_total=0,
                merge_diagnostics={"extraction_failed": True},
            )

        # N=1 degrades gracefully: every element is confirmed (vote_count=1/1),
        # matching how vision_debate_votes=1 already degrades the debate path.
        component_clusters: List[Dict[str, Any]] = []
        boundary_clusters: List[Dict[str, Any]] = []
        flow_clusters: List[Dict[str, Any]] = []
        component_id_maps: List[Dict[str, str]] = []

        for pass_index, extraction in enumerate(passes):
            components = extraction.get("components") or []
            id_map: Dict[str, str] = {}
            # A single pass must contribute at most one vote per cluster — two
            # distinct elements the SAME pass reported are never the same
            # real-world entity, even if they happen to look similar to each
            # other. Without this guard, same-pass elements can spuriously
            # merge together and inflate vote_count past votes_total.
            matched_in_this_pass: set = set()
            for component in components:
                if not isinstance(component, dict):
                    continue
                raw_id = str(component.get("id", ""))
                name = component.get("name", "")
                comp_type = str(component.get("type", "other"))
                best_cluster = None
                best_score = 0.0
                for cluster in component_clusters:
                    if id(cluster) in matched_in_this_pass:
                        continue
                    score = _similarity(name, cluster["name"])
                    if 0.55 <= score < self._fuzzy_match_threshold and comp_type == cluster["type"]:
                        label_overlap = _jaccard(component.get("labels") or [], cluster["labels"])
                        if label_overlap >= 0.5:
                            score = max(score, self._fuzzy_match_threshold)
                    if score > best_score:
                        best_score, best_cluster = score, cluster
                if best_cluster is not None and best_score >= self._fuzzy_match_threshold:
                    matched_in_this_pass.add(id(best_cluster))
                    best_cluster["vote_count"] += 1
                    best_cluster["labels"] = list(set(best_cluster["labels"]) | set(component.get("labels") or []))
                    existing_notes = best_cluster.get("notes", "")
                    new_notes = str(component.get("notes", "") or "")
                    best_cluster["notes"] = new_notes if len(new_notes) > len(existing_notes) else existing_notes
                    id_map[raw_id] = best_cluster["local_key"]
                else:
                    local_key = f"pending-c{len(component_clusters)}"
                    new_cluster = {
                        "local_key": local_key,
                        "name": name,
                        "type": comp_type,
                        "labels": list(component.get("labels") or []),
                        "notes": str(component.get("notes", "") or ""),
                        "vote_count": 1,
                    }
                    component_clusters.append(new_cluster)
                    matched_in_this_pass.add(id(new_cluster))
                    id_map[raw_id] = local_key
            component_id_maps.append(id_map)

        for cluster_index, cluster in enumerate(component_clusters, start=1):
            cluster["id"] = f"c{cluster_index}"
        local_to_final = {c["local_key"]: c["id"] for c in component_clusters}
        for id_map in component_id_maps:
            for raw_id, local_key in list(id_map.items()):
                id_map[raw_id] = local_to_final.get(local_key, local_key)

        for pass_index, extraction in enumerate(passes):
            id_map = component_id_maps[pass_index]
            # Same one-vote-per-pass guard as the component loop above,
            # tracked separately per cluster list since boundaries and flows
            # are independent categories.
            boundary_matched_in_this_pass: set = set()
            flow_matched_in_this_pass: set = set()

            for boundary in extraction.get("trust_boundaries") or []:
                if not isinstance(boundary, dict):
                    continue
                resolved_enclosed = sorted({
                    id_map.get(str(cid), str(cid)) for cid in (boundary.get("encloses_component_ids") or [])
                })
                label = boundary.get("label", "")
                best_cluster, best_score = None, 0.0
                for cluster in boundary_clusters:
                    if id(cluster) in boundary_matched_in_this_pass:
                        continue
                    overlap = _jaccard(resolved_enclosed, cluster["encloses_component_ids"])
                    label_sim = _similarity(label, cluster["label"])
                    score = max(overlap, label_sim) if overlap >= 0.5 else label_sim
                    if score > best_score:
                        best_score, best_cluster = score, cluster
                if best_cluster is not None and best_score >= 0.5:
                    boundary_matched_in_this_pass.add(id(best_cluster))
                    best_cluster["vote_count"] += 1
                    best_cluster["encloses_component_ids"] = sorted(
                        set(best_cluster["encloses_component_ids"]) | set(resolved_enclosed)
                    )
                else:
                    new_cluster = {
                        "label": label or "unlabeled boundary",
                        "encloses_component_ids": resolved_enclosed,
                        "boundary_style": boundary.get("boundary_style", "other"),
                        "vote_count": 1,
                    }
                    boundary_clusters.append(new_cluster)
                    boundary_matched_in_this_pass.add(id(new_cluster))

            for flow_item in extraction.get("flows") or []:
                if not isinstance(flow_item, dict):
                    continue
                resolved_source = id_map.get(str(flow_item.get("source_component_id", "")), str(flow_item.get("source_component_id", "")))
                resolved_target = id_map.get(str(flow_item.get("target_component_id", "")), str(flow_item.get("target_component_id", "")))
                label = flow_item.get("label", "")
                protocol = flow_item.get("protocol") or ""
                best_cluster, best_score = None, 0.0
                for cluster in flow_clusters:
                    if id(cluster) in flow_matched_in_this_pass:
                        continue
                    if cluster["source_component_id"] != resolved_source or cluster["target_component_id"] != resolved_target:
                        continue
                    score = _similarity(f"{label} {protocol}", f"{cluster['label']} {cluster['protocol']}")
                    if score > best_score:
                        best_score, best_cluster = score, cluster
                if best_cluster is not None and best_score >= self._fuzzy_match_threshold:
                    flow_matched_in_this_pass.add(id(best_cluster))
                    best_cluster["vote_count"] += 1
                    if flow_item.get("direction") != best_cluster.get("direction"):
                        best_cluster.setdefault("direction_disagreement", True)
                    best_cluster["security_annotations"] = list(
                        set(best_cluster["security_annotations"]) | set(flow_item.get("security_annotations") or [])
                    )
                else:
                    new_cluster = {
                        "source_component_id": resolved_source,
                        "target_component_id": resolved_target,
                        "direction": flow_item.get("direction", "unclear"),
                        "label": label,
                        "protocol": protocol or None,
                        "security_annotations": list(flow_item.get("security_annotations") or []),
                        "vote_count": 1,
                    }
                    flow_clusters.append(new_cluster)
                    flow_matched_in_this_pass.add(id(new_cluster))

        for cluster_index, cluster in enumerate(boundary_clusters, start=1):
            cluster["id"] = f"tb{cluster_index}"
        for cluster_index, cluster in enumerate(flow_clusters, start=1):
            cluster["id"] = f"f{cluster_index}"

        threshold = self._merge_threshold

        def finalize(clusters: List[Dict[str, Any]], drop_keys: Tuple[str, ...] = ("local_key",)) -> List[Dict[str, Any]]:
            finalized = []
            for cluster in clusters:
                item = {k: v for k, v in cluster.items() if k not in drop_keys}
                vote_count = item.pop("vote_count")
                item["vote_count"] = vote_count
                item["votes_total"] = n
                item["confirmed"] = (vote_count / n) >= threshold
                finalized.append(item)
            return finalized

        components_final = finalize(component_clusters)
        boundaries_final = finalize(boundary_clusters)
        flows_final = finalize(flow_clusters)

        scope_votes: Dict[str, int] = {}
        scope_reasoning_by_vote: Dict[str, str] = {}
        other_text: set = set()
        for extraction in passes:
            scope = str(extraction.get("diagram_scope_verdict", "uncertain")).strip().lower()
            if scope not in {"architecture_relevant", "non_architecture", "uncertain"}:
                scope = "uncertain"
            scope_votes[scope] = scope_votes.get(scope, 0) + 1
            scope_reasoning_by_vote.setdefault(scope, str(extraction.get("diagram_scope_reasoning", "")))
            other_text.update(str(t) for t in (extraction.get("other_visible_text") or []))

        max_votes = max(scope_votes.values()) if scope_votes else 0
        top_scopes = [scope for scope, count in scope_votes.items() if count == max_votes]
        final_scope = top_scopes[0] if len(top_scopes) == 1 else "uncertain"
        scope_agreement_ratio = round(max_votes / n, 4) if n else 0.0

        style_votes: Dict[str, int] = {}
        for extraction in passes:
            style = str(extraction.get("diagram_style", "other")).strip().lower()
            if style not in {"architecture_or_dfd", "sequence_or_flow", "other"}:
                style = "other"
            style_votes[style] = style_votes.get(style, 0) + 1
        max_style_votes = max(style_votes.values()) if style_votes else 0
        top_styles = [style for style, count in style_votes.items() if count == max_style_votes]
        final_style = top_styles[0] if len(top_styles) == 1 else "other"

        return MergedDiagramExtraction(
            components=components_final,
            trust_boundaries=boundaries_final,
            flows=flows_final,
            other_visible_text=sorted(other_text),
            diagram_scope_verdict=final_scope,
            diagram_scope_reasoning=scope_reasoning_by_vote.get(final_scope, ""),
            diagram_style=final_style,
            votes_total=n,
            raw_passes=passes,
            merge_diagnostics={
                "scope_agreement_ratio": scope_agreement_ratio,
                "scope_votes": scope_votes,
                "style_votes": style_votes,
            },
        )

    # ------------------------------------------------------------------
    # Stage 2: text-only reasoning + citation validation
    # ------------------------------------------------------------------

    def _run_reasoning_batches(
        self,
        *,
        merged: MergedDiagramExtraction,
        requirements: List[Any],
        caption: str,
        surrounding: str,
        cancel_check=None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        requirement_batches = self._batch_requirements(requirements, self._reasoner_batch_size)
        extraction_text = merged.to_reasoner_text()
        valid_element_ids = sorted(merged.all_element_ids())

        diagnostics = {
            "citation_retry_batches": 0,
            "citation_retry_exhausted_requirements": 0,
            "completeness_retry_batches": 0,
            "requirements_missing_after_retry": 0,
            "full_failure_retry_batches": 0,
            "full_failure_exhausted_batches": 0,
        }

        def run_batch(batch_index: int, requirement_batch: List[Any]) -> List[Dict[str, Any]]:
            batch_requirements_text = _format_requirements_with_hints(requirement_batch)
            batch_req_ids = [self._requirement_id(r) for r in requirement_batch]
            retry_context: Optional[Dict[str, Any]] = None
            assessments: List[Dict[str, Any]] = []
            invalid_met_ids: List[str] = []
            missing_ids: List[str] = []
            saw_total_failure = False

            # Retries cover THREE distinct failure modes the reasoner can hit:
            # (1) it answers every requirement but hallucinates a citation for
            # a "met" verdict, (2) it silently answers only a subset of the
            # batch despite the "assess every requirement" instruction, or
            # (3) the response fails entirely (empty/unparseable, nothing
            # answered at all) — this last one is a cheap, likely-transient
            # generation hiccup, not a persistent completeness/reasoning gap,
            # so it gets a larger retry budget than (1)/(2). The effective
            # attempt cap only expands once a total failure is actually
            # observed on this batch, so well-behaved batches still resolve
            # in at most `citation_retry_limit + 1` attempts.
            attempt = 0
            while True:
                prompt = build_vision_reasoner_prompt(
                    requirements_with_hints=batch_requirements_text,
                    extraction_text=extraction_text,
                    diagram_caption=caption,
                    surrounding_text=surrounding,
                    citation_retry_context=retry_context,
                )
                raw_result = self._reasoner_agent.run_text(
                    user_prompt=prompt,
                    system_prompt=VISION_REASONER_SYSTEM_PROMPT,
                    log_context=(
                        f"agent=reasoner batch={batch_index}/{len(requirement_batches)} attempt={attempt + 1}"
                    ),
                )
                is_total_failure = raw_result is None
                raw_assessments = (raw_result or {}).get("requirement_assessments") or []
                assessments, invalid_met_ids = self._validate_and_filter_citations(raw_assessments, merged)
                answered_ids = {str(a.get("requirement_id", "")).strip() for a in assessments}
                missing_ids = [req_id for req_id in batch_req_ids if req_id not in answered_ids]

                if not invalid_met_ids and not missing_ids:
                    break

                if is_total_failure:
                    saw_total_failure = True
                    diagnostics["full_failure_retry_batches"] += 1
                elif missing_ids:
                    diagnostics["completeness_retry_batches"] += 1
                if invalid_met_ids:
                    diagnostics["citation_retry_batches"] += 1

                effective_limit = (
                    max(self._citation_retry_limit, self._full_failure_retry_limit)
                    if saw_total_failure
                    else self._citation_retry_limit
                )
                if attempt >= effective_limit:
                    if saw_total_failure:
                        diagnostics["full_failure_exhausted_batches"] += 1
                    break

                self.logger.warning(
                    "DiagramExtractReasonService: reasoner batch incomplete/invalid batch=%d/%d attempt=%d "
                    "total_failure=%s missing_ids=%s invalid_met_ids=%s",
                    batch_index,
                    len(requirement_batches),
                    attempt + 1,
                    is_total_failure,
                    missing_ids,
                    invalid_met_ids,
                )
                # A total failure means nothing was answered at all — there's
                # no content to correct, so retry with a clean prompt instead
                # of a "missing ids" reminder (which would just list every id
                # in the batch and adds nothing useful to a generation-level
                # failure).
                retry_context = None if is_total_failure else {
                    "invalid_requirement_ids": invalid_met_ids,
                    "valid_element_ids": valid_element_ids,
                    "missing_requirement_ids": missing_ids,
                }
                attempt += 1

            # Retry budget exhausted — deterministically downgrade any
            # requirement still met-with-no-valid-citation to "na". Only
            # "met" is downgraded this way; "not_met" never needs a positive
            # citation, matching the existing scope-vs-absence asymmetry.
            # Requirements still missing entirely fall back via
            # _ensure_complete_batch below (verdict_policy_source
            # "reasoner_omitted_requirement").
            if invalid_met_ids:
                diagnostics["citation_retry_exhausted_requirements"] += len(invalid_met_ids)
                for assessment in assessments:
                    if str(assessment.get("requirement_id")) in invalid_met_ids:
                        assessment["verdict"] = VERDICT_NA
                        assessment["verdict_policy_source"] = "citation_retry_exhausted"
            if missing_ids:
                diagnostics["requirements_missing_after_retry"] += len(missing_ids)

            return self._ensure_complete_batch(assessments, requirement_batch)

        batch_results = self._run_parallel(
            items=requirement_batches,
            max_workers=self._reasoner_batch_max_concurrency,
            work_fn=run_batch,
            cancel_check=cancel_check,
        )
        flattened = [item for batch in batch_results for item in batch]
        return flattened, diagnostics

    def _validate_and_filter_citations(
        self,
        raw_assessments: List[Dict[str, Any]],
        merged: MergedDiagramExtraction,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        valid_ids = merged.all_element_ids()
        confirmed_ids = merged.confirmed_element_ids()
        validated: List[Dict[str, Any]] = []
        invalid_met_ids: List[str] = []

        for assessment in raw_assessments:
            if not isinstance(assessment, dict):
                continue
            item = dict(assessment)
            requirement_id = str(item.get("requirement_id", "")).strip()
            if not requirement_id:
                continue
            verdict = str(item.get("verdict", VERDICT_NA)).strip().lower()
            if verdict not in {VERDICT_MET, VERDICT_NOT_MET, VERDICT_NA}:
                verdict = VERDICT_NA
            raw_cited = [str(c) for c in (item.get("cited_element_ids") or [])]
            cited = [c for c in raw_cited if c in valid_ids]
            dropped = [c for c in raw_cited if c not in valid_ids]

            item["verdict"] = verdict
            item["cited_element_ids"] = cited
            if dropped:
                item["dropped_hallucinated_citations"] = dropped

            if verdict == VERDICT_MET and not cited:
                invalid_met_ids.append(requirement_id)
            elif verdict == VERDICT_NOT_MET and cited and not any(c in confirmed_ids for c in cited):
                item["evidence_quality"] = "unconfirmed_extraction"

            validated.append(item)

        return validated, invalid_met_ids

    def _ensure_complete_batch(
        self,
        assessments: List[Dict[str, Any]],
        requirement_batch: List[Any],
    ) -> List[Dict[str, Any]]:
        by_id = {str(a.get("requirement_id", "")).strip(): a for a in assessments}
        completed = []
        for requirement in requirement_batch:
            req_id = self._requirement_id(requirement)
            if req_id in by_id:
                completed.append(by_id[req_id])
            else:
                completed.append({
                    "requirement_id": req_id,
                    "verdict": VERDICT_NA,
                    "cited_element_ids": [],
                    "reasoning": "Reasoner omitted this requirement; synthesized conservative fallback.",
                    "verdict_policy_source": "reasoner_omitted_requirement",
                })
        return completed


__all__ = ["DiagramExtractReasonService"]
