from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

from sdr.apps.ai.retrieval.core import (
    HybridRetrievalTrace,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalStrategy,
    dedupe_candidates,
    merge_candidates,
    reciprocal_rank_fusion,
)
from sdr.apps.ai.retrieval.core.candidates import _key_for_candidate
from sdr.apps.ai.retrieval.searchers.raptor import RAPTOR_LEVEL_HIGH, RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTORSearchResponse
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob
from sdr.core.config import settings

from sdr.apps.ai.retrieval.core.keywords import extract_keywords as _extract_keywords

logger = logging.getLogger(__name__)


def _grade_with_secondary_search(
    router,
    candidates: List[RetrievalCandidate],
    *,
    query_text: str,
    keywords: List[str],
    raptor_tree: Optional[RAPTORTree],
    has_raptor: bool,
) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
    """Grade candidates, then — if nothing graded as implementation-grade evidence —
    run one extra cheap (no-LLM) BM25 + RAPTOR-dense round using a follow-up query
    built from the current top candidates' own keywords, to reach adjacent TSD
    sections that implement a concept the first pass only found discussed in the
    abstract (e.g. an access-control model whose enforcement lives in a separate
    session-management section)."""
    evidence_filtered, evidence_metadata = router._grade_and_filter_candidates(
        candidates,
        query_text=query_text,
        keywords=keywords,
    )
    evidence_metadata["secondary_search_triggered"] = False

    evidence_quality = evidence_metadata.get("evidence_quality") or {}
    implementation_count = int(evidence_quality.get("implementation_evidence_count") or 0)
    if (
        not bool(getattr(settings, "AI_RETRIEVAL_SECONDARY_SEARCH_ENABLED", True))
        or implementation_count > 0
        or not evidence_filtered
    ):
        return evidence_filtered, evidence_metadata

    followup_query = router._grader().generate_followup_query(query_text, evidence_filtered[:5], _extract_keywords)
    followup_query = (followup_query or "").strip()
    if not followup_query or followup_query.lower() == query_text.strip().lower():
        return evidence_filtered, evidence_metadata

    followup_embedding = router._generate_query_embedding(followup_query)
    extra_bm25 = _safe_execute(
        router._keyword_searcher.search,
        query_text=followup_query,
        tree=raptor_tree,
        top_k=max(router.vector_top_k, router.raptor_top_k, 20),
        allowed_levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
        on_error=lambda exc: [],
    )
    extra_dense_candidates: List[RetrievalCandidate] = []
    if has_raptor:
        extra_dense_response = _safe_execute(
            router._raptor_searcher.search_collapsed_raptor,
            query_text=followup_query,
            tree=raptor_tree,
            top_k=max(router.vector_top_k, 8),
            max_tokens=4000,
            allowed_levels=[RAPTOR_LEVEL_LOW],
            precomputed_embedding=followup_embedding or None,
            on_error=lambda exc: RAPTORSearchResponse(error=str(exc)),
        )
        extra_dense_candidates = router._dense_tsd_results_to_candidates(extra_dense_response)

    # Secondary-search top-up always uses the agreement-boost merge regardless
    # of router.advanced_config.fusion_method — it's a small supplementary
    # merge on top of already-fused candidates, not the primary fusion step.
    merged = merge_candidates(candidates, extra_bm25, extra_dense_candidates)
    deduped = dedupe_candidates(merged)
    followup_keywords = keywords + _extract_keywords(followup_query)
    boosted = router._apply_keyword_coverage_boost(deduped, followup_keywords)
    evidence_filtered, evidence_metadata = router._grade_and_filter_candidates(
        boosted,
        query_text=query_text,
        keywords=keywords,
    )
    evidence_metadata["secondary_search_triggered"] = True
    evidence_metadata["secondary_search_query"] = followup_query
    return evidence_filtered, evidence_metadata


_EVIDENCE_TIER_ORDER = ("implementation_evidence", "__fallback__", "hierarchical_summary")


def _rerank_within_tiers(
    router,
    query_text: str,
    candidates: List[RetrievalCandidate],
    max_context_chunks: int,
    trace: Optional[HybridRetrievalTrace] = None,
) -> List[RetrievalCandidate]:
    """Reranks candidates by evidence quality, with the cross-encoder (when
    enabled) used only to break ties WITHIN each tier.

    Tier priority — implementation_evidence > fallback > hierarchical_summary —
    is always the dominant signal. A hierarchical summary's embedding/relevance
    score can look deceptively strong against a broad or generically-worded
    question (it superficially touches many topics), which previously let a
    summary spanning dozens of blocks across several pages outrank the actual
    specific, citable leaf chunk once the cross-encoder was enabled. Letting
    the cross-encoder fully bypass tiering traded that away for no real gain:
    its semantic judgment is valuable for discriminating between several
    plausible literal chunks, not for promoting a whole summary tier above
    literal evidence.
    """
    tiers: Dict[str, List[RetrievalCandidate]] = {key: [] for key in _EVIDENCE_TIER_ORDER}
    for candidate in candidates:
        kind = candidate.metadata.get("evidence_kind")
        tiers[kind if kind in ("implementation_evidence", "hierarchical_summary") else "__fallback__"].append(
            candidate
        )

    if trace is not None:
        trace.pre_rerank_order = [
            _key_for_candidate(c)
            for tier_key in _EVIDENCE_TIER_ORDER
            for c in sorted(tiers[tier_key], key=lambda c: c.score, reverse=True)
        ]

    reranked: List[RetrievalCandidate] = []
    for tier_key in _EVIDENCE_TIER_ORDER:
        tier_candidates = tiers[tier_key]
        if not tier_candidates:
            continue
        reranked.extend(
            router._reranker.rerank(query=query_text, candidates=tier_candidates, top_k=len(tier_candidates))
        )

    if trace is not None:
        trace.post_rerank_order = [_key_for_candidate(c) for c in reranked]

    return reranked[:max_context_chunks]


def _protected_candidate_keys(
    *,
    primary_dense: List[RetrievalCandidate],
    primary_bm25: List[RetrievalCandidate],
    raptor_multi: List[RetrievalCandidate],
    dense_n: int,
    bm25_n: int,
    raptor_n: int,
) -> Dict[str, str]:
    """Top-N candidate keys per constituent signal (primary-query rankings
    only — expansion variants are supplementary), mapped to the signal name.
    These are the recall-safety floor: the hybrid final context must contain
    each of them unless the evidence grader legitimately rejected it."""
    protected: Dict[str, str] = {}
    for source, candidates, top_n in (
        ("dense", primary_dense, dense_n),
        ("bm25", primary_bm25, bm25_n),
        ("raptor", raptor_multi, raptor_n),
    ):
        for candidate in candidates[: max(0, int(top_n))]:
            protected.setdefault(_key_for_candidate(candidate), source)
    return protected


def _enforce_protected_slots(
    reranked: List[RetrievalCandidate],
    graded_pool: List[RetrievalCandidate],
    protected: Dict[str, str],
    max_context_chunks: int,
) -> List[RetrievalCandidate]:
    """Re-inserts missing protected candidates at the TAIL of the final list,
    evicting the lowest-ranked non-protected items to stay within budget.

    Tail insertion keeps the head of the fused/tiered/reranked ordering intact
    (context_precision/MRR unaffected) while guaranteeing coverage: a hybrid
    ensemble must never return a context strictly worse than any of its
    constituent signals. Candidates the grader rejected outright
    (baseline_requirement/empty — absent from graded_pool) stay excluded;
    those filters are legitimate and don't touch literal TSD leaves.
    """
    if not protected:
        return reranked

    present = {_key_for_candidate(c) for c in reranked}
    graded_by_key: Dict[str, RetrievalCandidate] = {}
    for candidate in graded_pool:
        graded_by_key.setdefault(_key_for_candidate(candidate), candidate)

    additions: List[RetrievalCandidate] = []
    for key, source in protected.items():
        if key in present:
            continue
        candidate = graded_by_key.get(key)
        if candidate is None:
            continue
        candidate.metadata["protected_slot_source"] = source
        additions.append(candidate)
        present.add(key)

    if not additions:
        return reranked

    result = list(reranked) + additions
    while len(result) > max_context_chunks:
        evictable = [i for i, c in enumerate(result) if _key_for_candidate(c) not in protected]
        if not evictable:
            result = result[:max_context_chunks]
            break
        del result[evictable[-1]]
    return result


def _leaf_descendants(node) -> List[Any]:
    """Recursively collects level-0 (literal, non-summarized) descendant nodes
    of a RAPTOR tree node. A leaf node is its own sole descendant."""
    if int(getattr(node, "level", 0) or 0) == 0:
        return [node]
    leaves: List[Any] = []
    for child in getattr(node, "children", None) or []:
        leaves.extend(_leaf_descendants(child))
    return leaves


def _ground_summaries_with_leaves(
    router,
    candidates: List[RetrievalCandidate],
    *,
    raptor_tree: Optional[RAPTORTree],
    query_embedding: List[float],
    keywords: List[str],
    max_context_chunks: int,
) -> List[RetrievalCandidate]:
    """A hierarchical_summary candidate's block_ids are a UNION over many
    literal blocks — a judge/reader never sees the actual block text, only the
    LLM-synthesized summary, which produces "coverage=1, recall=0" failures
    (the block is technically "covered" but its literal content never reaches
    the answer). For every summary candidate kept in the final list, this
    grounds it with its single highest-cosine literal leaf descendant (if not
    already present), so at least one literal chunk backs every summary that
    survives to the final context. Added candidates are treated exactly like
    protected slots: tail insertion, never evict each other or the protected
    set, subject to the same evidence-grader legitimacy check.
    """
    if raptor_tree is None or raptor_tree.is_empty():
        return candidates

    summary_ids = [
        c.id for c in candidates
        if int(c.metadata.get("level", 0) or 0) > 0 and c.metadata.get("evidence_kind") == "hierarchical_summary"
    ]
    if not summary_ids:
        return candidates

    from sdr.apps.ai.tsd_processing.raptor import _compute_cosine_similarity

    node_map = {n.node_id: n for n in raptor_tree.get_all_nodes()}
    present_ids = {c.id for c in candidates}
    groundings: List[RetrievalCandidate] = []

    for summary_id in summary_ids:
        node = node_map.get(summary_id)
        if node is None:
            continue
        leaves = [leaf for leaf in _leaf_descendants(node) if getattr(leaf, "has_embedding", False)]
        if not leaves:
            continue
        best_leaf = max(leaves, key=lambda leaf: _compute_cosine_similarity(query_embedding, leaf.embedding))
        if best_leaf.node_id in present_ids:
            continue

        grounding_candidate = RetrievalCandidate(
            id=best_leaf.node_id,
            source_type="raptor",
            text=best_leaf.text,
            score=_compute_cosine_similarity(query_embedding, best_leaf.embedding),
            block_ids=list(best_leaf.source_block_ids),
            metadata={
                "level": best_leaf.level,
                "page_numbers": list(best_leaf.page_numbers),
                "section_heading": best_leaf.section_heading,
            },
            token_count=best_leaf.token_estimate,
        )
        kind, reason = router._classify_candidate_evidence(grounding_candidate, keywords=keywords)
        if kind in {"baseline_requirement", "empty"}:
            continue
        grounding_candidate.metadata["evidence_kind"] = kind
        grounding_candidate.metadata["evidence_reason"] = reason
        grounding_candidate.metadata["leaf_grounded_for"] = summary_id
        groundings.append(grounding_candidate)
        present_ids.add(best_leaf.node_id)

    if not groundings:
        return candidates

    result = list(candidates) + groundings
    protected_and_grounded = {c.id for c in groundings} | {
        c.id for c in candidates if c.metadata.get("protected_slot_source")
    }
    while len(result) > max_context_chunks:
        evictable = [i for i, c in enumerate(result) if c.id not in protected_and_grounded]
        if not evictable:
            result = result[:max_context_chunks]
            break
        del result[evictable[-1]]
    return result


def _safe_execute(func, *args, on_error=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        logger.exception("Retrieval branch failed in thread pool: %s", exc)
        if on_error is not None:
            return on_error(exc)
        raise


def _bounded_pool_size(configured_max: int, branch_count: int) -> int:
    return max(1, min(max(1, int(configured_max)), max(1, int(branch_count))))


def _canonical_source_type(source_type: Optional[str]) -> str:
    normalized = str(source_type or "").strip().lower()
    if normalized in {"raptor", "bm25"}:
        return "raptor"
    if normalized in {"dense", "vector"}:
        return "vector"
    return normalized or "unknown"


def _source_descriptor_for_keys(source_keys: Set[str]) -> Dict[str, Any]:
    normalized = {key for key in (_canonical_source_type(item) for item in source_keys) if key}
    if normalized == {"raptor"}:
        return {"key": "raptor", "label": "RAPTOR"}
    if normalized == {"vector"}:
        return {"key": "vector", "label": "Vector"}
    if len(normalized) == 1:
        key = next(iter(normalized))
        return {"key": key, "label": key.replace("_", " ").title()}
    ordered = sorted(normalized)
    return {"key": "+".join(ordered), "label": " + ".join(item.replace("_", " ").title() for item in ordered)}


def _merge_block_source_entry(block_source_map: Dict[str, Dict[str, Any]], block_id: str, source_type: str) -> None:
    if not block_id:
        return
    canonical = _canonical_source_type(source_type)
    existing = dict(block_source_map.get(block_id) or {})
    source_keys = set(existing.get("source_keys") or [])
    if existing.get("retrieval_origin"):
        source_keys.add(str(existing["retrieval_origin"]))
    if canonical:
        source_keys.add(canonical)
    descriptor = _source_descriptor_for_keys(source_keys)
    block_source_map[block_id] = {
        "retrieval_origin": descriptor["key"],
        "retrieval_origin_label": descriptor["label"],
        "source_keys": sorted(source_keys),
    }


def _build_uniform_block_source_map(block_ids: List[str], source_type: str) -> Dict[str, Dict[str, Any]]:
    block_source_map: Dict[str, Dict[str, Any]] = {}
    for block_id in block_ids or []:
        _merge_block_source_entry(block_source_map, block_id, source_type)
    return block_source_map


def _build_block_source_map_from_candidates(candidates: List[RetrievalCandidate]) -> Dict[str, Dict[str, Any]]:
    block_source_map: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates or []:
        merged_sources = candidate.metadata.get("merged_sources", [candidate.source_type]) if isinstance(candidate.metadata, dict) else [candidate.source_type]
        for block_id in candidate.block_ids or []:
            for source_type in merged_sources:
                _merge_block_source_entry(block_source_map, block_id, str(source_type))
    return block_source_map


class RetrievalRouteExecutor:
    def execute_raptor_low(self, router, *, query_text: str, raptor_tree: Optional[RAPTORTree], query_embedding: List[float]) -> RetrievalResult:
        if raptor_tree is None or raptor_tree.is_empty():
            return RetrievalResult(
                strategy_used=RetrievalStrategy.RAPTOR_LOW,
                query_embedding=query_embedding,
                error="RAPTORTree is not available.",
            )
        raptor_response = router._raptor_searcher.search(
            query_text=query_text,
            tree=raptor_tree,
            level=RAPTOR_LEVEL_LOW,
            top_k=router.raptor_top_k,
            precomputed_embedding=query_embedding or None,
        )
        return RetrievalResult(
            context_chunks=raptor_response.get_context_chunks()[: router.max_context_chunks],
            context_chunk_block_ids=raptor_response.get_context_chunk_block_ids()[: router.max_context_chunks],
            source_block_ids=raptor_response.all_source_block_ids,
            block_source_map=_build_uniform_block_source_map(
                raptor_response.all_source_block_ids,
                "raptor",
            ),
            strategy_used=RetrievalStrategy.RAPTOR_LOW,
            query_embedding=query_embedding,
            raptor_response=raptor_response,
            error=raptor_response.error,
        )

    def execute_flat_topk(self, router, *, query_text: str, raptor_tree: Optional[RAPTORTree], query_embedding: List[float]) -> RetrievalResult:
        if raptor_tree is None or raptor_tree.is_empty():
            return RetrievalResult(
                strategy_used=RetrievalStrategy.FLAT_TOPK,
                query_embedding=query_embedding,
                error="RAPTORTree is not available.",
            )
        raptor_response = router._raptor_searcher.search_flat_topk(
            query_text=query_text,
            tree=raptor_tree,
            level=RAPTOR_LEVEL_LOW,
            top_k=router.raptor_top_k,
            precomputed_embedding=query_embedding or None,
        )
        return RetrievalResult(
            context_chunks=raptor_response.get_context_chunks()[: router.max_context_chunks],
            context_chunk_block_ids=raptor_response.get_context_chunk_block_ids()[: router.max_context_chunks],
            source_block_ids=raptor_response.all_source_block_ids,
            block_source_map=_build_uniform_block_source_map(
                raptor_response.all_source_block_ids,
                "raptor",
            ),
            strategy_used=RetrievalStrategy.FLAT_TOPK,
            query_embedding=query_embedding,
            raptor_response=raptor_response,
            error=raptor_response.error,
        )

    def execute_raptor_high(self, router, *, query_text: str, raptor_tree: Optional[RAPTORTree], query_embedding: List[float]) -> RetrievalResult:
        if raptor_tree is None or raptor_tree.is_empty():
            return RetrievalResult(
                strategy_used=RetrievalStrategy.RAPTOR_HIGH,
                query_embedding=query_embedding,
                error="RAPTORTree is not available.",
            )
        raptor_response = router._raptor_searcher.search_multi_level(
            query_text=query_text,
            tree=raptor_tree,
            levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
            top_k_per_level=3,
            precomputed_embedding=query_embedding or None,
        )
        return RetrievalResult(
            context_chunks=raptor_response.get_context_chunks()[: router.max_context_chunks],
            context_chunk_block_ids=raptor_response.get_context_chunk_block_ids()[: router.max_context_chunks],
            context_chunk_levels=raptor_response.get_context_chunk_levels()[: router.max_context_chunks],
            source_block_ids=raptor_response.all_source_block_ids,
            block_source_map=_build_uniform_block_source_map(
                raptor_response.all_source_block_ids,
                "raptor",
            ),
            strategy_used=RetrievalStrategy.RAPTOR_HIGH,
            query_embedding=query_embedding,
            raptor_response=raptor_response,
            error=raptor_response.error,
        )

    def execute_hybrid(
        self,
        router,
        *,
        query_text: str,
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        raptor_tree: Optional[RAPTORTree],
        query_embedding: List[float],
        keywords: List[str],
        query_variants: Optional[List[Tuple[str, List[float]]]] = None,
        trace: Optional[HybridRetrievalTrace] = None,
    ) -> RetrievalResult:
        has_raptor = bool(raptor_tree and not raptor_tree.is_empty())
        # query_variants are LLM-generated rephrasings of query_text aimed at bridging
        # the abstract-standard-language vs concrete-TSD-language vocabulary gap. They
        # are only run through the literal-text-matching branches (BM25, RAPTOR dense
        # leaf search) — structural/hierarchical branches stay single-query.
        all_queries: List[Tuple[str, List[float]]] = [(query_text, query_embedding)] + list(query_variants or [])
        branch_count = len(all_queries) * (2 if has_raptor else 1) + (1 if has_raptor else 0)
        max_workers = _bounded_pool_size(router.advanced_config.hybrid_max_workers, branch_count)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ThreadPoolExecutor-3") as executor:
            dense_futures = []
            bm25_futures = []
            if has_raptor:
                for q_text, q_embedding in all_queries:
                    dense_futures.append(
                        executor.submit(
                            _safe_execute,
                            router._raptor_searcher.search_collapsed_raptor,
                            query_text=q_text,
                            tree=raptor_tree,
                            top_k=max(router.vector_top_k, 8),
                            max_tokens=4000,
                            allowed_levels=[RAPTOR_LEVEL_LOW],
                            precomputed_embedding=q_embedding or None,
                            on_error=lambda exc: RAPTORSearchResponse(error=str(exc)),
                        )
                    )
                fut_rap = executor.submit(
                    _safe_execute,
                    router._raptor_searcher.search_collapsed_raptor,
                    query_text=query_text,
                    tree=raptor_tree,
                    top_k=max(router.raptor_top_k, 6),
                    max_tokens=4000,
                    allowed_levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
                    precomputed_embedding=query_embedding or None,
                    on_error=lambda exc: RAPTORSearchResponse(error=str(exc)),
                )
            else:
                fut_rap = None
            for q_text, _q_embedding in all_queries:
                bm25_futures.append(
                    executor.submit(
                        _safe_execute,
                        router._keyword_searcher.search,
                        query_text=q_text,
                        tree=raptor_tree,
                        top_k=max(router.vector_top_k, router.raptor_top_k, 20),
                        allowed_levels=[RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTOR_LEVEL_HIGH],
                        on_error=lambda exc: [],
                    )
                )
            dense_responses = [fut.result() for fut in dense_futures]
            raptor_response = fut_rap.result() if fut_rap else None
            bm25_candidate_lists = [fut.result() for fut in bm25_futures]

        dense_candidate_lists = [router._dense_tsd_results_to_candidates(resp) for resp in dense_responses]
        raptor_candidates = router._raptor_results_to_candidates(raptor_response)
        ranked_lists = [*bm25_candidate_lists, *dense_candidate_lists, raptor_candidates]

        if trace is not None:
            trace.queries = [q for q, _ in all_queries]
            trace.fusion_method = router.advanced_config.fusion_method
            trace.max_context_chunks = router.max_context_chunks
            for i, bm25_list in enumerate(bm25_candidate_lists):
                trace.record_list(f"bm25[{i}]", bm25_list)
            for i, dense_list in enumerate(dense_candidate_lists):
                trace.record_list(f"dense[{i}]", dense_list)
            trace.record_list("raptor_multi", raptor_candidates)

        if router.advanced_config.fusion_method == "rrf":
            # Primary fusion via Reciprocal Rank Fusion across each searcher's
            # own ranked list, replacing the dedupe+agreement-boost merge below.
            # Downstream tiering/rerank/keyword-boost logic is unchanged either way.
            deduped = reciprocal_rank_fusion(ranked_lists, k=router.advanced_config.rrf_k)
        else:
            deduped = dedupe_candidates(merge_candidates(*ranked_lists))
        scored = router._apply_keyword_coverage_boost(deduped, keywords)
        if trace is not None:
            trace.record_fused(scored)
        evidence_filtered, evidence_metadata = _grade_with_secondary_search(
            router,
            scored,
            query_text=query_text,
            keywords=keywords,
            raptor_tree=raptor_tree,
            has_raptor=has_raptor,
        )
        if trace is not None:
            trace.record_graded(evidence_filtered)
            trace.rejected = list((evidence_metadata.get("evidence_quality") or {}).get("rejected") or [])
            trace.secondary_search_triggered = bool(evidence_metadata.get("secondary_search_triggered"))
        reranked = _rerank_within_tiers(router, query_text, evidence_filtered, router.max_context_chunks, trace=trace)
        protected = _protected_candidate_keys(
            primary_dense=dense_candidate_lists[0] if dense_candidate_lists else [],
            primary_bm25=bm25_candidate_lists[0] if bm25_candidate_lists else [],
            raptor_multi=raptor_candidates,
            dense_n=router.advanced_config.protected_dense_top_n,
            bm25_n=router.advanced_config.protected_bm25_top_n,
            raptor_n=router.advanced_config.protected_raptor_top_n,
        )
        reranked = _enforce_protected_slots(reranked, evidence_filtered, protected, router.max_context_chunks)
        reranked = _ground_summaries_with_leaves(
            router,
            reranked,
            raptor_tree=raptor_tree,
            query_embedding=query_embedding,
            keywords=keywords,
            max_context_chunks=router.max_context_chunks,
        )
        kept = [c for c in reranked if c.text]
        if trace is not None:
            trace.record_final(kept)
        merged_chunks = [c.text for c in kept]
        merged_chunk_block_ids = [list(c.block_ids) for c in kept]
        merged_chunk_levels = [int(c.metadata.get("level", 0) or 0) for c in kept]
        all_source_block_ids = router._collect_candidate_block_ids(reranked)
        block_source_map = _build_block_source_map_from_candidates(reranked)
        return RetrievalResult(
            context_chunks=merged_chunks[: router.max_context_chunks],
            context_chunk_block_ids=merged_chunk_block_ids[: router.max_context_chunks],
            context_chunk_levels=merged_chunk_levels[: router.max_context_chunks],
            source_block_ids=all_source_block_ids,
            block_source_map=block_source_map,
            strategy_used=RetrievalStrategy.HYBRID,
            query_embedding=query_embedding,
            raptor_response=raptor_response,
            evidence_metadata={
                **evidence_metadata,
                "block_source_map": block_source_map,
                "protected_slots_added": [
                    {"id": c.id, "source": c.metadata.get("protected_slot_source")}
                    for c in kept
                    if c.metadata.get("protected_slot_source")
                ],
                "leaf_groundings_added": [
                    {"id": c.id, "grounds_summary": c.metadata.get("leaf_grounded_for")}
                    for c in kept
                    if c.metadata.get("leaf_grounded_for")
                ],
            },
        )

__all__ = ["RetrievalRouteExecutor"]
