from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

from sdr.apps.ai.retrieval.core import RetrievalCandidate, RetrievalResult, RetrievalStrategy, dedupe_candidates, merge_candidates
from sdr.apps.ai.retrieval.searchers.raptor import RAPTOR_LEVEL_HIGH, RAPTOR_LEVEL_LOW, RAPTOR_LEVEL_MID, RAPTORSearchResponse
from sdr.apps.ai.retrieval.searchers.vector import VectorSearchResponse
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob
from sdr.core.config import settings

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
) -> List[RetrievalCandidate]:
    """Reranks within each evidence tier (on the tier's full membership, not
    a pre-cut slice) and only concatenates+truncates afterward, so reranking
    can actually act on the full candidate pool instead of just whatever
    happened to survive an earlier per-tier cutoff. Tier priority is
    preserved — a hierarchical_summary can never outrank literal evidence
    just because a cross-encoder/score ranks it higher within its own tier."""
    tiers: Dict[str, List[RetrievalCandidate]] = {key: [] for key in _EVIDENCE_TIER_ORDER}
    for candidate in candidates:
        kind = candidate.metadata.get("evidence_kind")
        tiers[kind if kind in ("implementation_evidence", "hierarchical_summary") else "__fallback__"].append(
            candidate
        )

    reranked: List[RetrievalCandidate] = []
    for tier_key in _EVIDENCE_TIER_ORDER:
        tier_candidates = tiers[tier_key]
        if not tier_candidates:
            continue
        reranked.extend(
            router._reranker.rerank(query=query_text, candidates=tier_candidates, top_k=len(tier_candidates))
        )
    return reranked[:max_context_chunks]


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
    def execute_vector_only(
        self,
        router,
        *,
        query_text: str,
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        query_embedding: List[float],
    ) -> RetrievalResult:
        vector_response = router._vector_searcher.search(
            query_text=query_text,
            category=category,
            top_k=router.vector_top_k,
            ingestion_job=ingestion_job,
            precomputed_embedding=query_embedding or None,
        )
        context_chunks = router._build_chunks_from_vector(vector_response)
        source_block_ids = router._collect_block_ids_from_vector(vector_response)
        return RetrievalResult(
            context_chunks=context_chunks[: router.max_context_chunks],
            context_chunk_block_ids=[[] for _ in context_chunks[: router.max_context_chunks]],
            source_block_ids=source_block_ids,
            block_source_map={},
            strategy_used=RetrievalStrategy.VECTOR_ONLY,
            query_embedding=query_embedding,
            vector_response=vector_response,
            error=vector_response.error,
        )

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
        merged = merge_candidates(*bm25_candidate_lists, *dense_candidate_lists, raptor_candidates)
        deduped = dedupe_candidates(merged)
        scored = router._apply_keyword_coverage_boost(deduped, keywords)
        evidence_filtered, evidence_metadata = _grade_with_secondary_search(
            router,
            scored,
            query_text=query_text,
            keywords=keywords,
            raptor_tree=raptor_tree,
            has_raptor=has_raptor,
        )
        reranked = _rerank_within_tiers(router, query_text, evidence_filtered, router.max_context_chunks)
        kept = [c for c in reranked if c.text]
        merged_chunks = [c.text for c in kept]
        merged_chunk_block_ids = [list(c.block_ids) for c in kept]
        all_source_block_ids = router._collect_candidate_block_ids(reranked)
        block_source_map = _build_block_source_map_from_candidates(reranked)
        return RetrievalResult(
            context_chunks=merged_chunks[: router.max_context_chunks],
            context_chunk_block_ids=merged_chunk_block_ids[: router.max_context_chunks],
            source_block_ids=all_source_block_ids,
            block_source_map=block_source_map,
            strategy_used=RetrievalStrategy.HYBRID,
            query_embedding=query_embedding,
            vector_response=None,
            raptor_response=raptor_response,
            evidence_metadata={
                **evidence_metadata,
                "block_source_map": block_source_map,
            },
        )

__all__ = ["RetrievalRouteExecutor"]
