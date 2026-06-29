from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from sdr.apps.ai.client import get_embedding
from sdr.apps.ai.engine.classification.query_expansion import expand_retrieval_query_variants
from sdr.apps.ai.retrieval.core import AdvancedRetrievalConfig, RetrievalCandidate, RetrievalResult, RetrievalStrategy
from sdr.apps.ai.retrieval.postprocessing.chunk_builders import (
    build_chunks_from_vector,
    collect_block_ids_from_vector,
)
from sdr.apps.ai.retrieval.postprocessing.evidence_grader import EvidenceGrader
from sdr.apps.ai.retrieval.postprocessing.reranker import SafeOptionalReranker
from sdr.apps.ai.retrieval.routing.executors import RetrievalRouteExecutor
from sdr.apps.ai.retrieval.routing.strategy_selector import RetrievalStrategySelector
from sdr.apps.ai.retrieval.searchers.keyword import KeywordSearcher
from sdr.apps.ai.retrieval.searchers.raptor import RAPTORSearchResponse, RAPTORSearcher
from sdr.apps.ai.retrieval.searchers.vector import VectorSearchResponse, VectorSearcher
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree
from sdr.apps.standards.models import CategoryParameterChild, StandardCategory, StandardIngestionJob
from sdr.apps.standards.utils import build_parameter_analysis_text
from sdr.core.config import settings

logger = logging.getLogger(__name__)

_VECTOR_TOP_K = 8
_RAPTOR_TOP_K = 5
_MAX_CONTEXT_CHUNKS = 12
_EMBEDDING_DIMENSIONS = 1024

# Simple keyword extractor used for BM25 coverage boost — splits on
# non-alpha characters and drops short / stopword tokens.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "as", "or", "and", "that",
    "this", "it", "its", "not", "all", "any", "if", "when", "which",
    "used", "use", "using", "verify", "ensure", "check", "confirm",
})


def _extract_keywords(text: str) -> List[str]:
    import re as _re
    words = _re.split(r"[^a-zA-Z]+", text or "")
    return [w for w in words if len(w) >= 4 and w.lower() not in _STOPWORDS]


class HybridRetrievalRouter:
    def __init__(
        self,
        vector_top_k: int = _VECTOR_TOP_K,
        raptor_top_k: int = _RAPTOR_TOP_K,
        max_context_chunks: int = _MAX_CONTEXT_CHUNKS,
        advanced_config: Optional[AdvancedRetrievalConfig] = None,
    ) -> None:
        self.vector_top_k = vector_top_k
        self.raptor_top_k = raptor_top_k
        self.max_context_chunks = max_context_chunks
        self.advanced_config = advanced_config or AdvancedRetrievalConfig.from_settings()
        self._vector_searcher = VectorSearcher()
        self._raptor_searcher = RAPTORSearcher()
        self._keyword_searcher = KeywordSearcher()
        self._reranker = SafeOptionalReranker(
            enable_cross_encoder=self.advanced_config.enable_cross_encoder_rerank,
        )
        self._strategy_selector = RetrievalStrategySelector()
        self._route_executor = RetrievalRouteExecutor()
        self._evidence_grader = EvidenceGrader(max_context_chunks=self.max_context_chunks)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def retrieve(
        self,
        parameter: CategoryParameterChild,
        category: StandardCategory,
        raptor_tree: Optional[RAPTORTree] = None,
        ingestion_job: Optional[StandardIngestionJob] = None,
        force_strategy: Optional[RetrievalStrategy] = None,
        override_query_text: Optional[str] = None,
    ) -> RetrievalResult:
        query_text = (override_query_text or "").strip() or build_parameter_analysis_text(parameter).strip()
        if not query_text:
            msg = f"Parameter id={parameter.id} has empty requirement text — cannot retrieve context."
            self.logger.warning("HybridRetrievalRouter.retrieve: %s", msg)
            return RetrievalResult(error=msg)

        query_embedding = self._generate_query_embedding(query_text)
        keywords = _extract_keywords(query_text)

        strategy = force_strategy or self._select_strategy(
            query_text=query_text,
            keywords=keywords,
            raptor_tree=raptor_tree,
        )

        try:
            if strategy == RetrievalStrategy.VECTOR_ONLY:
                return self._execute_vector_only(
                    query_text=query_text,
                    category=category,
                    ingestion_job=ingestion_job,
                    query_embedding=query_embedding,
                )
            if strategy == RetrievalStrategy.RAPTOR_LOW:
                return self._execute_raptor_low(query_text=query_text, raptor_tree=raptor_tree, query_embedding=query_embedding)
            if strategy == RetrievalStrategy.RAPTOR_HIGH:
                return self._execute_raptor_high(query_text=query_text, raptor_tree=raptor_tree, query_embedding=query_embedding)
            query_variants = self._get_query_variants(
                parameter=parameter,
                ingestion_job=ingestion_job,
                query_text=query_text,
            )
            return self._execute_hybrid(
                query_text=query_text,
                category=category,
                ingestion_job=ingestion_job,
                raptor_tree=raptor_tree,
                query_embedding=query_embedding,
                keywords=keywords,
                query_variants=query_variants,
            )
        except Exception as exc:
            msg = f"Strategy execution failed for strategy={strategy.value}: {exc}"
            self.logger.exception("HybridRetrievalRouter.retrieve: %s for parameter id=%s", msg, parameter.id)
            return RetrievalResult(error=msg, strategy_used=strategy, query_embedding=query_embedding)

    def _select_strategy(self, *, query_text: str, keywords: List[str], raptor_tree) -> RetrievalStrategy:
        from sdr.apps.ai.retrieval.core.types import QueryType
        # inferred_relations is used only by classify_query_type for REASONING_BASED detection
        inferred_relations: Set[str] = set()
        _relation_keywords = {"authenticate", "authorize", "encrypt", "protocol", "communicate", "connect"}
        for kw in keywords:
            if kw.lower() in _relation_keywords:
                inferred_relations.add(kw.lower())
        qtype = self._strategy_selector.classify_query_type(query_text, keywords, inferred_relations)
        return self._strategy_selector.select_strategy(
            advanced_config=self.advanced_config,
            keywords=keywords,
            inferred_relations=inferred_relations,
            query_type=qtype,
            raptor_tree=raptor_tree,
        )

    def _execute_vector_only(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_vector_only(self, **kwargs)

    def _execute_raptor_low(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_raptor_low(self, **kwargs)

    def _execute_raptor_high(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_raptor_high(self, **kwargs)

    def _execute_hybrid(self, **kwargs) -> RetrievalResult:
        return self._executor().execute_hybrid(self, **kwargs)

    def _classify_candidate_evidence(self, candidate: RetrievalCandidate, *, keywords: List[str]):
        return self._grader().classify_candidate_evidence(candidate, keywords=keywords)

    def _grade_and_filter_candidates(self, candidates: List[RetrievalCandidate], *, query_text: str, keywords: List[str]):
        return self._grader().grade_and_filter_candidates(candidates, query_text=query_text, keywords=keywords)

    def _apply_keyword_coverage_boost(self, candidates: List[RetrievalCandidate], keywords: List[str]) -> List[RetrievalCandidate]:
        return self._grader().apply_keyword_coverage_boost(candidates, keywords)

    def _build_chunks_from_vector(self, vector_response: VectorSearchResponse) -> List[str]:
        return build_chunks_from_vector(vector_response)

    def _collect_block_ids_from_vector(self, vector_response: VectorSearchResponse) -> List[str]:
        return collect_block_ids_from_vector(vector_response)

    def _vector_results_to_candidates(self, vector_response: VectorSearchResponse) -> List[RetrievalCandidate]:
        dense_candidates: List[RetrievalCandidate] = []
        for idx, result in enumerate(vector_response.results):
            text = result.child.requirement_text or ""
            if not text:
                continue
            dense_candidates.append(
                RetrievalCandidate(
                    id=f"dense:{getattr(result.child, 'stable_key', idx)}",
                    source_type="dense",
                    text=text,
                    score=float(result.cosine_similarity),
                    block_ids=[],
                    metadata={
                        "sensitivity": "internal",
                        "evidence_kind": "baseline_requirement",
                        "non_tsd_evidence": True,
                    },
                    token_count=max(1, len(text) // 4),
                )
            )
        return dense_candidates

    def _raptor_results_to_candidates(self, raptor_response: Optional[RAPTORSearchResponse]) -> List[RetrievalCandidate]:
        candidates: List[RetrievalCandidate] = []
        if raptor_response and not raptor_response.is_empty:
            for result in raptor_response.results:
                candidates.append(
                    RetrievalCandidate(
                        id=result.node.node_id,
                        source_type="raptor",
                        text=result.node.text,
                        score=float(result.cosine_similarity),
                        block_ids=list(result.source_block_ids),
                        metadata={
                            "level": result.node.level,
                            "page_numbers": list(result.node.page_numbers),
                            "section_heading": result.node.section_heading,
                            "sensitivity": "internal",
                        },
                        token_count=result.node.token_estimate,
                    )
                )
        return candidates

    def _dense_tsd_results_to_candidates(self, raptor_response: Optional[RAPTORSearchResponse]) -> List[RetrievalCandidate]:
        candidates: List[RetrievalCandidate] = []
        if raptor_response and not raptor_response.is_empty:
            for result in raptor_response.results:
                candidates.append(
                    RetrievalCandidate(
                        id=f"dense:{result.node.node_id}",
                        source_type="dense",
                        text=result.node.text,
                        score=float(result.cosine_similarity),
                        block_ids=list(result.source_block_ids),
                        metadata={
                            "level": result.node.level,
                            "page_numbers": list(result.node.page_numbers),
                            "section_heading": result.node.section_heading,
                            "sensitivity": "internal",
                        },
                        token_count=result.node.token_estimate,
                    )
                )
        return candidates

    def _collect_candidate_block_ids(self, candidates: List[RetrievalCandidate]) -> List[str]:
        block_ids: List[str] = []
        seen: Set[str] = set()
        for candidate in candidates:
            for block_id in candidate.block_ids:
                if block_id not in seen:
                    seen.add(block_id)
                    block_ids.append(block_id)
        return block_ids

    def _get_query_variants(
        self,
        *,
        parameter: CategoryParameterChild,
        ingestion_job: Optional[StandardIngestionJob],
        query_text: str,
    ) -> List[Any]:
        if not bool(getattr(settings, "AI_RETRIEVAL_QUERY_EXPANSION_ENABLED", True)):
            return []
        cache_key = f"{getattr(parameter, 'id', '')}:{getattr(ingestion_job, 'id', 'none')}"
        try:
            variant_texts = expand_retrieval_query_variants(
                query_text,
                cache_key=cache_key,
                variant_count=int(getattr(settings, "AI_RETRIEVAL_QUERY_EXPANSION_VARIANT_COUNT", 3)),
                enabled=True,
            )
        except Exception:
            self.logger.exception("HybridRetrievalRouter._get_query_variants: expansion failed")
            return []
        return [(text, self._generate_query_embedding(text)) for text in variant_texts]

    def _generate_query_embedding(self, query_text: str) -> List[float]:
        try:
            return get_embedding(text=query_text, dimensions=_EMBEDDING_DIMENSIONS) or []
        except Exception as exc:
            self.logger.error("HybridRetrievalRouter._generate_query_embedding: unexpected error: %s", exc)
            return []

    def _executor(self) -> RetrievalRouteExecutor:
        helper = getattr(self, "_route_executor", None)
        if helper is None:
            helper = RetrievalRouteExecutor()
            self._route_executor = helper
        return helper

    def _grader(self) -> EvidenceGrader:
        helper = getattr(self, "_evidence_grader", None)
        if helper is None:
            helper = EvidenceGrader(max_context_chunks=getattr(self, "max_context_chunks", _MAX_CONTEXT_CHUNKS))
            self._evidence_grader = helper
        return helper


def retrieve_context_for_parameter(
    parameter: CategoryParameterChild,
    category: StandardCategory,
    raptor_tree: Optional[RAPTORTree] = None,
    ingestion_job: Optional[StandardIngestionJob] = None,
    force_strategy: Optional[RetrievalStrategy] = None,
) -> RetrievalResult:
    router = HybridRetrievalRouter()
    return router.retrieve(
        parameter=parameter,
        category=category,
        raptor_tree=raptor_tree,
        ingestion_job=ingestion_job,
        force_strategy=force_strategy,
    )


__all__ = [
    "AdvancedRetrievalConfig",
    "HybridRetrievalRouter",
    "RetrievalResult",
    "RetrievalStrategy",
    "retrieve_context_for_parameter",
]
