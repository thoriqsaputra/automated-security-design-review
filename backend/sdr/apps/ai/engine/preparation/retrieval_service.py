from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple, Iterable


from sdr.core.config import settings

from sdr.apps.ai.tsd_processing.document_models import TSDDocument
from sdr.apps.ai.tsd_processing.content_filter import (
    content_filter_enabled,
    iter_filtered_scope_parts,
)
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree, RAPTORTreeBuilder
from sdr.apps.ai.tsd_processing.graph_builder import TSDGraph, TSDGraphBuilder
from sdr.apps.ai.tsd_processing.prepared_view import prepare_tsd_view
from sdr.apps.ai.tsd_processing.raptor_graph_linker import RaptorGraphLinker
from sdr.apps.ai.retrieval.core import RetrievalCandidate, RetrievalResult
from sdr.apps.ai.retrieval.postprocessing.evidence_grader import EvidenceGrader
from sdr.apps.ai.retrieval.postprocessing.reranker import SafeOptionalReranker
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.retrieval.searchers.graph import _extract_keywords
from sdr.apps.ai.utils.parsing import strip_thinking_block
from sdr.apps.ai.client import chat_completion
from sdr.apps.standards.models import (
    CategoryParameterParent,
    CategoryParameterChild,
    StandardCategory,
    StandardIngestionJob,
)
from sdr.apps.standards.utils import build_parameter_analysis_text
from sdr.apps.ai.engine.dto import DebatableParameter, RetrievalIndexes

logger = logging.getLogger(__name__)

_MAX_SCOPE_EVIDENCE_TERMS = 6


class RetrievalService:
    """
    Orchestrates context retrieval and index building for debate.
    Dependency-injected for testability.
    """

    def __init__(
        self,
        raptor_builder: Optional[RAPTORTreeBuilder] = None,
        graph_builder: Optional[TSDGraphBuilder] = None,
        router: Optional[HybridRetrievalRouter] = None,
        linker: Optional[RaptorGraphLinker] = None,
    ) -> None:
        self.raptor_builder = raptor_builder or RAPTORTreeBuilder()
        self.graph_builder = graph_builder or TSDGraphBuilder()
        self.router = router or HybridRetrievalRouter()
        self.linker = linker or RaptorGraphLinker()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_retrieve_many_max_concurrency(self, override: Optional[int] = None) -> int:
        if override is not None:
            return max(1, int(override))
        config = getattr(self.router, "advanced_config", None)
        return max(1, int(getattr(config, "retrieve_many_max_concurrency", 2)))

    def get_parent_retrieval_max_child_snippets(self) -> int:
        return max(1, int(getattr(settings, "AI_PARENT_RETRIEVAL_MAX_CHILD_SNIPPETS", 4)))

    def get_parent_retrieval_max_child_snippet_chars(self) -> int:
        return max(80, int(getattr(settings, "AI_PARENT_RETRIEVAL_MAX_CHILD_SNIPPET_CHARS", 240)))

    def get_parent_retrieval_max_context_chunks(self) -> int:
        return max(1, int(getattr(settings, "AI_PARENT_RETRIEVAL_MAX_CONTEXT_CHUNKS", 6)))

    def get_parent_retrieval_retry_max_context_chunks(self) -> int:
        return max(1, int(getattr(settings, "AI_PARENT_RETRIEVAL_RETRY_MAX_CONTEXT_CHUNKS", 14)))

    def child_refinement_enabled(self) -> bool:
        return bool(getattr(settings, "AI_BATCH_DEBATE_CHILD_REFINE_ENABLED", True))

    def get_child_refinement_max_context_chunks(self) -> int:
        return max(1, int(getattr(settings, "AI_BATCH_DEBATE_CHILD_REFINE_MAX_CONTEXT_CHUNKS", 6)))

    def child_refinement_include_source_blocks(self) -> bool:
        return bool(getattr(settings, "AI_BATCH_DEBATE_CHILD_REFINE_INCLUDE_SOURCE_BLOCKS", True))

    def get_child_refinement_source_block_limit(self) -> int:
        return max(0, int(getattr(settings, "AI_BATCH_DEBATE_CHILD_REFINE_SOURCE_BLOCK_LIMIT", 8)))

    def child_refinement_enable_keyword_boost(self) -> bool:
        return bool(getattr(settings, "AI_BATCH_DEBATE_CHILD_REFINE_ENABLE_KEYWORD_BOOST", True))

    def child_refinement_enable_rerank(self) -> bool:
        return bool(getattr(settings, "AI_BATCH_DEBATE_CHILD_REFINE_ENABLE_RERANK", True))

    def build_indexes(self, tsd_document: TSDDocument, progress_callbacks: Optional[Dict[str, Any]] = None) -> RetrievalIndexes:
        self.logger.info(
            "RetrievalService.build_indexes: building for '%s'",
            tsd_document.document_name,
        )
        progress_callbacks = progress_callbacks or {}
        total_started = time.monotonic()
        prepared_view = prepare_tsd_view(tsd_document)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ThreadPoolExecutor-6") as executor:
            raptor_future = executor.submit(
                self._build_raptor_tree,
                tsd_document,
                progress_callbacks.get("raptor"),
                prepared_view,
            )
            graph_future = executor.submit(
                self._build_graph,
                tsd_document,
                progress_callbacks.get("graph"),
                prepared_view,
            )
            raptor_tree = raptor_future.result()
            tsd_graph = graph_future.result()

        linking_seconds = 0.0
        if tsd_graph and not tsd_graph.is_empty() and raptor_tree and not raptor_tree.is_empty():
            try:
                link_started = time.monotonic()
                self.linker.link(tsd_graph, raptor_tree)
                linking_seconds = time.monotonic() - link_started
            except Exception:
                self.logger.exception("RetrievalService.build_indexes: failed to build RAPTOR-graph cross indexes")

        total_seconds = time.monotonic() - total_started
        graph_build_stats = getattr(tsd_graph, "build_stats", {}) or {}
        self.logger.info(
            "RetrievalService.build_indexes: timing total=%.4fs raptor_total=%.4fs graph_extraction_merge=%.4fs graph_entity_embedding=%.4fs graph_relation_embedding=%.4fs raptor_graph_linking=%.4fs",
            total_seconds,
            float(getattr(raptor_tree, "build_seconds", 0.0) or 0.0),
            float(graph_build_stats.get("extraction_merge_seconds", 0.0) or 0.0),
            float(graph_build_stats.get("entity_embedding_seconds", 0.0) or 0.0),
            float(graph_build_stats.get("relation_embedding_seconds", 0.0) or 0.0),
            linking_seconds,
        )

        return RetrievalIndexes(
            raptor_tree=raptor_tree,
            tsd_graph=tsd_graph,
        )

    def retrieve_for_parameter(
        self,
        parameter: DebatableParameter,
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        indexes: RetrievalIndexes,
        tsd_document: Optional[TSDDocument] = None,
        query_details: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        self.logger.info(
            "RetrievalService.retrieve_for_parameter: parameter id=%s",
            parameter.id,
        )

        override_query_text = self._build_override_query_text(query_details)

        result = self.router.retrieve(
            parameter=parameter,
            category=category,
            raptor_tree=indexes.raptor_tree,
            graph=indexes.tsd_graph,
            ingestion_job=ingestion_job,
            override_query_text=override_query_text,
        )
        if hasattr(self.router, "_normalize_embedding_diagnostics"):
            diagnostics = self.router._normalize_embedding_diagnostics(
                result=result,
                graph=indexes.tsd_graph,
            )
        else:
            diagnostics = {
                "graph_embedding_rerank_applied": False,
                "graph_result_count": 0,
                "graph_embedding_stats": {
                    "entity_succeeded": 0,
                    "entity_attempted": 0,
                    "entity_failed": 0,
                    "relation_succeeded": 0,
                    "relation_attempted": 0,
                    "relation_failed": 0,
                },
            }

        # If router selected VECTOR_ONLY and there is no RAPTOR/Graph index
        # available, avoid letting parameter-baseline text (vector matches)
        # be treated as citation-grade evidence. Prefer TSD-derived chunks
        # when possible; otherwise return empty context so Hunter defaults
        # to not_met rather than hallucinating from baseline wording.
        try:
            if (
                result.strategy_used == result.strategy_used.VECTOR_ONLY
                and (not indexes.raptor_tree or indexes.raptor_tree.is_empty())
                and (not indexes.tsd_graph or indexes.tsd_graph.is_empty())
            ):
                if tsd_document is not None and getattr(tsd_document, "full_text", None):
                    from sdr.apps.ai.utils.chunking import chunk_text_with_context

                    chunks = chunk_text_with_context(tsd_document.full_text)
                    result.context_chunks = [c["text"] for c in chunks]
                    result.error = None
                else:
                    result.context_chunks = []
                    result.error = (
                        "No TSD-backed indexes available; vector-only matches "
                        "are not used as evidence."
                    )
        except Exception:
            self.logger.exception(
                "RetrievalService.retrieve_for_parameter: fallback handling failed for parameter id=%s",
                parameter.id,
            )

        try:
            explicit_diagrams = result.get_diagram_block_ids() or []
            skipped_diagrams: List[Dict[str, Any]] = []
            if explicit_diagrams and tsd_document is not None:
                explicit_diagrams, skipped_diagrams = self._filter_loadable_diagram_block_ids(
                    explicit_diagrams,
                    tsd_document=tsd_document,
                    max_diagrams=len(explicit_diagrams),
                )
                result.diagram_block_ids = explicit_diagrams
            if not explicit_diagrams and tsd_document is not None:
                result.diagram_block_ids = []
            elif explicit_diagrams:
                result.evidence_metadata = {
                    **(result.evidence_metadata or {}),
                    "diagram_id_source": "explicit",
                    "diagram_inference": {
                        "skipped_diagrams": skipped_diagrams,
                    },
                }
        except Exception:
            self.logger.exception(
                "RetrievalService.retrieve_for_parameter: explicit diagram block handling failed for parameter id=%s",
                parameter.id,
            )

        strategy_value = getattr(result.strategy_used, "value", result.strategy_used)
        self.logger.info(
            "RetrievalService.retrieve_for_parameter: [SUCCESS] "
            "parameter id=%s strategy=%s context_chunks=%d diagram_block_ids=%d "
            "graph_embed_rerank=%s graph_result_count=%d "
            "entity_embed=%d/%d(failed=%d) relation_embed=%d/%d(failed=%d)",
            parameter.id,
            strategy_value if strategy_value else "unknown",
            len(result.context_chunks or []),
            len(result.get_diagram_block_ids() or []),
            diagnostics["graph_embedding_rerank_applied"],
            diagnostics["graph_result_count"],
            diagnostics["graph_embedding_stats"]["entity_succeeded"],
            diagnostics["graph_embedding_stats"]["entity_attempted"],
            diagnostics["graph_embedding_stats"]["entity_failed"],
            diagnostics["graph_embedding_stats"]["relation_succeeded"],
            diagnostics["graph_embedding_stats"]["relation_attempted"],
            diagnostics["graph_embedding_stats"]["relation_failed"],
        )

        return result

    def refine_parent_result_for_child(
        self,
        *,
        parent_result: RetrievalResult,
        parameter: DebatableParameter,
        query_details: Optional[Dict[str, Any]] = None,
        tsd_document: Optional[TSDDocument] = None,
    ) -> RetrievalResult:
        if not self.child_refinement_enabled():
            return parent_result
        if parent_result is None:
            return RetrievalResult(error="Missing parent retrieval result for child refinement.")

        query_details = dict(query_details or {})
        query_text = self._build_override_query_text(query_details) or build_parameter_analysis_text(parameter).strip()
        keywords = _extract_keywords(query_text)
        candidates = self._build_child_refinement_candidates(
            parent_result=parent_result,
            tsd_document=tsd_document,
        )
        if not candidates:
            metadata = dict(getattr(parent_result, "evidence_metadata", {}) or {})
            metadata["child_refinement"] = {
                "applied": True,
                "query_text": query_text,
                "query_hash": hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:12],
                "candidate_count": 0,
                "selected_count": 0,
                "selected_block_ids": [],
                "from_parent_group": True,
            }
            return RetrievalResult(
                context_chunks=list(parent_result.context_chunks or []),
                source_block_ids=list(parent_result.source_block_ids or []),
                block_source_map=dict(parent_result.block_source_map or {}),
                diagram_block_ids=list(parent_result.diagram_block_ids or []),
                strategy_used=parent_result.strategy_used,
                query_embedding=list(parent_result.query_embedding or []),
                vector_response=parent_result.vector_response,
                raptor_response=parent_result.raptor_response,
                graph_response=parent_result.graph_response,
                graph_node_ids=list(parent_result.graph_node_ids or []),
                graph_edge_ids=list(parent_result.graph_edge_ids or []),
                grounded_texts=list(parent_result.grounded_texts or []),
                evidence_metadata=metadata,
                error=parent_result.error,
            )

        if self.child_refinement_enable_keyword_boost():
            candidates = self._build_refinement_grader().apply_keyword_coverage_boost(candidates, keywords)

        selected_candidates, evidence_metadata = self._build_refinement_grader().grade_and_filter_candidates(
            candidates,
            query_text=query_text,
            keywords=keywords,
        )
        if self.child_refinement_enable_rerank():
            selected_candidates = self._build_refinement_reranker().rerank(
                query=query_text,
                candidates=selected_candidates,
                top_k=self.get_child_refinement_max_context_chunks(),
            )
        else:
            selected_candidates = sorted(selected_candidates, key=lambda c: c.score, reverse=True)[
                : self.get_child_refinement_max_context_chunks()
            ]

        refined_context_chunks: List[str] = []
        seen_chunks = set()
        refined_source_block_ids: List[str] = []
        seen_block_ids = set()
        parent_block_source_map = dict(getattr(parent_result, "block_source_map", {}) or {})
        refined_block_source_map: Dict[str, Dict[str, Any]] = {}

        for candidate in selected_candidates:
            text = (candidate.text or "").strip()
            if text and text not in seen_chunks:
                refined_context_chunks.append(text)
                seen_chunks.add(text)
            for block_id in candidate.block_ids or []:
                if not block_id or block_id in seen_block_ids:
                    continue
                seen_block_ids.add(block_id)
                refined_source_block_ids.append(block_id)
                if block_id in parent_block_source_map:
                    refined_block_source_map[block_id] = dict(parent_block_source_map[block_id])
                else:
                    refined_block_source_map[block_id] = {
                        "retrieval_origin": candidate.source_type,
                        "retrieval_origin_label": str(candidate.source_type).replace("_", " ").title(),
                        "source_keys": [candidate.source_type],
                    }

        metadata = dict(getattr(parent_result, "evidence_metadata", {}) or {})
        metadata.update(evidence_metadata)
        metadata["block_source_map"] = refined_block_source_map
        metadata["child_refinement"] = {
            "applied": True,
            "query_text": query_text,
            "query_hash": hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:12],
            "candidate_count": len(candidates),
            "selected_count": len(selected_candidates),
            "selected_block_ids": list(refined_source_block_ids),
            "from_parent_group": True,
        }
        return RetrievalResult(
            context_chunks=refined_context_chunks[: self.get_child_refinement_max_context_chunks()],
            source_block_ids=refined_source_block_ids,
            block_source_map=refined_block_source_map,
            diagram_block_ids=list(parent_result.diagram_block_ids or []),
            strategy_used=parent_result.strategy_used,
            query_embedding=list(parent_result.query_embedding or []),
            vector_response=parent_result.vector_response,
            raptor_response=parent_result.raptor_response,
            graph_response=parent_result.graph_response,
            graph_node_ids=list(parent_result.graph_node_ids or []),
            graph_edge_ids=list(parent_result.graph_edge_ids or []),
            grounded_texts=list(parent_result.grounded_texts or []),
            evidence_metadata=metadata,
            error=parent_result.error,
        )

    def retrieve_many_for_parameters(
        self,
        *,
        parameters: List[DebatableParameter],
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        indexes: RetrievalIndexes,
        tsd_document: Optional[TSDDocument] = None,
        query_details_by_parameter_id: Optional[Dict[str, Dict[str, Any]]] = None,
        max_concurrency: Optional[int] = None,
    ) -> Dict[str, RetrievalResult]:
        if not parameters:
            return {}

        concurrency = min(self.get_retrieve_many_max_concurrency(max_concurrency), len(parameters))
        query_details_by_parameter_id = query_details_by_parameter_id or {}

        if concurrency <= 1 or len(parameters) <= 1:
            results: Dict[str, RetrievalResult] = {}
            for parameter in parameters:
                parameter_id = str(parameter.id)
                try:
                    results[parameter_id] = self.retrieve_for_parameter(
                        parameter=parameter,
                        category=category,
                        ingestion_job=ingestion_job,
                        indexes=indexes,
                        tsd_document=tsd_document,
                        query_details=query_details_by_parameter_id.get(parameter_id),
                    )
                except Exception as exc:
                    self.logger.warning(
                        "RetrievalService.retrieve_many_for_parameters: retrieval failed for parameter id=%s: %s",
                        parameter_id,
                        exc,
                    )
                    results[parameter_id] = RetrievalResult(error=str(exc))
            return results

        results: Dict[str, RetrievalResult] = {}
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="RetrieveMany") as executor:
            future_map = {}
            for parameter in parameters:
                parameter_id = str(parameter.id)
                future = executor.submit(
                    self.retrieve_for_parameter,
                    parameter=parameter,
                    category=category,
                    ingestion_job=ingestion_job,
                    indexes=indexes,
                    tsd_document=tsd_document,
                    query_details=query_details_by_parameter_id.get(parameter_id),
                )
                future_map[future] = parameter_id

            for future in as_completed(future_map):
                parameter_id = future_map[future]
                try:
                    results[parameter_id] = future.result()
                except Exception as exc:
                    self.logger.warning(
                        "RetrievalService.retrieve_many_for_parameters: retrieval failed for parameter id=%s: %s",
                        parameter_id,
                        exc,
                    )
                    results[parameter_id] = RetrievalResult(error=str(exc))
        return results

    def _filter_loadable_diagram_block_ids(
        self,
        diagram_block_ids: List[str],
        *,
        tsd_document: TSDDocument,
        max_diagrams: int,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        output: List[str] = []
        skipped: List[Dict[str, Any]] = []
        seen = set()
        for diagram_id in diagram_block_ids:
            if not diagram_id or diagram_id in seen:
                continue
            seen.add(diagram_id)
            diagram = tsd_document.get_diagram_by_id(diagram_id)
            if not diagram:
                skipped.append({"diagram_id": diagram_id, "reason": "not_found"})
                continue
            if not diagram.is_valid():
                skipped.append({"diagram_id": diagram_id, "reason": "too_small_or_unloadable"})
                continue
            output.append(diagram_id)
            if len(output) >= max_diagrams:
                break
        return output, skipped

    def retrieve_for_parent_group(
        self,
        *,
        parent,
        child_parameters: List[CategoryParameterChild],
        category: StandardCategory,
        ingestion_job: Optional[StandardIngestionJob],
        indexes: RetrievalIndexes,
        tsd_document: Optional[TSDDocument] = None,
        query_details: Optional[Dict[str, Any]] = None,
        max_context_chunks_override: Optional[int] = None,
    ) -> RetrievalResult:
        if not child_parameters:
            return RetrievalResult(error="No child parameters supplied for parent retrieval.")

        parent_title = (getattr(parent, "title", "") or "").strip()
        parent_description = (getattr(parent, "description", "") or "").strip()
        child_snippets = []
        max_child_snippets = self.get_parent_retrieval_max_child_snippets()
        max_child_snippet_chars = self.get_parent_retrieval_max_child_snippet_chars()
        # Sample evenly across ALL children rather than taking the first N, so a
        # family with more children than the snippet cap still gets query coverage
        # representative of the whole family, not just its first few entries.
        if len(child_parameters) <= max_child_snippets:
            sampled_children = child_parameters
        else:
            step = len(child_parameters) / max_child_snippets
            sampled_children = [
                child_parameters[min(int(idx * step), len(child_parameters) - 1)]
                for idx in range(max_child_snippets)
            ]
        for child in sampled_children:
            text = (
                build_parameter_analysis_text(child)
            ).strip()
            if text:
                child_snippets.append(text[:max_child_snippet_chars])

        details = dict(query_details or {})
        details.update(
            {
                "parent_title": parent_title,
                "parent_description": parent_description,
                "child_requirement": "\n".join(child_snippets),
            }
        )
        self.logger.info(
            "RetrievalService.retrieve_for_parent_group: parent=%s children=%d",
            getattr(parent, "id", None),
            len(child_parameters),
        )
        result = self.retrieve_for_parameter(
            parameter=child_parameters[0],
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
            query_details=details,
        )
        max_context_chunks = (
            max(1, int(max_context_chunks_override))
            if max_context_chunks_override is not None
            else self.get_parent_retrieval_max_context_chunks()
        )
        if len(result.context_chunks or []) > max_context_chunks:
            result.context_chunks = list(result.context_chunks[:max_context_chunks])
        self.logger.info(
            "RetrievalService.retrieve_for_parent_group: parent=%s prompt_child_snippets=%d context_chunks=%d",
            getattr(parent, "id", None),
            len(child_snippets),
            len(result.context_chunks or []),
        )
        return result

    def _build_override_query_text(self, query_details: Optional[Dict[str, Any]]) -> Optional[str]:
        if not query_details:
            return None
        parent_title = (query_details.get("parent_title") or "").strip()
        parent_description = (query_details.get("parent_description") or "").strip()
        child_requirement = (query_details.get("child_requirement") or "").strip()
        contract_then = (query_details.get("contract_then") or "").strip()
        not_sufficient = query_details.get("contract_not_sufficient") or []
        domain_keywords = query_details.get("domain_keywords") or []
        family_scope_terms = query_details.get("family_scope_terms") or []
        parts = [
            parent_title,
            parent_description,
            child_requirement,
            contract_then,
            " ".join([x for x in not_sufficient[:2] if isinstance(x, str)]),
            " ".join([x for x in family_scope_terms[:8] if isinstance(x, str)]),
            " ".join([x for x in domain_keywords if isinstance(x, str)]),
        ]
        return "\n".join([p for p in parts if p]).strip() or None

    def _build_refinement_grader(self) -> EvidenceGrader:
        return EvidenceGrader(max_context_chunks=self.get_child_refinement_max_context_chunks())

    def _build_refinement_reranker(self) -> SafeOptionalReranker:
        enable_cross_encoder = bool(
            getattr(getattr(self.router, "advanced_config", None), "enable_cross_encoder_rerank", False)
        )
        return SafeOptionalReranker(enable_cross_encoder=enable_cross_encoder)

    def _build_child_refinement_candidates(
        self,
        *,
        parent_result: RetrievalResult,
        tsd_document: Optional[TSDDocument],
    ) -> List[RetrievalCandidate]:
        candidates: List[RetrievalCandidate] = []
        parent_block_source_map = dict(getattr(parent_result, "block_source_map", {}) or {})
        source_block_limit = self.get_child_refinement_source_block_limit()
        if self.child_refinement_include_source_blocks() and tsd_document is not None and source_block_limit != 0:
            for idx, block_id in enumerate(parent_result.source_block_ids or []):
                if source_block_limit and idx >= source_block_limit:
                    break
                if not block_id or "_d" in block_id:
                    continue
                try:
                    block = tsd_document.get_block_by_id(block_id)
                except Exception:
                    block = None
                text = (getattr(block, "text", "") or "").strip() if block is not None else ""
                if not text:
                    continue
                provenance = parent_block_source_map.get(block_id) or {}
                source_type = str(
                    provenance.get("retrieval_origin")
                    or (provenance.get("source_keys") or ["parent_group"])[0]
                )
                candidates.append(
                    RetrievalCandidate(
                        id=f"block:{block_id}",
                        source_type=source_type,
                        text=text,
                        score=1.0,
                        block_ids=[block_id],
                        metadata={
                            "source": "parent_group_source_block",
                            "section_heading": getattr(block, "section_heading", None),
                            "page_numbers": [getattr(block, "page_number", None)] if getattr(block, "page_number", None) is not None else [],
                            "sensitivity": "internal",
                        },
                        token_count=max(1, len(text) // 4),
                    )
                )

        for idx, chunk in enumerate(parent_result.context_chunks or [], start=1):
            text = (chunk or "").strip()
            if not text:
                continue
            candidates.append(
                RetrievalCandidate(
                    id=f"context:{idx}",
                    source_type="parent_group_context",
                    text=text,
                    score=max(0.1, 0.5 - (idx * 0.01)),
                    block_ids=[],
                    metadata={
                        "source": "parent_group_context",
                        "sensitivity": "internal",
                    },
                    token_count=max(1, len(text) // 4),
                )
            )
        return candidates

    def _extract_json_payload(self, text: str) -> str:
        text = strip_thinking_block(text)
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _build_raptor_tree(self, tsd_document: TSDDocument, progress_callback=None, prepared_view=None) -> Optional[RAPTORTree]:
        try:
            started = time.monotonic()
            self.logger.info(
                "RetrievalService._build_raptor_tree: building for '%s'",
                tsd_document.document_name,
            )
            tree = self.raptor_builder.build(tsd_document, progress_callback=progress_callback, prepared_view=prepared_view)
            if tree.is_empty():
                self.logger.warning(
                    "RetrievalService._build_raptor_tree: empty tree for '%s'",
                    tsd_document.document_name,
                )
                return None
            self.logger.info(
                "RetrievalService._build_raptor_tree: [SUCCESS] %d node(s)",
                tree.total_nodes,
            )
            setattr(tree, "build_seconds", time.monotonic() - started)
            return tree
        except Exception as exc:
            self.logger.exception(
                "RetrievalService._build_raptor_tree: failed: %s", exc
            )
            if progress_callback:
                progress_callback(
                    {
                        "status": "failed",
                        "progress_percent": 100,
                        "current_step": "RAPTOR index failed",
                    }
                )
            return None

    def _build_graph(self, tsd_document: TSDDocument, progress_callback=None, prepared_view=None) -> Optional[TSDGraph]:
        try:
            self.logger.info(
                "RetrievalService._build_graph: building for '%s'",
                tsd_document.document_name,
            )
            graph = self.graph_builder.build(tsd_document, progress_callback=progress_callback, prepared_view=prepared_view)
            if graph.is_empty():
                self.logger.warning(
                    "RetrievalService._build_graph: empty graph for '%s'",
                    tsd_document.document_name,
                )
                return None
            self.logger.info(
                "RetrievalService._build_graph: [SUCCESS] %d entity(ies) %d relation(s)",
                graph.total_entities,
                graph.total_relations,
            )
            return graph
        except Exception as exc:
            self.logger.exception(
                "RetrievalService._build_graph: failed: %s", exc
            )
            if progress_callback:
                progress_callback(
                    {
                        "status": "failed",
                        "progress_percent": 100,
                        "current_step": "GraphRAG index failed",
                    }
                )
            return None
