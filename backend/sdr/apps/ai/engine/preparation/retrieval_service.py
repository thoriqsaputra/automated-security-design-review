from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
from sdr.apps.ai.retrieval.core import RetrievalResult
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.utils.parsing import strip_thinking_block
from sdr.apps.ai.client import chat_completion
from sdr.apps.standards.models import (
    CategoryParameterParent,
    CategoryParameterChild,
    StandardCategory,
    StandardIngestionJob,
)
from sdr.apps.standards.utils import build_parameter_analysis_text
from sdr.apps.ai.engine.dto import RetrievalIndexes

logger = logging.getLogger(__name__)

_TEXT_BLOCK_ID_PATTERN = re.compile(r"^p(?P<page>\d+)_b\d+$")
_MAX_INFERRED_DIAGRAMS = 3
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
        parameter: CategoryParameterChild,
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

        override_query_text = None
        if query_details:
            parent_title = (query_details.get("parent_title") or "").strip()
            parent_description = (query_details.get("parent_description") or "").strip()
            child_requirement = (query_details.get("child_requirement") or "").strip()
            contract_then = (query_details.get("contract_then") or "").strip()
            not_sufficient = query_details.get("contract_not_sufficient") or []
            domain_keywords = query_details.get("domain_keywords") or []
            parts = [
                parent_title,
                parent_description,
                child_requirement,
                contract_then,
                " ".join([x for x in not_sufficient[:2] if isinstance(x, str)]),
                " ".join([x for x in domain_keywords if isinstance(x, str)]),
            ]
            override_query_text = "\n".join([p for p in parts if p]).strip() or None

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
                inferred_diagrams, inferred_skipped = self._infer_diagram_block_ids(
                    source_block_ids=result.source_block_ids or [],
                    tsd_document=tsd_document,
                    max_diagrams=_MAX_INFERRED_DIAGRAMS,
                )
                skipped_diagrams.extend(inferred_skipped)
                if inferred_diagrams:
                    result.diagram_block_ids = inferred_diagrams
                    result.evidence_metadata = {
                        **(result.evidence_metadata or {}),
                        "diagram_id_source": "inferred",
                        "diagram_inference": {
                            "source_page_count": len(self._extract_source_pages(result.source_block_ids or [])),
                            "inferred_count": len(inferred_diagrams),
                            "skipped_diagrams": skipped_diagrams,
                        },
                    }
                    self.logger.info(
                        "RetrievalService.retrieve_for_parameter: inferred %d diagram block id(s) for parameter id=%s",
                        len(inferred_diagrams),
                        parameter.id,
                    )
                else:
                    result.evidence_metadata = {
                        **(result.evidence_metadata or {}),
                        "diagram_id_source": "none",
                        "diagram_inference": {
                            "source_page_count": len(self._extract_source_pages(result.source_block_ids or [])),
                            "inferred_count": 0,
                            "skipped_diagrams": skipped_diagrams,
                        },
                    }
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
                "RetrievalService.retrieve_for_parameter: diagram inference failed for parameter id=%s",
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

    def _extract_source_pages(self, source_block_ids: List[str]) -> List[int]:
        pages = set()
        for block_id in source_block_ids:
            if not isinstance(block_id, str):
                continue
            match = _TEXT_BLOCK_ID_PATTERN.match(block_id)
            if not match:
                continue
            pages.add(int(match.group("page")))
        return sorted(pages)

    def _infer_diagram_block_ids(
        self,
        *,
        source_block_ids: List[str],
        tsd_document: TSDDocument,
        max_diagrams: int,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        if max_diagrams <= 0:
            return [], []
        source_pages = self._extract_source_pages(source_block_ids)
        if not source_pages:
            return [], []
        diagrams = list(getattr(tsd_document, "all_diagrams", []) or [])
        if not diagrams:
            return [], []

        ranked: List[tuple[int, int, str]] = []
        for diagram in diagrams:
            diagram_id = getattr(diagram, "diagram_id", None)
            page_number = int(getattr(diagram, "page_number", 0) or 0)
            if not diagram_id or page_number <= 0:
                continue
            nearest = min(abs(page_number - source_page) for source_page in source_pages)
            if nearest > 1:
                continue
            ranked.append((nearest, page_number, diagram_id))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        ranked_ids = [diagram_id for _, _, diagram_id in ranked]
        return self._filter_loadable_diagram_block_ids(
            ranked_ids,
            tsd_document=tsd_document,
            max_diagrams=max_diagrams,
        )

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
    ) -> RetrievalResult:
        if not child_parameters:
            return RetrievalResult(error="No child parameters supplied for parent retrieval.")

        parent_title = (getattr(parent, "title", "") or "").strip()
        parent_description = (getattr(parent, "description", "") or "").strip()
        child_snippets = []
        for child in child_parameters[:8]:
            text = (
                build_parameter_analysis_text(child)
            ).strip()
            if text:
                child_snippets.append(text[:600])

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
        return self.retrieve_for_parameter(
            parameter=child_parameters[0],
            category=category,
            ingestion_job=ingestion_job,
            indexes=indexes,
            tsd_document=tsd_document,
            query_details=details,
        )

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
