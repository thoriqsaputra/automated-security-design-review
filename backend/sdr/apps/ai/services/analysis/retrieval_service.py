# apps/ai/services/retrieval_service.py
"""
Retrieval Service — Builds indexes and retrieves context for debate.
Responsibility: RAPTOR tree, GraphRAG, parameter pre-filtering.
No agent logic. No database writes.
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Tuple

from sdr.core.config import settings

from sdr.apps.ai.tsd_processing.ingestor import TSDDocument
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree, RAPTORTreeBuilder
from sdr.apps.ai.tsd_processing.graph_builder import TSDGraph, TSDGraphBuilder
from sdr.apps.ai.retrieval.router import HybridRetrievalRouter, RetrievalResult
from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.prompts.analysis_prompts import (
    PARAMETER_APPLICABILITY_SYSTEM_PROMPT,
    build_parameter_applicability_prompt,
)
from sdr.apps.standards.models import (
    CategoryParameterChild,
    StandardCategory,
    StandardIngestionJob,
)
from sdr.apps.standards.utils import build_parameter_analysis_text
from .dto import RetrievalIndexes, ParameterApplicabilityResult


logger = logging.getLogger(__name__)

_APPLICABILITY_TEMPERATURE = 0.0
_APPLICABILITY_MAX_TOKENS = 2048
_APPLICABILITY_DEFAULT_CONFIDENCE_THRESHOLD = 0.75
_APPLICABILITY_PREFILTER_SHARE_BREAKER = 0.8
_APPLICABILITY_PREFILTER_SHARE_BREAKER_MIN_BATCH = 4
_TEXT_BLOCK_ID_PATTERN = re.compile(r"^p(?P<page>\d+)_b\d+$")
_MAX_INFERRED_DIAGRAMS = 3


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
    ) -> None:
        self.raptor_builder = raptor_builder or RAPTORTreeBuilder()
        self.graph_builder = graph_builder or TSDGraphBuilder()
        self.router = router or HybridRetrievalRouter()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def build_indexes(self, tsd_document: TSDDocument, progress_callbacks: Optional[Dict[str, Any]] = None) -> RetrievalIndexes:
        """
        Builds RAPTOR and GraphRAG indexes from the TSD document.
        Both are best-effort — failures are logged but don't raise.

        Args:
            tsd_document: Parsed TSD document.

        Returns:
            RetrievalIndexes with (optional) RAPTOR tree and graph.
        """
        self.logger.info(
            "RetrievalService.build_indexes: building for '%s'",
            tsd_document.document_name,
        )
        progress_callbacks = progress_callbacks or {}
        total_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ThreadPoolExecutor-6") as executor:
            raptor_future = executor.submit(
                self._build_raptor_tree,
                tsd_document,
                progress_callbacks.get("raptor"),
            )
            graph_future = executor.submit(
                self._build_graph,
                tsd_document,
                progress_callbacks.get("graph"),
            )
            raptor_tree = raptor_future.result()
            tsd_graph = graph_future.result()

        linking_seconds = 0.0
        if tsd_graph and not tsd_graph.is_empty() and raptor_tree and not raptor_tree.is_empty():
            try:
                link_started = time.monotonic()
                self.graph_builder.link_raptor_entities(tsd_graph, raptor_tree)
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
        """
        Retrieves context for a single parameter using the hybrid router.

        Args:
            parameter: The CategoryParameterChild to retrieve context for.
            category: The StandardCategory scope.
            ingestion_job: The active StandardIngestionJob.
            indexes: Pre-built RetrievalIndexes.

        Returns:
            RetrievalResult with context chunks and diagram block_ids.
        """
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
        """
        Retrieves shared context for a parent section and its child parameters.

        The router still requires a parameter object for strategy selection, so
        the first child is used as the representative while the override query
        carries the parent title/description plus child requirement snippets.
        """
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

    def pre_filter_parameters(
        self,
        parameters: List[CategoryParameterChild],
        indexes: RetrievalIndexes,
        *,
        category_code: str = "",
    ) -> ParameterApplicabilityResult:
        """
        Pre-filters parameters that are clearly N/A using RAPTOR root summary.

        Args:
            parameters: All CategoryParameterChild records to evaluate.
            indexes: Pre-built RetrievalIndexes (for RAPTOR tree).

        Returns:
            ParameterApplicabilityResult with applicable and pre-filtered lists.
        """
        self.logger.info(
            "RetrievalService.pre_filter_parameters: filtering %d parameter(s)",
            len(parameters),
        )

        if not getattr(settings, "AI_PARAMETER_APPLICABILITY_PREFILTER_ENABLED", False):
            self.logger.debug(
                "RetrievalService.pre_filter_parameters: disabled by setting for category=%s",
                category_code or "unknown",
            )
            return ParameterApplicabilityResult(
                applicable_parameters=parameters,
                pre_filtered_parameters=[],
                pre_filter_details={},
            )

        if not indexes.raptor_tree or indexes.raptor_tree.is_empty():
            self.logger.debug(
                "RetrievalService.pre_filter_parameters: no RAPTOR tree — "
                "skipping pre-filter"
            )
            return ParameterApplicabilityResult(
                applicable_parameters=parameters,
                pre_filtered_parameters=[],
                pre_filter_details={},
            )

        root = indexes.raptor_tree.root_node
        if not root or not root.text:
            self.logger.debug(
                "RetrievalService.pre_filter_parameters: no root node text — "
                "skipping pre-filter"
            )
            return ParameterApplicabilityResult(
                applicable_parameters=parameters,
                pre_filtered_parameters=[],
                pre_filter_details={},
            )

        confidence_threshold = float(
            getattr(
                settings,
                "AI_PARAMETER_APPLICABILITY_CONFIDENCE_THRESHOLD",
                _APPLICABILITY_DEFAULT_CONFIDENCE_THRESHOLD,
            )
        )
        param_dicts = [
            {
                "id": str(p.id),
                "requirement_text": build_parameter_analysis_text(p),
                "parent_title": getattr(getattr(p, "parent", None), "title", "") or "",
                "domain_keywords": list(self._build_parameter_domain_keywords(p)),
                "contract_summary": self._build_parameter_contract_summary(p),
            }
            for p in parameters
        ]

        prompt = build_parameter_applicability_prompt(
            document_summary=root.text,
            parameters=param_dicts,
            category_code=category_code,
        )

        try:
            response = chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": PARAMETER_APPLICABILITY_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                component="tsd_ingestion",
                temperature=_APPLICABILITY_TEMPERATURE,
                max_tokens=_APPLICABILITY_MAX_TOKENS,
                response_format={"type": "json_object"},
            )

            if response.error or not response.content:
                self.logger.warning(
                    "RetrievalService.pre_filter_parameters: LLM error — "
                    "returning all parameters as applicable: %s",
                    response.error,
                )
                return ParameterApplicabilityResult(
                    applicable_parameters=parameters,
                    pre_filtered_parameters=[],
                    pre_filter_details={},
                )

            try:
                result = json.loads(response.content.strip())
            except json.JSONDecodeError:
                self.logger.warning("RetrievalService.pre_filter_parameters: JSON decode error. Attempting LLM repair.")
                repair_prompt = (
                    "The following JSON has syntax errors (e.g. missing commas, unescaped quotes). "
                    "Fix it and return ONLY valid JSON without any markdown or conversational text.\n\n"
                    f"{response.content.strip()}"
                )
                repair_resp = chat_completion(
                    messages=[{"role": "user", "content": repair_prompt}],
                    component="fallback",
                    temperature=0.0,
                    max_tokens=_APPLICABILITY_MAX_TOKENS
                )
                if repair_resp.error:
                    self.logger.warning("RetrievalService.pre_filter_parameters: JSON repair API error: %s", repair_resp.error)
                    return ParameterApplicabilityResult(applicable_parameters=parameters, pre_filtered_parameters=[], pre_filter_details={})
                try:
                    result = json.loads((repair_resp.content or "").strip())
                    self.logger.info("RetrievalService.pre_filter_parameters: successfully repaired JSON via LLM fallback.")
                except json.JSONDecodeError:
                    self.logger.warning("RetrievalService.pre_filter_parameters: JSON repair failed.")
                    return ParameterApplicabilityResult(applicable_parameters=parameters, pre_filtered_parameters=[], pre_filter_details={})
            pre_filter_details: Dict[str, Dict[str, Any]] = {}
            not_applicable_ids = {
                r["id"]
                for r in result.get("results", [])
                if not r.get("applicable", True)
                and float(r.get("confidence", 0)) >= confidence_threshold
            }
            for item in result.get("results", []):
                item_id = str(item.get("id", "")).strip()
                if not item_id:
                    continue
                pre_filter_details[item_id] = {
                    "applicable": bool(item.get("applicable", True)),
                    "prefilter_reason": item.get("reason"),
                    "prefilter_confidence": float(item.get("confidence", 0) or 0),
                    "category_code": category_code or None,
                }

            applicable = [
                p for p in parameters if str(p.id) not in not_applicable_ids
            ]
            pre_filtered = [
                p for p in parameters if str(p.id) in not_applicable_ids
            ]

            if (
                len(parameters) >= _APPLICABILITY_PREFILTER_SHARE_BREAKER_MIN_BATCH
                and pre_filtered
                and (len(pre_filtered) / len(parameters)) >= _APPLICABILITY_PREFILTER_SHARE_BREAKER
            ):
                self.logger.warning(
                    "RetrievalService.pre_filter_parameters: breaker tripped for category=%s total=%d pre_filtered=%d",
                    category_code or "unknown",
                    len(parameters),
                    len(pre_filtered),
                )
                return ParameterApplicabilityResult(
                    applicable_parameters=parameters,
                    pre_filtered_parameters=[],
                    pre_filter_details={},
                )

            self.logger.info(
                "RetrievalService.pre_filter_parameters: [SUCCESS] "
                "applicable=%d pre_filtered=%d",
                len(applicable),
                len(pre_filtered),
            )

            return ParameterApplicabilityResult(
                applicable_parameters=applicable,
                pre_filtered_parameters=pre_filtered,
                pre_filter_details=pre_filter_details,
            )

        except Exception as exc:
            self.logger.exception(
                "RetrievalService.pre_filter_parameters: error — returning all: %s",
                exc,
            )
            return ParameterApplicabilityResult(
                applicable_parameters=parameters,
                pre_filtered_parameters=[],
                pre_filter_details={},
            )

    # ---- Private helpers ----

    def _build_parameter_domain_keywords(self, parameter: CategoryParameterChild) -> List[str]:
        parent = getattr(parameter, "parent", None)
        query_details = {
            "parent_title": (getattr(parent, "title", "") or "").strip(),
            "parent_description": (getattr(parent, "description", "") or "").strip(),
            "child_requirement": build_parameter_analysis_text(parameter).strip(),
        }
        domain_keywords = [
            term
            for term in (
                query_details["parent_title"],
                query_details["parent_description"],
            )
            if term
        ]
        normalized_child = query_details["child_requirement"].lower()
        for keyword in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", normalized_child):
            if keyword not in {item.lower() for item in domain_keywords}:
                domain_keywords.append(keyword)
            if len(domain_keywords) >= 10:
                break
        return domain_keywords[:10]

    def _build_parameter_contract_summary(self, parameter: CategoryParameterChild) -> str:
        parent = getattr(parameter, "parent", None)
        parts = [
            (getattr(parent, "title", "") or "").strip(),
            (getattr(parent, "description", "") or "").strip(),
            build_parameter_analysis_text(parameter).strip(),
        ]
        summary = " | ".join(part for part in parts if part)
        return summary[:400]

    def _build_raptor_tree(self, tsd_document: TSDDocument, progress_callback=None) -> Optional[RAPTORTree]:
        try:
            started = time.monotonic()
            self.logger.info(
                "RetrievalService._build_raptor_tree: building for '%s'",
                tsd_document.document_name,
            )
            tree = self.raptor_builder.build(tsd_document, progress_callback=progress_callback)
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

    def _build_graph(self, tsd_document: TSDDocument, progress_callback=None) -> Optional[TSDGraph]:
        try:
            self.logger.info(
                "RetrievalService._build_graph: building for '%s'",
                tsd_document.document_name,
            )
            graph = self.graph_builder.build(tsd_document, progress_callback=progress_callback)
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
