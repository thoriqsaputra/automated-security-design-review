from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple, Iterable


from sdr.core.config import settings

from sdr.apps.ai.client.session import capture_current_context
from sdr.apps.ai.tsd_processing.document_models import TSDDocument
from sdr.apps.ai.tsd_processing.content_filter import (
    content_filter_enabled,
    iter_filtered_scope_parts,
)
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree, RAPTORTreeBuilder
from sdr.apps.ai.tsd_processing.prepared_view import prepare_tsd_view
from sdr.apps.ai.retrieval.core import RetrievalCandidate, RetrievalResult
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.utils.parsing import strip_thinking_block
from sdr.apps.ai.client import chat_completion
from sdr.apps.standards.models import (
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
        router: Optional[HybridRetrievalRouter] = None,
    ) -> None:
        self.raptor_builder = raptor_builder or RAPTORTreeBuilder()
        self.router = router or HybridRetrievalRouter()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_retrieve_many_max_concurrency(self, override: Optional[int] = None) -> int:
        if override is not None:
            return max(1, int(override))
        config = getattr(self.router, "advanced_config", None)
        return max(1, int(getattr(config, "retrieve_many_max_concurrency", 2)))

    def build_indexes(self, tsd_document: TSDDocument, progress_callbacks: Optional[Dict[str, Any]] = None) -> RetrievalIndexes:
        self.logger.info(
            "RetrievalService.build_indexes: building for '%s'",
            tsd_document.document_name,
        )
        progress_callbacks = progress_callbacks or {}
        total_started = time.monotonic()
        prepared_view = prepare_tsd_view(tsd_document)
        raptor_tree = self._build_raptor_tree(
            tsd_document,
            progress_callbacks.get("raptor"),
            prepared_view,
        )
        total_seconds = time.monotonic() - total_started
        self.logger.info(
            "RetrievalService.build_indexes: timing total=%.4fs raptor_total=%.4fs",
            total_seconds,
            float(getattr(raptor_tree, "build_seconds", 0.0) or 0.0),
        )
        return RetrievalIndexes(raptor_tree=raptor_tree)

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
            ingestion_job=ingestion_job,
            override_query_text=override_query_text,
        )

        # If router selected VECTOR_ONLY and there is no RAPTOR/Graph index
        # available, avoid letting parameter-baseline text (vector matches)
        # be treated as citation-grade evidence. Prefer TSD-derived chunks
        # when possible; otherwise return empty context so Hunter defaults
        # to not_met rather than hallucinating from baseline wording.
        try:
            if (
                result.strategy_used == result.strategy_used.VECTOR_ONLY
                and (not indexes.raptor_tree or indexes.raptor_tree.is_empty())
            ):
                if tsd_document is not None and getattr(tsd_document, "full_text", None):
                    from sdr.apps.ai.utils.chunking import chunk_text_with_context

                    chunks = chunk_text_with_context(tsd_document.full_text)
                    result.context_chunks = [c["text"] for c in chunks]
                    result.context_chunk_block_ids = [[] for _ in result.context_chunks]
                    result.error = None
                else:
                    result.context_chunks = []
                    result.context_chunk_block_ids = []
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
            "parameter id=%s strategy=%s context_chunks=%d diagram_block_ids=%d",
            parameter.id,
            strategy_value if strategy_value else "unknown",
            len(result.context_chunks or []),
            len(result.get_diagram_block_ids() or []),
        )

        return result

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
                    capture_current_context(self.retrieve_for_parameter),
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
                        "current_step": "RAPTOR index failed",
                    }
                )
            return None

