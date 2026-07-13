from __future__ import annotations

import base64
import logging
import re
from typing import Any, List

from sdr.apps.ai.client import get_embedding
from sdr.apps.ai.tsd_processing.diagram_ocr import extract_diagram_text
from sdr.apps.ai.agents.vision import VisionAgent
from sdr.apps.ai.prompts.agents import (
    DIAGRAM_GATEKEEPER_SYSTEM_PROMPT,
    build_diagram_gatekeeper_prompt,
)

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    BM25_AVAILABLE = True
except Exception:
    BM25_AVAILABLE = False

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_RRF_K = 60
_TYPE_MATCH_BONUS = 1.0 / _RRF_K

_MIN_RESULTS = 3

_GATEKEEPER_BATCH_SIZE = 15

_GATEKEEPER_CONFIDENCE_THRESHOLD = 0.5

_TYPE_KEYWORDS = (
    ("sequence", ("sequence diagram",)),
    ("data_flow", ("data flow diagram", "data-flow diagram", "dfd")),
    ("architecture", ("architecture diagram", "deployment diagram", "system architecture")),
)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _normalize_query_segment(text: str, *, max_chars: int | None = None) -> str:
    normalized = " ".join((text or "").split()).strip()
    if max_chars is not None:
        normalized = normalized[:max_chars].strip()
    return normalized


def _classify_diagram_type(*texts: str) -> str | None:
    combined = " ".join(t for t in texts if t).lower()
    if not combined:
        return None
    for diagram_type, keywords in _TYPE_KEYWORDS:
        if any(keyword in combined for keyword in keywords):
            return diagram_type
    return None


class DiagramRequirementSelector:
    def __init__(self, *, config, workflow_repository) -> None:
        self.config = config
        self.workflow_repository = workflow_repository
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._gatekeeper = VisionAgent()

    def select_for_diagram(
        self,
        *,
        diagram,
        tsd_document,
        category,
        ingestion_job,
        force_strategy: str | None = None,
    ) -> List[Any]:
        top_k = self.config.vision_diagram_requirements_max_items
        caption = (getattr(diagram, "caption", "") or "").strip()
        surrounding = (getattr(diagram, "surrounding_text", "") or "").strip()
        ocr_text = self._extract_ocr_text(diagram)  # also ensures diagram.image_b64 is loaded
        diagram_type = _classify_diagram_type(caption, surrounding, ocr_text)

        if force_strategy == "naive":
            return self._fallback_requirements(
                category_id=category.id,
                ingestion_job_id=ingestion_job.id,
                top_k=top_k,
            )

        image_b64 = getattr(diagram, "image_b64", "") or ""
        if force_strategy == "gatekeeper" and image_b64:
            try:
                pool = self.workflow_repository.list_diagram_requirements(
                    category_id=category.id,
                    ingestion_job_id=ingestion_job.id,
                )
                requirements = self._gatekeeper_search(
                    image_bytes=base64.b64decode(image_b64),
                    image_format=getattr(diagram, "image_format", "png") or "png",
                    pool=pool,
                    top_k=top_k,
                    diagram_id=getattr(diagram, "diagram_id", None),
                )
            except Exception as exc:
                self.logger.warning(
                    "DiagramRequirementSelector.select_for_diagram: gatekeeper search failed for diagram_id=%s: %s",
                    getattr(diagram, "diagram_id", None),
                    exc,
                )
                requirements = []
            if requirements or force_strategy == "gatekeeper":
                return requirements

        if force_strategy == "hybrid" or force_strategy is None:
            query_text = self._build_query_text(
                diagram=diagram, tsd_document=tsd_document,
                caption=caption, surrounding=surrounding, ocr_text=ocr_text,
            )
            if not query_text:
                if force_strategy == "hybrid":
                    return []
                return self._fallback_requirements(
                    category_id=category.id,
                    ingestion_job_id=ingestion_job.id,
                    top_k=top_k,
                )

            try:
                query_vector = get_embedding(text=query_text, dimensions=1024)
            except Exception as exc:
                self.logger.warning(
                    "DiagramRequirementSelector.select_for_diagram: embedding failed for diagram_id=%s: %s",
                    getattr(diagram, "diagram_id", None),
                    exc,
                )
                query_vector = []

            if query_vector:
                try:
                    requirements = self._hybrid_search(
                        category_id=category.id,
                        ingestion_job_id=ingestion_job.id,
                        query_text=query_text,
                        query_vector=query_vector,
                        top_k=top_k,
                        diagram_type=diagram_type,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "DiagramRequirementSelector.select_for_diagram: hybrid search failed for diagram_id=%s: %s",
                        getattr(diagram, "diagram_id", None),
                        exc,
                    )
                    requirements = []
                if requirements or force_strategy == "hybrid":
                    return requirements

        return self._fallback_requirements(
            category_id=category.id,
            ingestion_job_id=ingestion_job.id,
            top_k=top_k,
        )

    def _gatekeeper_search(
        self,
        *,
        image_bytes: bytes,
        image_format: str,
        pool: List[Any],
        top_k: int,
        diagram_id: Any = None,
    ) -> List[Any]:
        if not pool:
            return []

        ranked_relevances: dict[str, tuple[float, int]] = {}
        any_batch_succeeded = False
        for start in range(0, len(pool), _GATEKEEPER_BATCH_SIZE):
            batch = pool[start:start + _GATEKEEPER_BATCH_SIZE]
            prompt = build_diagram_gatekeeper_prompt(requirements_batch=batch)
            result = self._gatekeeper.run_multimodal(
                user_prompt=prompt,
                image_bytes=image_bytes,
                image_format=image_format,
                system_prompt=DIAGRAM_GATEKEEPER_SYSTEM_PROMPT,
                log_context=f"diagram_id={diagram_id} agent=gatekeeper batch={start // _GATEKEEPER_BATCH_SIZE}",
            )
            if result is None:
                self.logger.warning(
                    "DiagramRequirementSelector._gatekeeper_search: batch %d failed for diagram_id=%s — skipping batch",
                    start // _GATEKEEPER_BATCH_SIZE,
                    diagram_id,
                )
                continue
            any_batch_succeeded = True
            for offset, item in enumerate(result.get("assessments") or []):
                if not isinstance(item, dict) or item.get("relevant") is not True:
                    continue
                req_id = str(item.get("requirement_id", "")).strip().strip("[]")
                if req_id:
                    confidence = float(item.get("confidence", 0.5) or 0.5)
                    global_order = start + offset
                    previous = ranked_relevances.get(req_id)
                    candidate = (confidence, -global_order)
                    if previous is None or candidate > previous:
                        ranked_relevances[req_id] = candidate

        if not any_batch_succeeded:
            return []

        by_key = {req.stable_key: req for req in pool}
        selected = [by_key[k] for k in ranked_relevances if k in by_key]
        selected.sort(
            key=lambda requirement: (
                ranked_relevances.get(requirement.stable_key, (0.0, 0))[0],
                ranked_relevances.get(requirement.stable_key, (0.0, 0))[1],
            ),
            reverse=True,
        )

        above_threshold = [
            req for req in selected
            if ranked_relevances[req.stable_key][0] >= _GATEKEEPER_CONFIDENCE_THRESHOLD
        ]
        if len(above_threshold) >= _MIN_RESULTS:
            selected = above_threshold
        elif len(selected) >= _MIN_RESULTS:
            selected = selected[:_MIN_RESULTS]

        if len(selected) < _MIN_RESULTS:
            for req in pool:
                if len(selected) >= _MIN_RESULTS:
                    break
                if req not in selected:
                    selected.append(req)

        return selected[:top_k]

    def _hybrid_search(
        self,
        *,
        category_id: Any,
        ingestion_job_id: Any,
        query_text: str,
        query_vector: List[float],
        top_k: int,
        diagram_type: str | None = None,
    ) -> List[Any]:
        vector_pairs = self.workflow_repository.list_diagram_requirements_with_similarity(
            category_id=category_id,
            ingestion_job_id=ingestion_job_id,
            query_embedding=query_vector,
        )
        if not vector_pairs:
            return []

        pool = [req for req, _distance in vector_pairs]
        vector_rank = {req.id: rank for rank, (req, _distance) in enumerate(vector_pairs)}

        query_tokens = _tokenize(query_text)
        if BM25_AVAILABLE and query_tokens:
            corpus_tokens = [
                _tokenize(f"{r.parent_section} {r.requirement_text} {r.verification_hint}")
                for r in pool
            ]
            bm25 = BM25Okapi(corpus_tokens)
            bm25_scores = bm25.get_scores(query_tokens)
            bm25_order = sorted(range(len(pool)), key=lambda i: bm25_scores[i], reverse=True)
            bm25_rank = {pool[i].id: rank for rank, i in enumerate(bm25_order)}
        else:
            bm25_rank = {r.id: 0 for r in pool}

        def rrf_score(req) -> float:
            score = 1.0 / (_RRF_K + vector_rank[req.id]) + 1.0 / (_RRF_K + bm25_rank[req.id])
            if diagram_type and req.diagram_type == diagram_type:
                score += _TYPE_MATCH_BONUS
            return score

        fused = sorted(pool, key=rrf_score, reverse=True)
        effective_k = max(_MIN_RESULTS, min(len(fused), top_k))
        return fused[:effective_k]

    def _extract_ocr_text(self, diagram) -> str:
        try:
            if hasattr(diagram, "ensure_image_loaded"):
                diagram.ensure_image_loaded()
            image_b64 = getattr(diagram, "image_b64", "") or ""
            if image_b64:
                return extract_diagram_text(base64.b64decode(image_b64))
        except Exception:
            self.logger.debug(
                "DiagramRequirementSelector._extract_ocr_text: OCR failed for diagram_id=%s",
                getattr(diagram, "diagram_id", None),
                exc_info=True,
            )
        return ""

    def _build_query_text(self, *, diagram, tsd_document, caption: str, surrounding: str, ocr_text: str) -> str:
        parts = []
        caption_segment = _normalize_query_segment(caption)
        if caption_segment:
            parts.append(caption_segment)
        surrounding_segment = _normalize_query_segment(surrounding)
        if surrounding_segment:
            parts.append(surrounding_segment)

        ocr_segment = _normalize_query_segment(ocr_text, max_chars=1200)
        if ocr_segment:
            parts.append(f"Visible diagram elements: {ocr_segment}")

        page_number = getattr(diagram, "page_number", None)
        pages = getattr(tsd_document, "pages", None) or []
        if isinstance(page_number, int) and pages:
            page_window_parts = []
            for candidate_page_number in (page_number - 1, page_number, page_number + 1):
                if 1 <= candidate_page_number <= len(pages):
                    page_text = _normalize_query_segment(
                        getattr(pages[candidate_page_number - 1], "all_text", "") or ""
                    )
                    if page_text:
                        page_window_parts.append(page_text)
            page_window = _normalize_query_segment(" ".join(page_window_parts), max_chars=1800)
            if page_window:
                parts.append(f"Nearby page context: {page_window}")

        deduped_parts = []
        seen = set()
        for part in parts:
            normalized = _normalize_query_segment(part)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped_parts.append(normalized)

        return "\n\n".join(deduped_parts).strip()

    def _fallback_requirements(
        self,
        *,
        category_id: Any,
        ingestion_job_id: Any,
        top_k: int,
    ) -> List[Any]:
        requirements = self.workflow_repository.list_diagram_requirements(
            category_id=category_id,
            ingestion_job_id=ingestion_job_id,
        )
        return list(requirements[:top_k])
