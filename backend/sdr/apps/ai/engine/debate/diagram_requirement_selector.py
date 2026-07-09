from __future__ import annotations

import base64
import logging
import re
from typing import Any, List

from sdr.apps.ai.client import get_embedding
from sdr.apps.ai.tsd_processing.visual_marker import extract_diagram_text
from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate
from sdr.apps.ai.retrieval.postprocessing.reranker import SafeOptionalReranker
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
# Same order of magnitude as gaining one rank-1 vote from either ranker — a
# nudge, not a filter, so a different-typed-but-genuinely-relevant item can
# still outrank a same-typed-but-irrelevant one.
_TYPE_MATCH_BONUS = 1.0 / _RRF_K

# Adaptive-cutoff constants. `config.vision_diagram_requirements_max_items` is
# treated as a ceiling, not a forced count — a diagram with genuinely few
# relevant requirements shouldn't be padded with irrelevant ones just to hit
# a fixed number, and one with many shouldn't be truncated below the ceiling
# if more of them clear the relevance bar.
#
# Cutoff is a RELATIVE margin from the top-scoring candidate, not an absolute
# threshold: ms-marco-MiniLM-L-6-v2's raw logits aren't centered at 0 (checked
# empirically — a genuinely relevant pair scored -5.2, irrelevant pairs scored
# -11.2 to -11.4), so sigmoid(score) >= 0.5 rejects almost everything
# regardless of relevance. The ~6-point gap observed between a relevant and
# irrelevant pair is what a relative margin is calibrated against.
_RELEVANCE_MARGIN = 3.0
_MIN_RESULTS = 3

# Gatekeeper: batch size for the primary (real-reasoning) selection path.
# Kept well under typical pool sizes so each call stays focused rather than
# asking the model to judge dozens of candidates at once in one shot.
_GATEKEEPER_BATCH_SIZE = 15

# Keyword phrasing matched against caption/surrounding-text/OCR text to guess
# a diagram's type, using the exact vocabulary stored in
# CategoryDiagramRequirement.diagram_type ("data_flow" | "sequence" | "architecture").
# Order matters: first match wins. Unlike the earlier (reverted) attempt, this
# is only ever used as a soft score nudge below — never to exclude candidates,
# since most genuinely relevant items for a diagram carry a *different* type
# label than the diagram itself (confirmed empirically this session).
_TYPE_KEYWORDS = (
    ("sequence", ("sequence diagram",)),
    ("data_flow", ("data flow diagram", "data-flow diagram", "dfd")),
    ("architecture", ("architecture diagram", "deployment diagram", "system architecture")),
)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


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
        self._reranker = SafeOptionalReranker(enable_cross_encoder=True)
        self._gatekeeper = VisionAgent()

    def select_for_diagram(
        self,
        *,
        diagram,
        tsd_document,
        category,
        ingestion_job,
    ) -> List[Any]:
        top_k = self.config.vision_diagram_requirements_max_items
        caption = (getattr(diagram, "caption", "") or "").strip()
        surrounding = (getattr(diagram, "surrounding_text", "") or "").strip()
        ocr_text = self._extract_ocr_text(diagram)  # also ensures diagram.image_b64 is loaded
        diagram_type = _classify_diagram_type(caption, surrounding, ocr_text)

        # Primary path: real reasoning over the diagram image, batched. Only
        # falls through to the embedding/BM25 hybrid path below if every
        # batch errors out or the diagram has no usable image.
        image_b64 = getattr(diagram, "image_b64", "") or ""
        if image_b64:
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
            if requirements:
                return requirements

        query_text = self._build_query_text(
            diagram=diagram, tsd_document=tsd_document,
            caption=caption, surrounding=surrounding, ocr_text=ocr_text,
        )
        if not query_text:
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
            if requirements:
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
        """Real-reasoning primary path: batches the full candidate pool through
        VisionAgent (same model as Hunter/Critic/Mediator), asking a lean
        scope-only question per batch — same question the ground-truth judge
        asks: "is this requirement checkable from this diagram?" A distance
        metric (embeddings/BM25/cross-encoder) can only ever approximate this;
        actual reasoning is what the precision problem needed."""
        if not pool:
            return []

        relevant_keys: set[str] = set()
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
            for item in result.get("assessments") or []:
                if not isinstance(item, dict) or item.get("relevant") is not True:
                    continue
                req_id = str(item.get("requirement_id", "")).strip().strip("[]")
                if req_id:
                    relevant_keys.add(req_id)

        if not any_batch_succeeded:
            return []

        by_key = {req.stable_key: req for req in pool}
        selected = [by_key[k] for k in relevant_keys if k in by_key]
        selected.sort(key=lambda r: pool.index(r))  # preserve original pool ordering

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
        """BM25 + vector cosine, fused via Reciprocal Rank Fusion, optionally
        cross-encoder reranked. The category's requirement pool is small
        (tens of items), so both rankers score the FULL pool — no need for
        approximate/top-k-limited search at either stage."""
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
            # No BM25 signal available — treat every item as tied, so fusion
            # degrades gracefully to vector-only ranking.
            bm25_rank = {r.id: 0 for r in pool}

        def rrf_score(req) -> float:
            score = 1.0 / (_RRF_K + vector_rank[req.id]) + 1.0 / (_RRF_K + bm25_rank[req.id])
            if diagram_type and req.diagram_type == diagram_type:
                score += _TYPE_MATCH_BONUS
            return score

        fused = sorted(pool, key=rrf_score, reverse=True)
        # Must be >= top_k (the result-count ceiling) or we'd silently cap
        # recall below whatever `top_k` allows; must also stay below the pool
        # size or the fusion stage (BM25 + vector + type-boost) becomes a
        # no-op, since the cross-encoder would just see the entire pool
        # regardless of fusion order. 22 is a floor for when `top_k` itself is
        # small enough that fusion would otherwise barely filter anything.
        shortlist_size = max(top_k, 22)
        shortlist = fused[: min(len(fused), shortlist_size)]

        candidates = [
            RetrievalCandidate(
                id=str(req.id),
                source_type="diagram_requirement",
                text=req.requirement_text,
                score=rrf_score(req),
            )
            for req in shortlist
        ]
        by_id = {str(req.id): req for req in shortlist}
        # Rerank the whole shortlist (not just top_k) so every candidate gets
        # a cross-encoder score to threshold on below.
        ranked_candidates = self._reranker.rerank(query_text, candidates, len(candidates))

        # Adaptive cutoff: `top_k` is a ceiling, not a forced count. Keep
        # everything within _RELEVANCE_MARGIN of the top-scoring candidate
        # (relative, not absolute — see constant comment above for why), up
        # to the ceiling; always keep at least _MIN_RESULTS even if the margin
        # would otherwise leave fewer, so a diagram never comes back empty.
        if ranked_candidates:
            top_score = ranked_candidates[0].score
            selected = [c for c in ranked_candidates if c.score >= top_score - _RELEVANCE_MARGIN]
        else:
            selected = []
        if len(selected) < _MIN_RESULTS:
            selected = ranked_candidates[:_MIN_RESULTS]
        selected = selected[:top_k]

        return [by_id[c.id] for c in selected if c.id in by_id]

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
        if caption:
            parts.append(caption)
        if surrounding:
            parts.append(surrounding)

        if not parts:
            page_number = getattr(diagram, "page_number", None)
            pages = getattr(tsd_document, "pages", None) or []
            if isinstance(page_number, int) and 1 <= page_number <= len(pages):
                page_text = (getattr(pages[page_number - 1], "all_text", "") or "").strip()
                if page_text:
                    parts.append(page_text[:1200])

        # Caption/surrounding text describes what KIND of diagram this is, but
        # rarely names what's actually drawn in it (component/technology names
        # like "DMZ", "MySQL Database") — that's the vocabulary an embedding
        # needs to connect a diagram to the specific requirements it implicates.
        if ocr_text:
            parts.append(f"Visible diagram elements: {ocr_text}")

        return "\n\n".join(parts).strip()

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
