from __future__ import annotations

import logging
from typing import List, Optional

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate

logger = logging.getLogger(__name__)


class BaseReranker:
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int,
        extra_queries: Optional[List[str]] = None,
    ) -> List[RetrievalCandidate]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int,
        extra_queries: Optional[List[str]] = None,
    ) -> List[RetrievalCandidate]:
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]


class SafeOptionalReranker(BaseReranker):
    def __init__(
        self,
        enable_cross_encoder: bool = False,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        score_weight: float = 0.72,
        max_window_chars: int = 1800,
    ) -> None:
        self._fallback = NoOpReranker()
        self.enable_cross_encoder = bool(enable_cross_encoder)
        self.score_weight = min(1.0, max(0.0, float(score_weight)))
        self.max_window_chars = max(200, int(max_window_chars))
        self._cross_encoder = None
        if self.enable_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._cross_encoder = CrossEncoder(cross_encoder_model)
            except Exception:
                logger.exception("Failed to initialize cross-encoder reranker; using fallback.")
                self._cross_encoder = None
                self.enable_cross_encoder = False

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int,
        extra_queries: Optional[List[str]] = None,
    ) -> List[RetrievalCandidate]:
        try:
            if self._cross_encoder is not None and candidates:
                queries = [query] + [q for q in (extra_queries or []) if q and q != query]
                original_scores = [float(c.score) for c in candidates]
                semantic_scores = [
                    self._score_candidate(queries=queries, candidate=c)
                    for c in candidates
                ]
                normalized_original = _normalize_scores(original_scores)
                normalized_semantic = _normalize_scores(semantic_scores)
                for candidate, original, semantic, norm_original, norm_semantic in zip(
                    candidates,
                    original_scores,
                    semantic_scores,
                    normalized_original,
                    normalized_semantic,
                ):
                    candidate.metadata["pre_rerank_score"] = original
                    candidate.metadata["cross_encoder_score"] = float(semantic)
                    candidate.metadata["normalized_pre_rerank_score"] = float(norm_original)
                    candidate.metadata["normalized_cross_encoder_score"] = float(norm_semantic)
                    blended = (self.score_weight * norm_semantic) + ((1.0 - self.score_weight) * norm_original)
                    candidate.metadata["rerank_blended_score"] = float(blended)
                    candidate.score = float(blended)
                ranked = sorted(
                    candidates,
                    key=lambda c: (
                        c.score,
                        int(c.metadata.get("agreement_count") or 1),
                        float(c.metadata.get("pre_rerank_score") or 0.0),
                    ),
                    reverse=True,
                )
                return ranked[:top_k]
            return self._fallback.rerank(query=query, candidates=candidates, top_k=top_k)
        except Exception as exc:
            logger.warning("Reranker failed, using fallback order: %s", exc)
            return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]

    def _score_candidate(self, *, queries: List[str], candidate: RetrievalCandidate) -> float:
        text = candidate.text or ""
        windows = _candidate_windows(text, self.max_window_chars)
        pairs = [[q, window] for q in queries for window in windows]
        scores = self._cross_encoder.predict(pairs)
        return max(float(score) for score in scores)


def _candidate_windows(text: str, max_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]
    overlap = max(50, int(max_chars * 0.2))
    step = max(1, max_chars - overlap)
    windows: List[str] = []
    for start in range(0, len(text), step):
        chunk = text[start:start + max_chars].strip()
        if chunk:
            windows.append(chunk)
        if start + max_chars >= len(text):
            break
    return windows or [text[:max_chars]]


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(score - lo) / (hi - lo) for score in scores]


__all__ = ["BaseReranker", "NoOpReranker", "SafeOptionalReranker"]
