from __future__ import annotations

import logging
from typing import List

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder  # type: ignore

    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CROSS_ENCODER_AVAILABLE = False


class BaseReranker:
    def rerank(self, query: str, candidates: List[RetrievalCandidate], top_k: int) -> List[RetrievalCandidate]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    def rerank(self, query: str, candidates: List[RetrievalCandidate], top_k: int) -> List[RetrievalCandidate]:
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]


class SafeOptionalReranker(BaseReranker):
    def __init__(
        self,
        enable_cross_encoder: bool = False,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._fallback = NoOpReranker()
        self.enable_cross_encoder = enable_cross_encoder and CROSS_ENCODER_AVAILABLE
        self._cross_encoder = None
        if self.enable_cross_encoder:
            try:
                self._cross_encoder = CrossEncoder(cross_encoder_model)
            except Exception:
                logger.exception("Failed to initialize cross-encoder reranker; using fallback.")
                self._cross_encoder = None

    def rerank(self, query: str, candidates: List[RetrievalCandidate], top_k: int) -> List[RetrievalCandidate]:
        try:
            if self._cross_encoder is not None and candidates:
                pairs = [[query, c.text or ""] for c in candidates]
                scores = self._cross_encoder.predict(pairs)
                for candidate, score in zip(candidates, scores):
                    candidate.metadata["cross_encoder_score"] = float(score)
                    candidate.score = float(score)
                ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
                return ranked[:top_k]
            return self._fallback.rerank(query=query, candidates=candidates, top_k=top_k)
        except Exception as exc:
            logger.warning("Reranker failed, using fallback order: %s", exc)
            return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]


__all__ = ["BaseReranker", "NoOpReranker", "SafeOptionalReranker"]
