from __future__ import annotations

from dataclasses import dataclass

from sdr.core.config import settings


@dataclass(frozen=True)
class AnalysisPipelineConfig:
    batch_debate_max_concurrency: int = 3
    debate_context_supplemental_block_limit: int = 12
    debate_warn_context_chunk_threshold: int = 40
    vision_enabled: bool = True
    vision_min_diagram_bytes: int = 512
    vision_diagram_requirements_max_items: int = 32
    vision_max_concurrency: int = 2
    vision_skip_mediator_on_uphold: bool = True
    vision_debate_votes: int = 1
    vision_extraction_votes: int = 3
    vision_extraction_merge_threshold: float = 0.5
    vision_extraction_fuzzy_match_threshold: float = 0.75
    vision_reasoner_batch_size: int = 10
    # Diagram requirement selection (DiagramRequirementSelector._hybrid_search).
    # BM25 over short requirement texts is easily diluted by the broad
    # page-window segment, so BM25 sees only caption/OCR/surrounding while the
    # dense embedding keeps the full query. Weights/bonuses default to the
    # historical fixed values; score_floor_ratio 0.0 = no adaptive cutoff.
    diagram_query_page_window_chars: int = 1800
    diagram_rrf_k: int = 60
    diagram_rrf_vector_weight: float = 1.0
    diagram_rrf_bm25_weight: float = 1.0
    diagram_type_match_bonus: float = 1.0 / 60.0
    diagram_chapter_prior_bonus: float = 0.0
    diagram_score_floor_ratio: float = 0.0
    @classmethod
    def from_settings(cls) -> "AnalysisPipelineConfig":
        return cls(
            batch_debate_max_concurrency=max(1, int(getattr(settings, "AI_BATCH_DEBATE_MAX_CONCURRENCY", 3))),
            debate_context_supplemental_block_limit=max(
                0,
                int(getattr(settings, "AI_DEBATE_CONTEXT_SUPPLEMENTAL_BLOCK_LIMIT", 12)),
            ),
            debate_warn_context_chunk_threshold=max(
                1,
                int(getattr(settings, "AI_DEBATE_WARN_CONTEXT_CHUNK_THRESHOLD", 40)),
            ),
            vision_enabled=bool(getattr(settings, "AI_VISION_ENABLED", True)),
            vision_min_diagram_bytes=int(getattr(settings, "AI_VISION_MIN_DIAGRAM_BYTES", 512)),
            vision_diagram_requirements_max_items=int(
                getattr(settings, "AI_VISION_DIAGRAM_REQUIREMENTS_MAX_ITEMS", 16)
            ),
            vision_max_concurrency=int(getattr(settings, "AI_VISION_MAX_CONCURRENCY", 2)),
            vision_skip_mediator_on_uphold=bool(
                getattr(settings, "AI_VISION_SKIP_MEDIATOR_ON_UPHOLD", True)
            ),
            vision_debate_votes=max(1, int(getattr(settings, "AI_VISION_DEBATE_VOTES", 1))),
            vision_extraction_votes=max(1, int(getattr(settings, "AI_VISION_EXTRACTION_VOTES", 3))),
            vision_extraction_merge_threshold=float(
                getattr(settings, "AI_VISION_EXTRACTION_MERGE_THRESHOLD", 0.5)
            ),
            vision_extraction_fuzzy_match_threshold=float(
                getattr(settings, "AI_VISION_EXTRACTION_FUZZY_MATCH_THRESHOLD", 0.75)
            ),
            vision_reasoner_batch_size=max(1, int(getattr(settings, "AI_VISION_REASONER_BATCH_SIZE", 10))),
            diagram_query_page_window_chars=max(
                0, int(getattr(settings, "AI_VISION_DIAGRAM_QUERY_PAGE_WINDOW_CHARS", 1800))
            ),
            diagram_rrf_k=max(1, int(getattr(settings, "AI_VISION_DIAGRAM_RRF_K", 60))),
            diagram_rrf_vector_weight=float(getattr(settings, "AI_VISION_DIAGRAM_RRF_VECTOR_WEIGHT", 1.0)),
            diagram_rrf_bm25_weight=float(getattr(settings, "AI_VISION_DIAGRAM_RRF_BM25_WEIGHT", 1.0)),
            diagram_type_match_bonus=float(
                getattr(settings, "AI_VISION_DIAGRAM_TYPE_MATCH_BONUS", 1.0 / 60.0)
            ),
            diagram_chapter_prior_bonus=float(
                getattr(settings, "AI_VISION_DIAGRAM_CHAPTER_PRIOR_BONUS", 0.0)
            ),
            diagram_score_floor_ratio=float(
                getattr(settings, "AI_VISION_DIAGRAM_SCORE_FLOOR_RATIO", 0.0)
            ),
        )
