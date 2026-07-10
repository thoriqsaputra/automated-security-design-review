from __future__ import annotations

from dataclasses import dataclass

from sdr.core.config import settings


@dataclass(frozen=True)
class AnalysisPipelineConfig:
    batch_debate_max_concurrency: int = 3
    batch_debate_ungrounded_not_met_policy: str = "preserve_not_met"
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
    @classmethod
    def from_settings(cls) -> "AnalysisPipelineConfig":
        return cls(
            batch_debate_max_concurrency=max(1, int(getattr(settings, "AI_BATCH_DEBATE_MAX_CONCURRENCY", 3))),
            batch_debate_ungrounded_not_met_policy=str(
                getattr(settings, "AI_BATCH_DEBATE_UNGROUNDED_NOT_MET_POLICY", "preserve_not_met") or ""
            ).strip().lower(),
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
        )
