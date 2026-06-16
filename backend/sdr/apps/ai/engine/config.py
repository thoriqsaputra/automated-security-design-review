from __future__ import annotations

from dataclasses import dataclass

from sdr.core.config import settings


@dataclass(frozen=True)
class AnalysisPipelineConfig:
    batch_debate_enabled: bool = True
    batch_debate_batch_size: int = 3
    batch_debate_max_concurrency: int = 3
    batch_debate_fallback_enabled: bool = True
    batch_debate_parent_context_cache_enabled: bool = True
    batch_debate_confidence_threshold: float = 0.75
    batch_debate_soft_confidence_threshold: float = 0.65
    batch_debate_require_citations_for_not_met: bool = True
    batch_debate_ungrounded_not_met_policy: str = "preserve_not_met"
    parent_applicability_enabled: bool = True
    parent_applicability_confidence_threshold: float = 0.7
    parent_applicability_fallback_mode: str = "skip"
    parent_applicability_max_child_texts: int = 8
    vision_diagram_analysis_enabled: bool = True
    vision_enabled: bool = True
    vision_min_diagram_bytes: int = 512
    vision_diagram_requirements_max_items: int = 15
    vision_max_concurrency: int = 2

    @classmethod
    def from_settings(cls) -> "AnalysisPipelineConfig":
        return cls(
            batch_debate_enabled=bool(getattr(settings, "AI_BATCH_DEBATE_ENABLED", True)),
            batch_debate_batch_size=max(1, int(getattr(settings, "AI_BATCH_DEBATE_BATCH_SIZE", 3))),
            batch_debate_max_concurrency=max(1, int(getattr(settings, "AI_BATCH_DEBATE_MAX_CONCURRENCY", 3))),
            batch_debate_fallback_enabled=bool(getattr(settings, "AI_BATCH_DEBATE_FALLBACK_ENABLED", True)),
            batch_debate_parent_context_cache_enabled=bool(
                getattr(settings, "AI_BATCH_DEBATE_PARENT_CONTEXT_CACHE_ENABLED", True)
            ),
            batch_debate_confidence_threshold=float(
                getattr(settings, "AI_BATCH_DEBATE_CONFIDENCE_THRESHOLD", 0.75)
            ),
            batch_debate_soft_confidence_threshold=float(
                getattr(settings, "AI_BATCH_DEBATE_SOFT_CONFIDENCE_THRESHOLD", 0.65)
            ),
            batch_debate_require_citations_for_not_met=bool(
                getattr(settings, "AI_BATCH_DEBATE_REQUIRE_CITATIONS_FOR_NOT_MET", True)
            ),
            batch_debate_ungrounded_not_met_policy=str(
                getattr(settings, "AI_BATCH_DEBATE_UNGROUNDED_NOT_MET_POLICY", "preserve_not_met") or ""
            ).strip().lower(),
            parent_applicability_enabled=bool(
                getattr(settings, "AI_PARENT_APPLICABILITY_ENABLED", True)
            ),
            parent_applicability_confidence_threshold=float(
                getattr(settings, "AI_PARENT_APPLICABILITY_CONFIDENCE_THRESHOLD", 0.7)
            ),
            parent_applicability_fallback_mode=str(
                getattr(settings, "AI_PARENT_APPLICABILITY_FALLBACK_MODE", "skip") or ""
            ).strip().lower(),
            parent_applicability_max_child_texts=max(
                1, int(getattr(settings, "AI_PARENT_APPLICABILITY_MAX_CHILD_TEXTS", 8))
            ),
            vision_diagram_analysis_enabled=bool(
                getattr(settings, "AI_VISION_DIAGRAM_ANALYSIS_ENABLED", True)
            ),
            vision_enabled=bool(getattr(settings, "AI_VISION_ENABLED", True)),
            vision_min_diagram_bytes=int(getattr(settings, "AI_VISION_MIN_DIAGRAM_BYTES", 512)),
            vision_diagram_requirements_max_items=int(
                getattr(settings, "AI_VISION_DIAGRAM_REQUIREMENTS_MAX_ITEMS", 15)
            ),
            vision_max_concurrency=int(getattr(settings, "AI_VISION_MAX_CONCURRENCY", 2)),
        )
