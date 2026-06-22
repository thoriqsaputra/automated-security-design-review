from __future__ import annotations

from dataclasses import dataclass

from sdr.core.config import settings


@dataclass(frozen=True)
class ExtractionConfig:
    standard_extraction_max_workers: int = 3
    diagram_requirement_extraction_max_concurrency: int = 3
    standard_extraction_chunk_token_target: int = 4500

    @classmethod
    def from_settings(cls) -> "ExtractionConfig":
        return cls(
            standard_extraction_max_workers=max(
                1,
                int(getattr(settings, "AI_STANDARD_EXTRACTION_MAX_WORKERS", 3)),
            ),
            diagram_requirement_extraction_max_concurrency=max(
                1,
                int(getattr(settings, "AI_DIAGRAM_REQUIREMENT_EXTRACTION_MAX_CONCURRENCY", 3)),
            ),
            standard_extraction_chunk_token_target=max(
                1,
                int(getattr(settings, "AI_STANDARD_EXTRACTION_CHUNK_TOKEN_TARGET", 3200)),
            ),
        )
