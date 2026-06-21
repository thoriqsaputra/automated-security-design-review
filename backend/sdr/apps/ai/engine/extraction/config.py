from __future__ import annotations

from dataclasses import dataclass

from sdr.core.config import settings


@dataclass(frozen=True)
class ExtractionConfig:
    standard_extraction_max_workers: int = 3
    diagram_requirement_extraction_max_concurrency: int = 3
    cfsr_extraction_max_concurrency: int = 4
    cfsr_max_per_parent: int = 5
    standard_extraction_chunk_token_target: int = 3200

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
            cfsr_extraction_max_concurrency=max(
                1,
                int(getattr(settings, "AI_CFSR_EXTRACTION_MAX_CONCURRENCY", 4)),
            ),
            cfsr_max_per_parent=max(
                1,
                int(getattr(settings, "AI_CFSR_MAX_PER_PARENT", 5)),
            ),
            standard_extraction_chunk_token_target=max(
                1,
                int(getattr(settings, "AI_STANDARD_EXTRACTION_CHUNK_TOKEN_TARGET", 3200)),
            ),
        )
