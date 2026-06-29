from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from sdr.core.config import settings

_DEFAULT_DIAGRAM_TYPES = ["data_flow", "sequence", "architecture"]
_VALID_DIAGRAM_TYPES = {"data_flow", "sequence", "architecture"}


def _parse_diagram_types(raw: str) -> List[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    valid = [p for p in parts if p in _VALID_DIAGRAM_TYPES]
    return valid or _DEFAULT_DIAGRAM_TYPES


@dataclass(frozen=True)
class ExtractionConfig:
    standard_extraction_max_workers: int = 3
    diagram_requirement_extraction_max_concurrency: int = 3
    standard_extraction_chunk_token_target: int = 4500
    diagram_types: List[str] = field(default_factory=lambda: list(_DEFAULT_DIAGRAM_TYPES))

    @classmethod
    def from_settings(cls) -> "ExtractionConfig":
        raw_diagram_types = str(
            getattr(settings, "AI_DIAGRAM_TYPES", ",".join(_DEFAULT_DIAGRAM_TYPES))
        )
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
            diagram_types=_parse_diagram_types(raw_diagram_types),
        )
