from .config import ExtractionConfig
from .document_reader import StandardDocumentReader
from .normalizers import (
    _canonicalize_diagram_requirements,
    _extract_logical_id,
    canonicalize_requirement_items,
    canonicalize_structured_requirements,
)

_API_EXPORTS = {
    "detect_asvs_page_ranges",
    "extract_diagram_requirements",
    "extract_requirements_from_document",
    "extract_structured_requirements",
}
_SERVICE_EXPORTS = {
    "DiagramRequirementExtractionService",
    "RequirementDocumentExtractionService",
    "RequirementCategoryValidationService",
    "StructuredRequirementExtractionService",
}


def __getattr__(name):
    """Load document-processing dependencies only when their API is used."""
    if name in _API_EXPORTS:
        from . import api

        return getattr(api, name)
    if name == "ASVSPageRangeDetectionService":
        from .page_detection import ASVSPageRangeDetectionService

        return ASVSPageRangeDetectionService
    if name in _SERVICE_EXPORTS:
        from . import services

        return getattr(services, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ExtractionConfig",
    "StandardDocumentReader",
    "ASVSPageRangeDetectionService",
    "detect_asvs_page_ranges",
    "extract_structured_requirements",
    "extract_requirements_from_document",
    "extract_diagram_requirements",
    "StructuredRequirementExtractionService",
    "RequirementDocumentExtractionService",
    "RequirementCategoryValidationService",
    "_extract_logical_id",
    "_canonicalize_diagram_requirements",
    "canonicalize_requirement_items",
    "canonicalize_structured_requirements",
]
