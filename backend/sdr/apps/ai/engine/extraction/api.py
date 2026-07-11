from __future__ import annotations

from typing import Any, Dict, List, Optional

from sdr.apps.ai.client import chat_completion
from sdr.apps.standards.models import StandardSourceDocument
from sdr.apps.workspace.document_processing import get_document_content, get_local_file_path

from .config import ExtractionConfig
from .document_reader import StandardDocumentReader
from .llm_client import ExtractionLLMClient
from .page_detection import ASVSPageRangeDetectionService
from .services import (
    DiagramRequirementExtractionService,
    RequirementDocumentExtractionService,
    StructuredRequirementExtractionService,
)


def _build_config() -> ExtractionConfig:
    return ExtractionConfig.from_settings()


def _build_document_reader() -> StandardDocumentReader:
    return StandardDocumentReader(
        get_local_file_path=get_local_file_path,
        get_document_content=get_document_content,
    )


def _build_llm_client() -> ExtractionLLMClient:
    return ExtractionLLMClient(chat_completion=chat_completion)


def detect_asvs_page_ranges(
    source_doc: StandardSourceDocument,
) -> Dict[str, Any]:
    service = ASVSPageRangeDetectionService(
        get_local_file_path=get_local_file_path,
    )
    return service.detect(source_doc).to_dict()


def extract_structured_requirements(source_doc_text: str, source_name: str = "") -> Dict[str, List[Any]]:
    service = StructuredRequirementExtractionService(
        llm_client=_build_llm_client(),
        config=_build_config(),
    )
    return service.extract(source_doc_text, source_name=source_name)


def extract_requirements_from_document(
    source_doc: StandardSourceDocument,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    progress_callback=None,
) -> Dict[str, List[Any]]:
    service = RequirementDocumentExtractionService(
        document_reader=_build_document_reader(),
        structured_extractor=StructuredRequirementExtractionService(
            llm_client=_build_llm_client(),
            config=_build_config(),
        ),
        requirement_level_detector=None,
        config=_build_config(),
    )
    return service.extract(
        source_doc,
        start_page=start_page,
        end_page=end_page,
        progress_callback=progress_callback,
    )



def extract_diagram_requirements(
    parameters: list,
    category_id: int,
    ingestion_job_id: int,
) -> list:
    service = DiagramRequirementExtractionService(
        llm_client=_build_llm_client(),
        config=_build_config(),
    )
    return service.extract(
        parameters=parameters,
        category_id=category_id,
        ingestion_job_id=ingestion_job_id,
    )
