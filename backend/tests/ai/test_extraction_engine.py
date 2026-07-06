from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sdr.apps.ai.engine.extraction import (
    ExtractionConfig,
    ExtractionLLMClient,
    RequirementDocumentExtractionService,
    StandardDocumentReader,
    StructuredRequirementExtractionService,
    detect_asvs_page_ranges,
    extract_structured_requirements,
)


def test_structured_requirement_extraction_service_parses_details():
    response = SimpleNamespace(
        error=None,
        model="mock-model",
        provider="mock-provider",
        usage=None,
        content="""
        {
          "5.4 API and Web Service": [
            {
              "requirement": "5.4.1 Generic Web Service Security",
              "details": "All APIs must document interfaces and validate input/output.",
              "verbatim_quote": "a. semua API mendefinisikan ...",
              "context_marker": "Section 5.4.1"
            }
          ]
        }
        """,
    )
    llm_client = ExtractionLLMClient(chat_completion=lambda **_: response)
    service = StructuredRequirementExtractionService(llm_client=llm_client)

    result = service.extract("5.4.1 Generic Web Service Security")

    item = result["5.4 API and Web Service"][0]
    assert item["requirement"] == "5.4.1 Generic Web Service Security"
    assert item["details"] == "All APIs must document interfaces and validate input/output."
    assert item["verbatim_quote"] == "a. semua API mendefinisikan ..."
    assert item["context_marker"] == "Section 5.4.1"


def test_requirement_document_extraction_service_uses_injected_reader():
    source_doc = SimpleNamespace(id=1, name="ASVS", document="standards/asvs.pdf")
    document_reader = StandardDocumentReader(
        get_local_file_path=lambda document: nullcontext(f"/tmp/{document}"),
        get_document_content=lambda *args, **kwargs: {
            "text": "5.4.1 Generic Web Service Security\nAll APIs must validate input.",
            "conversion_method": "mock",
        },
    )

    class StubStructuredExtractor:
        def __init__(self) -> None:
            self.calls = []

        def extract(self, text: str, *, source_name: str = ""):
            self.calls.append((text, source_name))
            return {
                "5.4 API and Web Service": [
                    {
                        "requirement": "5.4.1 Generic Web Service Security",
                        "details": "All APIs must validate input.",
                    }
                ]
            }

    structured_extractor = StubStructuredExtractor()
    service = RequirementDocumentExtractionService(
        document_reader=document_reader,
        structured_extractor=structured_extractor,
        config=ExtractionConfig(standard_extraction_max_workers=1),
    )

    result = service.extract(source_doc)

    assert structured_extractor.calls
    assert result["5.4 API and Web Service"][0]["details"] == "All APIs must validate input."


def test_requirement_document_extraction_service_uses_configured_chunk_sizing():
    source_doc = SimpleNamespace(id=1, name="ASVS", document="standards/asvs.pdf")
    document_reader = StandardDocumentReader(
        get_local_file_path=lambda document: nullcontext(f"/tmp/{document}"),
        get_document_content=lambda *args, **kwargs: {
            "text": "5.4.1 Generic Web Service Security\nAll APIs must validate input.",
            "conversion_method": "mock",
        },
    )

    class StubStructuredExtractor:
        def extract(self, text: str, *, source_name: str = ""):
            return {}

    service = RequirementDocumentExtractionService(
        document_reader=document_reader,
        structured_extractor=StubStructuredExtractor(),
        config=ExtractionConfig(
            standard_extraction_max_workers=1,
            standard_extraction_chunk_token_target=4242,
        ),
    )

    with patch(
        "sdr.apps.ai.engine.extraction.services.chunk_text_semantically",
        return_value=[{"text": "chunk"}],
    ) as mock_chunker:
        service.extract(source_doc)

    mock_chunker.assert_called_once()
    _, kwargs = mock_chunker.call_args
    assert kwargs["chunk_size"] == 4242


def test_engine_function_wrapper_uses_api_patch_points():
    response = SimpleNamespace(
        error=None,
        model="mock-model",
        provider="mock-provider",
        usage=None,
        content="""
        {
          "5.4 API and Web Service": [
            {
              "requirement": "5.4.1 Generic Web Service Security",
              "details": "All APIs must document interfaces and validate input/output."
            }
          ]
        }
        """,
    )

    with patch("sdr.apps.ai.engine.extraction.api.chat_completion", return_value=response):
        result = extract_structured_requirements("5.4.1 Generic Web Service Security")

    assert result["5.4 API and Web Service"][0]["requirement"] == "5.4.1 Generic Web Service Security"


def test_detect_asvs_page_ranges_for_asvs_5_pdf():
    repo_root = Path(__file__).resolve().parents[3]
    pdf_path = repo_root / "dataset/Standard/OWASP_Application_Security_Verification_Standard_5.0.0_en.pdf"
    source_doc = SimpleNamespace(id=1, name=pdf_path.name, document=str(pdf_path))

    result = detect_asvs_page_ranges(source_doc)

    assert result["start_page"] == 23
    assert result["end_page"] == 95


def test_detect_asvs_page_ranges_for_asvs_4_pdf():
    repo_root = Path(__file__).resolve().parents[3]
    pdf_path = repo_root / "dataset/Standard/OWASP Application Security Verification Standard 4.0.3-en.pdf"
    source_doc = SimpleNamespace(id=1, name=pdf_path.name, document=str(pdf_path))

    result = detect_asvs_page_ranges(source_doc)

    assert result["start_page"] == 17
    assert result["end_page"] == 63
