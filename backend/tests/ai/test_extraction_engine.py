from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sdr.apps.ai.engine.extraction import (
    ASVSLevelDefinitionExtractionService,
    ASVSRequirementMarkdownExtractionService,
    ControlFamilySummaryExtractionService,
    detect_asvs_page_ranges,
    ExtractionConfig,
    ExtractionLLMClient,
    RequirementDocumentExtractionService,
    StandardDocumentReader,
    StructuredRequirementExtractionService,
    extract_structured_requirements,
)
from sdr.apps.ai.engine.extraction.normalizers import (
    _dedupe_near_duplicate_requirements,
    _merge_requirements,
    render_asvs_markdown_table,
    render_asvs_plain_text,
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
              "context_marker": "Section 5.4.1",
              "asvs_level": 2
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
    assert item["asvs_level"] == 2


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

        def extract(self, text: str):
            self.calls.append(text)
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
        def extract(self, text: str):
            return {}

    service = RequirementDocumentExtractionService(
        document_reader=document_reader,
        structured_extractor=StubStructuredExtractor(),
        config=ExtractionConfig(
            standard_extraction_max_workers=1,
            standard_extraction_chunk_token_target=4242,
            standard_extraction_chunk_overlap_tokens=111,
        ),
    )

    with patch(
        "sdr.apps.ai.engine.extraction.services.chunk_text_with_context",
        return_value=[{"text": "chunk"}],
    ) as mock_chunker:
        service.extract(source_doc)

    mock_chunker.assert_called_once()
    _, kwargs = mock_chunker.call_args
    assert kwargs["chunk_size"] == 4242
    assert kwargs["overlap"] == 111


def test_asvs_level_definition_service_uses_injected_dependencies():
    source_doc = SimpleNamespace(id=1, name="OWASP ASVS.pdf", document="standards/asvs.pdf")
    document_reader = StandardDocumentReader(
        get_local_file_path=lambda document: nullcontext(f"/tmp/{document}"),
        get_document_content=lambda *args, **kwargs: {"text": "ASVS Level 1 and Level 2 definitions"},
    )
    response = SimpleNamespace(
        error=None,
        content="""
        {
          "levels": [
            {
              "level": "L1",
              "name": "Opportunistic",
              "description": "Baseline verification",
              "classification_guidance": "Use L1 for ordinary applications."
            },
            {
              "level": 2,
              "name": "Standard",
              "description": "Sensitive data verification",
              "classification_guidance": "Use L2 for applications with sensitive data."
            }
          ]
        }
        """,
    )
    service = ASVSLevelDefinitionExtractionService(
        document_reader=document_reader,
        llm_client=ExtractionLLMClient(chat_completion=lambda **_: response),
    )

    result = service.extract(source_doc, start_page=8, end_page=10)

    assert [item["level"] for item in result] == [1, 2]


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

    assert result["level_definition_start_page"] == 12
    assert result["level_definition_end_page"] == 14
    assert result["start_page"] == 23
    assert result["end_page"] == 95


def test_detect_asvs_page_ranges_for_asvs_4_pdf():
    repo_root = Path(__file__).resolve().parents[3]
    pdf_path = repo_root / "dataset/Standard/OWASP Application Security Verification Standard 4.0.3-en.pdf"
    source_doc = SimpleNamespace(id=1, name=pdf_path.name, document=str(pdf_path))

    result = detect_asvs_page_ranges(source_doc)

    assert result["level_definition_start_page"] == 11
    assert result["level_definition_end_page"] == 12
    assert result["start_page"] == 17
    assert result["end_page"] == 63


def test_merge_requirements_collapses_reworded_duplicate_from_overlapping_chunks():
    base = {
        "V1 Architecture": [
            {
                "requirement": "Verify that the application enforces strict input validation on all untrusted input boundaries",
                "details": "",
                "verbatim_quote": "",
                "context_marker": "",
                "asvs_level": None,
            }
        ]
    }
    incoming = {
        "V1 Architecture": [
            {
                "requirement": "Verify the application enforces strict input validation on all untrusted input boundaries.",
                "details": "Applies to every external entry point.",
                "verbatim_quote": "",
                "context_marker": "",
                "asvs_level": 1,
            }
        ]
    }

    merged = _merge_requirements(base, incoming)

    assert len(merged["V1 Architecture"]) == 1
    assert merged["V1 Architecture"][0]["details"] == "Applies to every external entry point."


def test_dedupe_near_duplicate_requirements_keeps_genuinely_distinct_items():
    items = [
        {"requirement": "Verify that passwords are hashed using a memory-hard algorithm", "details": ""},
        {"requirement": "Verify that session tokens are invalidated on logout", "details": ""},
    ]

    deduped = _dedupe_near_duplicate_requirements(items)

    assert len(deduped) == 2


def test_dedupe_near_duplicate_requirements_merges_near_identical_text():
    items = [
        {"requirement": "Verify the use of unique low-privilege OS accounts for all application components", "details": ""},
        {"requirement": "Verify the use of unique low privilege OS accounts for all application components.", "details": "extra context"},
    ]

    deduped = _dedupe_near_duplicate_requirements(items)

    assert len(deduped) == 1
    assert deduped[0]["details"] == "extra context"


def test_render_asvs_plain_text_dedupes_prefixed_and_deletes_rows():
    rows = {
        "V1 Architecture": [
            {
                "requirement": "V1.1 - 1.1.1 Verify the use of a secure software development lifecycle that addresses security in all stages of development.",
                "details": "",
                "verbatim_quote": "",
                "context_marker": "V1.1",
                "asvs_level": 1,
            },
            {
                "requirement": "1.1.1 Verify the use of a secure software development lifecycle that addresses security in all stages of development.",
                "details": "page break duplicate",
                "verbatim_quote": "",
                "context_marker": "V1.1",
                "asvs_level": 1,
            },
        ],
        "V6 Stored Cryptography": [
            {
                "requirement": "6.1.1 Verify that stored passwords are hashed using a memory-hard algorithm.",
                "details": "",
                "verbatim_quote": "",
                "context_marker": "V6.1",
                "asvs_level": 2,
            },
            {
                "requirement": "[DELETED, DUPLICATE OF 6.1.1]",
                "details": "",
                "verbatim_quote": "[DELETED, DUPLICATE OF 6.1.1]",
                "context_marker": "V6.1",
                "asvs_level": 2,
            },
        ],
    }

    markdown = render_asvs_plain_text(rows)

    assert markdown.splitlines() == [
        "1.1.1 Verify the use of a secure software development lifecycle that addresses security in all stages of development. - page break duplicate",
        "6.1.1 Verify that stored passwords are hashed using a memory-hard algorithm.",
    ]


def test_asvs_markdown_extraction_service_renders_markdown_from_structured_rows():
    source_doc = SimpleNamespace(id=1, name="OWASP ASVS.pdf", document="standards/asvs.pdf")
    document_reader = StandardDocumentReader(
        get_local_file_path=lambda document: nullcontext(f"/tmp/{document}"),
        get_document_content=lambda *args, **kwargs: {
            "text": "1.1.1 Verify the use of a secure software development lifecycle that addresses security in all stages of development.",
            "conversion_method": "mock",
        },
    )

    class StubStructuredExtractor:
        def extract(self, text: str, *, source_name: str = ""):
            assert source_name == "OWASP ASVS.pdf"
            return {
                "V1 Architecture": [
                    {
                        "requirement": "V1.1 - 1.1.1 Verify the use of a secure software development lifecycle that addresses security in all stages of development.",
                        "details": "",
                        "verbatim_quote": "",
                        "context_marker": "V1.1",
                        "asvs_level": 1,
                    }
                ]
            }

    service = ASVSRequirementMarkdownExtractionService(
        document_reader=document_reader,
        structured_extractor=StubStructuredExtractor(),
        config=ExtractionConfig(standard_extraction_max_workers=1),
    )

    result = service.extract(source_doc)

    assert result == "1.1.1 Verify the use of a secure software development lifecycle that addresses security in all stages of development."


def _make_cfsr_param(stable_key: str, requirement_text: str, parent):
    return SimpleNamespace(
        stable_key=stable_key,
        requirement_text=requirement_text,
        details="",
        asvs_level=1,
        parent=parent,
    )


def test_cfsr_extraction_drops_zero_coverage_cfsr_and_reassigns_its_orphan():
    parent = SimpleNamespace(id=1, title="V2 Authentication")
    child_real_1 = _make_cfsr_param(
        "child-real-1", "Authentication tokens must expire within 15 minutes", parent
    )
    child_real_2 = _make_cfsr_param(
        "child-real-2", "Session tokens must be invalidated on logout", parent
    )

    response = SimpleNamespace(
        error=None,
        content="""
        {
          "summary_requirements": [
            {
              "stable_key": "cfsr-a",
              "requirement_text": "Completely unrelated control about audit logging",
              "asvs_level": 1,
              "covered_child_keys": ["child-999"]
            },
            {
              "stable_key": "cfsr-b",
              "requirement_text": "Session tokens must be invalidated on logout",
              "asvs_level": 1,
              "covered_child_keys": ["child-001"]
            }
          ]
        }
        """,
    )
    llm_client = ExtractionLLMClient(chat_completion=lambda **_: response)
    service = ControlFamilySummaryExtractionService(
        llm_client=llm_client,
        config=ExtractionConfig(cfsr_extraction_max_concurrency=1, cfsr_max_per_parent=5),
    )

    results = service.extract(
        parameters=[child_real_1, child_real_2],
        category_id=1,
        ingestion_job_id=1,
    )

    assert len(results) == 1
    assert results[0]["stable_key"].endswith("cfsr-b")
    assert set(results[0]["covered_child_keys"]) == {"child-real-1", "child-real-2"}


def test_cfsr_extraction_merges_near_duplicates_before_applying_cap():
    parent = SimpleNamespace(id=1, title="V4 Access Control")
    child_1 = _make_cfsr_param(
        "child-real-1", "All API endpoints require authentication and authorization", parent
    )
    child_2 = _make_cfsr_param(
        "child-real-2", "Administrative API endpoints require multi-factor authentication", parent
    )
    child_3 = _make_cfsr_param(
        "child-real-3", "Audit logs capture all administrative actions", parent
    )

    response = SimpleNamespace(
        error=None,
        content="""
        {
          "summary_requirements": [
            {
              "stable_key": "cfsr-a",
              "requirement_text": "Verify that all API endpoints require authentication and authorization checks",
              "asvs_level": 1,
              "covered_child_keys": ["child-001"]
            },
            {
              "stable_key": "cfsr-b",
              "requirement_text": "Verify that all API endpoints require authentication and authorization checks.",
              "asvs_level": 1,
              "covered_child_keys": ["child-002"]
            },
            {
              "stable_key": "cfsr-c",
              "requirement_text": "Verify that audit logs capture all administrative actions",
              "asvs_level": 1,
              "covered_child_keys": ["child-003"]
            }
          ]
        }
        """,
    )
    llm_client = ExtractionLLMClient(chat_completion=lambda **_: response)
    service = ControlFamilySummaryExtractionService(
        llm_client=llm_client,
        config=ExtractionConfig(cfsr_extraction_max_concurrency=1, cfsr_max_per_parent=2),
    )

    results = service.extract(
        parameters=[child_1, child_2, child_3],
        category_id=1,
        ingestion_job_id=1,
    )

    assert len(results) == 2
    merged = next(r for r in results if r["stable_key"].endswith("cfsr-a"))
    assert set(merged["covered_child_keys"]) == {"child-real-1", "child-real-2"}
