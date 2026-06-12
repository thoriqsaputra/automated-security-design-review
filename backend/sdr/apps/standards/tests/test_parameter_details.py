from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch

from sdr.apps.ai.services.extraction_services import (
    _remove_table_of_contents,
    extract_asvs_level_definitions_from_document,
    extract_structured_requirements,
)
from sdr.apps.standards.schemas import CategoryParameterChildSchema
from sdr.apps.standards.tasks import _coerce_requirement_details, _coerce_requirement_text
from sdr.apps.standards.tasks import _coerce_asvs_level
from sdr.apps.standards.utils import build_parameter_analysis_text, normalize_requirement_text


def test_build_parameter_analysis_text_combines_heading_and_details():
    child = SimpleNamespace(
        requirement_text="5.4.1 Generic Web Service Security",
        details="All APIs must document interfaces and validate schema-bound input/output.",
    )

    assert build_parameter_analysis_text(child) == (
        "5.4.1 Generic Web Service Security\n\n"
        "All APIs must document interfaces and validate schema-bound input/output."
    )


def test_build_parameter_analysis_text_supports_legacy_heading_only():
    child = SimpleNamespace(requirement_text="Use MFA", details="")

    assert build_parameter_analysis_text(child) == "Use MFA"


def test_coerce_requirement_text_and_details_from_structured_item():
    item = {
        "requirement": "5.4.2 HTTP Message Structure Validation",
        "details": "HTTP method, headers, and body must be validated.",
    }

    assert _coerce_requirement_text(item) == "5.4.2 HTTP Message Structure Validation"
    assert _coerce_requirement_details(item) == "HTTP method, headers, and body must be validated."


def test_coerce_asvs_level_from_structured_item():
    assert _coerce_asvs_level({"asvs_level": 2}) == 2
    assert _coerce_asvs_level({"asvs_level": "L3"}) == 3
    assert _coerce_asvs_level({"asvs_level": None}) is None
    assert _coerce_asvs_level({"asvs_level": "unknown"}) is None


def test_coerce_requirement_details_defaults_blank_for_legacy_string():
    assert _coerce_requirement_text("Use MFA") == "Use MFA"
    assert _coerce_requirement_details("Use MFA") == ""


def test_normalized_text_can_include_full_requirement_meaning():
    analysis_text = build_parameter_analysis_text(
        "5.4.3 GraphQL",
        "GraphQL APIs must enforce schema validation and depth limiting.",
    )

    assert normalize_requirement_text(analysis_text) == (
        "5.4.3 graphql graphql apis must enforce schema validation and depth limiting."
    )


def test_remove_table_of_contents_keeps_real_body_sections():
    text = """
DAFTAR ISI
Halaman
5.4 API dan Web Service .......... 12
5.4.1 Generic Web Service Security .......... 13

5.4 API dan Web Service
5.4.1 Generic Web Service Security
a. semua API mendefinisikan dan mendokumentasikan antarmuka.
"""

    cleaned = _remove_table_of_contents(text)

    assert ".........." not in cleaned
    assert "Halaman" not in cleaned
    assert "5.4 API dan Web Service" in cleaned
    assert "semua API mendefinisikan" in cleaned


def test_extract_structured_requirements_preserves_details():
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

    with patch("sdr.apps.ai.services.extraction_services.chat_completion", return_value=response):
        result = extract_structured_requirements("5.4.1 Generic Web Service Security")

    item = result["5.4 API and Web Service"][0]
    assert item["requirement"] == "5.4.1 Generic Web Service Security"
    assert item["details"] == "All APIs must document interfaces and validate input/output."
    assert item["verbatim_quote"] == "a. semua API mendefinisikan ..."
    assert item["context_marker"] == "Section 5.4.1"
    assert item["asvs_level"] == 2


def test_extract_asvs_level_definitions_from_document():
    source_doc = SimpleNamespace(id=1, name="OWASP ASVS.pdf", document="standards/asvs.pdf")
    response = SimpleNamespace(
        error=None,
        content="""
        {
          "levels": [
            {
              "level": "L1",
              "name": "Opportunistic",
              "description": "Baseline verification",
              "classification_guidance": "Use L1 for ordinary applications.",
              "source_quote": "Level 1 is for ordinary applications.",
              "context_marker": "Section 1.3"
            },
            {
              "level": 2,
              "name": "Standard",
              "description": "Sensitive data verification",
              "classification_guidance": "Use L2 for applications with sensitive data.",
              "source_quote": "Level 2 is for sensitive data.",
              "context_marker": "Section 1.3"
            }
          ]
        }
        """,
    )

    with (
        patch("sdr.apps.ai.services.extraction_services.get_local_file_path", return_value=nullcontext("/tmp/asvs.pdf")),
        patch(
            "sdr.apps.ai.services.extraction_services.get_document_content",
            return_value={"text": "ASVS Level 1 and Level 2 definitions"},
        ),
        patch("sdr.apps.ai.services.extraction_services.chat_completion", return_value=response),
    ):
        result = extract_asvs_level_definitions_from_document(source_doc, start_page=8, end_page=10)

    assert [item["level"] for item in result] == [1, 2]
    assert result[0]["code"] == "L1"
    assert result[1]["classification_guidance"] == "Use L2 for applications with sensitive data."


def test_category_parameter_child_schema_includes_details():
    child = SimpleNamespace(
        id=1,
        stable_key="child-key",
        asvs_level=2,
        requirement_text="5.4.1 Generic Web Service Security",
        details="All APIs must document interfaces.",
        requirement_text_normalized="5.4.1 generic web service security all apis must document interfaces.",
        ordinal=1,
    )

    data = CategoryParameterChildSchema.model_validate(child).model_dump()

    assert data["details"] == "All APIs must document interfaces."
    assert data["asvs_level"] == 2
