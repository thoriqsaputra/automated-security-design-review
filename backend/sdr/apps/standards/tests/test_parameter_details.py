from types import SimpleNamespace
from unittest.mock import patch

from sdr.apps.ai.services.extraction_services import (
    _remove_table_of_contents,
    extract_structured_requirements,
)
from sdr.apps.standards.models import CategoryParameterChild
from sdr.apps.standards.serializers import CategoryParameterChildSerializer
from sdr.apps.standards.tasks import _coerce_requirement_details, _coerce_requirement_text
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
              "context_marker": "Section 5.4.1"
            }
          ]
        }
        """,
    )

    with patch("apps.ai.services.extraction_services.chat_completion", return_value=response):
        result = extract_structured_requirements("5.4.1 Generic Web Service Security")

    item = result["5.4 API and Web Service"][0]
    assert item["requirement"] == "5.4.1 Generic Web Service Security"
    assert item["details"] == "All APIs must document interfaces and validate input/output."
    assert item["verbatim_quote"] == "a. semua API mendefinisikan ..."
    assert item["context_marker"] == "Section 5.4.1"


def test_category_parameter_child_serializer_includes_details():
    child = CategoryParameterChild(
        id=1,
        stable_key="child-key",
        requirement_text="5.4.1 Generic Web Service Security",
        details="All APIs must document interfaces.",
        requirement_text_normalized="5.4.1 generic web service security all apis must document interfaces.",
        ordinal=1,
    )

    data = CategoryParameterChildSerializer(child).data

    assert data["details"] == "All APIs must document interfaces."
