from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch

from sdr.apps.ai.engine.extraction import (
    _canonicalize_diagram_requirements,
    _remove_table_of_contents,
    extract_asvs_level_definitions_from_document,
    extract_diagram_requirements,
    extract_structured_requirements,
)
from sdr.apps.standards.schemas import CategoryParameterChildSchema
from sdr.apps.standards.tasks import _coerce_requirement_details, _coerce_requirement_text
from sdr.apps.standards.tasks import _coerce_asvs_level
from sdr.apps.standards.utils import (
    build_diagram_requirement_analysis_text,
    build_parameter_analysis_text,
    normalize_requirement_text,
)


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


def test_build_diagram_requirement_analysis_text_enriches_with_source_parameter():
    diagram_requirement = SimpleNamespace(
        parent_section="V1 Architecture",
        requirement_text="Show trust boundary",
        verification_hint="Boundary line must separate public and internal zones.",
    )
    source_parameter = SimpleNamespace(
        requirement_text="1.4.1 Trust Boundaries",
        details="Applications should identify and document trust boundaries.",
    )

    text = build_diagram_requirement_analysis_text(
        diagram_requirement,
        source_parameter=source_parameter,
    )

    assert "V1 Architecture" in text
    assert "Show trust boundary" in text
    assert "Applications should identify and document trust boundaries." in text


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

    with patch("sdr.apps.ai.engine.extraction.api.chat_completion", return_value=response):
        result = extract_structured_requirements("5.4.1 Generic Web Service Security")

    item = result["5.4 API and Web Service"][0]
    assert item["requirement"] == "5.4.1 Generic Web Service Security"
    assert item["details"] == "All APIs must document interfaces and validate input/output."
    assert item["verbatim_quote"] == "a. semua API mendefinisikan ..."
    assert item["context_marker"] == "Section 5.4.1"
    assert item["asvs_level"] == 2


def test_extract_structured_requirements_filters_note_paragraphs():
    response = SimpleNamespace(
        error=None,
        model="mock-model",
        provider="mock-provider",
        usage=None,
        content="""
        {
          "V5 Validation": [
            {
              "requirement": "Note: Using parameterized queries or escaping SQL is not always sufficient",
              "details": "Table and column names cannot be escaped safely.",
              "verbatim_quote": "Note: Using parameterized queries or escaping SQL is not always sufficient...",
              "context_marker": "V5.3"
            },
            {
              "requirement": "5.3.4 SQL queries must separate data values from query structure",
              "details": "Applications must treat identifiers and sort clauses as trusted-only inputs.",
              "verbatim_quote": "Verify that parameterized queries separate data values from query structure.",
              "context_marker": "V5.3.4",
              "asvs_level": 2
            }
          ]
        }
        """,
    )

    with patch("sdr.apps.ai.engine.extraction.api.chat_completion", return_value=response):
        result = extract_structured_requirements("V5 SQL requirements")

    assert [item["requirement"] for item in result["V5 Validation"]] == [
        "5.3.4 SQL queries must separate data values from query structure"
    ]


def test_extract_structured_requirements_filters_owasp_subsection_heading_only_items():
    response = SimpleNamespace(
        error=None,
        model="mock-model",
        provider="mock-provider",
        usage=None,
        content="""
        {
          "V2 Authentication": [
            {
              "requirement": "V2.1 Password Security",
              "details": "Passwords (memorized secrets) must be used as single-factor authenticators.",
              "verbatim_quote": "V2.1 Password Security",
              "context_marker": "V2.1"
            },
            {
              "requirement": "V2.1 - 2.1.1 Verify user set passwords are at least 12 characters in length",
              "details": "Applications must enforce a minimum password length for user-chosen passwords.",
              "verbatim_quote": "2.1.1 Verify user set passwords are at least 12 characters in length.",
              "context_marker": "V2.1",
              "asvs_level": 1
            }
          ]
        }
        """,
    )

    with patch("sdr.apps.ai.engine.extraction.api.chat_completion", return_value=response):
        result = extract_structured_requirements("V2 authentication requirements")

    assert [item["requirement"] for item in result["V2 Authentication"]] == [
        "V2.1 - 2.1.1 Verify user set passwords are at least 12 characters in length"
    ]


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
        patch("sdr.apps.ai.engine.extraction.api.get_local_file_path", return_value=nullcontext("/tmp/asvs.pdf")),
        patch(
            "sdr.apps.ai.engine.extraction.api.get_document_content",
            return_value={"text": "ASVS Level 1 and Level 2 definitions"},
        ),
        patch("sdr.apps.ai.engine.extraction.api.chat_completion", return_value=response),
    ):
        result = extract_asvs_level_definitions_from_document(source_doc, start_page=8, end_page=10)

    assert [item["level"] for item in result] == [1, 2]
    assert result[0]["code"] == "L1"
    assert result[1]["classification_guidance"] == "Use L2 for applications with sensitive data."


def test_extract_asvs_level_definitions_from_document_repairs_trailing_comma_json():
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
            }
          ]
        }
        """,
    )

    with (
        patch("sdr.apps.ai.engine.extraction.api.get_local_file_path", return_value=nullcontext("/tmp/asvs.pdf")),
        patch(
            "sdr.apps.ai.engine.extraction.api.get_document_content",
            return_value={"text": "ASVS Level 1 definitions"},
        ),
        patch("sdr.apps.ai.engine.extraction.api.chat_completion", return_value=response),
    ):
        result = extract_asvs_level_definitions_from_document(source_doc, start_page=8, end_page=10)

    assert [item["level"] for item in result] == [1]
    assert result[0]["classification_guidance"] == "Use L1 for ordinary applications."


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


def test_canonicalize_diagram_requirements_suffixes_duplicate_stable_keys():
    items = [
        {
            "stable_key": "job9-D-V1.4",
            "source_requirement_key": "child-1",
            "requirement_text": "Show trust boundary",
            "verification_hint": "Boundary visible",
            "asvs_level": 1,
            "parent_section": "V1 Architecture",
        },
        {
            "stable_key": "job9-D-V1.4",
            "source_requirement_key": "child-2",
            "requirement_text": "Show auth boundary",
            "verification_hint": "Auth visible",
            "asvs_level": 1,
            "parent_section": "V1 Architecture",
        },
    ]

    result = _canonicalize_diagram_requirements(items)

    assert [item["stable_key"] for item in result] == [
        "job9-D-V1.4",
        "job9-D-V1.4-2",
    ]


def test_canonicalize_diagram_requirements_drops_exact_duplicates():
    items = [
        {
            "stable_key": "job9-D-V1.4",
            "source_requirement_key": "child-1",
            "requirement_text": "Show trust boundary",
            "verification_hint": "Boundary visible",
            "asvs_level": 1,
            "parent_section": "V1 Architecture",
        },
        {
            "stable_key": "job9-D-V1.4",
            "source_requirement_key": "child-1",
            "requirement_text": "Show trust boundary",
            "verification_hint": "Boundary visible",
            "asvs_level": 1,
            "parent_section": "V1 Architecture",
        },
    ]

    result = _canonicalize_diagram_requirements(items)

    assert len(result) == 1
    assert result[0]["stable_key"] == "job9-D-V1.4"


def test_canonicalize_diagram_requirements_avoids_collision_with_pre_suffixed_keys():
    items = [
        {
            "stable_key": "job17-D-V3.2",
            "source_requirement_key": "child-1",
            "requirement_text": "Show control A",
            "verification_hint": "A visible",
            "asvs_level": 3,
            "parent_section": "V3 Session Management",
        },
        {
            "stable_key": "job17-D-V3.2-2",
            "source_requirement_key": "child-2",
            "requirement_text": "Show control B",
            "verification_hint": "B visible",
            "asvs_level": 3,
            "parent_section": "V3 Session Management",
        },
        {
            "stable_key": "job17-D-V3.2",
            "source_requirement_key": "child-3",
            "requirement_text": "Show control C",
            "verification_hint": "C visible",
            "asvs_level": 3,
            "parent_section": "V3 Session Management",
        },
    ]

    result = _canonicalize_diagram_requirements(items)

    assert [item["stable_key"] for item in result] == [
        "job17-D-V3.2",
        "job17-D-V3.2-2",
        "job17-D-V3.2-3",
    ]


def test_extract_diagram_requirements_canonicalizes_duplicate_llm_keys():
    parameters = [
        SimpleNamespace(
            stable_key="child-1",
            requirement_text="Trust boundary",
            details="Show zone split",
            asvs_level=1,
            parent=SimpleNamespace(title="V1 Architecture"),
        ),
        SimpleNamespace(
            stable_key="child-2",
            requirement_text="Authentication path",
            details="Show auth flow",
            asvs_level=1,
            parent=SimpleNamespace(title="V1 Architecture"),
        ),
    ]
    response = SimpleNamespace(
        error=None,
        content="""
        {
          "diagram_requirements": [
            {
              "stable_key": "D-V1.4",
              "source_requirement_id": "child-1",
              "requirement_text": "Show trust boundary",
              "verification_hint": "Boundary visible",
              "asvs_level": 1,
              "parent_section": "V1 Architecture"
            },
            {
              "stable_key": "D-V1.4",
              "source_requirement_id": "child-2",
              "requirement_text": "Show authentication path",
              "verification_hint": "Auth visible",
              "asvs_level": 1,
              "parent_section": "V1 Architecture"
            }
          ]
        }
        """,
    )

    with patch("sdr.apps.ai.engine.extraction.api.chat_completion", return_value=response):
        result = extract_diagram_requirements(parameters=parameters, category_id=1, ingestion_job_id=9)

    assert [item["stable_key"] for item in result] == [
        "job9-D-V1.4",
        "job9-D-V1.4-2",
    ]
