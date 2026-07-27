from types import SimpleNamespace
from unittest.mock import patch

from sdr.apps.ai.engine.extraction import (
    _canonicalize_diagram_requirements,
    canonicalize_requirement_items,
    extract_diagram_requirements,
    extract_structured_requirements,
)
from sdr.apps.standards.schemas import CategoryParameterChildSchema
from sdr.apps.standards.tasks import _coerce_requirement_text
from sdr.apps.standards.utils import (
    build_diagram_requirement_analysis_text,
    normalize_requirement_text,
)





def test_build_diagram_requirement_analysis_text_enriches_with_source_parameter():
    diagram_requirement = SimpleNamespace(
        parent_section="V1 Architecture",
        requirement_text="Show trust boundary",
        verification_hint="Boundary line must separate public and internal zones.",
    )
    source_parameter = SimpleNamespace(
        requirement_text="1.4.1 Trust Boundaries",
        description="Applications should identify and document trust boundaries.",
    )

    text = build_diagram_requirement_analysis_text(
        diagram_requirement,
        source_parameter=source_parameter,
    )

    assert "V1 Architecture" in text
    assert "Show trust boundary" in text
    assert "1.4.1 Trust Boundaries" in text


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
              "verbatim_quote": "a. semua API mendefinisikan ...",
              "context_marker": "Section 5.4.1",
              "requirement_category": "design"
            }
          ]
        }
        """,
    )

    with patch("sdr.apps.ai.engine.extraction.api.chat_completion", return_value=response):
        result = extract_structured_requirements("5.4.1 Generic Web Service Security")

    item = result["5.4 API and Web Service"][0]
    assert item["requirement"] == "5.4.1 Generic Web Service Security"
    assert item["verbatim_quote"] == "a. semua API mendefinisikan ..."
    assert item["context_marker"] == "Section 5.4.1"


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
              "verbatim_quote": "Note: Using parameterized queries or escaping SQL is not always sufficient...",
              "context_marker": "V5.3",
              "requirement_category": "code"
            },
            {
              "requirement": "5.3.4 SQL queries must separate data values from query structure",
              "verbatim_quote": "Verify that parameterized queries separate data values from query structure.",
              "context_marker": "V5.3.4",
              "requirement_category": "code"
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


def test_extract_structured_requirements_filters_heading_only_items():
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
              "verbatim_quote": "V2.1 Password Security",
              "context_marker": "V2.1",
              "requirement_category": "design"
            },
            {
              "requirement": "V2.1 - 2.1.1 Verify user set passwords are at least 12 characters in length",
              "verbatim_quote": "2.1.1 Verify user set passwords are at least 12 characters in length.",
              "context_marker": "V2.1",
              "requirement_category": "code"
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


def test_canonicalize_requirement_items_dedupes_same_control_id_and_keeps_richer_item():
    items = [
        {
            "requirement": "V1.1 - 1.1.1 Verify the use of a secure SDLC",
            "verbatim_quote": "",
            "context_marker": "V1.1",
        },
        {
            "requirement": "1.1.1 Verify the use of a secure SDLC",
            "verbatim_quote": "1.1.1 Verify the use of a secure SDLC",
            "context_marker": "V1.1",
        },
    ]

    result = canonicalize_requirement_items(items)

    assert len(result) == 1
    assert result[0]["requirement"] == "1.1.1 Verify the use of a secure SDLC"


def test_canonicalize_requirement_items_drops_deleted_placeholder_rows():
    items = [
        {
            "requirement": "6.1.1 Verify passwords are hashed using a memory-hard algorithm.",
            "verbatim_quote": "",
            "context_marker": "V6.1",
        },
        {
            "requirement": "[DELETED, DUPLICATE OF 6.1.1]",
            "verbatim_quote": "[DELETED, DUPLICATE OF 6.1.1]",
            "context_marker": "V6.1",
        },
    ]

    result = canonicalize_requirement_items(items)

    assert len(result) == 1
    assert result[0]["requirement"] == "6.1.1 Verify passwords are hashed using a memory-hard algorithm."





def test_canonicalize_diagram_requirements_suffixes_duplicate_stable_keys():
    items = [
        {
            "stable_key": "job9-D-V1.4",
            "source_requirement_key": "child-1",
            "requirement_text": "Show trust boundary",
            "verification_hint": "Boundary visible",
            "parent_section": "V1 Architecture",
        },
        {
            "stable_key": "job9-D-V1.4",
            "source_requirement_key": "child-2",
            "requirement_text": "Show auth boundary",
            "verification_hint": "Auth visible",
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
            "parent_section": "V1 Architecture",
        },
        {
            "stable_key": "job9-D-V1.4",
            "source_requirement_key": "child-1",
            "requirement_text": "Show trust boundary",
            "verification_hint": "Boundary visible",
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
            "parent_section": "V3 Session Management",
        },
        {
            "stable_key": "job17-D-V3.2-2",
            "source_requirement_key": "child-2",
            "requirement_text": "Show control B",
            "verification_hint": "B visible",
            "parent_section": "V3 Session Management",
        },
        {
            "stable_key": "job17-D-V3.2",
            "source_requirement_key": "child-3",
            "requirement_text": "Show control C",
            "verification_hint": "C visible",
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
            parent=SimpleNamespace(title="V1 Architecture"),
        ),
        SimpleNamespace(
            stable_key="child-2",
            requirement_text="Authentication path",
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
              "parent_section": "V1 Architecture"
            },
            {
              "stable_key": "D-V1.4",
              "source_requirement_id": "child-2",
              "requirement_text": "Show authentication path",
              "verification_hint": "Auth visible",
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
