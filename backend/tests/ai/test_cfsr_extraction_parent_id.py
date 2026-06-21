from types import SimpleNamespace
from unittest.mock import patch

from sdr.apps.ai.engine.extraction.api import extract_control_family_summary_requirements
from sdr.apps.ai.engine.extraction.config import ExtractionConfig
from sdr.apps.ai.engine.extraction.llm_client import ExtractionLLMClient
from sdr.apps.ai.engine.extraction.services import ControlFamilySummaryExtractionService


def _mock_response():
    return SimpleNamespace(
        error=None,
        model="mock-model",
        provider="mock-provider",
        usage=None,
        content="""
        {
          "summary_requirements": [
            {
              "stable_key": "CFSR-V2-1",
              "requirement_text": "The TSD must describe sequential step enforcement.",
              "analysis_hint": "Look for architecture descriptions or data-flow diagrams.",
              "covered_child_keys": ["child-001"],
              "asvs_level": 1
            }
          ]
        }
        """,
    )


def test_cfsr_extraction_uses_child_parent_id_when_parent_relationship_is_unavailable():
    response = _mock_response()
    service = ControlFamilySummaryExtractionService(
        llm_client=ExtractionLLMClient(chat_completion=lambda **_: response),
        config=ExtractionConfig(cfsr_extraction_max_concurrency=1, cfsr_max_per_parent=5),
    )
    parameters = [
        SimpleNamespace(
            id=101,
            stable_key="V2-1",
            requirement_text="Describe sequential step enforcement.",
            details="Flow cannot skip mandatory steps.",
            asvs_level=1,
            parent_id=44,
            parent=None,
        )
    ]

    result = service.extract(parameters=parameters, category_id=1, ingestion_job_id=44)

    assert len(result) == 1
    assert result[0]["parent_id"] == 44
    assert result[0]["covered_child_keys"] == ["V2-1"]


def test_cfsr_extraction_skips_groups_with_missing_parent_id():
    response = _mock_response()
    service = ControlFamilySummaryExtractionService(
        llm_client=ExtractionLLMClient(chat_completion=lambda **_: response),
        config=ExtractionConfig(cfsr_extraction_max_concurrency=1, cfsr_max_per_parent=5),
    )
    parameters = [
        SimpleNamespace(
            id=102,
            stable_key="V2-2",
            requirement_text="Describe anti-skip protections.",
            details="Users cannot bypass enforced sequence.",
            asvs_level=1,
            parent_id=None,
            parent=None,
        )
    ]

    result = service.extract(parameters=parameters, category_id=1, ingestion_job_id=44)

    assert result == []


def test_cfsr_api_wrapper_preserves_fix_for_detached_parent():
    parameters = [
        SimpleNamespace(
            id=103,
            stable_key="V2-3",
            requirement_text="Describe password reset confirmation.",
            details="Flow confirms identity before reset.",
            asvs_level=1,
            parent_id=55,
            parent=None,
        )
    ]

    with patch("sdr.apps.ai.engine.extraction.api.chat_completion", return_value=_mock_response()):
        result = extract_control_family_summary_requirements(
            parameters=parameters,
            category_id=1,
            ingestion_job_id=44,
        )

    assert len(result) == 1
    assert result[0]["parent_id"] == 55
