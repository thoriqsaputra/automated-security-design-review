import json
from types import SimpleNamespace

from sdr.apps.ai.engine.extraction.config import ExtractionConfig
from sdr.apps.ai.engine.extraction.llm_client import ExtractionLLMClient
from sdr.apps.ai.engine.extraction.services import (
    RequirementCategoryValidationService,
    StructuredRequirementExtractionService,
)


def _response(content: str, error=None):
    return SimpleNamespace(
        content=content,
        error=error,
        model="mock-model",
        provider="mock-provider",
        usage=None,
    )


def _config() -> ExtractionConfig:
    return ExtractionConfig(
        standard_extraction_max_workers=1,
        standard_category_validation_batch_size=25,
    )


def test_category_validator_replaces_every_category_in_a_complete_batch():
    client = ExtractionLLMClient(
        chat_completion=lambda **_: _response(
            json.dumps(
                {
                    "items": [
                        {"index": 0, "requirement_category": "infrastructure"},
                        {"index": 1, "requirement_category": "process"},
                    ]
                }
            )
        )
    )
    validator = RequirementCategoryValidationService(llm_client=client, config=_config())
    requirements = {
        "V1": [
            {"requirement": "1.6.2 Use a key vault", "requirement_category": "design"},
            {"requirement": "1.1.1 Maintain an SBOM", "requirement_category": "code"},
        ]
    }

    result = validator.validate(requirements)

    assert [item["requirement_category"] for item in result["V1"]] == [
        "infrastructure",
        "process",
    ]


def test_category_validator_preserves_extracted_categories_for_partial_response():
    client = ExtractionLLMClient(
        chat_completion=lambda **_: _response(
            json.dumps({"items": [{"index": 0, "requirement_category": "code"}]})
        )
    )
    validator = RequirementCategoryValidationService(llm_client=client, config=_config())
    requirements = {
        "V1": [
            {"requirement": "1.1.1 Maintain an SBOM", "requirement_category": "process"},
            {"requirement": "3.4.1 Set HSTS", "requirement_category": "infrastructure"},
        ]
    }

    result = validator.validate(requirements)

    assert [item["requirement_category"] for item in result["V1"]] == [
        "process",
        "infrastructure",
    ]


def test_structured_extraction_uses_validator_category_as_final_label():
    calls = []
    responses = iter(
        [
            _response(
                json.dumps(
                    {
                        "V3": [
                            {
                                "requirement": "3.4.1 Verify that HSTS is set.",
                                "context_marker": "V3.4",
                                "requirement_category": "code",
                            }
                        ]
                    }
                )
            ),
            _response(
                json.dumps(
                    {"items": [{"index": 0, "requirement_category": "infrastructure"}]}
                )
            ),
        ]
    )

    def chat_completion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    service = StructuredRequirementExtractionService(
        llm_client=ExtractionLLMClient(chat_completion=chat_completion),
        config=_config(),
    )

    result = service.extract("3.4.1 Verify that HSTS is set.")

    assert result["V3"][0]["requirement_category"] == "infrastructure"
    assert [call["component"] for call in calls] == [
        "standard_extraction",
        "standard_category_validation",
    ]


def test_category_validator_applies_local_overrides_after_llm_validation():
    client = ExtractionLLMClient(
        chat_completion=lambda **_: _response(
            json.dumps(
                {
                    "items": [
                        {"index": 0, "requirement_category": "design"},
                        {"index": 1, "requirement_category": "code"},
                        {"index": 2, "requirement_category": "infrastructure"},
                        {"index": 3, "requirement_category": "code"},
                    ]
                }
            )
        )
    )
    validator = RequirementCategoryValidationService(llm_client=client, config=_config())
    requirements = {
        "V": [
            {
                "requirement": "3.4.1 Verify that cookie-based session tokens have the 'Secure' attribute set.",
                "requirement_category": "design",
            },
            {
                "requirement": "14.1.1 Verify that the application build and deployment processes are performed in a secure and repeatable way.",
                "requirement_category": "design",
            },
            {
                "requirement": "11.1.3 Verify that cryptographic discovery mechanisms are employed to identify all instances of cryptography in the system.",
                "requirement_category": "design",
            },
            {
                "requirement": "7.3.1 Verify that there is an inactivity timeout such that re-authentication is enforced according to risk analysis and documented security decisions.",
                "requirement_category": "code",
            },
        ]
    }

    result = validator.validate(requirements)

    assert [item["requirement_category"] for item in result["V"]] == [
        "code",
        "infrastructure",
        "process",
        "design",
    ]


def test_category_validator_applies_local_overrides_when_llm_output_is_unusable():
    client = ExtractionLLMClient(
        chat_completion=lambda **_: _response('{"items":[{"index":0,"requirement_category":"code"}]}')
    )
    validator = RequirementCategoryValidationService(llm_client=client, config=_config())
    requirements = {
        "V": [
            {
                "requirement": "4.3.1 Verify that a query allowlist, depth limiting, or query cost analysis is used to prevent GraphQL denial of service.",
                "requirement_category": "design",
            },
            {
                "requirement": "14.2.2 Verify that the application prevents sensitive data from being cached in server components, such as load balancers and application caches, or ensures that the data is securely purged after use.",
                "requirement_category": "code",
            },
        ]
    }

    result = validator.validate(requirements)

    assert [item["requirement_category"] for item in result["V"]] == [
        "code",
        "infrastructure",
    ]


def test_category_validator_normalizes_unicode_dashes_and_ocr_variants_for_overrides():
    client = ExtractionLLMClient(
        chat_completion=lambda **_: _response(
            json.dumps(
                {
                    "items": [
                        {"index": 0, "requirement_category": "code"},
                        {"index": 1, "requirement_category": "code"},
                        {"index": 2, "requirement_category": "code"},
                    ]
                }
            )
        )
    )
    validator = RequirementCategoryValidationService(llm_client=client, config=_config())
    requirements = {
        "V": [
            {
                "requirement": "V3.2 - 3.2.1 Verify that security controls are in place to prevent browsers from rendering content in an incorrect context using Sec‑Fetch headers.",
                "requirement_category": "code",
            },
            {
                "requirement": "4.1.1 Verify that every HTTP response contains a Content‑Type header feld that matches the actual content of the response.",
                "requirement_category": "code",
            },
            {
                "requirement": "V7.2 - 7.2.1 Verify that the application performs all session token verifcation using a trusted, backend service.",
                "requirement_category": "code",
            },
        ]
    }

    result = validator.validate(requirements)

    assert [item["requirement_category"] for item in result["V"]] == [
        "infrastructure",
        "infrastructure",
        "design",
    ]
