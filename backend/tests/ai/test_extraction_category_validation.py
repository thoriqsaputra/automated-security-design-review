import json
from types import SimpleNamespace

import pytest

from sdr.apps.ai.engine.extraction.config import ExtractionConfig
from sdr.apps.ai.engine.extraction.services import (
    RequirementCategoryValidationError,
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
    chat_completion_fn = lambda **_: _response(
        json.dumps(
            {
                "items": [
                    {"index": 0, "requirement_category": "infrastructure"},
                    {"index": 1, "requirement_category": "process"},
                ]
            }
        )
    )
    validator = RequirementCategoryValidationService(chat_completion_fn=chat_completion_fn, config=_config())
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
    chat_completion_fn = lambda **_: _response(
        json.dumps({"items": [{"index": 0, "requirement_category": "code"}]})
    )
    validator = RequirementCategoryValidationService(chat_completion_fn=chat_completion_fn, config=_config())
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
        chat_completion_fn=chat_completion,
        config=_config(),
    )

    result = service.extract("3.4.1 Verify that HSTS is set.")

    assert result["V3"][0]["requirement_category"] == "infrastructure"
    assert [call["component"] for call in calls] == [
        "standard_extraction",
        "standard_category_validation",
    ]


def test_category_validator_uses_complete_llm_response_without_local_overrides():
    chat_completion_fn = lambda **_: _response(
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
    validator = RequirementCategoryValidationService(chat_completion_fn=chat_completion_fn, config=_config())
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
        "design",
        "code",
        "infrastructure",
        "code",
    ]


def test_category_validator_preserves_valid_initial_labels_when_llm_output_is_unusable():
    chat_completion_fn = lambda **_: _response('{"items":[{"index":0,"requirement_category":"code"}]}')
    validator = RequirementCategoryValidationService(chat_completion_fn=chat_completion_fn, config=_config())
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
        "design",
        "code",
    ]


def test_category_validator_does_not_override_complete_llm_labels_from_text_patterns():
    chat_completion_fn = lambda **_: _response(
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
    validator = RequirementCategoryValidationService(chat_completion_fn=chat_completion_fn, config=_config())
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
        "code",
        "code",
        "code",
    ]


def test_category_validator_rejects_missing_initial_label_when_llm_output_is_unusable():
    chat_completion_fn = lambda **_: _response(
        '{"items":[{"index":0,"requirement_category":"code"}]}'
    )
    validator = RequirementCategoryValidationService(
        chat_completion_fn=chat_completion_fn,
        config=_config(),
    )
    requirements = {
        "V": [
            {
                "requirement": "1.1.1 Verify that a security control is documented.",
                "requirement_category": "",
            },
            {
                "requirement": "1.1.2 Verify that another security control is documented.",
                "requirement_category": "design",
            },
        ]
    }

    with pytest.raises(RequirementCategoryValidationError):
        validator.validate(requirements)


def test_structured_extraction_propagates_category_validation_failure():
    responses = iter(
        [
            _response(
                json.dumps(
                    {
                        "V1": [
                            {
                                "requirement": "1.1.1 Verify that a security control is documented.",
                                "context_marker": "V1.1",
                            }
                        ]
                    }
                )
            ),
            _response("{}", error="validator unavailable"),
        ]
    )
    service = StructuredRequirementExtractionService(
        chat_completion_fn=lambda **_: next(responses),
        config=_config(),
    )

    with pytest.raises(RequirementCategoryValidationError):
        service.extract("1.1.1 Verify that a security control is documented.")
