from __future__ import annotations

import json
from types import SimpleNamespace

from sdr.apps.standards.tasks import (
    _generate_child_metadata_map,
    _generate_parent_metadata_map,
)


def _parent(parent_id: int, title: str, description: str = ""):
    return SimpleNamespace(id=parent_id, title=title, description=description)


def _child(child_id: int, parent_id: int, text: str, details: str = ""):
    return SimpleNamespace(
        id=child_id,
        parent_id=parent_id,
        requirement_text=text,
        details=details,
    )


def test_parent_and_child_metadata_generation_uses_llm_contracts(monkeypatch):
    parent = _parent(1, "Identity Controls", "Authentication and session requirements")
    children = [
        _child(11, 1, "Verify authentication uses MFA"),
        _child(12, 1, "Verify session cookies use secure flags"),
    ]

    responses = iter(
        [
            {
                "results": [
                    {
                        "id": "1",
                        "scope_summary": "Identity and session control family.",
                        "control_domain": "identity",
                        "when_applicable": ["The system authenticates users."],
                        "when_not_applicable": ["The system has no authentication boundary."],
                        "required_capabilities": ["auth"],
                        "optional_capabilities": ["session"],
                        "negative_scope_signals": ["no_session_management"],
                        "evidence_keywords": ["mfa", "session cookie"],
                    }
                ]
            },
            {
                "results": [
                    {
                        "id": "11",
                        "scope_summary": "MFA requirement.",
                        "control_domain": "auth",
                        "when_applicable": ["Users authenticate to the system."],
                        "when_not_applicable": [],
                        "required_capabilities": ["auth"],
                        "optional_capabilities": [],
                        "negative_scope_signals": [],
                        "evidence_keywords": ["mfa"],
                    },
                    {
                        "id": "12",
                        "scope_summary": "Session cookie security.",
                        "control_domain": "session",
                        "when_applicable": ["The system uses browser sessions."],
                        "when_not_applicable": ["The system is API-only."],
                        "required_capabilities": ["session", "browser"],
                        "optional_capabilities": [],
                        "negative_scope_signals": ["service_api_only", "no_browser_interface"],
                        "evidence_keywords": ["cookie", "secure flag"],
                    },
                ]
            },
        ]
    )

    monkeypatch.setattr(
        "sdr.apps.standards.tasks.chat_completion",
        lambda **kwargs: SimpleNamespace(
            error=None,
            content=json.dumps(next(responses)),
        ),
    )

    parent_map, parent_rate_limited = _generate_parent_metadata_map(
        [parent],
        {1: children},
        category_code="web_application",
        version_label="ASVS 5.0",
        generation_source="ingestion_llm",
    )
    child_map, child_rate_limited = _generate_child_metadata_map(
        parent,
        children,
        category_code="web_application",
        version_label="ASVS 5.0",
        generation_source="ingestion_llm",
    )

    assert parent_rate_limited is False
    assert child_rate_limited is False
    assert parent_map[1]["required_capabilities"] == ["auth"]
    assert parent_map[1]["generation_source"] == "ingestion_llm"
    assert child_map[11]["required_capabilities"] == ["auth"]
    assert child_map[12]["required_capabilities"] == ["session", "browser"]
    assert child_map[12]["negative_scope_signals"] == ["service_api_only", "no_browser_interface"]


def test_child_metadata_generation_circuit_breaks_to_heuristics_after_rate_limit(monkeypatch):
    parent = _parent(1, "Session Controls", "Browser session requirements")
    children = [_child(child_id, 1, f"Requirement {child_id}") for child_id in range(1, 23)]
    calls = {"count": 0}

    def _fake_completion(**kwargs):
        calls["count"] += 1
        return SimpleNamespace(
            error="429",
            error_code="rate_limit_exhausted",
            content="",
        )

    monkeypatch.setattr("sdr.apps.standards.tasks.chat_completion", _fake_completion)

    child_map, rate_limited = _generate_child_metadata_map(
        parent,
        children,
        category_code="web_application",
        version_label="ASVS 5.0",
        generation_source="ingestion_llm",
    )

    assert rate_limited is True
    assert calls["count"] == 1
    assert len(child_map) == len(children)
    assert all(metadata["generation_source"] == "heuristic_fallback" for metadata in child_map.values())
