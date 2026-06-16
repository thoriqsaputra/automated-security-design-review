from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.engine.classification.asvs_level import classify_tsd_asvs_level
from sdr.apps.ai.engine.classification.parent_applicability import classify_parent_applicability


def _levels():
    return [
        SimpleNamespace(level=1, name="L1", classification_guidance="baseline"),
        SimpleNamespace(level=2, name="L2", classification_guidance="moderate"),
        SimpleNamespace(level=3, name="L3", classification_guidance="strong"),
    ]


def test_asvs_level_classification_repairs_malformed_json(monkeypatch):
    calls = {"count": 0}

    def fake_chat_completion(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(
                content='{"level": 2, "confidence": 0.95, "reasoning": "missing comma" "evidence": ["login controls"]}',
                error=None,
            )
        return SimpleNamespace(
            content='{"level": 2, "confidence": 0.8, "reasoning": "repaired", "evidence": ["login controls"]}',
            error=None,
        )

    monkeypatch.setattr(
        "sdr.apps.ai.engine.classification.asvs_level.chat_completion",
        fake_chat_completion,
    )

    result = classify_tsd_asvs_level("Sample TSD text", _levels())

    assert result.level == 2
    assert result.confidence == 0.8
    assert result.evidence == ["login controls"]
    assert result.error is None
    assert calls["count"] == 2


def test_asvs_level_classification_falls_back_when_repair_still_invalid(monkeypatch):
    calls = {"count": 0}

    def fake_chat_completion(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(
                content='{"level": 3, "confidence": 0.95, "reasoning": "broken" "evidence": ["tls"]}',
                error=None,
            )
        return SimpleNamespace(content='{"level": 3, "confidence": 0.95,', error=None)

    monkeypatch.setattr(
        "sdr.apps.ai.engine.classification.asvs_level.chat_completion",
        fake_chat_completion,
    )

    result = classify_tsd_asvs_level("Sample TSD text", _levels())

    assert result.level == 1
    assert result.error is not None
    assert "repair_decode_failed" in result.error
    assert calls["count"] == 2


def test_parent_applicability_classification_repairs_malformed_json(monkeypatch):
    calls = {"count": 0}

    def fake_chat_completion(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(
                content='{"applicable": false, "confidence": 0.5, "reasoning": "bad" "evidence": ["missing scope"]}',
                error=None,
            )
        return SimpleNamespace(
            content='{"applicable": false, "confidence": 0.9, "reasoning": "repaired", "evidence": ["missing scope"]}',
            error=None,
        )

    monkeypatch.setattr(
        "sdr.apps.ai.engine.classification.parent_applicability.chat_completion",
        fake_chat_completion,
    )

    result = classify_parent_applicability(
        category_code="web_application",
        version_label="v1.0",
        parent_title="Authentication",
        parent_description="Authentication controls",
        child_requirements=["Require MFA"],
        retrieved_context="Authentication is enforced with MFA for administrator login flows.",
    )

    assert result.applicable is False
    assert result.confidence == 0.9
    assert result.evidence == ["missing scope"]
    assert result.decision_mode == "negative_match"
    assert result.error is None
    assert calls["count"] == 2


def test_parent_applicability_classification_skips_when_context_has_no_family_signal(monkeypatch):
    calls = {"count": 0}

    def fake_chat_completion(**_kwargs):
        calls["count"] += 1
        return SimpleNamespace(content="{}", error=None)

    monkeypatch.setattr(
        "sdr.apps.ai.engine.classification.parent_applicability.chat_completion",
        fake_chat_completion,
    )

    result = classify_parent_applicability(
        category_code="web_application",
        version_label="v1.0",
        parent_title="Session Management",
        parent_description="Browser session and cookie protections",
        child_requirements=["Use secure cookie flags."],
        retrieved_context="The service uses API tokens between backend services and has no browser workflow.",
    )

    assert result.applicable is False
    assert result.confidence == 0.9
    assert result.decision_mode == "no_scope_match"
    assert result.error is None
    assert calls["count"] == 0
