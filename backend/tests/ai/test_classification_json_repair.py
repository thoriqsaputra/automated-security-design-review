from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.engine.classification.asvs_level import classify_tsd_asvs_level
from sdr.apps.ai.engine.classification.json_utils import parse_json_with_repair


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


def test_parse_json_with_repair_sanitizes_newlines_without_llm_repair():
    calls = {"count": 0}

    def fake_chat_completion(**_kwargs):
        calls["count"] += 1
        return SimpleNamespace(content='{"unused": true}', error=None)

    parsed, error = parse_json_with_repair(
        '{"reasoning": "line 1\nline 2", "evidence": ["scope"]}',
        component="unit_test",
        max_tokens=100,
        chat_completion_fn=fake_chat_completion,
    )

    assert parsed == {"reasoning": "line 1\nline 2", "evidence": ["scope"]}
    assert error is None
    assert calls["count"] == 0


def test_parse_json_with_repair_sanitizes_trailing_commas_without_llm_repair():
    calls = {"count": 0}

    def fake_chat_completion(**_kwargs):
        calls["count"] += 1
        return SimpleNamespace(content='{"unused": true}', error=None)

    parsed, error = parse_json_with_repair(
        '{"level": 2, "evidence": ["tls",],}',
        component="unit_test",
        max_tokens=100,
        chat_completion_fn=fake_chat_completion,
    )

    assert parsed == {"level": 2, "evidence": ["tls"]}
    assert error is None
    assert calls["count"] == 0


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


def test_asvs_level_classification_sanitizes_trailing_comma_without_repair(monkeypatch):
    calls = {"count": 0}

    def fake_chat_completion(**_kwargs):
        calls["count"] += 1
        return SimpleNamespace(
            content='{"level": 2, "confidence": 0.95, "reasoning": "trailing comma fixed", "evidence": ["tls",],}',
            error=None,
        )

    monkeypatch.setattr(
        "sdr.apps.ai.engine.classification.asvs_level.chat_completion",
        fake_chat_completion,
    )

    result = classify_tsd_asvs_level("Sample TSD text", _levels())

    assert result.level == 2
    assert result.confidence == 0.95
    assert result.evidence == ["tls"]
    assert result.error is None
    assert calls["count"] == 1


