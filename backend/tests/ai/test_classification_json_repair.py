from __future__ import annotations

from sdr.apps.ai.engine.classification.json_utils import parse_json_with_repair


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

