from __future__ import annotations

from sdr.apps.ai.agents.base import BaseAgent, VERDICT_NA
from sdr.apps.ai.agents.critic import CriticAgent
from sdr.apps.ai.agents.hunter import HunterAgent
from sdr.apps.ai.client.base import AIProvider, AIResponse


def _response(content="{}", finish_reason=None, error=None):
    return AIResponse(
        content=content,
        model="test-model",
        provider=AIProvider.NVIDIA,
        finish_reason=finish_reason,
        error=error,
    )


def test_retry_returns_second_response_when_first_is_truncated(monkeypatch):
    agent = HunterAgent()
    calls = []

    def fake_call_llm(_self, user_prompt, **kwargs):
        calls.append(kwargs.get("max_tokens"))
        if len(calls) == 1:
            return _response(content="{partial", finish_reason="length")
        return _response(content='{"verdict": "met"}', finish_reason="stop")

    monkeypatch.setattr(BaseAgent, "_call_llm", fake_call_llm)

    result = agent._call_llm_with_truncation_retry("prompt")

    assert result.finish_reason == "stop"
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] == min(agent.max_tokens * 2, 16384)


def test_retry_returns_still_truncated_response_when_retry_also_fails(monkeypatch):
    agent = CriticAgent()
    calls = []

    def fake_call_llm(_self, user_prompt, **kwargs):
        calls.append(kwargs.get("max_tokens"))
        return _response(content="{partial", finish_reason="length")

    monkeypatch.setattr(BaseAgent, "_call_llm", fake_call_llm)

    result = agent._call_llm_with_truncation_retry("prompt")

    assert result.finish_reason == "length"
    assert len(calls) == 2

    parsed = agent._parse_json_response(result)
    assert parsed is None

    fallback = agent._critic_error("still truncated after retry")
    assert fallback.revised_verdict == VERDICT_NA


def test_no_retry_when_response_not_truncated(monkeypatch):
    agent = HunterAgent()
    calls = []

    def fake_call_llm(_self, user_prompt, **kwargs):
        calls.append(kwargs.get("max_tokens"))
        return _response(content='{"verdict": "met"}', finish_reason="stop")

    monkeypatch.setattr(BaseAgent, "_call_llm", fake_call_llm)

    result = agent._call_llm_with_truncation_retry("prompt")

    assert result.finish_reason == "stop"
    assert len(calls) == 1


def test_no_retry_when_response_has_error(monkeypatch):
    agent = CriticAgent()
    calls = []

    def fake_call_llm(_self, user_prompt, **kwargs):
        calls.append(kwargs.get("max_tokens"))
        return _response(content="", finish_reason="length", error="provider unavailable")

    monkeypatch.setattr(BaseAgent, "_call_llm", fake_call_llm)

    result = agent._call_llm_with_truncation_retry("prompt")

    assert result.error == "provider unavailable"
    assert len(calls) == 1


def test_hunter_error_defaults_to_na():
    agent = HunterAgent()
    result = agent._hunter_error("LLM call failed: timeout")
    assert result.verdict == VERDICT_NA
    assert result.confidence == 0.0


def test_critic_error_defaults_to_na():
    agent = CriticAgent()
    result = agent._critic_error("Failed to parse LLM response as JSON.")
    assert result.revised_verdict == VERDICT_NA
    assert result.revised_confidence == 0.0
