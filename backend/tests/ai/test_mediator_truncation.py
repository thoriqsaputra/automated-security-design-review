from __future__ import annotations

from sdr.apps.ai.agents.base import CriticResult, HunterResult
from sdr.apps.ai.agents.mediator import MediatorAgent
from sdr.apps.ai.client.base import AIProvider, AIResponse


def test_mediator_returns_degraded_result_when_llm_response_is_truncated(monkeypatch):
    agent = MediatorAgent()
    hunter_result = HunterResult(verdict="met", confidence=0.2)
    critic_result = CriticResult(
        outcome="OVERTURN",
        revised_verdict="not_met",
        revised_confidence=0.3,
        reasoning="Critic saw missing evidence.",
        logic_summary="Critic saw missing evidence.",
    )

    def fake_call_llm(user_prompt, **_kwargs):
        return AIResponse(
            content='{"final_verdict":"not_met","confidence":0.3',
            model="test-model",
            provider=AIProvider.NVIDIA,
            finish_reason="length",
        )

    def fail_parse(_response):
        raise AssertionError("Mediator should not attempt JSON parsing after truncation")

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "_parse_json_response", fail_parse)

    result = agent.run(
        parameter_text="Require MFA for admin access",
        parameter_section="Authentication",
        contract={"domain": "iam_access_control"},
        hunter_result=hunter_result,
        critic_result=critic_result,
        debate_history=[],
    )

    assert result.final_verdict == "not_met"
    assert result.error is not None
    assert "truncated" in result.error.lower()
    assert result.logic_summary.startswith("[DEGRADED")
