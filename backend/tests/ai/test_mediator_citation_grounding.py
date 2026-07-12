from __future__ import annotations

import json

from sdr.apps.ai.agents.base import Citation, CriticResult, HunterResult
from sdr.apps.ai.agents.mediator import MediatorAgent
from sdr.apps.ai.client.base import AIProvider, AIResponse


def _ai_response(payload: dict) -> AIResponse:
    return AIResponse(
        content=json.dumps(payload),
        model="test-model",
        provider=AIProvider.NVIDIA,
        finish_reason="stop",
    )


def test_run_downgrades_ungrounded_met_to_not_met(monkeypatch):
    agent = MediatorAgent()
    hunter_result = HunterResult(verdict="met", confidence=0.8)
    critic_result = CriticResult(
        outcome="OVERTURN",
        revised_verdict="not_met",
        revised_confidence=0.5,
        reasoning="Critic could not verify any citation.",
        logic_summary="Critic could not verify any citation.",
        valid_citations=[],
    )

    def fake_call_llm(user_prompt, **_kwargs):
        return _ai_response(
            {
                "final_verdict": "met",
                "confidence": 0.9,
                "reasoning": "Mediator believes the control is met.",
                "logic_summary": "Mediator believes the control is met.",
                "final_citations": [
                    {"block_id": "p1_b1", "page_number": 1, "quoted_text": "uses TLS"}
                ],
            }
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = agent.run(
        parameter_text="Require TLS for all traffic",
        parameter_section="Transport Security",
        hunter_result=hunter_result,
        critic_result=critic_result,
        debate_history=[],
    )

    assert result.final_verdict == "not_met"
    assert result.final_citations == []
    assert result.confidence <= 0.45
    assert result.severity is not None
    assert result.recommendation


def test_run_keeps_met_when_citations_survive(monkeypatch):
    agent = MediatorAgent()
    hunter_result = HunterResult(verdict="met", confidence=0.9)
    citation = Citation(block_id="p1_b1", page_number=1, quoted_text="uses TLS")
    critic_result = CriticResult(
        outcome="UPHOLD",
        revised_verdict="met",
        revised_confidence=0.9,
        reasoning="Critic verified the citation.",
        logic_summary="Critic verified the citation.",
        valid_citations=[citation],
    )

    def fake_call_llm(user_prompt, **_kwargs):
        return _ai_response(
            {
                "final_verdict": "met",
                "confidence": 0.9,
                "reasoning": "Mediator agrees the control is met.",
                "logic_summary": "Mediator agrees the control is met.",
                "final_citations": [
                    {"block_id": "p1_b1", "page_number": 1, "quoted_text": "uses TLS"}
                ],
            }
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = agent.run(
        parameter_text="Require TLS for all traffic",
        parameter_section="Transport Security",
        hunter_result=hunter_result,
        critic_result=critic_result,
        debate_history=[],
    )

    assert result.final_verdict == "met"
    assert len(result.final_citations) == 1
    assert result.severity is None


def test_fast_path_keeps_verified_met_with_lower_confidence():
    agent = MediatorAgent()
    hunter_result = HunterResult(verdict="met", confidence=0.62)
    citation = Citation(block_id="p1_b1", page_number=1, quoted_text="uses TLS")
    critic_result = CriticResult(
        outcome="UPHOLD",
        revised_verdict="met",
        revised_confidence=0.61,
        valid_citations=[citation],
    )

    fast_path_result = agent._try_fast_path(
        parameter_text="Require TLS for all traffic",
        hunter_result=hunter_result,
        critic_result=critic_result,
        debate_history=[],
    )

    assert fast_path_result is not None
    assert fast_path_result.final_verdict == "met"


def test_fast_path_falls_through_for_ungrounded_met():
    agent = MediatorAgent()
    hunter_result = HunterResult(verdict="met", confidence=0.9)
    critic_result = CriticResult(
        outcome="UPHOLD",
        revised_verdict="met",
        revised_confidence=0.9,
        valid_citations=[],
    )

    fast_path_result = agent._try_fast_path(
        parameter_text="Require TLS for all traffic",
        hunter_result=hunter_result,
        critic_result=critic_result,
        debate_history=[],
    )

    assert fast_path_result is None


def test_fallback_to_critic_downgrades_ungrounded_met():
    agent = MediatorAgent()
    critic_result = CriticResult(
        outcome="UPHOLD",
        revised_verdict="met",
        revised_confidence=0.8,
        reasoning="Critic upheld the met verdict.",
        logic_summary="Critic upheld the met verdict.",
        valid_citations=[],
    )

    result = agent._fallback_to_critic(
        critic_result=critic_result,
        error_msg="LLM call failed: timeout",
        raw=None,
        parameter_text="Require TLS for all traffic",
    )

    assert result.final_verdict == "not_met"
    assert result.final_citations == []
    assert result.confidence <= 0.45
    assert result.recommendation
