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
        contract={},
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
        contract={},
        hunter_result=hunter_result,
        critic_result=critic_result,
        debate_history=[],
    )

    assert result.final_verdict == "met"
    assert len(result.final_citations) == 1
    assert result.severity is None


def test_run_batch_downgrades_only_ungrounded_child(monkeypatch):
    agent = MediatorAgent()
    citation = Citation(block_id="p1_b1", page_number=1, quoted_text="uses TLS")

    hunter_results = {
        "1": HunterResult(verdict="met", confidence=0.6),
        "2": HunterResult(verdict="met", confidence=0.6),
    }
    critic_results = {
        "1": CriticResult(
            outcome="OVERTURN",
            revised_verdict="not_met",
            revised_confidence=0.5,
            valid_citations=[],
        ),
        "2": CriticResult(
            outcome="UPHOLD",
            revised_verdict="met",
            revised_confidence=0.6,
            valid_citations=[citation],
        ),
    }

    def fake_call_llm(user_prompt, **_kwargs):
        return _ai_response(
            {
                "results": [
                    {
                        "child_id": "1",
                        "final_verdict": "met",
                        "confidence": 0.9,
                        "reasoning": "Ungrounded met.",
                        "logic_summary": "Ungrounded met.",
                        "final_citations": [],
                    },
                    {
                        "child_id": "2",
                        "final_verdict": "met",
                        "confidence": 0.9,
                        "reasoning": "Grounded met.",
                        "logic_summary": "Grounded met.",
                        "final_citations": [
                            {"block_id": "p1_b1", "page_number": 1, "quoted_text": "uses TLS"}
                        ],
                    },
                ]
            }
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    results = agent.run_batch(
        child_inputs=[
            {"id": "1", "requirement": "Req 1"},
            {"id": "2", "requirement": "Req 2"},
        ],
        parameter_section="Transport Security",
        hunter_results=hunter_results,
        critic_results=critic_results,
    )

    assert results["1"].final_verdict == "not_met"
    assert results["1"].final_citations == []
    assert results["2"].final_verdict == "met"
    assert len(results["2"].final_citations) == 1


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
