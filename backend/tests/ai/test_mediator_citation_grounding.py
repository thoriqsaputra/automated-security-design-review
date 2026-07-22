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


def test_run_preserves_ungrounded_met_for_downstream_repair(monkeypatch):
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

    assert result.final_verdict == "met"
    assert result.final_citations == []
    assert result.confidence == 0.9
    assert result.severity is None
    assert result.recommendation is None


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


def test_run_adopts_hunter_citation_when_critic_verified_none(monkeypatch):
    """When the Critic verifies nothing but the Hunter has a genuinely
    grounded citation the Critic simply didn't adopt (not invalidated),
    the Mediator may explicitly select it as final evidence — this is the
    round-5 recall-recovery path."""
    agent = MediatorAgent()
    hunter_citation = Citation(block_id="p1_b1", page_number=1, quoted_text="uses TLS 1.3 for all traffic")
    hunter_result = HunterResult(verdict="met", confidence=0.85, citations=[hunter_citation])
    critic_result = CriticResult(
        outcome="PARTIAL",
        revised_verdict="not_met",
        revised_confidence=0.5,
        reasoning="Doubt about coverage, no contradicting evidence found.",
        logic_summary="Doubt about coverage, no contradicting evidence found.",
        valid_citations=[],
        invalid_citation_ids=[],
    )

    def fake_call_llm(user_prompt, **_kwargs):
        return _ai_response(
            {
                "final_verdict": "met",
                "confidence": 0.8,
                "reasoning": "Hunter's citation is genuinely grounded and the Critic's doubt cites no contradicting evidence.",
                "logic_summary": "Hunter's citation is genuinely grounded and the Critic's doubt cites no contradicting evidence.",
                "final_citations": [
                    {"block_id": "p1_b1", "page_number": 1, "quoted_text": "uses TLS 1.3 for all traffic"}
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
    assert result.final_citations[0].block_id == "p1_b1"


def test_run_never_adopts_critic_invalidated_hunter_citation(monkeypatch):
    """A block_id the Critic actively invalidated must never survive as a
    final citation, even if the Hunter cited it and the Mediator selects
    it — invalidation is a verified rejection, not a mere non-adoption."""
    agent = MediatorAgent()
    hunter_citation = Citation(block_id="p1_b1", page_number=1, quoted_text="uses TLS 1.3 for all traffic")
    hunter_result = HunterResult(verdict="met", confidence=0.85, citations=[hunter_citation])
    critic_result = CriticResult(
        outcome="OVERTURN",
        revised_verdict="not_met",
        revised_confidence=0.7,
        reasoning="The cited block does not actually mention TLS.",
        logic_summary="The cited block does not actually mention TLS.",
        valid_citations=[],
        invalid_citation_ids=["p1_b1"],
    )

    def fake_call_llm(user_prompt, **_kwargs):
        return _ai_response(
            {
                "final_verdict": "met",
                "confidence": 0.8,
                "reasoning": "Adopting Hunter's citation.",
                "logic_summary": "Adopting Hunter's citation.",
                "final_citations": [
                    {"block_id": "p1_b1", "page_number": 1, "quoted_text": "uses TLS 1.3 for all traffic"}
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

    assert result.final_citations == []
    assert result.final_verdict == "met"


def test_fallback_to_critic_preserves_ungrounded_met():
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

    assert result.final_verdict == "met"
    assert result.final_citations == []
    assert result.confidence == 0.8
    assert result.recommendation is None


def test_run_preserves_ungrounded_not_met_for_downstream_repair(monkeypatch):
    agent = MediatorAgent()
    hunter_result = HunterResult(verdict="not_met", confidence=0.7)
    critic_result = CriticResult(
        outcome="UPHOLD",
        revised_verdict="not_met",
        revised_confidence=0.7,
        reasoning="Critic believes the control is not met.",
        logic_summary="Critic believes the control is not met.",
        valid_citations=[],
    )

    def fake_call_llm(user_prompt, **_kwargs):
        return _ai_response(
            {
                "final_verdict": "not_met",
                "confidence": 0.85,
                "reasoning": "Mediator agrees the control is not met.",
                "logic_summary": "Mediator agrees the control is not met.",
                "recommendation": "Add the missing control.",
                "final_citations": [],
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
    assert result.confidence == 0.85
    assert result.recommendation == "Add the missing control."
