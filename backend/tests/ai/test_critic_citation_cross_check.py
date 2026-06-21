from __future__ import annotations

import json

from sdr.apps.ai.agents.base import Citation, HunterResult
from sdr.apps.ai.agents.critic import CriticAgent
from sdr.apps.ai.client.base import AIProvider, AIResponse
from sdr.apps.ai.prompts.agents.critic import build_batch_critic_prompt, build_critic_prompt


def _ai_response(payload: dict) -> AIResponse:
    return AIResponse(
        content=json.dumps(payload),
        model="test-model",
        provider=AIProvider.NVIDIA,
        finish_reason="stop",
    )


def _hunter_result() -> HunterResult:
    return HunterResult(
        verdict="met",
        confidence=0.8,
        citations=[Citation(block_id="p1_b1", page_number=1, quoted_text="uses TLS")],
    )


def test_build_critic_prompt_includes_block_ids_guardrail_when_supplied():
    prompt = build_critic_prompt(
        parameter_text="Require TLS for all traffic",
        parameter_section="Transport Security",
        contract={},
        context_chunks=["some context"],
        hunter_verdict="met",
        hunter_citation_ids=["p1_b1"],
        cited_blocks=[],
        hunter_confidence=0.8,
        available_block_ids=["p1_b1", "p1_b2"],
    )

    assert "VALID CITATION BLOCK IDS" in prompt
    assert "p1_b1, p1_b2" in prompt


def test_build_critic_prompt_omits_block_ids_guardrail_when_not_supplied():
    prompt = build_critic_prompt(
        parameter_text="Require TLS for all traffic",
        parameter_section="Transport Security",
        contract={},
        context_chunks=["some context"],
        hunter_verdict="met",
        hunter_citation_ids=["p1_b1"],
        cited_blocks=[],
        hunter_confidence=0.8,
    )

    assert "VALID CITATION BLOCK IDS" not in prompt


def test_build_batch_critic_prompt_includes_block_ids_guardrail_when_supplied():
    prompt = build_batch_critic_prompt(
        child_inputs=[{"id": "1", "requirement": "Req 1"}],
        parameter_section="Transport Security",
        context_chunks=["some context"],
        hunter_payload={"1": {"verdict": "met"}},
        available_block_ids=["p1_b1"],
    )

    assert "VALID CITATION BLOCK IDS" in prompt
    assert "p1_b1" in prompt


def test_cross_check_exact_match_keeps_citation():
    agent = CriticAgent()
    citation = Citation(block_id="p1_b1", page_number=1, quoted_text="uses TLS")

    confirmed, invalidated = agent._cross_check_citations(
        valid_citations=[citation],
        available_block_ids=["p1_b1", "p1_b2"],
    )

    assert confirmed == [citation]
    assert invalidated == []


def test_cross_check_rejects_coincidental_substring():
    agent = CriticAgent()
    citation = Citation(block_id="b1", page_number=1, quoted_text="hallucinated")

    # "b1" is a substring of "cab1net" in raw context text, but it is not a
    # real block id — exact set-membership must reject it.
    confirmed, invalidated = agent._cross_check_citations(
        valid_citations=[citation],
        available_block_ids=["p1_b2", "p1_b3"],
    )

    assert confirmed == []
    assert invalidated == ["b1"]


def test_cross_check_confirms_id_despite_raw_text_formatting_differences():
    agent = CriticAgent()
    citation = Citation(block_id="p1_b1", page_number=1, quoted_text="uses TLS")

    # Previously, substring search against joined raw chunk text could fail
    # to find an id rendered differently in the text. Set-membership against
    # the canonical id list is immune to that.
    confirmed, invalidated = agent._cross_check_citations(
        valid_citations=[citation],
        available_block_ids=["p1_b1"],
    )

    assert confirmed == [citation]
    assert invalidated == []


def test_cross_check_no_available_ids_passes_through_unchanged():
    agent = CriticAgent()
    citation = Citation(block_id="p1_b1", page_number=1, quoted_text="uses TLS")

    confirmed, invalidated = agent._cross_check_citations(
        valid_citations=[citation],
        available_block_ids=None,
    )

    assert confirmed == [citation]
    assert invalidated == []


def test_run_invalidates_hallucinated_citation_via_available_block_ids(monkeypatch):
    agent = CriticAgent()
    hunter_result = _hunter_result()

    def fake_call_llm(user_prompt, **_kwargs):
        return _ai_response(
            {
                "decision": "uphold",
                "outcome": "UPHOLD",
                "revised_verdict": "met",
                "revised_confidence": 0.8,
                "reasoning": "Looks fine.",
                "logic_summary": "Looks fine.",
                "valid_citations": [
                    {"block_id": "hallucinated_id", "page_number": 1, "quoted_text": "fabricated"}
                ],
                "invalid_citation_ids": [],
            }
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = agent.run(
        parameter_text="Require TLS for all traffic",
        parameter_section="Transport Security",
        contract={},
        context_chunks=["some unrelated text"],
        hunter_result=hunter_result,
        cited_blocks=[],
        available_block_ids=["p1_b1"],
    )

    assert result.valid_citations == []
    assert "hallucinated_id" in result.invalid_citation_ids


def test_run_keeps_real_citation_via_available_block_ids(monkeypatch):
    agent = CriticAgent()
    hunter_result = _hunter_result()

    def fake_call_llm(user_prompt, **_kwargs):
        return _ai_response(
            {
                "decision": "uphold",
                "outcome": "UPHOLD",
                "revised_verdict": "met",
                "revised_confidence": 0.8,
                "reasoning": "Verified.",
                "logic_summary": "Verified.",
                "valid_citations": [
                    {"block_id": "p1_b1", "page_number": 1, "quoted_text": "uses TLS"}
                ],
                "invalid_citation_ids": [],
            }
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = agent.run(
        parameter_text="Require TLS for all traffic",
        parameter_section="Transport Security",
        contract={},
        context_chunks=["unrelated raw text that does not literally contain the id"],
        hunter_result=hunter_result,
        cited_blocks=[],
        available_block_ids=["p1_b1"],
    )

    assert len(result.valid_citations) == 1
    assert result.valid_citations[0].block_id == "p1_b1"
    assert "p1_b1" not in result.invalid_citation_ids


def test_run_batch_applies_available_block_ids_per_result(monkeypatch):
    agent = CriticAgent()
    hunter_results = {
        "1": _hunter_result(),
        "2": _hunter_result(),
    }

    def fake_call_llm(user_prompt, **_kwargs):
        return _ai_response(
            {
                "results": [
                    {
                        "child_id": "1",
                        "decision": "uphold",
                        "outcome": "UPHOLD",
                        "revised_verdict": "met",
                        "revised_confidence": 0.8,
                        "reasoning": "Verified.",
                        "logic_summary": "Verified.",
                        "valid_citations": [
                            {"block_id": "p1_b1", "page_number": 1, "quoted_text": "uses TLS"}
                        ],
                        "invalid_citation_ids": [],
                    },
                    {
                        "child_id": "2",
                        "decision": "uphold",
                        "outcome": "UPHOLD",
                        "revised_verdict": "met",
                        "revised_confidence": 0.8,
                        "reasoning": "Hallucinated.",
                        "logic_summary": "Hallucinated.",
                        "valid_citations": [
                            {"block_id": "fake_id", "page_number": 1, "quoted_text": "fabricated"}
                        ],
                        "invalid_citation_ids": [],
                    },
                ]
            }
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    results = agent.run_batch(
        child_inputs=[
            {"id": "1", "requirement": "Req 1", "contract": {}},
            {"id": "2", "requirement": "Req 2", "contract": {}},
        ],
        parameter_section="Transport Security",
        context_chunks=["unrelated text"],
        hunter_results=hunter_results,
        available_block_ids=["p1_b1"],
    )

    assert len(results["1"].valid_citations) == 1
    assert results["1"].valid_citations[0].block_id == "p1_b1"
    assert results["2"].valid_citations == []
    assert "fake_id" in results["2"].invalid_citation_ids
