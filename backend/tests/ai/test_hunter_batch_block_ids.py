from types import SimpleNamespace
from unittest.mock import patch

from sdr.apps.ai.agents.hunter import HunterAgent
from sdr.apps.ai.prompts.agents.hunter import build_batch_hunter_prompt


def test_build_batch_hunter_prompt_renders_block_ids_when_provided():
    prompt = build_batch_hunter_prompt(
        child_inputs=[{"id": "1", "requirement": "Use TLS"}],
        parameter_section="Transport Security",
        context_chunks=["<chunk>some text</chunk>"],
        killed_assumptions=None,
        available_block_ids=["p2_b3", "p1_b1"],
    )
    assert "VALID CITATION BLOCK IDS" in prompt
    assert "p1_b1, p2_b3" in prompt


def test_build_batch_hunter_prompt_omits_block_ids_when_absent():
    prompt = build_batch_hunter_prompt(
        child_inputs=[{"id": "1", "requirement": "Use TLS"}],
        parameter_section="Transport Security",
        context_chunks=["<chunk>some text</chunk>"],
        killed_assumptions=None,
    )
    assert "VALID CITATION BLOCK IDS" not in prompt


def test_hunter_run_batch_forwards_available_block_ids_into_prompt():
    agent = HunterAgent()
    captured_prompt = {}

    def fake_call_llm(user_prompt, **kwargs):
        captured_prompt["prompt"] = user_prompt
        return SimpleNamespace(error=None, content='{"results": []}', finish_reason="stop")

    with patch.object(agent, "_call_llm", side_effect=fake_call_llm):
        agent.run_batch(
            child_inputs=[{"id": "1", "requirement": "Use TLS"}],
            parameter_section="Transport Security",
            context_chunks=["<chunk>some text</chunk>"],
            available_block_ids=["p1_b1", "p2_b3"],
        )

    assert "VALID CITATION BLOCK IDS" in captured_prompt["prompt"]
    assert "p1_b1, p2_b3" in captured_prompt["prompt"]


def test_hunter_run_batch_without_block_ids_omits_guardrail():
    agent = HunterAgent()
    captured_prompt = {}

    def fake_call_llm(user_prompt, **kwargs):
        captured_prompt["prompt"] = user_prompt
        return SimpleNamespace(error=None, content='{"results": []}', finish_reason="stop")

    with patch.object(agent, "_call_llm", side_effect=fake_call_llm):
        agent.run_batch(
            child_inputs=[{"id": "1", "requirement": "Use TLS"}],
            parameter_section="Transport Security",
            context_chunks=["<chunk>some text</chunk>"],
        )

    assert "VALID CITATION BLOCK IDS" not in captured_prompt["prompt"]
