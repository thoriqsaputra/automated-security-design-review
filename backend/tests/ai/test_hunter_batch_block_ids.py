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


