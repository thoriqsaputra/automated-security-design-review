"""RouteLLM's response_format=json_object handling: OpenAI enforces this as a
hard constraint, but RouteLLM's pass-through to other model families (e.g.
Claude) doesn't — those models can wrap valid JSON in explanatory prose
despite "Return ONLY a JSON object" instructions, causing a naive json.loads
on the raw content to fail deterministically (every retry fails identically,
since it's the model's consistent behavior, not a random glitch)."""
import json

from sdr.apps.ai.client.routellm.service import _extract_json_object


def test_clean_json_passes_through_unchanged():
    content = '{"faithfulness_score": 1.0, "claims": []}'
    assert _extract_json_object(content) == content


def test_markdown_fenced_json_is_stripped():
    content = '```json\n{"faithfulness_score": 0.8}\n```'
    result = _extract_json_object(content)
    assert json.loads(result) == {"faithfulness_score": 0.8}


def test_json_wrapped_in_prose_is_extracted():
    content = (
        "Here is the JSON response:\n\n"
        '{"context_recall_score": 0.75, "statements": []}\n\n'
        "Let me know if you need anything else."
    )
    result = _extract_json_object(content)
    assert json.loads(result) == {"context_recall_score": 0.75, "statements": []}


def test_json_with_leading_explanation_only():
    content = 'Sure, here you go: {"relevant": true, "reasoning": "test"}'
    result = _extract_json_object(content)
    assert json.loads(result) == {"relevant": True, "reasoning": "test"}


def test_unparseable_content_returns_best_effort_stripped():
    content = "I cannot provide a JSON response for this request."
    result = _extract_json_object(content)
    assert result == content


def test_nested_objects_within_prose_still_extract_correctly():
    content = (
        "Analysis:\n\n"
        '{"claims": [{"claim": "MFA is enforced", "is_faithful": true}], "faithfulness_score": 1.0}'
    )
    result = _extract_json_object(content)
    parsed = json.loads(result)
    assert parsed["faithfulness_score"] == 1.0
    assert parsed["claims"][0]["is_faithful"] is True
