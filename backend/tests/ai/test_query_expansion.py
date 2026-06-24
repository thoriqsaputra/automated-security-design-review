from types import SimpleNamespace

import sdr.apps.ai.engine.classification.query_expansion as query_expansion_module
from sdr.apps.ai.engine.classification.query_expansion import expand_retrieval_query_variants


def test_expand_retrieval_query_variants_parses_and_caches(monkeypatch):
    calls = {"count": 0}

    def fake_chat_completion(**_kwargs):
        calls["count"] += 1
        return SimpleNamespace(
            error=None,
            content='{"variants": ["server-side authorization check", "backend access control validation", "session-based request validation"]}',
        )

    monkeypatch.setattr(query_expansion_module, "chat_completion", fake_chat_completion)
    query_expansion_module._variant_cache.clear()

    variants = expand_retrieval_query_variants(
        "Verify that the application enforces access control rules on a trusted service layer.",
        cache_key="param:1:job:1",
        variant_count=3,
    )

    assert calls["count"] == 1
    assert len(variants) == 3
    assert "server-side authorization check" in variants

    # Second call with the same cache key + text must hit the cache, not the LLM.
    variants_again = expand_retrieval_query_variants(
        "Verify that the application enforces access control rules on a trusted service layer.",
        cache_key="param:1:job:1",
        variant_count=3,
    )
    assert calls["count"] == 1
    assert variants_again == variants


def test_expand_retrieval_query_variants_disabled_returns_empty(monkeypatch):
    def fake_chat_completion(**_kwargs):
        raise AssertionError("chat_completion should not be called when disabled")

    monkeypatch.setattr(query_expansion_module, "chat_completion", fake_chat_completion)

    variants = expand_retrieval_query_variants(
        "Verify that the application enforces access control rules.",
        cache_key="param:2:job:1",
        variant_count=3,
        enabled=False,
    )
    assert variants == []


def test_expand_retrieval_query_variants_handles_error_response(monkeypatch):
    def fake_chat_completion(**_kwargs):
        return SimpleNamespace(error="upstream_failure", content=None)

    monkeypatch.setattr(query_expansion_module, "chat_completion", fake_chat_completion)
    query_expansion_module._variant_cache.clear()

    variants = expand_retrieval_query_variants(
        "Verify that IDOR protections exist.",
        cache_key="param:3:job:1",
        variant_count=3,
    )
    assert variants == []
