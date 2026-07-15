"""Regression tests for the dead-settings bug: AdvancedRetrievalConfig.from_settings
used to read AI_RETRIEVAL_FUSION_METHOD / AI_RETRIEVAL_RRF_K via getattr, but the
fields were never declared on Settings — so RRF fusion was unreachable in
production. These tests pin the declared defaults and the settings passthrough.
"""
from sdr.apps.ai.retrieval.core.types import AdvancedRetrievalConfig
from sdr.core.config import settings


def test_fusion_settings_are_declared_on_settings_model():
    assert settings.AI_RETRIEVAL_FUSION_METHOD == "agreement_boost"
    assert settings.AI_RETRIEVAL_RRF_K == 60
    assert settings.AI_RETRIEVAL_MAX_CONTEXT_CHUNKS == 16
    assert settings.AI_RETRIEVAL_PROTECTED_DENSE_TOP_N == 7
    assert settings.AI_RETRIEVAL_PROTECTED_BM25_TOP_N == 2
    assert settings.AI_RETRIEVAL_PROTECTED_RAPTOR_TOP_N == 7
    assert settings.AI_RETRIEVAL_SUMMARY_LEAVES_PER_GROUNDING == 1
    assert settings.AI_RETRIEVAL_HYBRID_DENSE_TOP_K == 20
    assert settings.AI_RETRIEVAL_HYBRID_BM25_TOP_K == 20
    assert settings.AI_RETRIEVAL_RERANK_SCORE_WEIGHT == 0.72


def test_from_settings_picks_up_declared_fusion_settings(monkeypatch):
    monkeypatch.setattr(settings, "AI_RETRIEVAL_FUSION_METHOD", "rrf")
    monkeypatch.setattr(settings, "AI_RETRIEVAL_RRF_K", 30)
    config = AdvancedRetrievalConfig.from_settings()
    assert config.fusion_method == "rrf"
    assert config.rrf_k == 30


def test_from_settings_picks_up_protected_slot_settings(monkeypatch):
    monkeypatch.setattr(settings, "AI_RETRIEVAL_PROTECTED_DENSE_TOP_N", 5)
    monkeypatch.setattr(settings, "AI_RETRIEVAL_PROTECTED_BM25_TOP_N", 0)
    monkeypatch.setattr(settings, "AI_RETRIEVAL_PROTECTED_RAPTOR_TOP_N", -1)
    monkeypatch.setattr(settings, "AI_RETRIEVAL_HYBRID_DENSE_TOP_K", 31)
    monkeypatch.setattr(settings, "AI_RETRIEVAL_HYBRID_BM25_TOP_K", 29)
    monkeypatch.setattr(settings, "AI_RETRIEVAL_RERANK_SCORE_WEIGHT", 2.0)
    config = AdvancedRetrievalConfig.from_settings()
    assert config.protected_dense_top_n == 5
    assert config.protected_bm25_top_n == 0
    assert config.protected_raptor_top_n == 0  # clamped to >= 0
    assert config.hybrid_dense_top_k == 31
    assert config.hybrid_bm25_top_k == 29
    assert config.rerank_score_weight == 1.0  # clamped to <= 1


def test_dataclass_defaults_match_settings_defaults():
    config = AdvancedRetrievalConfig()
    assert config.fusion_method == "agreement_boost"
    assert config.rrf_k == 60
    assert config.protected_dense_top_n == 7
    assert config.protected_bm25_top_n == 2
    assert config.protected_raptor_top_n == 7
    assert config.summary_leaves_per_grounding == 1
    assert config.hybrid_dense_top_k == 20
    assert config.hybrid_bm25_top_k == 20
    assert config.rerank_score_weight == 0.72
