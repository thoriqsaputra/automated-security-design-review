from __future__ import annotations

from sdr.apps.ai.retrieval.core.types import QueryType
from sdr.apps.ai.retrieval.routing.strategy_selector import RetrievalStrategySelector


def test_classifies_paraphrased_multi_hop_phrasing_via_markers():
    selector = RetrievalStrategySelector()
    # No literal "bypass"/"tenant data" substring, but still a multi-hop
    # security concern phrased with a synonym from the expanded marker list.
    query = "users can access another tenant's data via cross-tenant object reference"
    result = selector.classify_query_type(query, keywords=["tenant", "object", "reference"], inferred_relations=set())
    assert result == QueryType.MULTI_HOP_SECURITY


def test_classifies_structurally_multi_hop_query_without_any_marker():
    selector = RetrievalStrategySelector()
    query = "describe how the payment service and ledger service interact"
    inferred_relations = {"authenticates_with"}
    result = selector.classify_query_type(
        query,
        keywords=["payment", "ledger", "service", "interact"],
        inferred_relations=inferred_relations,
        query_entities=["payment_service", "ledger_service"],
    )
    assert result == QueryType.MULTI_HOP_SECURITY


def test_word_boundary_marker_avoids_substring_false_positive():
    selector = RetrievalStrategySelector()
    # "leak" is a multi_hop_markers entry. With plain substring matching this
    # would also match inside an unrelated word like "leakage-proof", which
    # the word-boundary fix should avoid (no "leak" token on its own here).
    query = "the container uses a leakage-proof seal during transport"
    result = selector.classify_query_type(query, keywords=["container", "transport"], inferred_relations=set())
    assert result != QueryType.MULTI_HOP_SECURITY


def test_classifies_attribute_manipulation_query_via_integrity_markers():
    selector = RetrievalStrategySelector()
    # ASVS 4.1.2-style wording: no multi_hop_markers substring present, but
    # "manipulated" should now route this like other multi-hop security
    # queries so it can pull in the session/state-management mechanism that
    # actually enforces the access-control model it describes.
    query = "verify that policy attributes used in authorization decisions cannot be manipulated by end users"
    result = selector.classify_query_type(
        query,
        keywords=["policy", "attributes", "authorization", "decisions"],
        inferred_relations=set(),
    )
    assert result == QueryType.MULTI_HOP_SECURITY
