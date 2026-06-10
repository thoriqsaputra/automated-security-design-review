from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from sdr.apps.ai.agents.base import Citation, CriticResult, HunterResult, MediatorResult
from sdr.apps.ai.retrieval.router import RetrievalResult
from sdr.apps.ai.services.analysis.debate_service import DebateService
from sdr.apps.ai.services.analysis.dto import DebateOutput
from sdr.apps.ai.services.analysis.pipeline import TSDAnalysisPipeline


def _pipeline():
    return TSDAnalysisPipeline(
        ingestion_service=SimpleNamespace(),
        retrieval_service=SimpleNamespace(),
        debate_service=SimpleNamespace(),
        persistence_service=SimpleNamespace(),
    )


def _parent(parent_id=1, *, title="Authentication", description="Parent scope"):
    return SimpleNamespace(id=parent_id, title=title, description=description)


def _parameter(child_id, parent=None, *, text=None, details="", ordinal=0):
    parent = parent or _parent()
    requirement_text = text or f"Requirement {child_id}"
    return SimpleNamespace(
        id=child_id,
        parent=parent,
        requirement_text=requirement_text,
        details=details,
        requirement_text_normalized=requirement_text,
        ordinal=ordinal,
    )


def _debate_output(parameter, *, verdict="not_met", confidence=0.9, citations=None, reasoning=None):
    citations = citations if citations is not None else [Citation(block_id="p1_b1", page_number=1)]
    if verdict != "met" and citations is None:
        citations = []
    reason = reasoning or "Explicit implementation evidence is missing from the retrieved TSD context."
    return DebateOutput.model_construct(
        parameter=parameter,
        hunter_result=HunterResult(
            verdict=verdict,
            confidence=confidence,
            reasoning=reason,
            logic_summary=reason,
            evidence_found=bool(citations),
            citations=list(citations),
        ),
        critic_result=CriticResult(
            revised_verdict=verdict,
            revised_confidence=confidence,
            reasoning=reason,
            logic_summary=reason,
            valid_citations=list(citations),
        ),
        mediator_result=MediatorResult(
            final_verdict=verdict,
            confidence=confidence,
            reasoning=reason,
            logic_summary=reason,
            final_citations=list(citations),
            severity="medium" if verdict == "not_met" else None,
            recommendation="Investigate the missing control" if verdict == "not_met" else None,
        ),
        retrieval_result=RetrievalResult(context_chunks=["<CONTEXT_CHUNK id=\"p1_b1\">evidence</CONTEXT_CHUNK>"]),
        analysis_trace={"retrieved_chunk_ids": ["p1_b1"]},
    )


def test_contract_builder_returns_rule_based_shape():
    contract = _pipeline()._build_contract(
        parameter_text="Use MFA for all admin access.",
        parameter_section="Authentication",
    )

    assert contract["given"].startswith("A TSD excerpt")
    assert contract["when"].startswith("Evaluating requirement:")
    assert contract["then"]
    assert contract["not_sufficient"]
    assert contract["specific_enough"] is True


def test_xml_context_builder_uses_stable_ids():
    xml_chunks = _pipeline()._build_xml_context_chunks(
        ["--- DOCUMENT CHUNK 1 OF 1 ---\np4_b2 Service A calls Service B over mTLS."]
    )

    assert len(xml_chunks) == 1
    assert '<CONTEXT_CHUNK id="p4_b2"' in xml_chunks[0]
    assert "Service A calls Service B over mTLS." in xml_chunks[0]


def test_context_chunk_map_uses_source_block_id_when_chunk_has_no_id():
    chunk_map = _pipeline()._build_context_chunk_map(
        ["Token validation is enforced at the API gateway."],
        source_block_ids=["p8_b4"],
    )

    assert "p8_b4" in chunk_map
    assert chunk_map["p8_b4"]["citation_grade"] is True


def test_citation_validator_rejects_unknown_ids():
    debate = DebateService()
    citations = [
        Citation(block_id="p1_b1", page_number=1),
        Citation(block_id="x999", page_number=1),
    ]

    valid, rejected = debate._validate_citations(citations, ["p1_b1"], "hunter")

    assert [c.block_id for c in valid] == ["p1_b1"]
    assert [c.block_id for c in rejected] == ["x999"]


def test_cold_start_payload_contains_only_cited_blocks():
    debate = DebateService()
    payload = debate._build_cold_start_cited_blocks(
        [Citation(block_id="p1_b1", page_number=1)],
        {
            "p1_b1": {"source": "retrieval_context", "section": "unknown", "text": "mTLS is enabled"},
            "p2_b2": {"source": "retrieval_context", "section": "unknown", "text": "unused"},
        },
    )

    assert payload == [
        {
            "block_id": "p1_b1",
            "source": "retrieval_context",
            "section": "unknown",
            "text": "mTLS is enabled",
        }
    ]


def test_mediator_evidence_policy_adjusts_verdicts():
    debate = DebateService()
    contract = {"in_scope": True, "specific_enough": True}

    grounded_not_met = debate._apply_mediator_evidence_policy(
        MediatorResult(final_verdict="not_met", confidence=0.5, final_citations=[]),
        CriticResult(
            revised_verdict="not_met",
            valid_citations=[Citation(block_id="p1_b1", page_number=1)],
        ),
        contract,
    )
    unsupported_met = debate._apply_mediator_evidence_policy(
        MediatorResult(final_verdict="met", confidence=0.5, final_citations=[]),
        CriticResult(revised_verdict="not_met", valid_citations=[]),
        contract,
    )
    out_of_scope = debate._apply_mediator_evidence_policy(
        MediatorResult(final_verdict="met", confidence=0.5, final_citations=[]),
        CriticResult(
            revised_verdict="met",
            valid_citations=[Citation(block_id="p1_b1", page_number=1)],
        ),
        {"in_scope": False, "specific_enough": True},
    )

    assert grounded_not_met.final_verdict == "not_met"
    assert [c.block_id for c in grounded_not_met.final_citations] == ["p1_b1"]
    assert unsupported_met.final_verdict == "na"
    assert out_of_scope.final_verdict == "na"


def test_group_parameters_by_parent_preserves_order():
    pipeline = _pipeline()
    parent_a = _parent(1, title="Parent A")
    parent_b = _parent(2, title="Parent B")
    grouped = pipeline._group_parameters_by_parent(
        [
            _parameter(1, parent_a),
            _parameter(2, parent_a),
            _parameter(3, parent_b),
            _parameter(4, parent_a),
        ]
    )

    assert [group[0] for group in grouped] == [parent_a, parent_b]
    assert [[p.id for p in group[1]] for group in grouped] == [[1, 2, 4], [3]]


def test_split_batches_uses_requested_size():
    batches = _pipeline()._split_batches([_parameter(i) for i in range(1, 8)], 3)
    assert [[p.id for p in batch] for batch in batches] == [[1, 2, 3], [4, 5, 6], [7]]


def test_validate_batch_outputs_rejects_missing_and_generic_results(settings_override):
    settings_override(
        AI_BATCH_ANALYSIS_CONFIDENCE_THRESHOLD=0.75,
        AI_BATCH_ANALYSIS_SOFT_CONFIDENCE_THRESHOLD=0.65,
        AI_BATCH_ANALYSIS_REQUIRE_CITATIONS_FOR_NOT_MET=True,
        AI_BATCH_UNGROUNDED_NOT_MET_POLICY="selective_fallback",
    )
    pipeline = _pipeline()
    params = [_parameter(i) for i in range(1, 5)]
    valid = _debate_output(params[0], verdict="not_met")
    valid.mediator_result.final_citations = [Citation(block_id="p1_b1", page_number=1)]
    valid.critic_result.valid_citations = [Citation(block_id="p1_b1", page_number=1)]
    low_conf = _debate_output(params[1], confidence=0.4)
    weak = _debate_output(params[2], reasoning="Too short.")
    generic = _debate_output(
        params[3],
        reasoning="All children in the batch are covered by the same evidence and therefore pass.",
    )
    accepted, invalid = pipeline._validate_batch_outputs(
        params,
        {
            "1": valid,
            "2": low_conf,
            "3": weak,
            "4": generic,
            "999": _debate_output(_parameter(999)),
        },
    )

    assert set(accepted) == {"1"}
    assert "low_confidence" in invalid["2"]
    assert "weak_or_generic_reasoning" in invalid["3"]
    assert "generic_multi_child_result" in invalid["4"]
    assert "unknown_child_id" in invalid["999"]


def test_evidence_gate_preserves_not_met_without_retry_context(settings_override):
    settings_override(AI_BATCH_UNGROUNDED_NOT_MET_POLICY="downgrade_na")
    pipeline = _pipeline()
    parameter = _parameter(1, text="Use locking and idempotency keys")
    output = _debate_output(parameter, verdict="not_met", citations=[])
    output.mediator_result.final_citations = []
    output.critic_result.valid_citations = []

    gated = pipeline._apply_not_met_evidence_gate(
        category=SimpleNamespace(),
        ingestion_job=None,
        parameter=parameter,
        indexes=None,
        tsd_document=None,
        debate_output=output,
    )

    assert gated.mediator_result.final_verdict == "not_met"
    assert gated.analysis_trace["evidence_gate_outcome"] == "no_retry_context_preserved_not_met"
    assert gated.analysis_trace["downgraded_due_to_missing_citations"] is False


def test_resolve_parameters_prefers_first_selected_category(monkeypatch):
    pipeline = _pipeline()
    category_a = SimpleNamespace(id=1, code="web_application")
    category_b = SimpleNamespace(id=2, code="mobile")
    review = SimpleNamespace(
        selected_categories=[category_a, category_b],
        ingestion_job=None,
        standards=[],
    )

    class _ScalarResult:
        def __init__(self, value):
            self.value = value

        def scalars(self):
            return self

        def first(self):
            return self.value

        def all(self):
            return self.value

    class _Session:
        def execute(self, _statement):
            return _ScalarResult(None)

    class _SessionContext:
        def __enter__(self):
            return _Session()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("sdr.core.database.SessionLocal", lambda: _SessionContext())

    category, ingestion_job, parameters = pipeline._resolve_parameters(review)

    assert category is category_a
    assert ingestion_job is None
    assert parameters == []
