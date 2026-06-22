from __future__ import annotations

from datetime import datetime, timezone
from collections import deque
from types import SimpleNamespace
import networkx as nx
import pytest

from sdr.apps.ai.agents.base import Citation, CriticResult, HunterResult, MediatorResult
from sdr.apps.ai.agents.hunter import HunterAgent
from sdr.apps.ai.retrieval.core import RetrievalResult
from sdr.apps.ai.engine.debate.debate_service import DebateService
from sdr.apps.ai.engine.debate.category_analysis_coordinator import CategoryAnalysisCoordinator
from sdr.apps.ai.engine.dto import AnalysisSummary, DebateOutput
from sdr.apps.ai.engine.classification.parent_applicability import ParentApplicabilityResult
from sdr.apps.ai.engine.persistence.persistence_service import PersistenceService
from sdr.apps.ai.engine.persistence.review_run_state_service import AnalysisCancelledError
from sdr.apps.ai.engine.pipeline import TSDAnalysisPipeline
from sdr.apps.standards.models.parameters import CategoryParameterChild, CategoryParameterParent
from unittest.mock import Mock


def _pipeline():
    return TSDAnalysisPipeline(
        ingestion_service=SimpleNamespace(),
        retrieval_service=SimpleNamespace(
            get_retrieve_many_max_concurrency=lambda *args, **kwargs: 1
        ),
        debate_service=SimpleNamespace(),
        persistence_service=SimpleNamespace(),
    )


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self

    def first(self):
        return self.value


class _Session:
    def __init__(self, execute_value=None):
        self.execute_value = execute_value

    def execute(self, _statement):
        return _ScalarResult(self.execute_value)


class _SessionContext:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class _SessionFactory:
    def __init__(self, sessions):
        self._sessions = list(sessions)

    def __call__(self):
        return _SessionContext(self._sessions.pop(0))


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


def test_hunter_citation_parser_accepts_aliases():
    agent = HunterAgent()
    citations = agent._extract_citations(
        [
            {"id": "p1_b1", "page_number": 2, "quoted_text": "alpha"},
            "p1_b2",
            {"chunk_id": "p1_b3", "page": 4, "quote": "beta"},
        ],
        field_name="citations",
    )

    assert [citation.block_id for citation in citations] == ["p1_b1", "p1_b2", "p1_b3"]
    assert [citation.page_number for citation in citations] == [2, 0, 4]
    assert citations[0].quoted_text == "alpha"
    assert citations[2].quoted_text == "beta"


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


def test_stabilize_hunter_grounding_downgrades_met_without_valid_citations():
    debate = DebateService()
    hunter_result = HunterResult(
        verdict="met",
        confidence=0.91,
        reasoning="All good.",
        logic_summary="All good.",
        evidence_found=True,
        citations=[],
        evidence_assessment="All good.",
    )

    stabilized = debate._stabilize_hunter_grounding(hunter_result, rejected_citations=[Citation(block_id="x999", page_number=1)])

    assert stabilized.verdict == "not_met"
    assert stabilized.evidence_found is False
    assert stabilized.confidence <= 0.45
    assert "Rejected citation ids: x999" in stabilized.logic_summary


def test_resolve_citations_for_anchoring_falls_back_to_validated_block_ids():
    service = PersistenceService()
    citations = [Citation(block_id="p1_b1", page_number=3, quoted_text="wrong quote")]
    analysis_trace = {
        "context_chunk_map": {
            "p1_b1": {
                "citation_grade": True,
                "text": "The system requires MFA for privileged access.",
                "section": "Authentication",
                "page_number": 3,
                "bbox": {"x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 40.0},
            }
        }
    }

    resolved, mode = service._resolve_citations_for_anchoring(citations, analysis_trace)

    assert mode == "validated_block_fallback"
    assert [citation.block_id for citation in resolved] == ["p1_b1"]
    assert resolved[0].quoted_text == "wrong quote"
    assert resolved[0].page_number == 3


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


def test_is_cancelled_detects_cancelled_status(monkeypatch):
    review = SimpleNamespace(id=77)
    latest = SimpleNamespace(status="cancelled", error_message="Analysis was cancelled by user.")
    monkeypatch.setattr(
        "sdr.core.database.SessionLocal",
        _SessionFactory([_Session(execute_value=latest)]),
    )

    assert _pipeline()._is_cancelled(review) is True


def test_is_cancelled_supports_legacy_failed_marker(monkeypatch):
    review = SimpleNamespace(id=78)
    latest = SimpleNamespace(status="failed", error_message="Analysis was cancelled by user.")
    monkeypatch.setattr(
        "sdr.core.database.SessionLocal",
        _SessionFactory([_Session(execute_value=latest)]),
    )

    assert _pipeline()._is_cancelled(review) is True


def test_parent_applicability_gate_marks_children_na(monkeypatch, settings_override):
    settings_override(
        AI_PARENT_APPLICABILITY_ENABLED=True,
        AI_PARENT_APPLICABILITY_CONFIDENCE_THRESHOLD=0.7,
    )
    pipeline = _pipeline()
    review = SimpleNamespace(id=81)
    category = SimpleNamespace(id=7, code="web_application")
    ingestion_job = SimpleNamespace(id=11, version_no=5)
    parent = _parent(1, title="Session Controls", description="Browser session requirements")
    parameters = [
        _parameter(1, parent=parent, text="Use secure cookie flags."),
        _parameter(2, parent=parent, text="Expire browser sessions after inactivity."),
    ]
    summary = AnalysisSummary()
    persisted = []

    monkeypatch.setattr(TSDAnalysisPipeline, "_is_cancelled", lambda self, review: False)
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_persist_summary_snapshot",
        lambda self, review_obj, summary_obj: setattr(review_obj, "summary_json", summary_obj.to_dict()),
    )
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_persist_debate_output",
        lambda self, **kwargs: persisted.append(
            (
                kwargs["parameter"].id,
                kwargs["debate_output"].mediator_result.final_verdict,
                kwargs["debate_output"].analysis_trace["parent_applicability"]["reasoning"],
            )
        ),
    )
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_get_parent_retrieval_result",
        lambda self, **kwargs: RetrievalResult(
            context_chunks=["The design is API-only and has no browser session state."],
            source_block_ids=["p1_b1"],
        ),
    )
    monkeypatch.setattr(
        "sdr.apps.ai.engine.pipeline.classify_parent_applicability",
        lambda **kwargs: ParentApplicabilityResult(
            applicable=False,
            confidence=0.92,
            reasoning="The design is API-only and has no browser sessions.",
            evidence=["API-only", "no browser session state"],
        ),
    )

    applicable_parameters, parent_cache = pipeline._apply_parent_applicability_gate(
        review=review,
        category=category,
        ingestion_job=ingestion_job,
        parameters=parameters,
        indexes=SimpleNamespace(),
        tsd_document=SimpleNamespace(),
        summary=summary,
    )

    assert applicable_parameters == []
    assert parent_cache == {}
    assert [item[0] for item in persisted] == [1, 2]
    assert all(item[1] == "na" for item in persisted)
    assert summary.applicability["parents_total"] == 1
    assert summary.applicability["parents_not_applicable"] == 1
    assert summary.applicability["parents_applicable"] == 0
    assert summary.applicability["children_marked_na_by_parent"] == 2


def test_parent_applicability_gate_skips_on_low_confidence_by_default(monkeypatch, settings_override):
    settings_override(
        AI_PARENT_APPLICABILITY_ENABLED=True,
        AI_PARENT_APPLICABILITY_CONFIDENCE_THRESHOLD=0.7,
    )
    pipeline = _pipeline()
    review = SimpleNamespace(id=82)
    category = SimpleNamespace(id=7, code="web_application")
    ingestion_job = SimpleNamespace(id=11, version_no=5)
    parent = _parent(1, title="Session Controls", description="Browser session requirements")
    parameters = [_parameter(1, parent=parent, text="Use secure cookie flags.")]
    summary = AnalysisSummary()

    monkeypatch.setattr(TSDAnalysisPipeline, "_is_cancelled", lambda self, review: False)
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_persist_summary_snapshot",
        lambda self, review_obj, summary_obj: setattr(review_obj, "summary_json", summary_obj.to_dict()),
    )
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_get_parent_retrieval_result",
        lambda self, **kwargs: RetrievalResult(
            context_chunks=["The design documentation is ambiguous about browser usage."],
            source_block_ids=["p1_b1"],
        ),
    )
    monkeypatch.setattr(
        "sdr.apps.ai.engine.pipeline.classify_parent_applicability",
        lambda **kwargs: ParentApplicabilityResult(
            applicable=False,
            confidence=0.45,
            reasoning="Browser use is unclear.",
            evidence=["ambiguous scope"],
            decision_mode="unclear",
        ),
    )
    persisted = []
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_persist_debate_output",
        lambda self, **kwargs: persisted.append(kwargs["parameter"].id),
    )

    applicable_parameters, _ = pipeline._apply_parent_applicability_gate(
        review=review,
        category=category,
        ingestion_job=ingestion_job,
        parameters=parameters,
        indexes=SimpleNamespace(),
        tsd_document=SimpleNamespace(),
        summary=summary,
    )

    assert applicable_parameters == []
    assert persisted == [1]
    assert summary.applicability["parents_total"] == 1
    assert summary.applicability["parents_not_applicable"] == 1
    assert summary.applicability["parents_applicable"] == 0
    assert summary.applicability["parents"][0]["fallback_mode"] == "skip"


def test_parent_applicability_gate_can_assume_applicable_on_low_confidence(monkeypatch, settings_override):
    settings_override(
        AI_PARENT_APPLICABILITY_ENABLED=True,
        AI_PARENT_APPLICABILITY_CONFIDENCE_THRESHOLD=0.7,
        AI_PARENT_APPLICABILITY_FALLBACK_MODE="assume_applicable",
    )
    pipeline = _pipeline()
    review = SimpleNamespace(id=82)
    category = SimpleNamespace(id=7, code="web_application")
    ingestion_job = SimpleNamespace(id=11, version_no=5)
    parent = _parent(1, title="Session Controls", description="Browser session requirements")
    parameters = [_parameter(1, parent=parent, text="Use secure cookie flags.")]
    summary = AnalysisSummary()

    monkeypatch.setattr(TSDAnalysisPipeline, "_is_cancelled", lambda self, review: False)
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_persist_summary_snapshot",
        lambda self, review_obj, summary_obj: setattr(review_obj, "summary_json", summary_obj.to_dict()),
    )
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_get_parent_retrieval_result",
        lambda self, **kwargs: RetrievalResult(
            context_chunks=["The design documentation is ambiguous about browser usage."],
            source_block_ids=["p1_b1"],
        ),
    )
    monkeypatch.setattr(
        "sdr.apps.ai.engine.pipeline.classify_parent_applicability",
        lambda **kwargs: ParentApplicabilityResult(
            applicable=False,
            confidence=0.45,
            reasoning="Browser use is unclear.",
            evidence=["ambiguous scope"],
            decision_mode="unclear",
        ),
    )

    applicable_parameters, _ = pipeline._apply_parent_applicability_gate(
        review=review,
        category=category,
        ingestion_job=ingestion_job,
        parameters=parameters,
        indexes=SimpleNamespace(),
        tsd_document=SimpleNamespace(),
        summary=summary,
    )

    assert [parameter.id for parameter in applicable_parameters] == [1]
    assert summary.applicability["parents_total"] == 1
    assert summary.applicability["parents_not_applicable"] == 0
    assert summary.applicability["parents_applicable"] == 1
    assert summary.applicability["parents"][0]["fallback_mode"] == "assume_applicable"


def test_build_retrieval_snapshot_serializes_raptor_and_graph():
    pipeline = _pipeline()
    leaf = SimpleNamespace(
        node_id="leaf-1",
        level=0,
        text="Leaf text preview",
        page_numbers=[1],
        source_block_ids=["p1_b1"],
        children=[],
        section_heading="Authentication",
    )
    root = SimpleNamespace(
        node_id="root-1",
        level=1,
        text="Root summary text",
        page_numbers=[1, 2],
        source_block_ids=["p1_b1", "p2_b1"],
        children=[leaf],
        section_heading="Overview",
    )
    raptor_tree = SimpleNamespace(
        total_nodes=2,
        max_level=1,
        root_node=root,
        is_empty=lambda: False,
        get_all_nodes=lambda: [root, leaf],
    )

    entity = SimpleNamespace(
        entity_id="api_gateway",
        name="API Gateway",
        entity_type="service",
        source_pages=[1],
        source_block_ids=["p1_b1"],
    )
    relation_obj = SimpleNamespace(
        confidence=0.88,
        protocol="HTTPS",
        is_encrypted=True,
        requires_auth=True,
        source_pages=[1],
    )
    graph_obj = nx.DiGraph()
    graph_obj.add_node("api_gateway")
    graph_obj.add_node("auth_service")
    graph_obj.add_edge("api_gateway", "auth_service", relation="calls", relation_obj=relation_obj)
    tsd_graph = SimpleNamespace(
        total_entities=1,
        total_relations=1,
        entities={"api_gateway": entity},
        graph=graph_obj,
        is_empty=lambda: False,
    )

    snapshot = pipeline._build_retrieval_snapshot(
        SimpleNamespace(raptor_tree=raptor_tree, tsd_graph=tsd_graph)
    )

    assert snapshot is not None
    assert snapshot["status"] == "ready"
    assert snapshot["raptor"]["status"] == "ready"
    assert snapshot["raptor"]["root_node_id"] == "root-1"
    assert snapshot["raptor"]["nodes"][1]["parent_id"] == "root-1"
    assert snapshot["graph"]["status"] == "ready"
    assert snapshot["graph"]["total_entities"] == 1
    assert snapshot["graph"]["edges"][0]["relation_type"] == "calls"


def test_batched_analysis_uses_wrapped_concurrency_probes(monkeypatch, settings_override):
    settings_override(
        AI_BATCH_DEBATE_ENABLED=True,
        AI_BATCH_DEBATE_BATCH_SIZE=10,
        AI_BATCH_DEBATE_MAX_CONCURRENCY=2,
    )

    parent = _parent(1, title="Authentication")
    parameters = [
        _parameter(1, parent=parent, text="Require MFA for admin access."),
        _parameter(2, parent=parent, text="Require session timeout."),
    ]
    review = SimpleNamespace(id=99)
    category = SimpleNamespace(id=7, code="web_application")
    ingestion_job = SimpleNamespace(id=11)
    tsd_document = SimpleNamespace(
        get_diagram_by_id=lambda *_args, **_kwargs: None,
    )
    indexes = SimpleNamespace()
    summary = AnalysisSummary()

    parent_result = RetrievalResult(
        context_chunks=["--- DOCUMENT CHUNK 1 OF 1 ---\np1_b1 Authentication evidence."],
        source_block_ids=["p1_b1"],
        evidence_metadata={},
    )

    class _BatchRetrievalService:
        def retrieve_for_parent_group(self, **kwargs):
            return parent_result

    class _BatchDebateService:
        def run_batch_debate(self, **kwargs):
            return {}

    pipeline = TSDAnalysisPipeline(
        ingestion_service=SimpleNamespace(),
        retrieval_service=_BatchRetrievalService(),
        debate_service=_BatchDebateService(),
        persistence_service=SimpleNamespace(),
    )

    persisted = []

    monkeypatch.setattr(TSDAnalysisPipeline, "_is_cancelled", lambda self, review: False)
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_persist_summary_snapshot",
        lambda self, review_obj, summary_obj: setattr(review_obj, "summary_json", summary_obj.to_dict()),
    )
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_persist_debate_output",
        lambda self, **kwargs: persisted.append(kwargs["parameter"].id),
    )
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_analyze_single_child_with_retrieval_result",
        lambda self, **kwargs: _debate_output(
            kwargs["parameter"],
            verdict="met",
            citations=[Citation(block_id="p1_b1", page_number=1)],
            reasoning="Explicit implementation evidence is present.",
        ),
    )

    pipeline._run_batched_analysis_for_category(
        review=review,
        category=category,
        ingestion_job=ingestion_job,
        parameters=parameters,
        indexes=indexes,
        tsd_document=tsd_document,
        summary=summary,
        killed_assumptions_memory=deque(maxlen=16),
    )

    assert persisted == [1, 2]
    assert summary.debate_total_parameters == 2
    assert summary.debate_completed_parameters == 2
    assert summary.debate_remaining_parameters == 0
    assert summary.persistence_total_parameters == 2
    assert summary.persistence_completed_parameters == 2
    assert summary.persistence_remaining_parameters == 0
    assert summary.analysis_total_parameters == 2
    assert summary.analysis_processed_parameters == 2
    assert summary.analysis_remaining_parameters == 0
    assert summary.asvs["categories"]["web_application"]["debate_total_count"] == 2
    assert summary.asvs["categories"]["web_application"]["debate_completed_count"] == 2
    assert summary.asvs["categories"]["web_application"]["debate_remaining_count"] == 0
    assert summary.asvs["categories"]["web_application"]["persistence_total_count"] == 2
    assert summary.asvs["categories"]["web_application"]["persistence_completed_count"] == 2
    assert summary.asvs["categories"]["web_application"]["persistence_remaining_count"] == 0
    assert summary.asvs["categories"]["web_application"]["analysis_total_count"] == 2
    assert summary.asvs["categories"]["web_application"]["analysis_processed_count"] == 2
    assert summary.asvs["categories"]["web_application"]["analysis_remaining_count"] == 0
    assert pipeline._last_batch_concurrency_stats["parent_retrieval"]["submitted"] == 1
    assert pipeline._last_batch_concurrency_stats["batch_debate"]["submitted"] == 1
    assert pipeline._last_batch_concurrency_stats["fallback"]["submitted"] == 2


def test_validate_batch_outputs_rejects_missing_and_generic_results(settings_override):
    settings_override(
        AI_BATCH_DEBATE_CONFIDENCE_THRESHOLD=0.75,
        AI_BATCH_DEBATE_SOFT_CONFIDENCE_THRESHOLD=0.65,
        AI_BATCH_DEBATE_REQUIRE_CITATIONS_FOR_NOT_MET=True,
        AI_BATCH_DEBATE_UNGROUNDED_NOT_MET_POLICY="selective_fallback",
    )
    pipeline = _pipeline()
    params = [_parameter(i) for i in range(1, 5)]
    valid = _debate_output(params[0], verdict="not_met")
    valid.mediator_result.final_citations = [Citation(block_id="p1_b1", page_number=1)]
    valid.critic_result.valid_citations = [Citation(block_id="p1_b1", page_number=1)]
    low_conf = _debate_output(params[1], confidence=0.4)
    weak = _debate_output(params[2], confidence=0.8, reasoning="Too short.")
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
    settings_override(AI_BATCH_DEBATE_UNGROUNDED_NOT_MET_POLICY="downgrade_na")
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


def test_debate_service_stops_before_critic_when_cancelled():
    parameter = _parameter(1, text="Use MFA for all admin access.")
    debate_input = SimpleNamespace(
        parameter=parameter,
        parameter_text=parameter.requirement_text,
        parameter_section="Authentication",
        contract={"domain": "iam_access_control"},
        killed_assumptions=[],
        hunter_plan={},
        retrieval_query_details={},
        context_chunks=["--- DOCUMENT CHUNK 1 OF 1 ---\np1_b1 Evidence."],
        context_chunk_map={
            "p1_b1": {
                "text": "Evidence.",
                "section": "Authentication",
                "source": "retrieval_context",
                "citation_grade": True,
            }
        },
    )
    tsd_document = SimpleNamespace(get_diagram_by_id=lambda *_args, **_kwargs: None)
    retrieval_result = RetrievalResult(
        context_chunks=list(debate_input.context_chunks),
        source_block_ids=["p1_b1"],
        evidence_metadata={},
    )

    class _Hunter:
        def run(self, **_kwargs):
            return HunterResult(
                verdict="not_met",
                confidence=0.42,
                reasoning="Hunter found evidence.",
                logic_summary="Hunter found evidence.",
                citations=[Citation(block_id="p1_b1", page_number=1)],
                evidence_found=True,
            )

    class _Critic:
        def run(self, **_kwargs):
            raise AssertionError("Critic should not run after cancellation")

    class _Mediator:
        def run(self, **_kwargs):
            raise AssertionError("Mediator should not run after cancellation")

    service = DebateService(hunter=_Hunter(), critic=_Critic(), mediator=_Mediator())
    cancel_calls = {"count": 0}

    def cancel_check():
        cancel_calls["count"] += 1
        return cancel_calls["count"] >= 5

    with pytest.raises(AnalysisCancelledError):
        service.run_debate(
            debate_input=debate_input,
            retrieval_result=retrieval_result,
            tsd_document=tsd_document,
            cancel_check=cancel_check,
        )

    assert cancel_calls["count"] >= 5


def test_debate_output_can_embed_retrieval_result_without_forward_ref_error():
    parameter = _parameter(1, text="Use MFA for all admin access.")
    output = DebateOutput.model_construct(
        parameter=parameter,
        hunter_result=HunterResult(verdict="not_met", confidence=0.1),
        critic_result=CriticResult(),
        mediator_result=MediatorResult(),
        retrieval_result=RetrievalResult(context_chunks=["evidence"]),
    )

    assert output.retrieval_result.context_chunks == ["evidence"]


def _category_analysis_coordinator():
    return CategoryAnalysisCoordinator(
        config=SimpleNamespace(batch_debate_enabled=True),
        workflow_repository=SimpleNamespace(),
        progress_service=SimpleNamespace(),
        run_state_service=SimpleNamespace(),
        text_debate_coordinator=SimpleNamespace(),
        diagram_analysis_coordinator=SimpleNamespace(),
    )


def test_category_analysis_coordinator_skips_diagrams_in_text_only_mode(monkeypatch):
    coordinator = _category_analysis_coordinator()
    parameter = _parameter(1)
    calls = {"text": 0, "diagram": 0}
    stages = []
    summary = AnalysisSummary()
    summary.asvs["categories"] = {}

    coordinator.workflow_repository.get_latest_active_ingestion_job = lambda _category_id: SimpleNamespace(id=11)
    coordinator.workflow_repository.list_category_parameters = lambda **_kwargs: [parameter]
    coordinator.progress_service.prepare_category_stats = lambda **_kwargs: None
    coordinator.progress_service.initialize_category_progress = lambda **_kwargs: None
    coordinator.progress_service.sync_analysis_aliases = lambda **_kwargs: None
    coordinator.run_state.update_stage = lambda *_args, **kwargs: stages.append(kwargs.get("stage") or _args[2])
    coordinator.run_state.persist_summary_snapshot = lambda *_args, **_kwargs: None
    coordinator.run_state.is_cancelled = lambda _review: False
    coordinator.text_debate.apply_parent_applicability_gate = lambda **_kwargs: ([parameter], {})
    coordinator.text_debate.run_batched_analysis_for_category = lambda **_kwargs: calls.__setitem__("text", calls["text"] + 1)
    coordinator.text_debate.run_single_analysis_for_category = lambda **_kwargs: None
    coordinator.diagram_analysis.run = lambda **_kwargs: calls.__setitem__("diagram", calls["diagram"] + 1)

    coordinator.run_category(
        review=SimpleNamespace(analysis_mode="text_only", ingestion_job=None),
        category=SimpleNamespace(id=5, code="web_application"),
        indexes=None,
        tsd_document=None,
        summary=summary,
        killed_assumptions_memory=deque(),
    )

    assert calls["text"] == 1
    assert calls["diagram"] == 0
    assert "7_diagram_debate" not in stages


def test_category_analysis_coordinator_skips_text_path_in_diagram_only_mode(monkeypatch):
    coordinator = _category_analysis_coordinator()
    parameter = _parameter(1)
    calls = {"text_gate": 0, "text": 0, "diagram": 0}
    stages = []
    summary = AnalysisSummary()
    summary.asvs["categories"] = {}

    coordinator.workflow_repository.get_latest_active_ingestion_job = lambda _category_id: SimpleNamespace(id=11)
    coordinator.workflow_repository.list_category_parameters = lambda **_kwargs: [parameter]
    coordinator.progress_service.prepare_category_stats = lambda **_kwargs: None
    coordinator.progress_service.initialize_category_progress = lambda **_kwargs: None
    coordinator.progress_service.sync_analysis_aliases = lambda **_kwargs: None
    coordinator.run_state.update_stage = lambda *_args, **kwargs: stages.append(kwargs.get("stage") or _args[2])
    coordinator.run_state.persist_summary_snapshot = lambda *_args, **_kwargs: None
    coordinator.run_state.is_cancelled = lambda _review: False
    coordinator.text_debate.apply_parent_applicability_gate = lambda **_kwargs: calls.__setitem__("text_gate", calls["text_gate"] + 1)
    coordinator.text_debate.run_batched_analysis_for_category = lambda **_kwargs: calls.__setitem__("text", calls["text"] + 1)
    coordinator.text_debate.run_single_analysis_for_category = lambda **_kwargs: None
    coordinator.diagram_analysis.run = lambda **_kwargs: calls.__setitem__("diagram", calls["diagram"] + 1)

    coordinator.run_category(
        review=SimpleNamespace(analysis_mode="diagram_only", ingestion_job=None),
        category=SimpleNamespace(id=5, code="web_application"),
        indexes=None,
        tsd_document=None,
        summary=summary,
        killed_assumptions_memory=deque(),
    )

    assert calls["text_gate"] == 0
    assert calls["text"] == 0
    assert calls["diagram"] == 1
    assert stages[-1] == "7_diagram_debate"
    assert summary.total_parameters == 0


def test_parent_applicability_failure_modes_fail_open():
    from sdr.apps.ai.engine.classification.parent_applicability import classify_parent_applicability

    # Missing child requirements
    result = classify_parent_applicability(
        category_code="web_application",
        version_label="4.0",
        parent_title="Session Controls",
        parent_description="Browser session requirements",
        child_requirements=[],
        retrieved_context="Some retrieved context.",
    )
    assert result.applicable is True
    assert result.confidence == 0.0
    assert result.decision_mode == "missing_child_requirements"

    # Missing retrieved context
    result = classify_parent_applicability(
        category_code="web_application",
        version_label="4.0",
        parent_title="Session Controls",
        parent_description="Browser session requirements",
        child_requirements=["Use secure cookie flags."],
        retrieved_context="",
    )
    assert result.applicable is True
    assert result.confidence == 0.0
    assert result.decision_mode == "missing_context"


def test_parent_applicability_llm_error_fails_open(monkeypatch):
    import sdr.apps.ai.engine.classification.parent_applicability as parent_applicability_module

    monkeypatch.setattr(
        parent_applicability_module,
        "chat_completion",
        lambda **_kwargs: SimpleNamespace(error="provider unavailable", content=None),
    )

    result = parent_applicability_module.classify_parent_applicability(
        category_code="web_application",
        version_label="4.0",
        parent_title="Session Controls",
        parent_description="Browser session requirements",
        child_requirements=["Use secure cookie flags."],
        retrieved_context="The application uses browser sessions with secure cookie flags.",
    )

    assert result.applicable is True
    assert result.confidence == 0.0
    assert result.decision_mode == "error"
    assert result.error == "provider unavailable"


def test_parent_applicability_parse_error_fails_open(monkeypatch):
    import sdr.apps.ai.engine.classification.parent_applicability as parent_applicability_module

    monkeypatch.setattr(
        parent_applicability_module,
        "chat_completion",
        lambda **_kwargs: SimpleNamespace(error=None, content="not valid json{{{"),
    )
    monkeypatch.setattr(
        parent_applicability_module,
        "parse_json_with_repair",
        lambda *_args, **_kwargs: (None, "invalid_json"),
    )

    result = parent_applicability_module.classify_parent_applicability(
        category_code="web_application",
        version_label="4.0",
        parent_title="Session Controls",
        parent_description="Browser session requirements",
        child_requirements=["Use secure cookie flags."],
        retrieved_context="The application uses browser sessions with secure cookie flags.",
    )

    assert result.applicable is True
    assert result.confidence == 0.0
    assert result.decision_mode == "parse_error"


def test_parent_applicability_exception_fails_open(monkeypatch):
    import sdr.apps.ai.engine.classification.parent_applicability as parent_applicability_module

    def _raise(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(parent_applicability_module, "chat_completion", _raise)

    result = parent_applicability_module.classify_parent_applicability(
        category_code="web_application",
        version_label="4.0",
        parent_title="Session Controls",
        parent_description="Browser session requirements",
        child_requirements=["Use secure cookie flags."],
        retrieved_context="The application uses browser sessions with secure cookie flags.",
    )

    assert result.applicable is True
    assert result.confidence == 0.0
    assert result.decision_mode == "exception"
    assert result.error == "boom"


def test_parent_applicability_unclear_decision_mode_clamps_confidence(monkeypatch):
    import sdr.apps.ai.engine.classification.parent_applicability as parent_applicability_module

    monkeypatch.setattr(
        parent_applicability_module,
        "chat_completion",
        lambda **_kwargs: SimpleNamespace(
            error=None,
            content='{"applicable": false, "confidence": 0.9, "decision_mode": "unclear", "reasoning": "Ambiguous scope.", "evidence": []}',
        ),
    )

    result = parent_applicability_module.classify_parent_applicability(
        category_code="web_application",
        version_label="4.0",
        parent_title="Session Controls",
        parent_description="Browser session requirements",
        child_requirements=["Use secure cookie flags."],
        retrieved_context="The application uses browser sessions with secure cookie flags.",
    )

    assert result.decision_mode == "unclear"
    assert result.confidence <= 0.4


def test_parent_applicability_consults_llm_even_without_keyword_overlap(monkeypatch):
    import sdr.apps.ai.engine.classification.parent_applicability as parent_applicability_module

    calls = {"count": 0}

    def fake_chat_completion(**_kwargs):
        calls["count"] += 1
        return SimpleNamespace(
            error=None,
            content='{"applicable": false, "confidence": 0.6, "decision_mode": "negative_match", "reasoning": "No session handling described.", "evidence": []}',
        )

    monkeypatch.setattr(parent_applicability_module, "chat_completion", fake_chat_completion)

    result = parent_applicability_module.classify_parent_applicability(
        category_code="web_application",
        version_label="4.0",
        parent_title="Session Controls",
        parent_description="Browser session requirements",
        child_requirements=["Use secure cookie flags."],
        retrieved_context="The design is a headless batch job with no user interface at all.",
    )

    assert calls["count"] == 1
    assert result.applicable is False
    assert result.decision_mode == "negative_match"


def test_parent_applicability_does_not_override_positive_llm_verdict_without_keyword_overlap(monkeypatch):
    import sdr.apps.ai.engine.classification.parent_applicability as parent_applicability_module

    monkeypatch.setattr(
        parent_applicability_module,
        "chat_completion",
        lambda **_kwargs: SimpleNamespace(
            error=None,
            content='{"applicable": true, "confidence": 0.75, "decision_mode": "positive_match", "reasoning": "MFA is described even though the literal term differs.", "evidence": ["multi-factor authentication"]}',
        ),
    )

    result = parent_applicability_module.classify_parent_applicability(
        category_code="web_application",
        version_label="4.0",
        parent_title="Authentication",
        parent_description="MFA requirements",
        child_requirements=["Require MFA for admin access."],
        retrieved_context="Operators must verify identity with a one-time code from a registered device before reaching the control panel.",
    )

    assert result.applicable is True
    assert result.confidence == 0.75
    assert result.decision_mode == "positive_match"


def test_match_scope_terms_uses_word_boundaries():
    from sdr.apps.ai.engine.classification.parent_applicability import _match_scope_terms

    matches = _match_scope_terms(["log"], "Users authenticate via the login form.")

    assert matches == []


def test_should_continue_debate_grants_one_escalation_round_on_overturn():
    service = DebateService(hunter=SimpleNamespace(), critic=SimpleNamespace(), mediator=SimpleNamespace())
    hunter_result = HunterResult(verdict="met", confidence=0.4)
    critic_result = CriticResult(outcome="OVERTURN", revised_verdict="not_met", revised_confidence=0.5)

    should_continue, escalation_round_granted = service._should_continue_debate(
        hunter_result, critic_result, service.max_debate_rounds, False
    )

    assert should_continue is True
    assert escalation_round_granted is True


def test_should_continue_debate_does_not_grant_escalation_twice():
    service = DebateService(hunter=SimpleNamespace(), critic=SimpleNamespace(), mediator=SimpleNamespace())
    hunter_result = HunterResult(verdict="met", confidence=0.4)
    critic_result = CriticResult(outcome="OVERTURN", revised_verdict="not_met", revised_confidence=0.5)

    should_continue, escalation_round_granted = service._should_continue_debate(
        hunter_result, critic_result, service.max_debate_rounds, True
    )

    assert should_continue is False
    assert escalation_round_granted is True


def test_run_debate_executes_escalation_round_on_low_confidence_overturn(monkeypatch):
    monkeypatch.setattr(
        "sdr.apps.ai.engine.debate.debate_service.settings",
        SimpleNamespace(
            AI_DEBATE_MAX_HUNTER_CALLS_PER_PARAMETER=8,
            AI_DEBATE_MAX_DEBATE_ROUNDS=1,
        ),
    )
    requirement_text = "Use MFA for all admin access."
    mock_parent = Mock(spec=CategoryParameterParent)
    mock_parent.id = 1
    mock_parent.title = "Authentication"
    mock_parent.description = "Parent scope"
    parameter = Mock(spec=CategoryParameterChild)
    parameter.id = 1
    parameter.parent = mock_parent
    parameter.requirement_text = requirement_text
    parameter.details = ""
    parameter.requirement_text_normalized = requirement_text
    parameter.ordinal = 0
    debate_input = SimpleNamespace(
        parameter=parameter,
        parameter_text=parameter.requirement_text,
        parameter_section="Authentication",
        contract={"domain": "iam_access_control"},
        killed_assumptions=[],
        hunter_plan={},
        retrieval_query_details={},
        context_chunks=["--- DOCUMENT CHUNK 1 OF 1 ---\np1_b1 Evidence."],
        context_chunk_map={
            "p1_b1": {
                "text": "Evidence.",
                "section": "Authentication",
                "source": "retrieval_context",
                "citation_grade": True,
            }
        },
    )
    tsd_document = SimpleNamespace(get_diagram_by_id=lambda *_args, **_kwargs: None)
    retrieval_result = RetrievalResult(
        context_chunks=list(debate_input.context_chunks),
        source_block_ids=["p1_b1"],
        evidence_metadata={},
    )

    hunter_calls = {"count": 0}

    class _Hunter:
        def run(self, **_kwargs):
            hunter_calls["count"] += 1
            return HunterResult(
                verdict="not_met",
                confidence=0.3,
                reasoning="Hunter found weak evidence.",
                logic_summary="Hunter found weak evidence.",
                citations=[Citation(block_id="p1_b1", page_number=1)],
                evidence_found=True,
            )

    critic_calls = {"count": 0}

    class _Critic:
        def run(self, **_kwargs):
            critic_calls["count"] += 1
            return CriticResult(
                outcome="OVERTURN",
                revised_verdict="met",
                revised_confidence=0.8,
                reasoning="Critic disagrees with Hunter.",
                logic_summary="Critic disagrees with Hunter.",
                valid_citations=[Citation(block_id="p1_b1", page_number=1)],
            )

    class _Mediator:
        def run(self, **_kwargs):
            return MediatorResult(
                final_verdict="not_met",
                confidence=0.5,
                reasoning="Mediator decision.",
                logic_summary="Mediator decision.",
            )

    service = DebateService(hunter=_Hunter(), critic=_Critic(), mediator=_Mediator())
    debate_output = service.run_debate(
        debate_input=debate_input,
        retrieval_result=retrieval_result,
        tsd_document=tsd_document,
    )

    assert hunter_calls["count"] == 2
    assert critic_calls["count"] == 2
    assert debate_output.debate_rounds == 2
