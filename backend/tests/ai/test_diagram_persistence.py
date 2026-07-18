from __future__ import annotations

import base64
from types import SimpleNamespace

import sdr.apps.designs.models  # noqa: F401

from sdr.apps.ai.agents.base import Citation, CriticResult, HunterResult, MediatorResult
from sdr.apps.ai.engine.dto import AnalysisSummary, PersistenceInput
from sdr.apps.ai.engine.persistence.persistence_service import PersistenceService


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self

    def first(self):
        return self.value


class _Session:
    def __init__(self):
        self.finding = None
        self.findings = []
        self.added = []
        self.commits = 0

    def execute(self, _statement):
        return _ScalarResult(self.finding)

    def add(self, obj):
        self.finding = obj
        self.added.append(obj)

    def add_all(self, objs):
        objs = list(objs)
        self.added.extend(objs)
        self.findings.extend(objs)

    def flush(self):
        return None

    def commit(self):
        self.commits += 1
        if self.finding is not None and getattr(self.finding, "id", None) is None:
            self.finding.id = 101

    def refresh(self, _obj):
        return None

    def rollback(self):
        return None


class _SessionContext:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


def test_persist_diagram_debate_finding_creates_top_level_diagram_finding(monkeypatch):
    service = PersistenceService()
    session = _Session()
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.SessionLocal",
        lambda: _SessionContext(session),
    )
    uploaded = {}
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.storage_service.upload_file",
        lambda content, object_name, content_type: uploaded.update(
            {
                "size": len(content),
                "object_name": object_name,
                "content_type": content_type,
            }
        ),
    )

    review = SimpleNamespace(id=77)
    category = SimpleNamespace(id=4, code="web_application")
    diagram = SimpleNamespace(
        diagram_id="d-1",
        caption="Authentication flow",
        page_number=3,
        bbox_x0=1.0,
        bbox_y0=2.0,
        bbox_x1=3.0,
        bbox_y1=4.0,
        image_format="png",
        image_b64=base64.b64encode(b"x" * 600).decode("ascii"),
    )
    diagram_output = SimpleNamespace(
        diagram=diagram,
        hunter_result={
            "reasoning": "Gateway is visible but no MFA step is shown.",
            "missing_controls": ["MFA step"],
            "requirement_assessments": [
                {"requirement_id": "D-V2", "verdict": "not_met", "reasoning": "No MFA node is shown."}
            ],
        },
        critic_result={
            "reasoning": "The missing MFA conclusion is grounded by the image.",
            "hallucinated_claims": [],
        },
        mediator_result={
            "final_verdict": "not_met",
            "confidence": 0.82,
            "finding_description": "The authentication flow omits a visible MFA step.",
            "reasoning": "The diagram shows credential flow but no second factor or gateway MFA enforcement.",
            "recommendation": "Add the MFA control to the authentication path.",
            "assessed_requirements": [
                {"requirement_id": "D-V2", "verdict": "not_met", "summary": "No MFA step is visible."}
            ],
        },
        debate_rounds=1,
        error=None,
    )
    summary = AnalysisSummary()

    finding = service.persist_diagram_debate_finding(
        review=review,
        category=category,
        diagram_debate_output=diagram_output,
        summary=summary,
    )

    assert finding is session.finding
    assert finding.child_parameter_id is None
    assert finding.parent_parameter_id is None
    assert finding.finding_type == "diagram"
    assert finding.diagram_id == "d-1"
    assert finding.met_status == "not_met"
    assert finding.requirement_reference == "D-V2"
    assert finding.requirement_metadata["assessed_requirements"][0]["requirement_id"] == "D-V2"
    assert finding.requirement_metadata["analysis_trace"]["mediator_result"]["final_verdict"] == "not_met"
    assert summary.diagram_findings_count == 1
    assert summary.not_met_count == 1
    assert uploaded["object_name"].endswith("/d-1.png")


def test_persist_diagram_debate_finding_caps_requirement_reference_to_db_limit(monkeypatch):
    service = PersistenceService()
    session = _Session()
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.SessionLocal",
        lambda: _SessionContext(session),
    )
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.storage_service.upload_file",
        lambda content, object_name, content_type: None,
    )

    review = SimpleNamespace(id=77)
    category = SimpleNamespace(id=4, code="web_application")
    diagram = SimpleNamespace(
        diagram_id="d-1",
        caption="Authentication flow",
        page_number=3,
        bbox_x0=1.0,
        bbox_y0=2.0,
        bbox_x1=3.0,
        bbox_y1=4.0,
        image_format="png",
        image_b64=base64.b64encode(b"x" * 600).decode("ascii"),
    )
    diagram_output = SimpleNamespace(
        diagram=diagram,
        hunter_result={
            "reasoning": "Gateway is visible but no MFA step is shown.",
            "missing_controls": ["MFA step"],
            "requirement_assessments": [
                {"requirement_id": f"D-{idx:03d}", "verdict": "not_met", "summary": "No MFA step is visible."}
                for idx in range(1, 40)
            ],
        },
        critic_result={
            "reasoning": "The missing MFA conclusion is grounded by the image.",
            "hallucinated_claims": [],
        },
        mediator_result={
            "final_verdict": "not_met",
            "confidence": 0.82,
            "finding_description": "The authentication flow omits a visible MFA step.",
            "reasoning": "The diagram shows credential flow but no second factor or gateway MFA enforcement.",
            "recommendation": "Add the MFA control to the authentication path.",
            "assessed_requirements": [
                {"requirement_id": f"D-{idx:03d}", "verdict": "not_met", "summary": "No MFA step is visible."}
                for idx in range(1, 40)
            ],
        },
        debate_rounds=1,
        error=None,
    )
    summary = AnalysisSummary()

    finding = service.persist_diagram_debate_finding(
        review=review,
        category=category,
        diagram_debate_output=diagram_output,
        summary=summary,
    )

    assert finding is not None
    assert finding.requirement_reference is not None
    assert len(finding.requirement_reference) <= 128


def test_persist_finding_keeps_met_when_citation_quotes_are_missing(monkeypatch):
    service = PersistenceService()
    session = _Session()
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.SessionLocal",
        lambda: _SessionContext(session),
    )

    review = SimpleNamespace(id=99)
    category = SimpleNamespace(id=8, code="web_application")
    parent = SimpleNamespace(id=17, title="Authentication", category_id=8)
    parameter = SimpleNamespace(
        id=23,
        parent=parent,
        stable_key="AUTH-1.2.3",
        ordinal=1,
        requirement_text="Use MFA for privileged access.",
    )
    citation = Citation(block_id="p1_b1", page_number=3, quoted_text="")
    debate_output = SimpleNamespace(
        retrieval_result=None,
        hunter_result=HunterResult(
            verdict="met",
            confidence=0.91,
            reasoning="The context names the control.",
            logic_summary="The context names the control.",
            evidence_found=True,
            citations=[citation],
        ),
        critic_result=CriticResult(
            revised_verdict="met",
            revised_confidence=0.9,
            reasoning="The control is grounded in the retrieved block.",
            logic_summary="The control is grounded in the retrieved block.",
            valid_citations=[citation],
        ),
        mediator_result=MediatorResult(
            final_verdict="met",
            confidence=0.92,
            reasoning="The control is shown in the TSD.",
            logic_summary="The control is shown in the TSD.",
            final_citations=[citation],
            finding_description="MFA is present for privileged access.",
            recommendation=None,
        ),
        analysis_trace={
            "context_chunk_map": {
                "p1_b1": {
                    "citation_grade": True,
                    "text": "The system requires MFA for privileged access.",
                    "section": "Authentication",
                    "page_number": 3,
                    "bbox": {"x0": 10.0, "y0": 20.0, "x1": 30.0, "y1": 40.0},
                }
            }
        },
    )
    summary = AnalysisSummary()

    finding = service.persist_finding(
        review,
        PersistenceInput.model_construct(
            parameter=parameter,
            category=category,
            ingestion_job=None,
            debate_output=debate_output,
        ),
        summary,
    )

    assert finding is session.finding
    assert finding.met_status == "met"
    assert finding.requirement_metadata["analysis_trace"]["citation_resolution_mode"] == "quote_matched"
    assert finding.requirement_metadata["structured_citations"][0]["chunk_id"] == "p1_b1"
    assert finding.requirement_metadata["structured_citations"][0]["page"] == 3
    assert session.findings and session.findings[0].quoted_text == "The system requires MFA for privileged access."
    assert summary.met_count == 1
    assert summary.citation_count == 1


def test_persist_finding_keeps_met_when_citations_cannot_be_anchored(monkeypatch):
    # Regression: findings 1839/1846 (review 57), 1884/1906/1911/1898 (review
    # 58) — the debate's own verdict_policy said "met" with Critic-verified
    # citations, but the persisted finding came out "na" because
    # _resolve_citations_for_anchoring couldn't place a UI bounding box for
    # any of them (e.g. the cited block isn't in context_chunk_map at all, or
    # the source text no longer matches). Anchoring failure is a UI/display
    # concern, not evidence invalidity — a "met" verdict the debate already
    # verified must survive it.
    service = PersistenceService()
    session = _Session()
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.SessionLocal",
        lambda: _SessionContext(session),
    )

    review = SimpleNamespace(id=99)
    category = SimpleNamespace(id=8, code="web_application")
    parent = SimpleNamespace(id=17, title="Authentication", category_id=8)
    parameter = SimpleNamespace(
        id=24,
        parent=parent,
        stable_key="AUTH-1.2.4",
        ordinal=1,
        requirement_text="Use MFA for privileged access.",
    )
    citation = Citation(block_id="p9_b3", page_number=9, quoted_text="Completely unrelated fabricated text.")
    debate_output = SimpleNamespace(
        retrieval_result=None,
        hunter_result=HunterResult(
            verdict="met",
            confidence=0.9,
            reasoning="The context names the control.",
            logic_summary="The context names the control.",
            evidence_found=True,
            citations=[citation],
        ),
        critic_result=CriticResult(
            revised_verdict="met",
            revised_confidence=0.9,
            reasoning="The control is grounded in the retrieved block.",
            logic_summary="The control is grounded in the retrieved block.",
            valid_citations=[citation],
        ),
        mediator_result=MediatorResult(
            final_verdict="met",
            confidence=0.9,
            reasoning="The control is shown in the TSD.",
            logic_summary="The control is shown in the TSD.",
            final_citations=[citation],
            finding_description="MFA is present for privileged access.",
            recommendation=None,
        ),
        analysis_trace={
            # p9_b3 is absent entirely — anchoring cannot resolve it to any
            # location, so _resolve_citations_for_anchoring returns [].
            "context_chunk_map": {}
        },
    )
    summary = AnalysisSummary()

    finding = service.persist_finding(
        review,
        PersistenceInput.model_construct(
            parameter=parameter,
            category=category,
            ingestion_job=None,
            debate_output=debate_output,
        ),
        summary,
    )

    assert finding is session.finding
    assert finding.met_status == "met"
    assert summary.met_count == 1


def test_persist_finding_repairs_met_when_no_citations_at_all(monkeypatch):
    # Grounded verdicts must now be repaired from retrieved context rather than
    # silently downgraded when the agent output omitted citations.
    service = PersistenceService()
    session = _Session()
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.SessionLocal",
        lambda: _SessionContext(session),
    )

    review = SimpleNamespace(id=99)
    category = SimpleNamespace(id=8, code="web_application")
    parent = SimpleNamespace(id=17, title="Authentication", category_id=8)
    parameter = SimpleNamespace(
        id=25,
        parent=parent,
        stable_key="AUTH-1.2.5",
        ordinal=1,
        requirement_text="Use MFA for privileged access.",
    )
    debate_output = SimpleNamespace(
        retrieval_result=None,
        hunter_result=HunterResult(
            verdict="met",
            confidence=0.9,
            reasoning="The context names the control.",
            logic_summary="The context names the control.",
            evidence_found=True,
            citations=[],
        ),
        critic_result=CriticResult(
            revised_verdict="met",
            revised_confidence=0.9,
            reasoning="No citations to verify.",
            logic_summary="No citations to verify.",
            valid_citations=[],
        ),
        mediator_result=MediatorResult(
            final_verdict="met",
            confidence=0.9,
            reasoning="The control is shown in the TSD.",
            logic_summary="The control is shown in the TSD.",
            final_citations=[],
            finding_description="MFA is present for privileged access.",
            recommendation=None,
        ),
        analysis_trace={
            "context_chunk_map": {
                "p4_b7": {
                    "citation_grade": True,
                    "text": "Privileged administrators must authenticate with MFA before access is granted.",
                    "section": "Authentication",
                    "page_number": 4,
                    "bbox": {"x0": 3.0, "y0": 11.0, "x1": 40.0, "y1": 18.0},
                }
            }
        },
    )
    summary = AnalysisSummary()

    finding = service.persist_finding(
        review,
        PersistenceInput.model_construct(
            parameter=parameter,
            category=category,
            ingestion_job=None,
            debate_output=debate_output,
        ),
        summary,
    )

    assert finding is session.finding
    assert finding.met_status == "met"
    assert finding.requirement_metadata["analysis_trace"]["citation_resolution_mode"] == "top_context_fallback"
    assert session.findings and session.findings[0].block_id == "p4_b7"


def test_persist_finding_raises_when_grounded_verdict_has_no_citable_context(monkeypatch):
    service = PersistenceService()
    session = _Session()
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.SessionLocal",
        lambda: _SessionContext(session),
    )

    review = SimpleNamespace(id=99)
    category = SimpleNamespace(id=8, code="web_application")
    parent = SimpleNamespace(id=17, title="Authentication", category_id=8)
    parameter = SimpleNamespace(
        id=26,
        parent=parent,
        stable_key="AUTH-1.2.6",
        ordinal=1,
        requirement_text="Use MFA for privileged access.",
    )
    debate_output = SimpleNamespace(
        retrieval_result=None,
        hunter_result=HunterResult(verdict="met", confidence=0.9, reasoning="Found control.", logic_summary="Found control.", evidence_found=True, citations=[]),
        critic_result=CriticResult(revised_verdict="met", revised_confidence=0.9, reasoning="Found control.", logic_summary="Found control.", valid_citations=[]),
        mediator_result=MediatorResult(
            final_verdict="met",
            confidence=0.9,
            reasoning="The control is shown in the TSD.",
            logic_summary="The control is shown in the TSD.",
            final_citations=[],
            finding_description="MFA is present for privileged access.",
            recommendation=None,
        ),
        analysis_trace={"context_chunk_map": {}},
    )
    summary = AnalysisSummary()

    with pytest.raises(ValueError):
        service.persist_finding(
            review,
            PersistenceInput.model_construct(
                parameter=parameter,
                category=category,
                ingestion_job=None,
                debate_output=debate_output,
            ),
            summary,
        )


def test_persist_diagram_debate_finding_stores_pipeline_mode_and_extraction(monkeypatch):
    service = PersistenceService()
    session = _Session()
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.SessionLocal",
        lambda: _SessionContext(session),
    )
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.storage_service.upload_file",
        lambda content, object_name, content_type: None,
    )

    review = SimpleNamespace(id=77)
    category = SimpleNamespace(id=4, code="web_application")
    diagram = SimpleNamespace(
        diagram_id="d-3",
        caption="Network diagram",
        page_number=2,
        bbox_x0=1.0,
        bbox_y0=2.0,
        bbox_x1=3.0,
        bbox_y1=4.0,
        image_format="png",
        image_b64=base64.b64encode(b"x" * 600).decode("ascii"),
    )
    diagram_output = SimpleNamespace(
        diagram=diagram,
        pipeline_mode="extract_reason",
        hunter_result={
            "extraction_summary": "Frontend talks to backend over HTTPS.",
            "reasoning": "Frontend talks to backend over HTTPS.",
            "components": [{"id": "c1", "name": "Frontend", "type": "service"}],
            "trust_boundaries": [],
            "flows": [{"id": "f1", "source_component_id": "c1", "target_component_id": "c2", "protocol": "HTTPS"}],
            "other_visible_text": [],
            "requirement_assessments": [],
        },
        critic_result={},
        mediator_result={
            "final_verdict": "not_met",
            "confidence": 0.7,
            "finding_description": "No MFA step is visible.",
            "reasoning": "No MFA step is visible.",
            "recommendation": None,
            "assessed_requirements": [
                {"requirement_id": "D-V2", "verdict": "not_met", "summary": "No MFA step is visible."}
            ],
        },
        debate_rounds=1,
        error=None,
    )
    summary = AnalysisSummary()

    finding = service.persist_diagram_debate_finding(
        review=review,
        category=category,
        diagram_debate_output=diagram_output,
        summary=summary,
    )

    assert finding is session.finding
    assert finding.requirement_metadata["pipeline_mode"] == "extract_reason"
    assert finding.requirement_metadata["diagram_extraction"]["components"] == [
        {"id": "c1", "name": "Frontend", "type": "service"}
    ]
    assert finding.requirement_metadata["diagram_extraction"]["flows"][0]["protocol"] == "HTTPS"
    assert finding.description == "No MFA step is visible."
    assert finding.reason == "No MFA step is visible."


def test_persist_diagram_debate_finding_preserves_non_architecture_scope(monkeypatch):
    service = PersistenceService()
    session = _Session()
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.SessionLocal",
        lambda: _SessionContext(session),
    )
    monkeypatch.setattr(
        "sdr.apps.ai.engine.persistence.persistence_service.storage_service.upload_file",
        lambda content, object_name, content_type: None,
    )

    review = SimpleNamespace(id=88)
    category = SimpleNamespace(id=4, code="web_application")
    diagram = SimpleNamespace(
        diagram_id="d-2",
        caption="Login screenshot",
        page_number=5,
        bbox_x0=1.0,
        bbox_y0=2.0,
        bbox_x1=3.0,
        bbox_y1=4.0,
        image_format="png",
        image_b64=base64.b64encode(b"x" * 600).decode("ascii"),
    )
    diagram_output = SimpleNamespace(
        diagram=diagram,
        hunter_result={
            "diagram_scope_verdict": "non_architecture",
            "diagram_scope_reasoning": "The image is a UI screenshot.",
            "reasoning": "This is not an architecture diagram.",
            "missing_controls": [],
            "requirement_assessments": [
                {"requirement_id": "D-V2", "verdict": "na", "reasoning": "Screenshots are out of scope."}
            ],
        },
        critic_result={
            "diagram_scope_verdict": "non_architecture",
            "diagram_scope_reasoning": "The image is not architecture/security-relevant.",
            "reasoning": "The Hunter should not treat a screenshot as a missing control.",
            "hallucinated_claims": [],
        },
        mediator_result={
            "diagram_scope_verdict": "non_architecture",
            "diagram_scope_reasoning": "The image is a screenshot, so the requirement is not applicable.",
            "final_verdict": "na",
            "confidence": 0.58,
            "finding_description": "The image is a screenshot rather than an architecture/security-relevant diagram.",
            "reasoning": "The requirement cannot be assessed from a UI screenshot.",
            "recommendation": None,
            "assessed_requirements": [
                {"requirement_id": "D-V2", "verdict": "na", "summary": "The screenshot is out of scope for this diagram requirement."}
            ],
        },
        debate_rounds=1,
        error=None,
    )
    summary = AnalysisSummary()

    finding = service.persist_diagram_debate_finding(
        review=review,
        category=category,
        diagram_debate_output=diagram_output,
        summary=summary,
    )

    assert finding is session.finding
    assert finding.met_status == "na"
    assert finding.requirement_metadata["analysis_trace"]["diagram_scope_verdict"] == "non_architecture"
    assert finding.requirement_metadata["analysis_trace"]["diagram_scope_reasoning"] == (
        "The image is a screenshot, so the requirement is not applicable."
    )
    assert finding.requirement_metadata["analysis_trace"]["mediator_result"]["diagram_scope_verdict"] == "non_architecture"
