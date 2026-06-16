from __future__ import annotations

import base64
from types import SimpleNamespace

import sdr.apps.designs.models  # noqa: F401

from sdr.apps.ai.engine.dto import AnalysisSummary
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
        self.commits = 0

    def execute(self, _statement):
        return _ScalarResult(self.finding)

    def add(self, obj):
        self.finding = obj

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
