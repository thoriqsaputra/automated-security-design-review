from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

from sdr.apps.ai.client.base import AIProvider, AIResponse
from sdr.apps.ai.engine.dto import AnalysisSummary
from sdr.apps.ai.engine.pipeline import TSDAnalysisPipeline


def _requirement(ordinal=1, stable_key="D-1", text="Gateway enforces MFA", hint="Look for MFA step"):
    return SimpleNamespace(
        ordinal=ordinal,
        stable_key=stable_key,
        requirement_text=text,
        verification_hint=hint,
    )


def _diagram_bytes(size=600):
    return base64.b64encode(b"x" * size).decode("ascii")


def test_agents_vision_exports_and_runs_mocked_debate(monkeypatch):
    from sdr.apps.ai.agents.base import BaseAgent
    from sdr.apps.ai.agents.vision import DiagramDebateOutput, DiagramDebateService, DiagramInput

    responses = iter(
        [
            AIResponse(
                content='{"overall_verdict":"not_met","confidence":0.61,"reasoning":"No MFA step is visible.","requirement_assessments":[{"requirement_id":"D-1","verdict":"not_met"}],"visual_elements_cited":["login form"]}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
            AIResponse(
                content='{"outcome":"uphold","validated_requirements":[{"requirement_id":"D-1","verdict":"not_met"}],"hallucinated_claims":[]}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
            AIResponse(
                content='{"final_verdict":"not_met","confidence":0.62,"finding_description":"The authentication flow omits MFA.","reasoning":"No second factor is shown.","assessed_requirements":[{"requirement_id":"D-1","verdict":"not_met"}]}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
        ]
    )

    monkeypatch.setattr(BaseAgent, "_call_llm", lambda self, **kwargs: next(responses))

    diagram = DiagramInput(
        diagram_id="p1_d1",
        image_b64=_diagram_bytes(),
        page_number=1,
        caption="Authentication flow",
    )
    output = DiagramDebateService().run_diagram_debate(
        diagram=diagram,
        requirements=[_requirement()],
        tsd_context="The gateway authenticates users.",
    )

    assert isinstance(output, DiagramDebateOutput)
    assert output.error is None
    assert output.diagram.diagram_id == "p1_d1"
    assert output.mediator_result["final_verdict"] == "not_met"
    assert output.mediator_result["confidence"] == 0.72


def test_diagram_debate_service_shim_reexports_symbols():
    from sdr.apps.ai.agents import vision as vision_module
    from sdr.apps.ai.engine import diagram_debate_service as shim_module

    assert shim_module.DiagramInput is vision_module.DiagramInput
    assert shim_module.DiagramDebateOutput is vision_module.DiagramDebateOutput
    assert shim_module.DiagramDebateService is vision_module.DiagramDebateService


class _FakeDiagramBlock:
    def __init__(self):
        self.diagram_id = "p2_d1"
        self.image_b64 = _diagram_bytes()
        self.page_number = 2
        self.caption = "Network layout"
        self.surrounding_text = "Traffic reaches the gateway first."
        self.image_format = "png"
        self.bbox_x0 = 1.0
        self.bbox_y0 = 2.0
        self.bbox_x1 = 3.0
        self.bbox_y1 = 4.0

    def ensure_image_loaded(self, _min_bytes):
        return None

    def is_valid(self):
        return True


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self

    def all(self):
        if isinstance(self.value, list):
            return list(self.value)
        return [self.value] if self.value is not None else []


class _Session:
    def __init__(self, execute_value):
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


def test_pipeline_diagram_analysis_uses_diagram_input_and_debate_service(monkeypatch, settings_override):
    settings_override(
        AI_VISION_DIAGRAM_ANALYSIS_ENABLED=True,
        AI_VISION_ENABLED=True,
        AI_VISION_MIN_DIAGRAM_BYTES=512,
        AI_VISION_MAX_CONCURRENCY=1,
        AI_VISION_DIAGRAM_REQUIREMENTS_MAX_ITEMS=15,
    )

    captured = {}
    persisted = []

    class _FakePersistence:
        def persist_diagram_debate_finding(self, review, category, diagram_debate_output, summary):
            persisted.append((review, category, diagram_debate_output))
            summary.diagram_findings_count += 1
            return object()

    pipeline = TSDAnalysisPipeline(
        ingestion_service=SimpleNamespace(),
        retrieval_service=SimpleNamespace(),
        debate_service=SimpleNamespace(),
        persistence_service=_FakePersistence(),
    )

    def _fake_run_diagram_debate(*, diagram, requirements, tsd_context):
        captured["diagram_cls"] = diagram.__class__.__name__
        captured["diagram_module"] = diagram.__class__.__module__
        captured["requirements"] = requirements
        captured["tsd_context"] = tsd_context
        return SimpleNamespace(
            diagram=diagram,
            hunter_result={"overall_verdict": "met"},
            critic_result={"outcome": "uphold"},
            mediator_result={"final_verdict": "met", "confidence": 0.8},
            requirements=requirements,
            debate_rounds=1,
            error=None,
        )

    pipeline.diagram_debate_service = SimpleNamespace(run_diagram_debate=_fake_run_diagram_debate)

    monkeypatch.setattr(
        "sdr.core.database.SessionLocal",
        lambda: _SessionContext(_Session([_requirement()])),
    )

    tsd_document = SimpleNamespace(
        all_diagrams=[_FakeDiagramBlock()],
        full_text="Gateway architecture with ingress traffic and authentication.",
    )
    review = SimpleNamespace(id=1)
    category = SimpleNamespace(id=10, code="web_application")
    ingestion_job = SimpleNamespace(id=11)
    summary = AnalysisSummary()

    pipeline._run_diagram_analysis(
        review=review,
        tsd_document=tsd_document,
        category=category,
        ingestion_job=ingestion_job,
        effective_asvs_level=2,
        summary=summary,
    )

    assert captured["diagram_cls"] == "DiagramInput"
    assert captured["diagram_module"] == "sdr.apps.ai.agents.vision"
    assert len(captured["requirements"]) == 1
    assert persisted
    assert summary.diagram_findings_count == 1


def test_backend_runtime_source_has_no_legacy_vision_agent_usage():
    apps_root = Path(__file__).resolve().parents[2] / "sdr" / "apps"

    for path in apps_root.rglob("*.py"):
        if path.name in {"vision.py", "diagram_debate_service.py"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert "VisionAgent" not in source
        assert "audit_diagrams_for_parameter" not in source

