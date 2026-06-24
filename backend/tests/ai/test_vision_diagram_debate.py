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
    from sdr.apps.ai.agents.vision import (
        DiagramDebateOutput,
        DiagramDebateService,
        DiagramInput,
        _apply_diagram_evidence_policy,
    )

    responses = iter(
        [
            AIResponse(
                content='{"diagram_scope_verdict":"architecture_relevant","diagram_scope_reasoning":"The image shows an authentication flow between system components.","overall_verdict":"not_met","confidence":0.61,"reasoning":"No MFA step is visible.","requirement_assessments":[{"requirement_id":"D-1","verdict":"not_met"}],"visual_elements_cited":["login form"]}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
            AIResponse(
                content='{"diagram_scope_verdict":"architecture_relevant","diagram_scope_reasoning":"The image depicts an authentication flow between components.","outcome":"uphold","validated_requirements":[{"requirement_id":"D-1","verdict":"not_met"}],"hallucinated_claims":[]}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
            AIResponse(
                content='{"diagram_scope_verdict":"architecture_relevant","diagram_scope_reasoning":"Both agents describe an authentication architecture flow.","final_verdict":"not_met","confidence":0.62,"finding_description":"The authentication flow omits MFA.","reasoning":"No second factor is shown.","assessed_requirements":[{"requirement_id":"D-1","verdict":"not_met"}]}',
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
    assert output.mediator_result["diagram_scope_verdict"] == "architecture_relevant"
    assert output.mediator_result["confidence"] == 0.72

    forced = _apply_diagram_evidence_policy(
        {
            "final_verdict": "not_met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "not_met"}],
        },
        {
            "diagram_scope_verdict": "non_architecture",
            "diagram_scope_reasoning": "This is a screenshot, not a system diagram.",
            "outcome": "overturn",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
            "requirement_assessments": [{"requirement_id": "D-1", "verdict": "not_met"}],
        },
    )
    assert forced["final_verdict"] == "na"
    assert forced["verdict_policy_source"] == "diagram_non_architecture_image"


def test_diagram_evidence_policy_forces_na_when_all_requirements_are_na():
    from sdr.apps.ai.agents.vision import _apply_diagram_evidence_policy

    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "not_met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "na"}],
        },
        {
            "outcome": "uphold",
            "validated_requirements": [],
            "diagram_scope_verdict": "uncertain",
        },
        {
            "diagram_scope_verdict": "uncertain",
        },
    )

    assert result["final_verdict"] == "na"
    assert result["verdict_policy_source"] == "diagram_all_requirements_not_applicable"


def test_diagram_debate_service_forces_na_for_non_architecture_images(monkeypatch):
    from sdr.apps.ai.agents.base import BaseAgent
    from sdr.apps.ai.agents.vision import DiagramDebateService, DiagramInput

    responses = iter(
        [
            AIResponse(
                content='{"diagram_scope_verdict":"non_architecture","diagram_scope_reasoning":"The image is a product UI screenshot, not a system architecture diagram.","overall_verdict":"not_met","confidence":0.61,"reasoning":"No MFA control is visible.","requirement_assessments":[{"requirement_id":"D-1","verdict":"not_met","reasoning":"The screenshot does not show MFA."}],"visual_elements_cited":["login screen"]}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
            AIResponse(
                content='{"diagram_scope_verdict":"non_architecture","diagram_scope_reasoning":"The image is a UI screenshot rather than an architecture diagram.","outcome":"overturn","invalidated_requirements":[{"requirement_id":"D-1","verdict":"na","reason":"The requirement is not applicable to a screenshot."}],"validated_requirements":[],"hallucinated_claims":[],"reasoning":"The image is outside architecture/security diagram scope."}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
            AIResponse(
                content='{"diagram_scope_verdict":"architecture_relevant","diagram_scope_reasoning":"The login page may represent an application flow.","final_verdict":"not_met","confidence":0.62,"finding_description":"The screenshot does not show MFA.","reasoning":"No second factor is shown.","assessed_requirements":[{"requirement_id":"D-1","verdict":"not_met","summary":"No MFA control is visible."}]}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
        ]
    )

    monkeypatch.setattr(BaseAgent, "_call_llm", lambda self, **kwargs: next(responses))

    output = DiagramDebateService().run_diagram_debate(
        diagram=DiagramInput(
            diagram_id="p1_d2",
            image_b64=_diagram_bytes(),
            page_number=1,
            caption="Login page screenshot",
        ),
        requirements=[_requirement()],
        tsd_context="The TSD includes a login UI screenshot.",
    )

    assert output.error is None
    assert output.mediator_result["final_verdict"] == "na"
    assert output.mediator_result["diagram_scope_verdict"] == "non_architecture"
    assert output.mediator_result["verdict_policy_source"] == "diagram_non_architecture_image"
    assert output.mediator_result["assessed_requirements"][0]["verdict"] == "na"
    assert output.mediator_result["confidence"] == 0.52


def test_diagram_debate_service_shim_reexports_symbols():
    from sdr.apps.ai.agents import vision as vision_module
    from sdr.apps.ai.engine.debate import diagram_debate_service as shim_module

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


def test_diagram_requirement_selector_retrieves_per_diagram_and_respects_cap(monkeypatch):
    from sdr.apps.ai.engine.debate.diagram_requirement_selector import DiagramRequirementSelector

    class _Repo:
        def __init__(self):
            self.search_calls = []

        def search_diagram_requirements(
            self,
            *,
            category_id,
            ingestion_job_id,
            effective_asvs_level,
            query_embedding,
            top_k,
        ):
            self.search_calls.append((tuple(query_embedding), top_k))
            if query_embedding == [1.0]:
                return [_requirement(stable_key="D-auth-1"), _requirement(stable_key="D-auth-2")][:top_k]
            return [_requirement(stable_key="D-net-1"), _requirement(stable_key="D-net-2")][:top_k]

        def list_diagram_requirements(self, *, category_id, ingestion_job_id, effective_asvs_level):
            return [_requirement(stable_key="D-fallback-1"), _requirement(stable_key="D-fallback-2")]

    selector = DiagramRequirementSelector(
        config=SimpleNamespace(vision_diagram_requirements_max_items=1),
        workflow_repository=_Repo(),
    )

    monkeypatch.setattr(
        "sdr.apps.ai.engine.debate.diagram_requirement_selector.get_embedding",
        lambda *, text, dimensions: [1.0] if "Authentication" in text else [2.0],
    )

    auth_diagram = SimpleNamespace(
        diagram_id="d-auth",
        caption="Authentication flow",
        surrounding_text="Authentication gateway validates MFA.",
        page_number=1,
    )
    network_diagram = SimpleNamespace(
        diagram_id="d-net",
        caption="Network layout",
        surrounding_text="Traffic enters through a DMZ gateway.",
        page_number=1,
    )
    tsd_document = SimpleNamespace(pages=[])
    category = SimpleNamespace(id=1)
    ingestion_job = SimpleNamespace(id=2)

    auth_requirements = selector.select_for_diagram(
        diagram=auth_diagram,
        tsd_document=tsd_document,
        category=category,
        ingestion_job=ingestion_job,
        effective_asvs_level=2,
    )
    network_requirements = selector.select_for_diagram(
        diagram=network_diagram,
        tsd_document=tsd_document,
        category=category,
        ingestion_job=ingestion_job,
        effective_asvs_level=2,
    )

    assert [item.stable_key for item in auth_requirements] == ["D-auth-1"]
    assert [item.stable_key for item in network_requirements] == ["D-net-1"]


def test_diagram_requirement_selector_falls_back_when_embedding_or_query_is_missing(monkeypatch):
    from sdr.apps.ai.engine.debate.diagram_requirement_selector import DiagramRequirementSelector

    class _Repo:
        def search_diagram_requirements(self, **kwargs):
            raise AssertionError("vector search should not run in fallback case")

        def list_diagram_requirements(self, *, category_id, ingestion_job_id, effective_asvs_level):
            return [_requirement(stable_key="D-fallback-1"), _requirement(stable_key="D-fallback-2")]

    selector = DiagramRequirementSelector(
        config=SimpleNamespace(vision_diagram_requirements_max_items=1),
        workflow_repository=_Repo(),
    )

    monkeypatch.setattr(
        "sdr.apps.ai.engine.debate.diagram_requirement_selector.get_embedding",
        lambda **kwargs: [],
    )

    requirements = selector.select_for_diagram(
        diagram=SimpleNamespace(diagram_id="d-empty", caption="", surrounding_text="", page_number=1),
        tsd_document=SimpleNamespace(pages=[]),
        category=SimpleNamespace(id=1),
        ingestion_job=SimpleNamespace(id=2),
        effective_asvs_level=2,
    )

    assert [item.stable_key for item in requirements] == ["D-fallback-1"]


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

    def _fake_run_diagram_debate(*, diagram, requirements, tsd_context, cancel_check=None):
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
    pipeline.diagram_analysis.diagram_debate_service = pipeline.diagram_debate_service

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
    assert summary.total_parameters == 1


def test_pipeline_diagram_analysis_persists_na_for_non_architecture_images(monkeypatch, settings_override):
    from sdr.apps.ai.agents.base import BaseAgent

    settings_override(
        AI_VISION_DIAGRAM_ANALYSIS_ENABLED=True,
        AI_VISION_ENABLED=True,
        AI_VISION_MIN_DIAGRAM_BYTES=512,
        AI_VISION_MAX_CONCURRENCY=1,
        AI_VISION_DIAGRAM_REQUIREMENTS_MAX_ITEMS=15,
    )

    responses = iter(
        [
            AIResponse(
                content='{"diagram_scope_verdict":"non_architecture","diagram_scope_reasoning":"The image is a screenshot of a product page, not a system diagram.","overall_verdict":"not_met","confidence":0.64,"reasoning":"No boundary or MFA control is visible.","requirement_assessments":[{"requirement_id":"D-1","verdict":"not_met","reasoning":"The screenshot does not show the required control."}],"visual_elements_cited":["browser window"]}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
            AIResponse(
                content='{"diagram_scope_verdict":"non_architecture","diagram_scope_reasoning":"This is a screenshot, not architecture/security scope.","outcome":"overturn","invalidated_requirements":[{"requirement_id":"D-1","verdict":"na","reason":"A screenshot cannot establish the diagram requirement."}],"validated_requirements":[],"hallucinated_claims":[],"reasoning":"The image should be treated as non-architecture."}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
            AIResponse(
                content='{"diagram_scope_verdict":"uncertain","diagram_scope_reasoning":"The screenshot might relate to the application.","final_verdict":"not_met","confidence":0.60,"finding_description":"The image does not show the required control.","reasoning":"The image lacks the control.","assessed_requirements":[{"requirement_id":"D-1","verdict":"not_met","summary":"The control is not visible."}]}',
                model="test",
                provider=AIProvider.LOCAL,
            ),
        ]
    )
    monkeypatch.setattr(BaseAgent, "_call_llm", lambda self, **kwargs: next(responses))

    persisted = []

    class _FakePersistence:
        def persist_diagram_debate_finding(self, review, category, diagram_debate_output, summary):
            persisted.append(diagram_debate_output)
            summary.diagram_findings_count += 1
            if diagram_debate_output.mediator_result["final_verdict"] == "na":
                summary.na_count += 1
            return object()

    pipeline = TSDAnalysisPipeline(
        ingestion_service=SimpleNamespace(),
        retrieval_service=SimpleNamespace(),
        debate_service=SimpleNamespace(),
        persistence_service=_FakePersistence(),
    )

    monkeypatch.setattr(
        "sdr.core.database.SessionLocal",
        lambda: _SessionContext(_Session([_requirement()])),
    )

    tsd_document = SimpleNamespace(
        all_diagrams=[_FakeDiagramBlock()],
        full_text="The document embeds a screenshot of the product login page.",
    )
    tsd_document.all_diagrams[0].caption = "Login screenshot"
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

    assert persisted
    assert persisted[0].mediator_result["final_verdict"] == "na"
    assert persisted[0].mediator_result["diagram_scope_verdict"] == "non_architecture"
    assert summary.diagram_findings_count == 1
    assert summary.total_parameters == 1


class _FakeDiagramBlockTwo(_FakeDiagramBlock):
    def __init__(self):
        super().__init__()
        self.diagram_id = "p3_d1"
        self.caption = "Deployment topology"


def test_pipeline_diagram_analysis_counts_total_parameters_for_errors_and_successes(
    monkeypatch, settings_override
):
    settings_override(
        AI_VISION_DIAGRAM_ANALYSIS_ENABLED=True,
        AI_VISION_ENABLED=True,
        AI_VISION_MIN_DIAGRAM_BYTES=512,
        AI_VISION_MAX_CONCURRENCY=1,
        AI_VISION_DIAGRAM_REQUIREMENTS_MAX_ITEMS=15,
    )

    persisted = []

    class _FakePersistence:
        def persist_diagram_debate_finding(self, review, category, diagram_debate_output, summary):
            persisted.append(diagram_debate_output)
            summary.diagram_findings_count += 1
            summary.not_met_count += 1
            return object()

    pipeline = TSDAnalysisPipeline(
        ingestion_service=SimpleNamespace(),
        retrieval_service=SimpleNamespace(),
        debate_service=SimpleNamespace(),
        persistence_service=_FakePersistence(),
    )

    def _fake_run_diagram_debate(*, diagram, requirements, tsd_context, cancel_check=None):
        if diagram.diagram_id == "p3_d1":
            return SimpleNamespace(
                diagram=diagram,
                hunter_result=None,
                critic_result=None,
                mediator_result=None,
                requirements=requirements,
                debate_rounds=0,
                error="vision call failed",
            )
        return SimpleNamespace(
            diagram=diagram,
            hunter_result={"overall_verdict": "not_met"},
            critic_result={"outcome": "uphold"},
            mediator_result={"final_verdict": "not_met", "confidence": 0.7},
            requirements=requirements,
            debate_rounds=1,
            error=None,
        )

    pipeline.diagram_debate_service = SimpleNamespace(run_diagram_debate=_fake_run_diagram_debate)
    pipeline.diagram_analysis.diagram_debate_service = pipeline.diagram_debate_service

    monkeypatch.setattr(
        "sdr.core.database.SessionLocal",
        lambda: _SessionContext(_Session([_requirement()])),
    )

    tsd_document = SimpleNamespace(
        all_diagrams=[_FakeDiagramBlock(), _FakeDiagramBlockTwo()],
        full_text="Gateway architecture with ingress traffic and authentication.",
    )
    review = SimpleNamespace(id=1)
    category = SimpleNamespace(id=10, code="web_application")
    ingestion_job = SimpleNamespace(id=11)
    summary = AnalysisSummary()

    pipeline.diagram_analysis.run(
        review=review,
        tsd_document=tsd_document,
        category=category,
        ingestion_job=ingestion_job,
        summary=summary,
    )

    assert len(persisted) == 1
    assert summary.error_count == 1
    assert summary.total_parameters == 2
    assert summary.met_count + summary.not_met_count + summary.na_count <= summary.total_parameters


def test_backend_runtime_source_has_no_legacy_vision_agent_usage():
    apps_root = Path(__file__).resolve().parents[2] / "sdr" / "apps"

    for path in apps_root.rglob("*.py"):
        if path.name in {"vision.py", "diagram_debate_service.py"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert "VisionAgent" not in source
        assert "audit_diagrams_for_parameter" not in source

