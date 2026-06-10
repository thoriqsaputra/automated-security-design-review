from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from sdr.apps.ai.agents.base import Citation, CriticResult, HunterResult, MediatorResult
from sdr.apps.ai.retrieval.router import RetrievalResult
from sdr.apps.ai.services.analysis.dto import AnalysisSummary, DebateOutput, IngestionOutput, ParameterApplicabilityResult
from sdr.apps.ai.services.analysis.pipeline import TSDAnalysisPipeline


def _parent(parent_id=1):
    return SimpleNamespace(id=parent_id, title="Authentication", description="Identity controls")


def _parameter(child_id, *, text, parent=None):
    parent = parent or _parent()
    return SimpleNamespace(
        id=child_id,
        parent=parent,
        requirement_text=text,
        requirement_text_normalized=text,
        details="",
        ordinal=child_id,
    )


class _FakeTSDDocument:
    def __init__(self):
        self.cleaned_up = False
        self._blocks = {
            "p1_b1": SimpleNamespace(
                text="MFA is enforced for all administrative access.",
                page_number=1,
                section_heading="Authentication",
                bbox_x0=0.0,
                bbox_y0=0.0,
                bbox_x1=1.0,
                bbox_y1=1.0,
            )
        }

    def get_block_by_id(self, block_id):
        return self._blocks.get(block_id)

    def get_diagram_by_id(self, block_id):
        return None

    def cleanup_temporary_artifacts(self):
        self.cleaned_up = True


class _FakeIngestionService:
    def __init__(self, tsd_document):
        self.tsd_document = tsd_document

    def ingest(self, review):
        return IngestionOutput.model_construct(tsd_document=self.tsd_document, is_valid_tsd=True)


class _FakeRetrievalService:
    def __init__(self, applicable_parameters, pre_filtered_parameters):
        self.applicable_parameters = applicable_parameters
        self.pre_filtered_parameters = pre_filtered_parameters
        self.parameter_queries = []

    def build_indexes(self, tsd_document):
        return SimpleNamespace(raptor_tree=None, tsd_graph=None)

    def pre_filter_parameters(self, parameters, indexes, category_code):
        return ParameterApplicabilityResult.model_construct(
            applicable_parameters=self.applicable_parameters,
            pre_filtered_parameters=self.pre_filtered_parameters,
            pre_filter_details={str(param.id): {"reason": "out_of_scope"} for param in self.pre_filtered_parameters},
        )

    def retrieve_for_parameter(self, **kwargs):
        self.parameter_queries.append(kwargs["query_details"])
        return RetrievalResult(
            context_chunks=["--- DOCUMENT CHUNK 1 OF 1 ---\np1_b1 MFA is enforced for all administrative access."],
            source_block_ids=["p1_b1"],
            evidence_metadata={
                "evidence_quality": {
                    "implementation_evidence_count": 1,
                    "applicability_signal": True,
                    "counts": {"implementation_or_scope_context": 1},
                    "applicability_terms": ["administrative access", "mfa"],
                }
            },
        )


class _FakeDebateService:
    def __init__(self):
        self.calls = []

    def run_debate(self, *, debate_input, retrieval_result, tsd_document, enable_vision):
        self.calls.append(debate_input)
        reason = "The TSD explicitly states that MFA is enforced for administrative access."
        citation = Citation(block_id="p1_b1", page_number=1, quoted_text="MFA is enforced")
        return DebateOutput.model_construct(
            parameter=debate_input.parameter,
            hunter_result=HunterResult(
                verdict="not_met",
                confidence=0.91,
                reasoning=reason,
                logic_summary=reason,
                evidence_found=True,
                citations=[citation],
            ),
            critic_result=CriticResult(
                revised_verdict="not_met",
                revised_confidence=0.91,
                reasoning=reason,
                logic_summary=reason,
                valid_citations=[citation],
            ),
            mediator_result=MediatorResult(
                final_verdict="not_met",
                confidence=0.91,
                reasoning=reason,
                logic_summary=reason,
                final_citations=[citation],
                severity="medium",
                recommendation="Investigate the admin MFA control gap.",
            ),
            retrieval_result=retrieval_result,
            analysis_trace={
                "retrieved_chunk_ids": ["p1_b1"],
                "contract": debate_input.contract,
                "retrieval_query_details": debate_input.retrieval_query_details,
            },
        )


class _FakePersistenceService:
    def __init__(self):
        self.pre_filtered = []
        self.persisted = []

    def persist_pre_filtered_finding(self, review, parameter, summary, prefilter_details):
        self.pre_filtered.append((review, parameter, prefilter_details))
        summary.na_count += 1

    def persist_finding(self, review, persistence_input, summary):
        self.persisted.append((review, persistence_input))
        verdict = persistence_input.debate_output.mediator_result.final_verdict
        if verdict == "met":
            summary.met_count += 1
        elif verdict == "not_met":
            summary.not_met_count += 1
        else:
            summary.na_count += 1
        summary.citation_count += len(persistence_input.debate_output.mediator_result.final_citations or [])


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self

    def first(self):
        if isinstance(self.value, list):
            return self.value[0] if self.value else None
        return self.value

    def all(self):
        if isinstance(self.value, list):
            return list(self.value)
        return [self.value] if self.value is not None else []


class _Session:
    def __init__(self, execute_value=None):
        self.execute_value = execute_value
        self.executed = []
        self.commits = 0

    def execute(self, statement):
        self.executed.append(statement)
        return _ScalarResult(self.execute_value)

    def commit(self):
        self.commits += 1


class _SessionContext:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class _SessionFactory:
    def __init__(self, sessions):
        self.sessions = list(sessions)

    def __call__(self):
        return _SessionContext(self.sessions.pop(0))


def test_pipeline_dry_run_executes_without_real_llm_or_db(monkeypatch, settings_override):
    settings_override(AI_BATCH_ANALYSIS_ENABLED=False)
    category = SimpleNamespace(id=4, code="web_application", name="Web Application")
    ingestion_job = SimpleNamespace(id=8, category=category)
    parent = _parent()
    pre_filtered = _parameter(1, text="Document OAuth scopes for external clients.", parent=parent)
    applicable = _parameter(2, text="Use MFA for all admin access.", parent=parent)
    tsd_document = _FakeTSDDocument()
    review = SimpleNamespace(
        id=21,
        design=SimpleNamespace(name="Identity Service"),
        selected_categories=[category],
        ingestion_job=ingestion_job,
        status="pending",
        started_at=None,
        completed_at=None,
        error_message=None,
        overview=None,
        summary_json={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    retrieval = _FakeRetrievalService([applicable], [pre_filtered])
    debate = _FakeDebateService()
    persistence = _FakePersistenceService()
    pipeline = TSDAnalysisPipeline(
        ingestion_service=_FakeIngestionService(tsd_document),
        retrieval_service=retrieval,
        debate_service=debate,
        persistence_service=persistence,
    )

    status_session = _Session()
    ingestion_job_session = _Session()
    parameter_session = _Session(execute_value=[pre_filtered, applicable])
    overview_session = _Session()
    monkeypatch.setattr(
        "sdr.core.database.SessionLocal",
        _SessionFactory([status_session, ingestion_job_session, parameter_session, overview_session]),
    )
    completed = {}
    failures = []
    overview_calls = []
    monkeypatch.setattr(TSDAnalysisPipeline, "_is_cancelled", lambda self, review: False)

    def _complete(self, review_obj, summary):
        completed["summary"] = summary.to_dict()
        review_obj.status = "completed_with_findings"
        review_obj.summary_json = summary.to_dict()

    def _fail(self, review_obj, error_message):
        failures.append(error_message)

    def _overview(self, review_obj, summary):
        overview_calls.append((review_obj.id, summary.not_met_count))
        return "Dry run overview"

    monkeypatch.setattr(TSDAnalysisPipeline, "_complete_review", _complete)
    monkeypatch.setattr(TSDAnalysisPipeline, "_fail_review", _fail)
    monkeypatch.setattr(TSDAnalysisPipeline, "_generate_overview", _overview)
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_build_debate_input_for_parameter",
        lambda self, **kwargs: SimpleNamespace(
            parameter=kwargs["parameter"],
            parameter_text=kwargs["parameter"].requirement_text,
            parameter_section=kwargs["parameter"].parent.title,
            contract=kwargs.get("contract") or {},
            retrieval_query_details=kwargs.get("retrieval_query_details") or {},
            context_chunks=[],
            context_chunk_map={},
            diagram_captions=[],
            killed_assumptions=list(kwargs.get("killed_assumptions", [])),
        ),
    )
    monkeypatch.setattr(
        TSDAnalysisPipeline,
        "_persist_debate_output",
        lambda self, **kwargs: persistence.persist_finding(
            kwargs["review"],
            SimpleNamespace(
                parameter=kwargs["parameter"],
                category=kwargs["category"],
                ingestion_job=kwargs["ingestion_job"],
                debate_output=kwargs["debate_output"],
            ),
            kwargs["summary"],
        ),
    )

    summary = pipeline.run(review)

    assert isinstance(summary, AnalysisSummary)
    assert summary.total_parameters == 2
    assert summary.pre_filtered_count == 1
    assert summary.not_met_count == 1
    assert summary.na_count == 1
    assert summary.citation_count == 1
    assert failures == []
    assert review.overview == "Dry run overview"
    assert completed["summary"]["not_met_count"] == 1
    assert persistence.pre_filtered[0][1] is pre_filtered
    assert persistence.persisted[0][1].parameter is applicable
    assert debate.calls[0].parameter is applicable
    assert retrieval.parameter_queries[0]["child_requirement"].startswith("Use MFA")
    assert tsd_document.cleaned_up is True
    assert overview_calls == [(review.id, 1)]
