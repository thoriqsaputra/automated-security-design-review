from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.agents.vision import MergedDiagramExtraction
from sdr.apps.ai.engine.debate.diagram_extract_reason_service import (
    DiagramExtractReasonService,
    _normalize_assessment_rows,
    _strict_confirmation_votes,
)


def test_strict_confirmation_votes_requires_real_majority():
    assert _strict_confirmation_votes(1, 0.5) == 1
    assert _strict_confirmation_votes(2, 0.5) == 2
    assert _strict_confirmation_votes(3, 0.5) == 2
    assert _strict_confirmation_votes(4, 0.75) == 3


def test_normalize_assessment_rows_accepts_assessed_requirements_alias():
    rows = _normalize_assessment_rows(
        {
            "assessed_requirements": [
                {"requirement_id": "R1", "verdict": "met", "cited_element_ids": ["c1"]},
            ]
        }
    )

    assert rows == [{"requirement_id": "R1", "verdict": "met", "cited_element_ids": ["c1"]}]


def test_normalize_assessment_rows_accepts_id_alias_for_requirement_id():
    rows = _normalize_assessment_rows(
        {
            "requirement_assessments": [
                {"id": "R1", "verdict": "met", "cited_element_ids": ["c1"]},
            ]
        }
    )

    assert rows == [{"id": "R1", "requirement_id": "R1", "verdict": "met", "cited_element_ids": ["c1"]}]


def test_reasoning_retries_merge_partial_attempts(monkeypatch):
    service = DiagramExtractReasonService()
    service._reasoner_batch_size = 10
    service._reasoner_batch_max_concurrency = 1
    service._citation_retry_limit = 1
    service._full_failure_retry_limit = 1

    merged = MergedDiagramExtraction(
        components=[{"id": "c1", "name": "Gateway", "confirmed": True}],
        diagram_scope_verdict="architecture_relevant",
        votes_total=1,
    )
    requirements = [
        SimpleNamespace(ordinal=1, stable_key="R1", requirement_text="req 1", verification_hint="hint"),
        SimpleNamespace(ordinal=2, stable_key="R2", requirement_text="req 2", verification_hint="hint"),
    ]

    responses = iter(
        [
            {"requirement_assessments": [{"requirement_id": "R1", "verdict": "met", "cited_element_ids": ["c1"]}]},
            {"assessed_requirements": [{"requirement_id": "R2", "verdict": "na", "cited_element_ids": []}]},
        ]
    )

    monkeypatch.setattr(service._reasoner_agent, "run_text", lambda **_: next(responses))

    assessments, diagnostics = service._run_reasoning_batches(
        merged=merged,
        requirements=requirements,
        caption="",
        surrounding="",
    )

    by_id = {item["requirement_id"]: item for item in assessments}
    assert by_id["R1"]["verdict"] == "met"
    assert by_id["R2"]["verdict"] == "na"
    assert diagnostics["completeness_retry_batches"] == 1
