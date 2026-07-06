from __future__ import annotations

from sdr.apps.ai.agents.vision import (
    _apply_diagram_evidence_policy,
    _calibrate_diagram_confidence,
)


def test_validated_met_requirement_stays_met():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
        },
        {
            "outcome": "uphold",
            "validated_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
        },
    )

    assert result["assessed_requirements"][0]["verdict"] == "met"
    assert result["final_verdict"] == "met"
    assert result.get("verdict_policy_source") != "diagram_requirement_not_corroborated"


def test_unvalidated_met_requirement_downgrades_to_na():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
        },
        {
            "outcome": "uphold",
            "validated_requirements": [],
            "invalidated_requirements": [],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
        },
    )

    assert result["assessed_requirements"][0]["verdict"] == "na"
    assert result["final_verdict"] == "na"
    assert result["verdict_policy_source"] == "diagram_requirement_not_corroborated"


def test_invalidated_met_requirement_downgrades_to_na():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
        },
        {
            "outcome": "overturn",
            "validated_requirements": [],
            "invalidated_requirements": [{"requirement_id": "D-1", "verdict": "met", "reason": "fabricated"}],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
        },
    )

    assert result["assessed_requirements"][0]["verdict"] == "na"
    assert result["final_verdict"] == "na"
    assert result["verdict_policy_source"] == "diagram_requirement_not_corroborated"


def test_mixed_ungrounded_met_and_not_met_keeps_not_met_aggregate():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "not_met",
            "assessed_requirements": [
                {"requirement_id": "D-1", "verdict": "met"},
                {"requirement_id": "D-2", "verdict": "met"},
                {"requirement_id": "D-3", "verdict": "not_met"},
            ],
        },
        {
            "outcome": "overturn",
            "validated_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
            "invalidated_requirements": [{"requirement_id": "D-3", "verdict": "not_met"}],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
        },
    )

    by_id = {a["requirement_id"]: a["verdict"] for a in result["assessed_requirements"]}
    assert by_id["D-1"] == "met"
    assert by_id["D-2"] == "na"
    assert by_id["D-3"] == "not_met"
    assert result["final_verdict"] == "not_met"
    assert result["verdict_policy_source"] == "diagram_requirement_not_corroborated"


def test_topline_verdict_disagreeing_with_assessments_is_corrected_to_aggregate():
    # Regression for the actual reported bug: the Mediator's own top-level
    # final_verdict claimed "met" while one of its own assessed_requirements
    # says "not_met", and no citation grounding downgrade occurred (both
    # assessments are already validated by the Critic) — previously nothing
    # would have caught this, and the wrong "met" claim would have been
    # persisted verbatim. The topline verdict must always be corrected to the
    # deterministic aggregate of the assessments, regardless of what the LLM
    # itself claimed at the top level.
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [
                {"requirement_id": "D-1", "verdict": "met"},
                {"requirement_id": "D-2", "verdict": "not_met"},
            ],
        },
        {
            "outcome": "uphold",
            "validated_requirements": [
                {"requirement_id": "D-1", "verdict": "met"},
                {"requirement_id": "D-2", "verdict": "not_met"},
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
        },
    )

    assert result["final_verdict"] == "not_met"
    assert result["verdict_policy_source"] == "diagram_verdict_aggregated_from_assessments"


def test_all_met_assessments_aggregate_to_met():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [
                {"requirement_id": "D-1", "verdict": "met"},
                {"requirement_id": "D-2", "verdict": "met"},
            ],
        },
        {
            "outcome": "uphold",
            "validated_requirements": [
                {"requirement_id": "D-1", "verdict": "met"},
                {"requirement_id": "D-2", "verdict": "met"},
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["final_verdict"] == "met"


def test_all_not_met_assessments_aggregate_to_not_met():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "not_met",
            "assessed_requirements": [
                {"requirement_id": "D-1", "verdict": "not_met"},
                {"requirement_id": "D-2", "verdict": "not_met"},
            ],
        },
        {
            "outcome": "uphold",
            "validated_requirements": [],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["final_verdict"] == "not_met"


def test_mixed_met_and_na_assessments_aggregate_to_na():
    # Documents the resolved ambiguity: a met+na mix with zero not_met
    # resolves to "na", matching the existing _worst_case_diagram_verdict
    # convention and the vision mediator prompt's own stated rule
    # ("not_met > na > met"), rather than "met".
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [
                {"requirement_id": "D-1", "verdict": "met"},
                {"requirement_id": "D-2", "verdict": "na"},
            ],
        },
        {
            "outcome": "uphold",
            "validated_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["final_verdict"] == "na"


def test_no_assessments_with_validated_requirements_keeps_met():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
        },
        {
            "outcome": "uphold",
            "validated_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
        },
    )

    assert result["final_verdict"] == "met"


def test_no_assessments_without_validated_requirements_downgrades_to_na():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
        },
        {
            "outcome": "uphold",
            "validated_requirements": [],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
        },
    )

    assert result["final_verdict"] == "na"
    assert result["verdict_policy_source"] == "diagram_met_without_validated_evidence"


def test_confidence_boost_suppressed_when_hallucinated_claims_present():
    boosted = _calibrate_diagram_confidence(
        {"final_verdict": "met", "confidence": 0.5},
        {
            "overall_verdict": "met",
            "visual_elements_cited": ["lock icon", "TLS badge", "firewall icon"],
        },
        {"outcome": "uphold", "hallucinated_claims": ["fabricated TLS lock icon"]},
    )
    unboosted = _calibrate_diagram_confidence(
        {"final_verdict": "met", "confidence": 0.5},
        {
            "overall_verdict": "met",
            "visual_elements_cited": ["lock icon", "TLS badge", "firewall icon"],
        },
        {"outcome": "uphold", "hallucinated_claims": []},
    )

    assert unboosted["confidence"] > boosted["confidence"]
