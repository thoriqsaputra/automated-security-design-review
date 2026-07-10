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


def test_unvalidated_met_requirement_stays_met_when_not_explicitly_invalidated():
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

    assert result["assessed_requirements"][0]["verdict"] == "met"
    assert result["final_verdict"] == "met"


def test_critic_verifier_can_downgrade_unsupported_met_to_na():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
        },
        {
            "outcome": "overturn",
            "requirement_reviews": [
                {
                    "requirement_id": "D-1",
                    "hunter_verdict": "met",
                    "critic_verdict": "met",
                    "disposition": "overturn",
                    "supports_met": False,
                    "supports_not_met": False,
                    "failure_mode": "weak_positive_evidence",
                    "verification_checks": [
                        {"question": "Is the required security control explicitly shown?", "answer": "unclear", "evidence": "No explicit label was found."}
                    ],
                    "reason": "The visible evidence is too weak to support a met verdict.",
                }
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["assessed_requirements"][0]["verdict"] == "na"
    assert result["final_verdict"] == "na"
    assert result["assessed_requirements"][0]["verdict_policy_source"] == "diagram_met_not_supported_by_critic_verifier"


def test_partial_compound_met_is_downgraded_to_not_met_when_scope_is_shown():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
        },
        {
            "outcome": "overturn",
            "requirement_reviews": [
                {
                    "requirement_id": "D-1",
                    "hunter_verdict": "met",
                    "critic_verdict": "met",
                    "disposition": "overturn",
                    "supports_met": False,
                    "supports_not_met": True,
                    "failure_mode": "partial_compound",
                    "scope_evidence": "Components and data flows are visible, but no trust boundary is shown.",
                    "absence_evidence": "The full diagram was inspected and no trust boundary or trust-zone annotation is shown.",
                    "compound_subelements_checked": [
                        "trust boundaries: absent",
                        "components: present",
                        "significant data flows: present",
                    ],
                    "verification_checks": [
                        {"question": "Are trust boundaries visibly shown?", "answer": "absent", "evidence": "No dashed boundary or labeled trust zone is visible."},
                        {"question": "Are components labeled?", "answer": "present", "evidence": "Multiple components are labeled."},
                    ],
                    "reason": "The compound requirement is only partially satisfied because trust boundaries are missing.",
                }
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["assessed_requirements"][0]["verdict"] == "not_met"
    assert result["final_verdict"] == "not_met"
    assert result["assessed_requirements"][0]["verdict_policy_source"] == "diagram_met_rejected_by_critic_verifier"


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


def test_critic_requirement_review_can_correct_met_to_not_met():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
        },
        {
            "outcome": "overturn",
            "requirement_reviews": [
                {
                    "requirement_id": "D-1",
                    "hunter_verdict": "met",
                    "critic_verdict": "not_met",
                    "disposition": "overturn",
                    "scope_evidence": "External-facing gateway and internet-facing request path are visible.",
                    "absence_evidence": "The full governed path was inspected and no control element is shown.",
                    "reason": "The governed boundary is shown, but the expected control is absent.",
                }
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["assessed_requirements"][0]["verdict"] == "not_met"
    assert result["final_verdict"] == "not_met"
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
            "requirement_reviews": [
                {
                    "requirement_id": "D-3",
                    "hunter_verdict": "not_met",
                    "critic_verdict": "not_met",
                    "disposition": "uphold",
                    "scope_evidence": "The governed integration path is visible, but the control is absent.",
                    "absence_evidence": "The full integration path was inspected and no control component is shown.",
                    "reason": "The visible integration path lacks the required control.",
                }
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
        },
    )

    by_id = {a["requirement_id"]: a["verdict"] for a in result["assessed_requirements"]}
    assert by_id["D-1"] == "met"
    assert by_id["D-2"] == "met"
    assert by_id["D-3"] == "not_met"
    assert result["final_verdict"] == "not_met"


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
            "requirement_reviews": [
                {
                    "requirement_id": "D-1",
                    "hunter_verdict": "met",
                    "critic_verdict": "met",
                    "disposition": "uphold",
                    "reason": "Visible control evidence is present.",
                },
                {
                    "requirement_id": "D-2",
                    "hunter_verdict": "not_met",
                    "critic_verdict": "not_met",
                    "disposition": "uphold",
                    "scope_evidence": "The governed component and flow are visible, but the control is absent.",
                    "absence_evidence": "The governed scope was inspected and no control representation is visible.",
                    "reason": "The required control is not shown on the visible governed scope.",
                },
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
        },
    )

    assert result["final_verdict"] == "not_met"
    assert result["verdict_policy_source"] == "diagram_verdict_aggregated_from_assessments"


def test_critic_not_met_without_absence_evidence_is_downgraded_to_na():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "not_met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "not_met"}],
        },
        {
            "outcome": "overturn",
            "requirement_reviews": [
                {
                    "requirement_id": "D-1",
                    "hunter_verdict": "na",
                    "critic_verdict": "not_met",
                    "disposition": "overturn",
                    "scope_evidence": "The external API gateway path is visible.",
                    "reason": "The control is not shown.",
                }
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["assessed_requirements"][0]["verdict"] == "na"
    assert result["assessed_requirements"][0]["verdict_policy_source"] == "diagram_mediator_not_met_without_absence_evidence"
    assert result["final_verdict"] == "na"


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


def test_mixed_met_and_na_assessments_aggregate_to_met():
    # Diagram findings now roll up as:
    # any not_met => not_met; else any met => met; else na.
    # So a met+na mix must persist as met.
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

    assert result["final_verdict"] == "met"


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


def test_single_critic_non_architecture_dissent_does_not_force_na_when_validated_requirements_exist():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "outcome": "uphold",
            "validated_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
            "diagram_scope_verdict": "non_architecture",
            "diagram_scope_reasoning": "This may be a screenshot.",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
        },
    )

    assert result["final_verdict"] == "met"
    assert result["diagram_scope_verdict"] == "architecture_relevant"


def test_missing_mediator_requirement_entries_preserve_hunter_assessments():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "met"}],
        },
        {
            "outcome": "overturn",
            "validated_requirements": [],
            "invalidated_requirements": [],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {
            "diagram_scope_verdict": "architecture_relevant",
            "requirement_assessments": [
                {"requirement_id": "D-1", "verdict": "met"},
                {"requirement_id": "D-2", "verdict": "not_met"},
            ],
        },
    )

    by_id = {a["requirement_id"]: a["verdict"] for a in result["assessed_requirements"]}
    assert by_id["D-1"] == "met"
    assert by_id["D-2"] == "not_met"
    assert result["final_verdict"] == "not_met"


def test_missing_critic_requirement_review_does_not_downgrade_not_met():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "not_met",
            "assessed_requirements": [{"requirement_id": "D-1", "verdict": "not_met"}],
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

    assert result["assessed_requirements"][0]["verdict"] == "not_met"
    assert result["final_verdict"] == "not_met"


def test_mediator_judged_not_met_is_not_overwritten_by_critic_met():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "not_met",
            "assessed_requirements": [
                {
                    "requirement_id": "D-1",
                    "verdict": "not_met",
                    "resolution_basis": "mediator_tiebreak",
                    "winning_side": "critic",
                    "judge_reason": "The judge accepted the Critic's scope evidence and rejected the Hunter's met claim.",
                }
            ],
        },
        {
            "outcome": "overturn",
            "requirement_reviews": [
                {
                    "requirement_id": "D-1",
                    "hunter_verdict": "met",
                    "critic_verdict": "not_met",
                    "disposition": "overturn",
                    "scope_evidence": "The protected flow is visible and lacks the required control.",
                    "reason": "The Critic established visible governed scope and no control on that path.",
                }
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["assessed_requirements"][0]["verdict"] == "not_met"
    assert result["final_verdict"] == "not_met"
    assert result["assessed_requirements"][0]["final_decision_source"] == "mediator_tiebreak_to_critic"


def test_mediator_judged_met_is_not_forced_back_to_critic_na():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [
                {
                    "requirement_id": "D-1",
                    "verdict": "met",
                    "resolution_basis": "mediator_tiebreak",
                    "winning_side": "hunter",
                    "judge_reason": "The judge accepted the Hunter rebuttal as satisfying the missing subclaim.",
                }
            ],
        },
        {
            "outcome": "overturn",
            "requirement_reviews": [
                {
                    "requirement_id": "D-1",
                    "hunter_verdict": "met",
                    "critic_verdict": "na",
                    "disposition": "overturn",
                    "supports_met": False,
                    "supports_not_met": False,
                    "failure_mode": "weak_positive_evidence",
                    "verification_checks": [
                        {"question": "Was the control explicitly labeled?", "answer": "unclear", "evidence": "The label was initially missed."}
                    ],
                    "reason": "The Critic thought the evidence was too weak.",
                }
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["assessed_requirements"][0]["verdict"] == "met"
    assert result["final_verdict"] == "met"
    assert result["assessed_requirements"][0]["final_decision_source"] == "mediator_tiebreak_to_hunter"


def test_same_verdict_critic_explanation_still_counts_as_hunter_preserved():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "met",
            "assessed_requirements": [
                {
                    "requirement_id": "D-1",
                    "verdict": "met",
                    "resolution_basis": "same_verdict_after_cross_exam",
                    "winning_side": "critic",
                    "judge_reason": "The Critic improved the rationale but did not disprove the met control.",
                }
            ],
        },
        {
            "outcome": "uphold",
            "requirement_reviews": [
                {
                    "requirement_id": "D-1",
                    "hunter_verdict": "met",
                    "critic_verdict": "met",
                    "disposition": "uphold",
                    "reason": "The visible control remains supported.",
                }
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["assessed_requirements"][0]["verdict"] == "met"
    assert result["assessed_requirements"][0]["final_decision_source"] == "hunter_preserved"


def test_unsupported_not_met_downgrade_is_labeled_as_policy_na():
    result = _apply_diagram_evidence_policy(
        {
            "final_verdict": "not_met",
            "assessed_requirements": [
                {
                    "requirement_id": "D-1",
                    "verdict": "not_met",
                    "resolution_basis": "mediator_tiebreak",
                    "winning_side": "critic",
                    "judge_reason": "The judge accepted the Critic's absence claim.",
                }
            ],
        },
        {
            "outcome": "overturn",
            "requirement_reviews": [
                {
                    "requirement_id": "D-1",
                    "hunter_verdict": "met",
                    "critic_verdict": "not_met",
                    "disposition": "overturn",
                    "scope_evidence": "",
                    "reason": "The Critic named no scope evidence.",
                }
            ],
            "diagram_scope_verdict": "architecture_relevant",
        },
        {"diagram_scope_verdict": "architecture_relevant"},
    )

    assert result["assessed_requirements"][0]["verdict"] == "na"
    assert result["assessed_requirements"][0]["final_decision_source"] == "policy_downgraded_to_na"


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
