from sdr.apps.ai.engine.classification.severity import calculate_deterministic_severity


def test_non_not_met_findings_have_no_severity():
    result = calculate_deterministic_severity(
        met_status="met",
        confidence_score=0.95,
        domain="iam_access_control",
        finding_type="requirement",
    )

    assert result.severity is None
    assert result.score is None
    assert result.analysis is None


def test_general_low_context_not_met_stays_medium():
    result = calculate_deterministic_severity(
        met_status="not_met",
        confidence_score=0.6,
        domain="general",
        finding_type="requirement",
        requirement_text="Document operational monitoring.",
        citation_count=1,
    )

    assert result.severity == "medium"
    assert result.score == 4.6
    assert result.analysis["version"] == "v2"


def test_high_impact_admin_auth_finding_can_be_critical():
    result = calculate_deterministic_severity(
        met_status="not_met",
        confidence_score=0.95,
        domain="iam_access_control",
        finding_type="requirement",
        requirement_text="MFA and token protection are missing for admin authentication.",
        analysis_trace={
            "contract": {
                "required_capabilities": ["admin", "auth", "secrets"],
                "optional_capabilities": ["api"],
            }
        },
        citation_count=2,
    )

    assert result.severity == "critical"
    assert result.score >= 9.0
    assert result.analysis["dimensions"]["impact_capabilities"]["matched"] == [
        "admin",
        "auth",
        "secrets",
    ]


def test_partial_requirement_verdict_is_capped_at_medium():
    result = calculate_deterministic_severity(
        met_status="not_met",
        confidence_score=0.95,
        domain="transaction_integrity",
        finding_type="requirement",
        raw_final_verdict="partial",
        requirement_text="Payment authorization and transaction integrity are partially documented.",
        analysis_trace={"contract": {"required_capabilities": ["transaction", "api"]}},
        citation_count=1,
    )

    assert result.severity == "medium"
    assert result.score == 6.0
    assert any(cap["kind"] == "partial_verdict_cap" and cap["applied"] for cap in result.analysis["caps"])


def test_requirement_without_verified_citations_is_capped_at_medium():
    result = calculate_deterministic_severity(
        met_status="not_met",
        confidence_score=0.95,
        domain="iam_access_control",
        finding_type="requirement",
        requirement_text="Admin MFA, token, and secret handling evidence is missing.",
        analysis_trace={"contract": {"required_capabilities": ["admin", "auth", "secrets", "api"]}},
        citation_count=0,
    )

    assert result.severity == "medium"
    assert result.score == 6.5
    assert any(cap["kind"] == "zero_verified_citations_cap" and cap["applied"] for cap in result.analysis["caps"])


def test_diagram_ambiguity_is_capped_at_medium():
    result = calculate_deterministic_severity(
        met_status="not_met",
        confidence_score=0.95,
        domain="architecture_network",
        finding_type="diagram",
        requirement_text="Network trust boundaries must be explicit.",
        ambiguous_elements=["unlabelled trust boundary"],
        missing_information=[],
    )

    assert result.severity == "medium"
    assert result.score == 6.5
    assert any(cap["kind"] == "diagram_ambiguity_cap" and cap["applied"] for cap in result.analysis["caps"])
