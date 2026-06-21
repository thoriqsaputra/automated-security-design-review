from types import SimpleNamespace

import sdr.apps.designs.models  # noqa: F401  (registers Design mapper used by Review relationship)
from sdr.apps.ai.engine.debate.category_analysis_coordinator import (
    CategoryAnalysisCoordinator,
)
from sdr.apps.standards.models.parameters import CategoryParameterChild


def _child(stable_key: str) -> CategoryParameterChild:
    return CategoryParameterChild(
        id=1,
        parent_id=1,
        stable_key=stable_key,
        requirement_text="Access control enforcement must occur server-side.",
        requirement_text_normalized="access control enforcement must occur server-side.",
        ordinal=0,
    )


def _anchor(block_id: str, page_number: int = 3, quoted_text: str = "evidence text"):
    return SimpleNamespace(
        block_id=block_id,
        page_number=page_number,
        quoted_text=quoted_text,
        bbox_x0=1.0,
        bbox_y0=2.0,
        bbox_x1=3.0,
        bbox_y1=4.0,
    )


def test_build_cascade_debate_output_propagates_parent_citations():
    cfsr_finding = SimpleNamespace(
        reason="The TSD lacks server-side enforcement evidence.",
        description="",
        confidence_score=0.8,
        severity="high",
        recommendation="Add server-side enforcement.",
        citations=[_anchor("p3_b12"), _anchor("p5_b1", page_number=5)],
    )
    cfsr = SimpleNamespace(stable_key="CFSR-V4-1")
    child = _child("child-042")

    output = CategoryAnalysisCoordinator._build_cascade_debate_output(
        child=child,
        cfsr=cfsr,
        cfsr_finding=cfsr_finding,
    )

    final_citations = output.mediator_result.final_citations
    assert [c.block_id for c in final_citations] == ["p3_b12", "p5_b1"]
    assert all(c.quoted_text == "evidence text" for c in final_citations)
    assert [c.block_id for c in output.hunter_result.citations] == ["p3_b12", "p5_b1"]
    assert [c.block_id for c in output.critic_result.valid_citations] == ["p3_b12", "p5_b1"]


def test_build_cascade_debate_output_handles_no_parent_citations():
    cfsr_finding = SimpleNamespace(
        reason="not met",
        description="",
        confidence_score=0.8,
        severity="medium",
        recommendation=None,
        citations=[],
    )
    cfsr = SimpleNamespace(stable_key="CFSR-V4-2")
    child = _child("child-007")

    output = CategoryAnalysisCoordinator._build_cascade_debate_output(
        child=child,
        cfsr=cfsr,
        cfsr_finding=cfsr_finding,
    )

    assert output.mediator_result.final_citations == []
