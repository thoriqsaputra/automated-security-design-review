from sdr.apps.ai.engine.persistence.persistence_service import PersistenceService
from sdr.apps.ai.agents.base import Citation
from sdr.apps.reviews.models import CitationAnchor, Finding
from sdr.apps.reviews.models.choices import FindingType


def test_diagram_not_met_recommendation_uses_generator_when_missing():
    captured = {}

    def recommendation_generator(**kwargs):
        captured.update(kwargs)
        return "Add the missing trust boundary and label the authenticated flow in the diagram."

    service = PersistenceService(recommendation_generator=recommendation_generator)

    recommendation = service._ensure_not_met_recommendation(
        finding_type=FindingType.DIAGRAM.value,
        met_status="not_met",
        recommendation=None,
        parameter_section="Authentication",
        parameter_text="Show the authentication boundary and protected data flow.",
        finding_description="The diagram does not show the boundary.",
        reasoning="The architecture view omits the control.",
        severity="high",
        source="diagram_debate",
        source_map={},
    )

    assert recommendation == "Add the missing trust boundary and label the authenticated flow in the diagram."
    assert captured["finding_type"] == FindingType.DIAGRAM.value
    assert captured["source"] == "diagram_debate"


def test_diagram_not_met_recommendation_falls_back_to_default_when_generator_returns_empty():
    service = PersistenceService(recommendation_generator=lambda **_: "   ")

    recommendation = service._ensure_not_met_recommendation(
        finding_type=FindingType.DIAGRAM.value,
        met_status="not_met",
        recommendation="",
        parameter_section="Network Security",
        parameter_text="Show the protected network segment and trust boundary.",
        finding_description="The relevant segment is missing.",
        reasoning="The diagram omits the required visual control.",
        severity="medium",
        source="diagram_debate",
        source_map={},
    )

    assert recommendation is not None
    assert "Update the TSD and diagrams" in recommendation
    assert "Network Security" in recommendation


def test_existing_diagram_not_met_recommendation_is_preserved():
    service = PersistenceService(recommendation_generator=lambda **_: "should not be used")

    recommendation = service._ensure_not_met_recommendation(
        finding_type=FindingType.DIAGRAM.value,
        met_status="not_met",
        recommendation="Add the missing encryption marker to the data flow.",
        parameter_section="Transport Security",
        parameter_text="Show encrypted traffic between services.",
        finding_description="Encryption is not visible.",
        reasoning="The connection is unlabeled.",
        severity="medium",
        source="diagram_debate",
        source_map={},
    )

    assert recommendation == "Add the missing encryption marker to the data flow."


def test_diagram_recommendation_is_cleared_for_non_not_met_status():
    service = PersistenceService(recommendation_generator=lambda **_: "should not be used")

    assert service._ensure_not_met_recommendation(
        finding_type=FindingType.DIAGRAM.value,
        met_status="met",
        recommendation="Keep this",
        parameter_section="Transport Security",
        parameter_text="Show encrypted traffic between services.",
        finding_description="Encryption is visible.",
        reasoning="The control is present.",
        severity=None,
        source="diagram_debate",
        source_map={},
    ) is None

    assert service._ensure_not_met_recommendation(
        finding_type=FindingType.DIAGRAM.value,
        met_status="na",
        recommendation="Keep this",
        parameter_section="Transport Security",
        parameter_text="Show encrypted traffic between services.",
        finding_description="Not applicable.",
        reasoning="The diagram does not cover this area.",
        severity=None,
        source="diagram_debate",
        source_map={},
    ) is None


def test_citation_source_map_includes_retrieval_origin():
    service = PersistenceService()

    source_map = service._build_citation_source_map(
        [
            Citation(
                block_id="p2_b4",
                page_number=2,
                quoted_text="mTLS is enforced",
            )
        ],
        {
            "context_chunk_map": {
                "p2_b4": {
                    "section": "Transport Security",
                    "retrieval_origin": "graph",
                    "retrieval_origin_label": "Graph",
                }
            }
        },
    )

    assert source_map["p2_b4"]["retrieval_origin"] == "graph"
    assert source_map["p2_b4"]["retrieval_origin_label"] == "Graph"


def test_finding_and_citation_expose_evidence_source_provenance():
    finding = Finding(
        title="Transport Security",
        description="mTLS is not shown.",
        requirement_metadata={
            "structured_citations": [
                {
                    "chunk_id": "p2_b4",
                    "retrieval_origin": "graph",
                    "retrieval_origin_label": "Graph",
                }
            ]
        },
    )
    citation = CitationAnchor(
        block_id="p2_b4",
        page_number=2,
        anchor_type="text",
    )
    citation.finding = finding
    finding.citations = [citation]

    assert finding.has_citations is True
    assert finding.citation_count == 1
    assert finding.evidence_sources == [{"key": "graph", "label": "Graph", "count": 1}]
    assert citation.retrieval_origin == "graph"
    assert citation.retrieval_origin_label == "Graph"
