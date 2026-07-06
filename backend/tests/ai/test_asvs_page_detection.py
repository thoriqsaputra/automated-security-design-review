from __future__ import annotations

from sdr.apps.ai.engine.extraction.page_detection import ASVSPageRangeDetectionService
from sdr.apps.standards.tasks import _resolve_detected_page_ranges


def _make_service() -> ASVSPageRangeDetectionService:
    return ASVSPageRangeDetectionService(get_local_file_path=lambda document: None)


def test_flag_low_confidence_requirement_range_below_threshold():
    service = _make_service()

    req_anchors, req_source = service._flag_low_confidence_requirement_range(
        req_start=10,
        req_end=15,
        req_anchors={"method": "heading_plus_appendix"},
        req_source="heuristic",
    )

    assert req_source == "heuristic_low_confidence"
    assert "warning" in req_anchors
    assert req_anchors["method"] == "heading_plus_appendix"


def test_flag_low_confidence_requirement_range_above_threshold_unchanged():
    service = _make_service()

    req_anchors, req_source = service._flag_low_confidence_requirement_range(
        req_start=17,
        req_end=63,
        req_anchors={"method": "heading_plus_appendix"},
        req_source="heuristic",
    )

    assert req_source == "heuristic"
    assert "warning" not in req_anchors


def test_resolve_detected_page_ranges_applies_manual_overrides_per_field():
    result = _resolve_detected_page_ranges(
        detected_ranges={
            "start_page": 23,
            "end_page": 95,
            "source": "toc",
            "matched_anchors": {"requirement_range": {"method": "toc_v1_plus_appendix"}},
        },
        requested_ranges={
            "start_page": None,
            "end_page": 90,
        },
    )

    assert result["effective"] == {
        "start_page": 23,
        "end_page": 90,
    }
    assert result["page_detection"]["field_sources"] == {
        "start_page": "auto_detected",
        "end_page": "manual_override",
    }
    assert result["page_detection"]["matched_anchors"] == {
        "requirement_range": {"method": "toc_v1_plus_appendix"}
    }


def test_end_page_from_appendix_excludes_appendix_start():
    service = _make_service()
    # appendix at page 97 → last requirements page is 96
    end_page = service._end_page_from_appendix(start_page=23, appendix_page=97)
    assert end_page == 96


def test_end_page_from_appendix_clamps_to_start_page():
    service = _make_service()
    # appendix detected before start (bad detection) → clamp to start
    end_page = service._end_page_from_appendix(start_page=23, appendix_page=10)
    assert end_page == 23


def test_end_page_from_appendix_none_returns_none():
    service = _make_service()
    assert service._end_page_from_appendix(start_page=23, appendix_page=None) is None
