from __future__ import annotations

from sdr.apps.standards.tasks import _resolve_detected_page_ranges


def test_resolve_detected_page_ranges_applies_manual_overrides_per_field():
    result = _resolve_detected_page_ranges(
        detected_ranges={
            "start_page": 23,
            "end_page": 95,
            "level_definition_start_page": 12,
            "level_definition_end_page": 14,
            "source": "toc",
            "matched_anchors": {"definition_range": {"method": "toc_section"}},
        },
        requested_ranges={
            "start_page": None,
            "end_page": 90,
            "level_definition_start_page": 10,
            "level_definition_end_page": None,
        },
    )

    assert result["effective"] == {
        "start_page": 23,
        "end_page": 90,
        "level_definition_start_page": 10,
        "level_definition_end_page": 14,
    }
    assert result["page_detection"]["field_sources"] == {
        "start_page": "auto_detected",
        "end_page": "manual_override",
        "level_definition_start_page": "manual_override",
        "level_definition_end_page": "auto_detected",
    }
    assert result["page_detection"]["matched_anchors"] == {
        "definition_range": {"method": "toc_section"}
    }
