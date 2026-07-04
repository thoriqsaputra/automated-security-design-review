from sdr.apps.ai.engine.extraction.normalizers import canonicalize_requirement_items


def _kept_requirements(items):
    return [item["requirement"] for item in canonicalize_requirement_items(items)]


def test_fused_chapter_letter_id_is_kept():
    items = [
        {
            "requirement": "V8.1.1 Verify the application protects sensitive data from being cached...",
            "context_marker": "V8.1.1",
            "requirement_category": "code",
        }
    ]

    kept = _kept_requirements(items)

    assert kept == ["V8.1.1 Verify the application protects sensitive data from being cached..."]


def test_title_between_marker_and_hyphen_is_kept():
    items = [
        {
            "requirement": (
                "V1.7 Errors, Logging and Auditing Architecture - 1.7.1 Verify that a "
                "common logging format and approach is used across the system. (C9)"
            ),
            "context_marker": "V1.7 Errors, Logging and Auditing Architecture",
            "requirement_category": "design",
        }
    ]

    kept = _kept_requirements(items)

    assert len(kept) == 1


def test_previously_working_formats_still_kept():
    items = [
        {
            "requirement": "V1.1 - 1.1.1 Verify the use of a secure SDLC that includes security in all stages.",
            "context_marker": "V1.1",
            "requirement_category": "process",
        },
        {
            "requirement": "5.4.1 Generic Web Service Security",
            "context_marker": "Section 5.4.1",
            "requirement_category": "design",
        },
        {
            "requirement": "REQ-INP-01 Verify that all input is validated.",
            "context_marker": "REQ-INP-01",
            "requirement_category": "code",
        },
        {
            "requirement": "PCI-3.4 Verify that cardholder data is rendered unreadable.",
            "context_marker": "PCI-3.4",
            "requirement_category": "code",
        },
    ]

    kept = _kept_requirements(items)

    assert len(kept) == 4


def test_non_items_are_still_skipped():
    items = [
        {
            "requirement": "Note: this is guidance without a control id.",
            "context_marker": "",
            "requirement_category": "design",
        },
        {
            "requirement": "V1.3 This is a placeholder for future architectural requirements.",
            "context_marker": "V1.3",
            "requirement_category": "design",
        },
        {
            "requirement": "Availability: Data should be available when needed.",
            "context_marker": "",
            "requirement_category": "design",
        },
    ]

    kept = _kept_requirements(items)

    assert kept == []
