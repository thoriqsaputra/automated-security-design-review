from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from sdr.apps.ai.engine.extraction import (
    _backfill_requirement_levels,
    _merge_requirements,
)
from sdr.apps.ai.engine.extraction.page_detection import ASVSRequirementLevelDetectionService


def _detector() -> ASVSRequirementLevelDetectionService:
    return ASVSRequirementLevelDetectionService(
        get_local_file_path=lambda document: nullcontext(document),
    )


def test_detect_asvs_5_requirement_levels_from_pdf_geometry():
    repo_root = Path(__file__).resolve().parents[3]
    pdf_path = repo_root / "dataset/Standard/OWASP_Application_Security_Verification_Standard_5.0.0_en.pdf"

    result = _detector().detect(
        SimpleNamespace(document=str(pdf_path)),
        start_page=23,
        end_page=95,
    )

    assert result.levels["1.1.1"] == 2
    assert result.levels["1.2.2"] == 1
    assert result.levels["2.4.2"] == 3


def test_detect_asvs_4_requirement_levels_from_pdf_geometry():
    repo_root = Path(__file__).resolve().parents[3]
    pdf_path = repo_root / "dataset/Standard/OWASP Application Security Verification Standard 4.0.3-en.pdf"

    result = _detector().detect(
        SimpleNamespace(document=str(pdf_path)),
        start_page=17,
        end_page=63,
    )

    assert result.levels["1.1.1"] == 2
    assert result.levels["1.10.1"] == 2
    assert result.levels["2.2.4"] == 3


def test_backfill_requirement_levels_sets_missing_asvs_levels_by_logical_id():
    requirements = {
        "V1 Architecture": [
            {
                "requirement": "V1.1 - 1.1.1 Verify secure SDLC",
                "details": "Use a secure SDLC.",
                "asvs_level": None,
            },
            {
                "requirement": "V1.2 - 1.2.1 Verify component auth",
                "details": "Authenticate components.",
                "asvs_level": 1,
            },
        ]
    }

    backfilled = _backfill_requirement_levels(
        requirements,
        {
            "1.1.1": 2,
            "1.2.1": 3,
        },
    )

    assert backfilled == 1
    assert requirements["V1 Architecture"][0]["asvs_level"] == 2
    assert requirements["V1 Architecture"][1]["asvs_level"] == 1


def test_merge_requirements_preserves_known_level_when_longer_duplicate_lacks_it():
    base = {
        "V1 Architecture": [
            {
                "requirement": "V1.1 - 1.1.1 Verify secure SDLC",
                "details": "Short detail.",
                "asvs_level": 2,
                "verbatim_quote": "1.1.1 Verify secure SDLC",
                "context_marker": "V1.1",
            }
        ]
    }
    incoming = {
        "V1 Architecture": [
            {
                "requirement": "V1.1 - 1.1.1 Verify secure SDLC",
                "details": "Longer detail that should replace the shorter version during merge.",
                "asvs_level": None,
                "verbatim_quote": "",
                "context_marker": "",
            }
        ]
    }

    result = _merge_requirements(base, incoming)
    item = result["V1 Architecture"][0]

    assert item["details"].startswith("Longer detail")
    assert item["asvs_level"] == 2
    assert item["verbatim_quote"] == "1.1.1 Verify secure SDLC"
