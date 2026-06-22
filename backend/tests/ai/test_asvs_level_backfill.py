from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

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
