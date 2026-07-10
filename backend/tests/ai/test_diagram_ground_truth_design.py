from __future__ import annotations

import json
from pathlib import Path

from sdr.apps.ai.evaluations.shared import diagram_ground_truth as gt
from sdr.apps.ai.evaluations.vision.real_diagram_source import load_labeled_samples


def test_diagram_ground_truth_filename_uses_design_scope():
    assert gt.diagram_ground_truth_filename(15) == "diagram_ground_truth_design_15.json"
    assert gt.diagram_ground_truth_filename(14, llm_judged=True) == "diagram_ground_truth_design_14_llm_judged.json"


def test_resolve_diagram_ground_truth_path_prefers_canonical(tmp_path, monkeypatch):
    canonical = tmp_path / "diagram_ground_truth_design_15.json"
    llm_judged = tmp_path / "diagram_ground_truth_design_15_llm_judged.json"
    canonical.write_text("{}")
    llm_judged.write_text("{}")
    monkeypatch.setattr(gt, "data_path", lambda filename: str(tmp_path / filename))

    assert gt.resolve_diagram_ground_truth_path(15) == str(canonical)


def test_design_scoped_ground_truth_has_no_review_only_fields():
    path = Path("backend/sdr/apps/ai/evaluations/data/diagram_ground_truth_design_15.json")
    data = json.loads(path.read_text())

    assert data["design_id"] == 15
    assert "review_id" not in data
    for item in data["items"]:
        assert "finding_id" not in item
        assert "system_assessed_requirement_ids" not in item
        for req in item["candidate_requirements"]:
            assert "was_assessed_by_system" not in req


def test_load_labeled_samples_reads_design_scoped_ground_truth():
    path = Path("backend/sdr/apps/ai/evaluations/data/diagram_ground_truth_design_15.json")
    data = json.loads(path.read_text())
    samples = load_labeled_samples(data)

    assert samples
    assert {sample.label for sample in samples} <= {"met", "not_met"}
