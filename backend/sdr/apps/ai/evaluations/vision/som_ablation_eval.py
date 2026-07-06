"""
Set-of-Mark (SoM) Ablation eval (real diagrams).

Tests whether Set-of-Mark visual marker annotation (numbered boxes/labels
burned into the diagram image before it reaches the LLM, see
`tsd_processing/visual_marker.py`) actually improves the diagram debate
(Hunter -> Critic -> Mediator), rather than just measuring how often the
Hunter cites a marker once one exists (that's `grounding_eval.py`'s job).

Methodology: uses the same labeled `diagram_ground_truth_review_<id>.json`
file as `blindness_eval.py`/`grounding_eval.py` (see
evaluations/data/build_diagram_ground_truth_template.py). For each labeled,
relevant (diagram, requirement) pair, run the real
DiagramDebateService.run_diagram_debate twice on independent DiagramInput
copies of the SAME real diagram image (caption/surrounding_text left
intact — unlike blindness_eval.py, this isn't testing vision-only
isolation, it's testing marker-on vs marker-off, so the real text context
production would normally see is kept):
  - "som"  condition: apply_markers=True  (production default)
  - "raw"  condition: apply_markers=False (SoM annotation skipped)

Same requirements, same TSD context (empty), same debate pipeline — the only
variable is whether the image pixels carry SoM markers. Confusion matrices
and FPR are computed per condition via the shared binary-confusion helper,
and delta_fpr / delta_accuracy quantify whether SoM measurably helps.

`visual_marker.apply_visual_markers` fails open (returns the original bytes
unchanged) if Tesseract OCR is unavailable or errors — this eval guards
against silently comparing "raw vs raw" by warning if the two conditions'
image bytes for a sample are byte-identical.

Usage:
    python som_ablation_eval.py \\
        --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_review_48.json \\
        [--output som_ablation_results.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from sdr.apps.ai.agents.vision import DiagramInput
from sdr.apps.ai.engine.debate.diagram_debate_service import DiagramDebateService
from sdr.apps.ai.evaluations.shared import results_path
from sdr.apps.ai.evaluations.shared.metrics import calculate_binary_confusion
from sdr.apps.ai.evaluations.vision.real_diagram_source import (
    build_diagram_input,
    load_ground_truth,
    load_labeled_samples,
    load_tsd_document,
    save_image_b64,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"\[\d+\]")


def _has_marker_ref(hunter_result: dict) -> bool:
    assessments = (hunter_result or {}).get("requirement_assessments") or []
    if any(
        _MARKER_RE.search(a.get("visual_evidence") or "") or _MARKER_RE.search(a.get("reasoning") or "")
        for a in assessments
    ):
        return True
    return bool((hunter_result or {}).get("marker_ids_cited"))


def _run_condition(service: DiagramDebateService, diagram: DiagramInput, requirement, apply_markers: bool):
    output = service.run_diagram_debate(
        diagram=diagram, requirements=[requirement], tsd_context="", apply_markers=apply_markers
    )
    mediator = output.mediator_result or {}
    hunter = output.hunter_result or {}
    critic = output.critic_result or {}
    return {
        "final_verdict": mediator.get("final_verdict"),
        "confidence": mediator.get("confidence"),
        "diagram_scope_verdict": mediator.get("diagram_scope_verdict"),
        "hunter_verdict": hunter.get("overall_verdict"),
        "critic_outcome": critic.get("outcome"),
        "has_marker_ref": _has_marker_ref(hunter),
        "image_b64": diagram.image_b64,
        "error": output.error,
    }


def main():
    parser = argparse.ArgumentParser(description="Set-of-Mark (SoM) ablation eval for the diagram debate (real diagrams).")
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to a labeled diagram ground-truth JSON (see build_diagram_ground_truth_template.py)"
    )
    parser.add_argument("--output", type=str, default="som_ablation_results.json")
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        return

    gt_data = load_ground_truth(args.ground_truth)
    design_id = gt_data.get("design_id")
    if design_id is None:
        logger.error("Ground truth file is missing 'design_id' — regenerate it with build_diagram_ground_truth_template.py.")
        return

    samples = load_labeled_samples(gt_data, labels=("met", "not_met"))
    if not samples:
        logger.error("No (diagram, requirement) rows labeled met/not_met found in ground truth.")
        return
    logger.info(f"Loaded {len(samples)} labeled (diagram, requirement) samples")

    tsd_doc = load_tsd_document(design_id)

    images_dir = results_path("images", subdir="vision")
    os.makedirs(images_dir, exist_ok=True)

    service = DiagramDebateService()

    per_item = []
    disagreement_cases = []
    identical_bytes_warnings = 0

    for i, sample in enumerate(samples):
        sample_id = f"som_{sample.diagram_id}_{sample.requirement_id}"
        expected_verdict = sample.label

        logger.info("[%d/%d] %s (expecting %s)", i + 1, len(samples), sample_id, expected_verdict)

        som_diagram = build_diagram_input(tsd_doc, sample.diagram_id, f"{sample_id}_som", blank_text=False)
        raw_diagram = build_diagram_input(tsd_doc, sample.diagram_id, f"{sample_id}_raw", blank_text=False)
        if som_diagram is None or raw_diagram is None:
            logger.warning(f"  {sample_id}: diagram not found or invalid in prepared TSD — skipping")
            continue

        requirement = SimpleNamespace(
            ordinal=1,
            stable_key=sample.requirement_id,
            requirement_text=sample.requirement_text,
            verification_hint=sample.verification_hint,
        )

        som_run = _run_condition(service, som_diagram, requirement, apply_markers=True)
        raw_run = _run_condition(service, raw_diagram, requirement, apply_markers=False)

        if som_run["image_b64"] == raw_run["image_b64"]:
            identical_bytes_warnings += 1
            logger.warning(
                "%s: SoM and raw condition images are byte-identical — "
                "apply_visual_markers may have failed open (Tesseract missing/erroring).",
                sample_id,
            )

        save_image_b64(raw_run["image_b64"], os.path.join(images_dir, f"{sample_id}_raw.png"))
        save_image_b64(som_run["image_b64"], os.path.join(images_dir, f"{sample_id}_marked.png"))

        som_correct = som_run["final_verdict"] == expected_verdict
        raw_correct = raw_run["final_verdict"] == expected_verdict
        verdict_changed = som_run["final_verdict"] != raw_run["final_verdict"]

        row = {
            "sample_id": sample_id,
            "diagram_id": sample.diagram_id,
            "requirement_id": sample.requirement_id,
            "expected_verdict": expected_verdict,
            "som": {k: v for k, v in som_run.items() if k != "image_b64"},
            "raw": {k: v for k, v in raw_run.items() if k != "image_b64"},
            "som_correct": som_correct,
            "raw_correct": raw_correct,
            "verdict_changed_by_markers": verdict_changed,
        }
        per_item.append(row)

        if verdict_changed:
            disagreement_cases.append(row)

    if not per_item:
        logger.error("No samples produced a valid diagram — nothing to evaluate.")
        return

    som_labels = [r["expected_verdict"] for r in per_item]
    som_preds = [r["som"]["final_verdict"] for r in per_item]
    raw_labels = [r["expected_verdict"] for r in per_item]
    raw_preds = [r["raw"]["final_verdict"] for r in per_item]

    som_cm = calculate_binary_confusion(som_labels, som_preds)
    raw_cm = calculate_binary_confusion(raw_labels, raw_preds)

    som_accuracy = sum(1 for r in per_item if r["som_correct"]) / len(per_item)
    raw_accuracy = sum(1 for r in per_item if r["raw_correct"]) / len(per_item)

    som_confidences = [r["som"]["confidence"] for r in per_item if isinstance(r["som"]["confidence"], (int, float))]
    raw_confidences = [r["raw"]["confidence"] for r in per_item if isinstance(r["raw"]["confidence"], (int, float))]
    som_avg_confidence = sum(som_confidences) / len(som_confidences) if som_confidences else 0.0
    raw_avg_confidence = sum(raw_confidences) / len(raw_confidences) if raw_confidences else 0.0

    marker_cited = sum(1 for r in per_item if r["som"]["has_marker_ref"])
    marker_utilization_rate = marker_cited / len(per_item)

    delta_fpr = round(raw_cm["fpr"] - som_cm["fpr"], 4)
    delta_accuracy = round(som_accuracy - raw_accuracy, 4)
    delta_avg_confidence = round(som_avg_confidence - raw_avg_confidence, 4)

    summary = {
        "design_id": design_id,
        "total_samples": len(per_item),
        "som_final": {**som_cm, "accuracy": round(som_accuracy, 4), "average_confidence": round(som_avg_confidence, 4)},
        "raw_final": {**raw_cm, "accuracy": round(raw_accuracy, 4), "average_confidence": round(raw_avg_confidence, 4)},
        "delta_fpr": delta_fpr,
        "som_suppresses_fpr": delta_fpr >= 0,
        "delta_accuracy": delta_accuracy,
        "delta_avg_confidence": delta_avg_confidence,
        "marker_utilization_rate": round(marker_utilization_rate, 4),
        "verdict_changes": len(disagreement_cases),
        "identical_image_bytes_warnings": identical_bytes_warnings,
        "per_item_results": per_item,
        "disagreement_cases": disagreement_cases,
    }

    output_path = results_path(args.output, subdir="vision")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("SoM ablation eval complete.")
    logger.info(
        "SoM: accuracy=%.2f fpr=%.2f | Raw: accuracy=%.2f fpr=%.2f",
        som_accuracy, som_cm["fpr"], raw_accuracy, raw_cm["fpr"],
    )
    logger.info("delta_fpr=%.4f delta_accuracy=%.4f", delta_fpr, delta_accuracy)
    logger.info("%s", "SoM suppresses FPR ✓" if delta_fpr >= 0 else "SoM does not suppress FPR ✗")
    logger.info("marker_utilization_rate=%.2f", marker_utilization_rate)
    if identical_bytes_warnings:
        logger.warning(
            "%d/%d samples had identical SoM/raw image bytes — check Tesseract availability.",
            identical_bytes_warnings, len(per_item),
        )
    logger.info("Marked/raw images saved to %s/", images_dir)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
