"""
Visual Blindness Mitigation eval (real diagrams).

Tests whether the Vision Agent (Hunter -> Critic -> Mediator) can detect a
security control's presence/absence purely from a REAL diagram image pulled
from a design's parsed TSD document, with caption/surrounding_text blanked
out to isolate the vision channel from the text channel — a real diagram's
caption may otherwise describe the very control under test, leaking the
answer.

Ground truth is the same labeled `diagram_ground_truth_review_<id>.json`
file used by retrieval/diagram_retrieval_eval.py and
debate/diagram_ablation_eval.py (see
evaluations/data/build_diagram_ground_truth_template.py): for each
(diagram, requirement) row labeled relevant=true, "met" is treated as the
"control present" class and "not_met" as the "control absent" class. This
replaces the previous synthetic present/absent-pair methodology (drawing
the same diagram twice with a control box included/omitted), which cannot
be done on a real, already-rendered diagram — design_id is read directly
from the ground-truth file.

Usage:
    python blindness_eval.py \\
        --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_review_48.json
"""
import argparse
import json
import logging
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

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


def main():
    parser = argparse.ArgumentParser(description="Visual blindness mitigation eval (real diagrams) for the Vision Agent.")
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to a labeled diagram ground-truth JSON (see build_diagram_ground_truth_template.py)"
    )
    parser.add_argument("--output", type=str, default="eval_vision_blindness.json")
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
    results = []

    for i, sample in enumerate(samples):
        sample_id = f"vb_{sample.diagram_id}_{sample.requirement_id}"
        logger.info(f"[{i + 1}/{len(samples)}] {sample_id} (expecting {sample.label})")

        diagram = build_diagram_input(tsd_doc, sample.diagram_id, sample_id, blank_text=True)
        if diagram is None:
            logger.warning(f"  {sample_id}: diagram not found or invalid in prepared TSD — skipping")
            continue

        save_image_b64(diagram.image_b64, os.path.join(images_dir, f"{sample_id}_input.png"))

        requirement = SimpleNamespace(
            ordinal=1,
            stable_key=sample.requirement_id,
            requirement_text=sample.requirement_text,
            verification_hint=sample.verification_hint,
        )
        output = service.run_diagram_debate(diagram=diagram, requirements=[requirement], tsd_context="")

        # diagram.image_b64 is updated to the marked version inside run_diagram_debate
        save_image_b64(diagram.image_b64, os.path.join(images_dir, f"{sample_id}_marked.png"))

        expected_verdict = sample.label
        mediator = output.mediator_result or {}
        actual_verdict = mediator.get("final_verdict")
        correct = actual_verdict == expected_verdict

        results.append({
            "diagram_id": sample.diagram_id,
            "requirement_id": sample.requirement_id,
            "sample_id": sample_id,
            "expected_verdict": expected_verdict,
            "actual_verdict": actual_verdict,
            "correct": correct,
            "confidence": mediator.get("confidence"),
            "diagram_scope_verdict": mediator.get("diagram_scope_verdict"),
            "finding_description": mediator.get("finding_description"),
            "error": output.error,
            "hunter_verdict": (output.hunter_result or {}).get("overall_verdict"),
            "critic_outcome": (output.critic_result or {}).get("outcome"),
        })

    if not results:
        logger.error("No samples produced a valid diagram — nothing to evaluate.")
        return

    # Detection accuracy/precision/recall: positive class = correctly flagging
    # an absent control as not_met (the "blindness mitigation" signal).
    labels = [r["expected_verdict"] for r in results]
    preds = [r["actual_verdict"] for r in results]
    cm = calculate_binary_confusion(labels, preds, positive_label="not_met")

    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    confidences = [r["confidence"] for r in results if isinstance(r["confidence"], (int, float))]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    in_scope_rate = sum(
        1 for r in results if r["diagram_scope_verdict"] == "architecture_relevant"
    ) / len(results)

    summary = {
        "design_id": design_id,
        "total_samples": len(results),
        "accuracy": round(accuracy, 4),
        "precision": cm["precision"],
        "recall": cm["recall"],
        "f1": cm["f1"],
        "average_confidence": round(avg_confidence, 4),
        "architecture_relevant_rate": round(in_scope_rate, 4),
        "confusion": {"tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"]},
        "results": results,
    }

    output_path = results_path(args.output, subdir="vision")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Vision blindness eval complete.")
    logger.info(f"Accuracy={accuracy:.2f} Precision={cm['precision']:.2f} Recall={cm['recall']:.2f} F1={cm['f1']:.2f}")
    logger.info(f"Architecture-relevant scope rate: {in_scope_rate:.2f}")
    logger.info(f"Marked images saved to {images_dir}/")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
