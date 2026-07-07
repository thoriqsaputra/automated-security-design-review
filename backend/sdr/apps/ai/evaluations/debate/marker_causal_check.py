"""
Marker causal check: does Set-of-Mark (SoM) marking actually change diagram
debate accuracy, at minimal cost?

Cost trick: production always runs with `apply_markers=True`, so every
completed vision-enabled review's `Finding.requirement_metadata.analysis_trace`
already contains a real, live "marked" verdict per (diagram_id, requirement_id)
— this is read for free, no LLM calls (same trick `diagram_ablation_eval.py`
uses for Hunter-only vs. debate). The only missing condition is "raw"
(unmarked); this script pays for that once.

Further cost cut vs. the old `som_ablation_eval.py`: rather than one live
debate call per (diagram, requirement) sample, this batches by diagram — one
raw debate call per diagram covers every labeled requirement for that diagram
at once (`DiagramDebateService.run_diagram_debate` already accepts a list of
requirements per call). For a ground truth with 53 labeled samples across 3
diagrams, that's ~3 raw debate calls (Hunter + Critic + occasional Mediator)
instead of ~53 — roughly the same batching production itself uses.

Usage:
    python marker_causal_check.py --review-id 53 \\
        --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_review_53.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import sdr.apps.standards.models  # noqa: F401 — resolve SQLAlchemy FK
import sdr.apps.designs.models    # noqa: F401
import sdr.apps.reviews.models.finding  # noqa: F401
import sdr.apps.reviews.models.review   # noqa: F401

from sdr.core.database import SessionLocal
from sdr.apps.reviews.models.finding import Finding
from sdr.apps.reviews.models.choices import FindingType

from sdr.apps.ai.engine.debate.diagram_debate_service import DiagramDebateService
from sdr.apps.ai.evaluations.shared import results_path
from sdr.apps.ai.evaluations.shared.metrics import calculate_binary_confusion
from sdr.apps.ai.evaluations.debate.diagram_ablation_eval import (
    _normalize_requirement_id,
    _per_requirement_verdicts,
)
from sdr.apps.ai.evaluations.vision.real_diagram_source import (
    load_ground_truth,
    load_labeled_samples,
    load_tsd_document,
    build_diagram_input,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_marked_verdicts(review_id: int) -> dict[tuple[str, str], str]:
    """(diagram_id, requirement_id) -> stored marked-condition final verdict,
    read from already-completed Finding records — zero LLM calls."""
    marked: dict[tuple[str, str], str] = {}
    with SessionLocal() as db:
        findings = (
            db.query(Finding)
            .filter(
                Finding.review_id == review_id,
                Finding.finding_type == FindingType.DIAGRAM.value,
            )
            .all()
        )
    for f in findings:
        trace = (f.requirement_metadata or {}).get("analysis_trace", {})
        verdicts = _per_requirement_verdicts(trace.get("assessed_requirements", []))
        for requirement_id, verdict in verdicts.items():
            marked[(f.diagram_id, requirement_id)] = verdict
    return marked


def main():
    parser = argparse.ArgumentParser(
        description="Marker causal check: raw (unmarked) vs. stored marked-condition diagram debate accuracy."
    )
    parser.add_argument("--review-id", type=int, required=True, help="Review whose stored Finding data supplies the marked condition.")
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to a labeled diagram ground-truth JSON (see build_diagram_ground_truth_template.py)"
    )
    parser.add_argument("--output", type=str, default="marker_causal_check_results.json")
    parser.add_argument(
        "--votes", type=int, default=1,
        help="Self-consistency votes for the live raw-condition call only (marked condition is a single stored production run either way).",
    )
    parser.add_argument(
        "--cheap-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip mediator LLM when Critic upholds, to reduce cost.",
    )
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

    marked_verdicts = _load_marked_verdicts(args.review_id)
    logger.info(f"Loaded {len(marked_verdicts)} stored marked-condition verdicts for review_id={args.review_id}")

    by_diagram: dict[str, list] = defaultdict(list)
    for sample in samples:
        by_diagram[sample.diagram_id].append(sample)
    logger.info(f"Loaded {len(samples)} labeled samples across {len(by_diagram)} diagrams")

    tsd_doc = load_tsd_document(design_id)
    service = DiagramDebateService()

    per_item = []
    for diagram_id, diagram_samples in by_diagram.items():
        diagram = build_diagram_input(tsd_doc, diagram_id, f"raw_{diagram_id}", blank_text=False)
        if diagram is None:
            logger.warning(f"  {diagram_id}: diagram not found or invalid in prepared TSD — skipping")
            continue

        requirements = [
            SimpleNamespace(
                ordinal=i + 1,
                stable_key=s.requirement_id,
                requirement_text=s.requirement_text,
                verification_hint=s.verification_hint,
            )
            for i, s in enumerate(diagram_samples)
        ]

        logger.info(f"Raw (unmarked) debate: diagram_id={diagram_id} requirements={len(requirements)}")
        output = service.run_diagram_debate_voted(
            diagram=diagram,
            requirements=requirements,
            tsd_context="",
            votes=args.votes,
            apply_markers=False,
            skip_mediator_on_uphold=args.cheap_mode,
        )
        raw_verdicts = _per_requirement_verdicts(
            (output.mediator_result or {}).get("assessed_requirements", [])
        )

        for sample in diagram_samples:
            requirement_id = _normalize_requirement_id(sample.requirement_id)
            marked_verdict = marked_verdicts.get((diagram_id, requirement_id))
            raw_verdict = raw_verdicts.get(requirement_id)
            if marked_verdict is None:
                logger.debug(f"  [{diagram_id}/{requirement_id}] no stored marked verdict — skipping (not assessed in review {args.review_id})")
                continue

            true_label = sample.label
            per_item.append({
                "diagram_id": diagram_id,
                "requirement_id": requirement_id,
                "true_label": true_label,
                "marked_verdict": marked_verdict,
                "raw_verdict": raw_verdict,
                "marked_correct": marked_verdict == true_label,
                "raw_correct": raw_verdict == true_label,
                "verdict_changed": marked_verdict != raw_verdict,
            })

    if not per_item:
        logger.error("No labeled samples matched a stored marked-condition verdict — nothing to compare.")
        return

    true_labels = [r["true_label"] for r in per_item]
    marked_preds = [r["marked_verdict"] for r in per_item]
    raw_preds = [r["raw_verdict"] for r in per_item]

    marked_cm = calculate_binary_confusion(true_labels, marked_preds)
    raw_cm = calculate_binary_confusion(true_labels, raw_preds)
    marked_accuracy = sum(1 for r in per_item if r["marked_correct"]) / len(per_item)
    raw_accuracy = sum(1 for r in per_item if r["raw_correct"]) / len(per_item)
    delta_accuracy = round(marked_accuracy - raw_accuracy, 4)
    delta_fpr = round(raw_cm["fpr"] - marked_cm["fpr"], 4)
    disagreements = [r for r in per_item if r["verdict_changed"]]

    summary = {
        "review_id": args.review_id,
        "design_id": design_id,
        "votes": args.votes,
        "cheap_mode": args.cheap_mode,
        "matched_pairs": len(per_item),
        "diagrams_evaluated": len(by_diagram),
        "verdict_changes": len(disagreements),
        "marked_final": {**marked_cm, "accuracy": round(marked_accuracy, 4)},
        "raw_final": {**raw_cm, "accuracy": round(raw_accuracy, 4)},
        "delta_accuracy": delta_accuracy,
        "delta_fpr": delta_fpr,
        "markers_improve_accuracy": delta_accuracy > 0,
        "per_item_results": per_item,
        "disagreement_cases": disagreements,
    }

    output_path = results_path(args.output, subdir="debate")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Marker Causal Check Results ===")
    logger.info(f"  Matched pairs: {len(per_item)} across {len(by_diagram)} diagrams")
    logger.info(f"  Marked (stored, production): accuracy={marked_accuracy:.3f} FPR={marked_cm['fpr']:.3f}")
    logger.info(f"  Raw (live, unmarked):        accuracy={raw_accuracy:.3f} FPR={raw_cm['fpr']:.3f}")
    logger.info(f"  Delta accuracy (marked - raw): {delta_accuracy:+.4f}")
    logger.info(f"  Delta FPR (raw - marked): {delta_fpr:+.4f}")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
