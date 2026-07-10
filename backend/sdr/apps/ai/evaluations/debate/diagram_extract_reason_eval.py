"""
Diagram extract-then-reason eval: runs ONLY DiagramExtractReasonService
(no DiagramDebateService rerun — Hunter-only/debate numbers are expensive
live LLM reruns and are pulled from a previously-saved baseline result file
instead, via --baseline-result).

Usage:
    python diagram_extract_reason_eval.py --design-id 14 \\
        --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_design_14_llm_judged.json \\
        --baseline-result /app/sdr/apps/ai/evaluations/results/debate/diagram_three_way_design14.json
"""
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
from sdr.apps.reviews.models.review import Review

from sdr.apps.ai.engine.debate.diagram_extract_reason_service import DiagramExtractReasonService
from sdr.apps.ai.evaluations.shared import results_path
from sdr.apps.ai.evaluations.shared.diagram_ground_truth import (
    get_latest_diagram_review_for_design,
    resolve_diagram_ground_truth_path,
)
from sdr.apps.ai.evaluations.shared.metrics import calculate_binary_confusion
from sdr.apps.ai.evaluations.vision.real_diagram_source import (
    build_diagram_input,
    load_ground_truth,
    load_labeled_samples,
    load_tsd_document,
)
from sdr.apps.ai.evaluations.debate.diagram_ablation_eval import (
    _load_ground_truth_labels,
    _normalize_requirement_id,
    _per_requirement_items,
    _per_requirement_verdicts,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Diagram extract-then-reason eval (extract_reason only; baseline pulled from a prior result file)."
    )
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument("--review-id", type=int, default=None)
    parser.add_argument("--ground-truth", type=str, default=None)
    parser.add_argument(
        "--baseline-result",
        type=str,
        default=None,
        help="Path to a previously-saved ablation/three-way result JSON to pull hunter_only/debate_final from for comparison (no rerun).",
    )
    parser.add_argument("--output", type=str, default="diagram_extract_reason_results.json")
    args = parser.parse_args()

    gt_path = args.ground_truth or resolve_diagram_ground_truth_path(args.design_id)
    if not os.path.exists(gt_path):
        logger.error(f"Ground truth file not found: {gt_path}")
        return

    gt = _load_ground_truth_labels(gt_path)
    logger.info(f"Loaded {len(gt)} ground-truth (diagram_id, requirement_id) labels")

    baseline = None
    if args.baseline_result and os.path.exists(args.baseline_result):
        with open(args.baseline_result) as f:
            baseline = json.load(f)
        logger.info(f"Loaded baseline (hunter_only/debate_final) from {args.baseline_result}")
    elif args.baseline_result:
        logger.warning(f"--baseline-result path not found: {args.baseline_result} — comparison will be omitted")

    with SessionLocal() as db:
        if args.review_id is not None:
            review = db.query(Review).filter(Review.id == args.review_id).first()
            if not review:
                logger.error(f"Review with ID {args.review_id} not found.")
                return
        else:
            review = get_latest_diagram_review_for_design(args.design_id, db=db)
            if not review:
                logger.error(f"No completed diagram-bearing review found for design_id={args.design_id}.")
                return

    per_item = []
    extract_reason_labels, extract_reason_preds = [], []

    gt_data = load_ground_truth(gt_path)
    samples = load_labeled_samples(gt_data, labels=("met", "not_met", "na"))
    by_diagram: dict[str, list] = defaultdict(list)
    for sample in samples:
        by_diagram[sample.diagram_id].append(sample)

    tsd_doc = load_tsd_document(args.design_id)
    extract_reason_service = DiagramExtractReasonService()

    for diagram_id, diagram_samples in by_diagram.items():
        diagram = build_diagram_input(tsd_doc, diagram_id, f"extract_reason_eval_{diagram_id}", blank_text=False)
        if diagram is None:
            logger.warning(f"  {diagram_id}: diagram not found or invalid in prepared TSD — skipping")
            continue

        requirements = [
            SimpleNamespace(
                ordinal=i + 1,
                stable_key=sample.requirement_id,
                requirement_text=sample.requirement_text,
                verification_hint=sample.verification_hint,
            )
            for i, sample in enumerate(diagram_samples)
        ]

        logger.info(f"Live diagram extract-reason: diagram_id={diagram_id} requirements={len(requirements)}")
        extract_reason_output = extract_reason_service.run_diagram_extract_reason(
            diagram=diagram,
            requirements=requirements,
            tsd_context="",
        )

        extract_reason_verdicts = _per_requirement_verdicts(
            (extract_reason_output.mediator_result or {}).get("assessed_requirements", [])
        )
        extract_reason_items = _per_requirement_items(
            (extract_reason_output.mediator_result or {}).get("assessed_requirements", [])
        )
        extraction_diagnostics = (extract_reason_output.mediator_result or {}).get("extraction_diagnostics", {})

        for sample in diagram_samples:
            requirement_id = _normalize_requirement_id(sample.requirement_id)
            extract_reason_verdict = extract_reason_verdicts.get(requirement_id)
            if extract_reason_verdict is None:
                logger.warning(f"  [diagram={diagram_id} req={requirement_id}] no extract_reason verdict — skipping")
                continue

            true_label = sample.label
            extract_reason_labels.append(true_label)
            extract_reason_preds.append(extract_reason_verdict)

            item = extract_reason_items.get(requirement_id) or {}
            per_item.append({
                "diagram_id": diagram_id,
                "requirement_id": requirement_id,
                "true_label": true_label,
                "extract_reason_verdict": extract_reason_verdict,
                "extract_reason_correct": extract_reason_verdict == true_label,
                "verdict_policy_source": item.get("verdict_policy_source"),
                "evidence_quality": item.get("evidence_quality"),
                "cited_element_ids": item.get("cited_element_ids"),
                "dropped_hallucinated_citations": item.get("dropped_hallucinated_citations"),
                "reasoning": item.get("reasoning"),
                "extraction_diagnostics": extraction_diagnostics,
            })

    if not per_item:
        logger.error("No (diagram_id, requirement_id) pairs matched ground-truth labels.")
        return

    extract_reason_cm = calculate_binary_confusion(extract_reason_labels, extract_reason_preds)

    hunter_cm = (baseline or {}).get("hunter_only")
    debate_cm = (baseline or {}).get("debate_final")

    summary = {
        "design_id": args.design_id,
        "review_id": review.id,
        "ground_truth_path": gt_path,
        "baseline_result_path": args.baseline_result,
        "ground_truth_items": len(gt),
        "matched_pairs": len(per_item),
        "extract_reason": extract_reason_cm,
        "hunter_only_baseline": hunter_cm,
        "debate_final_baseline": debate_cm,
        "delta_fpr_extract_reason_vs_hunter_baseline": (
            round(hunter_cm["fpr"] - extract_reason_cm["fpr"], 4) if hunter_cm else None
        ),
        "delta_fpr_extract_reason_vs_debate_baseline": (
            round(debate_cm["fpr"] - extract_reason_cm["fpr"], 4) if debate_cm else None
        ),
        "extract_reason_success": (
            extract_reason_cm["f1"] >= 0.9 and extract_reason_cm["fpr"] <= 0.05
        ),
        "per_item_results": per_item,
    }

    output_path = results_path(args.output, subdir="debate")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Diagram Extract-Reason Eval Results ===")
    logger.info(f"  Matched (diagram, requirement) pairs: {len(per_item)}")
    if hunter_cm:
        logger.info(
            f"\n  Hunter-only (baseline): P={hunter_cm['precision']:.3f}  R={hunter_cm['recall']:.3f}  "
            f"F1={hunter_cm['f1']:.3f}  FPR={hunter_cm['fpr']:.3f}  "
            f"(TP={hunter_cm['tp']} FP={hunter_cm['fp']} FN={hunter_cm['fn']} TN={hunter_cm['tn']})"
        )
    if debate_cm:
        logger.info(
            f"  Debate final (baseline): P={debate_cm['precision']:.3f}  R={debate_cm['recall']:.3f}  "
            f"F1={debate_cm['f1']:.3f}  FPR={debate_cm['fpr']:.3f}  "
            f"(TP={debate_cm['tp']} FP={debate_cm['fp']} FN={debate_cm['fn']} TN={debate_cm['tn']})"
        )
    logger.info(
        f"  Extract-reason (NEW):    P={extract_reason_cm['precision']:.3f}  R={extract_reason_cm['recall']:.3f}  "
        f"F1={extract_reason_cm['f1']:.3f}  FPR={extract_reason_cm['fpr']:.3f}  "
        f"(TP={extract_reason_cm['tp']} FP={extract_reason_cm['fp']} FN={extract_reason_cm['fn']} TN={extract_reason_cm['tn']})"
    )
    logger.info(f"  Target met (F1>=0.9, FPR<=0.05): {summary['extract_reason_success']}")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
