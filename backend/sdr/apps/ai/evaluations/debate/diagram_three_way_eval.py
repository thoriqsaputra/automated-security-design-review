"""
Diagram three-way eval: Hunter-only vs full multi-agent debate vs the
extract-then-reason pipeline (DiagramExtractReasonService), for diagram
findings.

This is a LIVE paired rerun: all three methods are taken from the same
current-code execution over the same diagram/requirement batch, so the
comparison is a real ablation. Reuses ground-truth loading and per-item row
helpers from diagram_ablation_eval.py (same-package import) rather than
duplicating them.

Usage:
    python diagram_three_way_eval.py --design-id 14 \\
        --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_design_14_llm_judged.json
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

from sdr.apps.ai.engine.debate.diagram_debate_service import DiagramDebateService
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
    _build_diag_counters,
    _build_per_item_row,
    _load_ground_truth_labels,
    _normalize_requirement_id,
    _per_requirement_items,
    _per_requirement_verdicts,
    _update_diag_counters,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Diagram three-way eval: Hunter-only vs debate vs extract-then-reason (F1/FPR comparison)."
    )
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument(
        "--review-id",
        type=int,
        default=None,
        help="Optional review override. Defaults to the latest completed review with diagram findings for the design.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to a labeled diagram ground-truth JSON (defaults to the canonical design-scoped file)",
    )
    parser.add_argument("--votes", type=int, default=1, help="Debate self-consistency votes (extract-reason votes at the extraction layer, see AI_VISION_EXTRACTION_VOTES)")
    parser.add_argument(
        "--cheap-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow the debate service to skip the mediator on Critic uphold. Defaults to false for a real full-debate ablation.",
    )
    parser.add_argument("--output", type=str, default="diagram_three_way_results.json")
    args = parser.parse_args()

    gt_path = args.ground_truth or resolve_diagram_ground_truth_path(args.design_id)
    if not os.path.exists(gt_path):
        logger.error(f"Ground truth file not found: {gt_path}")
        return

    gt = _load_ground_truth_labels(gt_path)
    logger.info(f"Loaded {len(gt)} ground-truth (diagram_id, requirement_id) labels")

    with SessionLocal() as db:
        review = None
        if args.review_id is not None:
            review = db.query(Review).filter(Review.id == args.review_id).first()
            if not review:
                logger.error(f"Review with ID {args.review_id} not found.")
                return
            if review.design_id != args.design_id:
                logger.error(
                    f"Review {args.review_id} belongs to design_id={review.design_id}, "
                    f"not design_id={args.design_id}."
                )
                return
        else:
            review = get_latest_diagram_review_for_design(args.design_id, db=db)
            if not review:
                logger.error(f"No completed diagram-bearing review found for design_id={args.design_id}.")
                return

    per_item = []
    hunter_labels, hunter_preds = [], []
    debate_labels, debate_preds = [], []
    extract_reason_labels, extract_reason_preds = [], []
    disagreements = []
    diag_counters = _build_diag_counters()

    gt_data = load_ground_truth(gt_path)
    samples = load_labeled_samples(gt_data, labels=("met", "not_met", "na"))
    by_diagram: dict[str, list] = defaultdict(list)
    for sample in samples:
        by_diagram[sample.diagram_id].append(sample)

    tsd_doc = load_tsd_document(args.design_id)
    debate_service = DiagramDebateService()
    extract_reason_service = DiagramExtractReasonService()

    for diagram_id, diagram_samples in by_diagram.items():
        diagram = build_diagram_input(tsd_doc, diagram_id, f"three_way_eval_{diagram_id}", blank_text=False)
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

        logger.info(f"Live diagram debate: diagram_id={diagram_id} requirements={len(requirements)}")
        debate_output = debate_service.run_diagram_debate_voted(
            diagram=diagram,
            requirements=requirements,
            tsd_context="",
            votes=args.votes,
            skip_mediator_on_uphold=args.cheap_mode,
        )
        logger.info(f"Live diagram extract-reason: diagram_id={diagram_id} requirements={len(requirements)}")
        extract_reason_output = extract_reason_service.run_diagram_extract_reason(
            diagram=diagram,
            requirements=requirements,
            tsd_context="",
        )

        hunter_verdicts = _per_requirement_verdicts(
            (debate_output.hunter_result or {}).get("requirement_assessments", [])
        )
        debate_verdicts = _per_requirement_verdicts(
            (debate_output.mediator_result or {}).get("assessed_requirements", [])
        )
        hunter_items = _per_requirement_items(
            (debate_output.hunter_result or {}).get("requirement_assessments", [])
        )
        debate_items = _per_requirement_items(
            (debate_output.mediator_result or {}).get("assessed_requirements", [])
        )
        extract_reason_verdicts = _per_requirement_verdicts(
            (extract_reason_output.mediator_result or {}).get("assessed_requirements", [])
        )
        extract_reason_items = _per_requirement_items(
            (extract_reason_output.mediator_result or {}).get("assessed_requirements", [])
        )
        _update_diag_counters(
            diag_counters,
            debate_output.critic_result or {},
            debate_output.hunter_rebuttal_result or {},
            debate_output.mediator_result or {},
        )

        for sample in diagram_samples:
            requirement_id = _normalize_requirement_id(sample.requirement_id)
            hunter_verdict = hunter_verdicts.get(requirement_id)
            debate_verdict = debate_verdicts.get(requirement_id)
            extract_reason_verdict = extract_reason_verdicts.get(requirement_id)
            if hunter_verdict is None:
                logger.warning(f"  [diagram={diagram_id} req={requirement_id}] no hunter verdict — skipping")
                continue

            true_label = sample.label
            hunter_labels.append(true_label)
            hunter_preds.append(hunter_verdict)
            debate_labels.append(true_label)
            debate_preds.append(debate_verdict)
            extract_reason_labels.append(true_label)
            extract_reason_preds.append(extract_reason_verdict)

            hunter_correct = hunter_verdict == true_label
            debate_correct = debate_verdict == true_label
            extract_reason_correct = extract_reason_verdict == true_label
            changed = hunter_verdict != debate_verdict

            row = _build_per_item_row(
                diagram_id=diagram_id,
                requirement_id=requirement_id,
                true_label=true_label,
                hunter_verdict=hunter_verdict,
                debate_verdict=debate_verdict,
                hunter_item=hunter_items.get(requirement_id),
                debate_item=debate_items.get(requirement_id),
            )
            extract_reason_item = extract_reason_items.get(requirement_id) or {}
            row["extract_reason_verdict"] = extract_reason_verdict
            row["extract_reason_correct"] = extract_reason_correct
            row["extract_reason_verdict_policy_source"] = extract_reason_item.get("verdict_policy_source")
            row["extract_reason_evidence_quality"] = extract_reason_item.get("evidence_quality")
            row["extract_reason_cited_element_ids"] = extract_reason_item.get("cited_element_ids")

            per_item.append(row)
            if changed:
                disagreements.append(row)

    if not per_item:
        logger.error("No (diagram_id, requirement_id) pairs matched ground-truth labels.")
        return

    hunter_cm = calculate_binary_confusion(hunter_labels, hunter_preds)
    debate_cm = calculate_binary_confusion(debate_labels, debate_preds)
    extract_reason_cm = calculate_binary_confusion(extract_reason_labels, extract_reason_preds)

    delta_fpr_debate_vs_hunter = round(hunter_cm["fpr"] - debate_cm["fpr"], 4)
    delta_fpr_extract_reason_vs_hunter = round(hunter_cm["fpr"] - extract_reason_cm["fpr"], 4)
    delta_fpr_extract_reason_vs_debate = round(debate_cm["fpr"] - extract_reason_cm["fpr"], 4)

    # Flip analysis for extract-reason vs Hunter-only: report BOTH directions,
    # not just the net delta, so a regression can't hide behind an unrelated
    # improvement.
    extract_reason_fixed = [
        row for row in per_item
        if not row["hunter_correct"] and row["extract_reason_correct"]
    ]
    extract_reason_regressed = [
        row for row in per_item
        if row["hunter_correct"] and not row["extract_reason_correct"]
    ]

    summary = {
        "design_id": args.design_id,
        "review_id": review.id,
        "ground_truth_path": gt_path,
        "votes": args.votes,
        "cheap_mode": args.cheap_mode,
        "ground_truth_items": len(gt),
        "matched_pairs": len(per_item),
        "verdict_changes_hunter_to_debate": len(disagreements),
        "hunter_only": hunter_cm,
        "debate_final": debate_cm,
        "extract_reason": extract_reason_cm,
        "delta_fpr_debate_vs_hunter": delta_fpr_debate_vs_hunter,
        "delta_fpr_extract_reason_vs_hunter": delta_fpr_extract_reason_vs_hunter,
        "delta_fpr_extract_reason_vs_debate": delta_fpr_extract_reason_vs_debate,
        "extract_reason_fixed_cases": len(extract_reason_fixed),
        "extract_reason_regressed_cases": len(extract_reason_regressed),
        "extract_reason_success": (
            extract_reason_cm["f1"] >= 0.9 and extract_reason_cm["fpr"] <= 0.05
        ),
        "extract_reason_citation_retry_batches": getattr(
            extract_reason_service, "_citation_retry_limit", None
        ),
        "per_item_results": per_item,
        "disagreement_cases_hunter_vs_debate": disagreements,
        "extract_reason_fixed_case_rows": extract_reason_fixed,
        "extract_reason_regressed_case_rows": extract_reason_regressed,
    }

    output_path = results_path(args.output, subdir="debate")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Diagram Three-Way Eval Results ===")
    logger.info(f"  Matched (diagram, requirement) pairs: {len(per_item)}")
    logger.info(
        f"\n  Hunter-only:      P={hunter_cm['precision']:.3f}  R={hunter_cm['recall']:.3f}  "
        f"F1={hunter_cm['f1']:.3f}  FPR={hunter_cm['fpr']:.3f}  "
        f"(TP={hunter_cm['tp']} FP={hunter_cm['fp']} FN={hunter_cm['fn']} TN={hunter_cm['tn']})"
    )
    logger.info(
        f"  Debate final:     P={debate_cm['precision']:.3f}  R={debate_cm['recall']:.3f}  "
        f"F1={debate_cm['f1']:.3f}  FPR={debate_cm['fpr']:.3f}  "
        f"(TP={debate_cm['tp']} FP={debate_cm['fp']} FN={debate_cm['fn']} TN={debate_cm['tn']})"
    )
    logger.info(
        f"  Extract-reason:   P={extract_reason_cm['precision']:.3f}  R={extract_reason_cm['recall']:.3f}  "
        f"F1={extract_reason_cm['f1']:.3f}  FPR={extract_reason_cm['fpr']:.3f}  "
        f"(TP={extract_reason_cm['tp']} FP={extract_reason_cm['fp']} FN={extract_reason_cm['fn']} TN={extract_reason_cm['tn']})"
    )
    logger.info(
        f"\n  Extract-reason vs Hunter: fixed={len(extract_reason_fixed)}  regressed={len(extract_reason_regressed)}"
    )
    logger.info(
        f"  Target met (F1>=0.9, FPR<=0.05): {summary['extract_reason_success']}"
    )
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
