"""
Pools per-design extract-then-reason eval results (design 14 + design 15) into
one cross-design confusion matrix per method (extract_reason / hunter_only /
debate_final), and surfaces per-diagram extraction confirmation ratios and
retry counts — the "extract-reason is the overall winner" defense table.

Usage:
    python diagram_extract_reason_aggregate_summary.py \\
        --design14 /app/sdr/apps/ai/evaluations/results/debate/diagram_extract_reason_design14_v5_retryfix.json \\
        --design15 /app/sdr/apps/ai/evaluations/results/debate/diagram_extract_reason_design15_v6_reverify.json \\
        --output diagram_extract_reason_aggregate_summary.json
"""
import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from sdr.apps.ai.evaluations.shared import results_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METHODS = ("extract_reason", "hunter_only_baseline", "debate_final_baseline")


def _metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
    }


def _pool(results: list[dict], method: str) -> dict:
    tp = fp = fn = tn = 0
    for result in results:
        cm = result.get(method)
        if not cm:
            continue
        tp += cm["tp"]
        fp += cm["fp"]
        fn += cm["fn"]
        tn += cm["tn"]
    return _metrics_from_counts(tp, fp, fn, tn)


def _diagram_extraction_summary(result: dict) -> dict:
    by_diagram: dict[str, dict] = {}
    for item in result.get("per_item_results", []):
        diagram_id = item.get("diagram_id")
        diag = item.get("extraction_diagnostics") or {}
        if diagram_id not in by_diagram and diag:
            by_diagram[diagram_id] = {
                "confirmed_element_count": diag.get("confirmed_element_count"),
                "total_element_count": diag.get("total_element_count"),
                "confirmation_ratio": (
                    round(diag["confirmed_element_count"] / diag["total_element_count"], 4)
                    if diag.get("total_element_count")
                    else None
                ),
                "completeness_retry_batches": diag.get("completeness_retry_batches"),
                "citation_retry_batches": diag.get("citation_retry_batches"),
                "full_failure_retry_batches": diag.get("full_failure_retry_batches"),
                "unconfirmed_evidence_reverify_candidates": diag.get("unconfirmed_evidence_reverify_candidates"),
                "unconfirmed_evidence_reverify_changed": diag.get("unconfirmed_evidence_reverify_changed"),
            }
    return by_diagram


def main():
    parser = argparse.ArgumentParser(description="Aggregate extract-reason eval results across designs 14 + 15.")
    parser.add_argument("--design14", type=str, required=True)
    parser.add_argument("--design15", type=str, required=True)
    parser.add_argument("--output", type=str, default="diagram_extract_reason_aggregate_summary.json")
    args = parser.parse_args()

    with open(args.design14) as f:
        design14 = json.load(f)
    with open(args.design15) as f:
        design15 = json.load(f)

    results = [design14, design15]

    pooled = {method: _pool(results, method) for method in METHODS}

    delta_fpr_vs_hunter = round(
        pooled["hunter_only_baseline"]["fpr"] - pooled["extract_reason"]["fpr"], 4
    )
    delta_fpr_vs_debate = round(
        pooled["debate_final_baseline"]["fpr"] - pooled["extract_reason"]["fpr"], 4
    )

    per_diagram = {
        "design_14": _diagram_extraction_summary(design14),
        "design_15": _diagram_extraction_summary(design15),
    }

    summary = {
        "sources": {"design_14": args.design14, "design_15": args.design15},
        "pooled": pooled,
        "delta_fpr_pooled_extract_reason_vs_hunter_baseline": delta_fpr_vs_hunter,
        "delta_fpr_pooled_extract_reason_vs_debate_baseline": delta_fpr_vs_debate,
        "per_diagram_extraction_confirmation": per_diagram,
    }

    output_path = results_path(args.output, subdir="debate")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Pooled Extract-Reason Aggregate Summary (design 14 + 15) ===")
    for method in METHODS:
        cm = pooled[method]
        logger.info(
            f"  {method:<20} P={cm['precision']:.3f}  R={cm['recall']:.3f}  F1={cm['f1']:.3f}  "
            f"FPR={cm['fpr']:.3f}  (TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']})"
        )
    logger.info(f"\n  Pooled delta FPR vs Hunter-only: {delta_fpr_vs_hunter}")
    logger.info(f"  Pooled delta FPR vs Debate-final: {delta_fpr_vs_debate}")

    for design_key, diagrams in per_diagram.items():
        logger.info(f"\n  {design_key} extraction confirmation ratios:")
        for diagram_id, diag in diagrams.items():
            logger.info(
                f"    {diagram_id}: {diag['confirmed_element_count']}/{diag['total_element_count']} "
                f"({diag['confirmation_ratio']}) completeness_retries={diag['completeness_retry_batches']} "
                f"citation_retries={diag['citation_retry_batches']} "
                f"full_failure_retries={diag['full_failure_retry_batches']}"
            )

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
