"""
Exp 6 — Single-Agent vs Multi-Agent Ablation (False Positive Rate).

Compares Hunter-only baseline against the full 3-agent debate (Mediator final verdict)
using manual ground truth labels to compute confusion matrices and FPR for both conditions.

The core thesis claim: the Critic→Mediator loop suppresses false positives that the Hunter
alone would produce, without sacrificing recall.

Hunter-only verdict:  analysis_trace.debate_history[0].hunter.verdict
Debate final verdict: Finding.met_status  (Mediator output)

Usage:
    python debate_ablation_eval.py --review-id 1 --ground-truth debate_ground_truth.json
    python debate_ablation_eval.py --review-id 1 --ground-truth debate_ground_truth.json \\
        --output ablation_results.json
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import sdr.apps.standards.models  # noqa: F401 — resolve SQLAlchemy FK
import sdr.apps.designs.models    # noqa: F401
import sdr.apps.reviews.models.finding  # noqa: F401
import sdr.apps.reviews.models.review   # noqa: F401

from sdr.core.database import SessionLocal
from sdr.apps.reviews.models.finding import Finding

from sdr.apps.ai.evaluations.shared import results_path
from sdr.apps.ai.evaluations.shared.metrics import calculate_binary_confusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VALID_VERDICTS = {"met", "not_met", "na"}

# verdict_policy.source values that indicate the final verdict was forced to
# "na" by a policy gate rather than by the debate itself concluding "na" — see
# DebateService._apply_mediator_evidence_policy. Tracked separately so a
# regression in these gates (silently discarding a real met/not_met finding)
# shows up as its own number instead of being buried in overall recall/FPR.
NA_OVERRIDE_SOURCES = {"contract_not_applicable", "not_assessable"}


def _load_ground_truth(gt_path: str) -> dict[str, str]:
    """Load {requirement_id_str: label} from the manual ground-truth JSON."""
    with open(gt_path) as f:
        data = json.load(f)
    items = data.get("items", data)
    if isinstance(items, list):
        return {str(item["requirement_id"]): item["label"].lower().strip() for item in items}
    return {str(k): v.lower().strip() for k, v in items.items()}


def _extract_hunter_verdict(finding: Finding) -> str | None:
    """Pull the Hunter's round-0 verdict from analysis_trace."""
    meta = finding.requirement_metadata or {}
    trace = meta.get("analysis_trace", {})
    history = trace.get("debate_history", [])
    if not history:
        return None
    round0 = history[0]
    hunter = round0.get("hunter", {})
    verdict = hunter.get("verdict", "")
    return verdict.lower().strip() if verdict else None


def _extract_verdict_policy_source(finding: Finding) -> str | None:
    """Pull DebateService._apply_mediator_evidence_policy's verdict_policy.source
    from analysis_trace, so a forced-"na" override can be distinguished from the
    debate genuinely concluding "na" on its own."""
    meta = finding.requirement_metadata or {}
    trace = meta.get("analysis_trace", {})
    policy = trace.get("verdict_policy", {}) or {}
    source = policy.get("source")
    return str(source).strip() if source else None


def main():
    parser = argparse.ArgumentParser(
        description="Exp 6: Single-agent vs multi-agent ablation (FPR comparison)."
    )
    parser.add_argument("--review-id", type=int, required=True)
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to manual ground-truth JSON: {items: [{requirement_id, label}]}"
    )
    parser.add_argument("--output", type=str, default="debate_ablation_results.json")
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        return

    gt = _load_ground_truth(args.ground_truth)
    logger.info(f"Loaded {len(gt)} ground-truth labels")

    with SessionLocal() as db:
        findings = (
            db.query(Finding)
            .filter_by(review_id=args.review_id)
            .all()
        )
    logger.info(f"Loaded {len(findings)} findings for review_id={args.review_id}")

    if not findings:
        logger.error("No findings found. Run a security design review first.")
        return

    # Per-finding analysis
    per_item = []
    hunter_labels, hunter_preds = [], []
    debate_labels, debate_preds = [], []
    disagreements = []

    for f in findings:
        req_ref = str(f.requirement_reference or "")
        if not req_ref or req_ref not in gt:
            continue

        true_label = gt[req_ref]
        hunter_verdict = _extract_hunter_verdict(f)
        debate_verdict = (f.met_status or "").lower().strip()
        verdict_policy_source = _extract_verdict_policy_source(f)

        if hunter_verdict is None:
            logger.warning(f"  [req {req_ref}] no hunter verdict in analysis_trace — skipping")
            continue

        hunter_labels.append(true_label)
        hunter_preds.append(hunter_verdict)
        debate_labels.append(true_label)
        debate_preds.append(debate_verdict)

        hunter_correct = hunter_verdict == true_label
        debate_correct = debate_verdict == true_label
        changed = hunter_verdict != debate_verdict

        row = {
            "finding_id": f.id,
            "requirement_reference": f.requirement_reference,
            "true_label": true_label,
            "hunter_verdict": hunter_verdict,
            "debate_verdict": debate_verdict,
            "hunter_correct": hunter_correct,
            "debate_correct": debate_correct,
            "verdict_changed_by_debate": changed,
            "verdict_policy_source": verdict_policy_source,
        }
        per_item.append(row)

        if changed:
            disagreements.append(row)
            direction = "✓ fixed" if (not hunter_correct and debate_correct) else (
                "✗ broken" if (hunter_correct and not debate_correct) else "→ still wrong"
            )
            logger.info(
                f"  [finding {f.id}] hunter={hunter_verdict} → debate={debate_verdict} "
                f"(true={true_label}) {direction}"
            )

    if not per_item:
        logger.error("No findings matched ground-truth labels. Check finding IDs in ground truth file.")
        return

    hunter_cm = calculate_binary_confusion(hunter_labels, hunter_preds)
    debate_cm = calculate_binary_confusion(debate_labels, debate_preds)
    delta_fpr = round(hunter_cm["fpr"] - debate_cm["fpr"], 4)

    # False negatives specifically caused by a policy gate forcing "na" (as
    # opposed to the debate genuinely reasoning its way to a wrong verdict) —
    # tracked separately so a regression in the na-override gates is visible as
    # its own number instead of being buried in overall recall.
    false_negatives = [row for row in per_item if not row["debate_correct"] and row["true_label"] != "na"]
    na_override_false_negatives = [
        row for row in false_negatives
        if row["debate_verdict"] == "na" and row["verdict_policy_source"] in NA_OVERRIDE_SOURCES
    ]
    other_false_negatives = [row for row in false_negatives if row not in na_override_false_negatives]

    summary = {
        "review_id": args.review_id,
        "ground_truth_items": len(gt),
        "matched_findings": len(per_item),
        "verdict_changes": len(disagreements),
        "hunter_only": hunter_cm,
        "debate_final": debate_cm,
        "delta_fpr": delta_fpr,
        "fpr_suppression_achieved": delta_fpr >= 0,
        "false_negatives_total": len(false_negatives),
        "na_override_false_negatives": {
            "count": len(na_override_false_negatives),
            "finding_ids": [row["finding_id"] for row in na_override_false_negatives],
        },
        "other_false_negatives": {
            "count": len(other_false_negatives),
            "finding_ids": [row["finding_id"] for row in other_false_negatives],
        },
        "per_item_results": per_item,
        "disagreement_cases": disagreements,
    }

    output_path = results_path(args.output, subdir="debate")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Exp 6: Ablation Results ===")
    logger.info(f"  Matched findings:        {len(per_item)}")
    logger.info(f"  Verdict changes:         {len(disagreements)} ({100*len(disagreements)/len(per_item):.1f}%)")
    logger.info(f"\n  Hunter-only:  P={hunter_cm['precision']:.3f}  R={hunter_cm['recall']:.3f}  "
                f"F1={hunter_cm['f1']:.3f}  FPR={hunter_cm['fpr']:.3f}  "
                f"(TP={hunter_cm['tp']} FP={hunter_cm['fp']} FN={hunter_cm['fn']} TN={hunter_cm['tn']})")
    logger.info(f"  Debate final: P={debate_cm['precision']:.3f}  R={debate_cm['recall']:.3f}  "
                f"F1={debate_cm['f1']:.3f}  FPR={debate_cm['fpr']:.3f}  "
                f"(TP={debate_cm['tp']} FP={debate_cm['fp']} FN={debate_cm['fn']} TN={debate_cm['tn']})")
    logger.info(f"\n  Delta FPR (hunter - debate): {delta_fpr:+.4f}  "
                f"({'FPR not increased ✓' if delta_fpr >= 0 else 'FPR increased ✗'})")
    logger.info(
        f"\n  False negatives: {len(false_negatives)} total — "
        f"{len(na_override_false_negatives)} caused by an na-override policy gate, "
        f"{len(other_false_negatives)} other"
    )
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
