"""
Exp 6b — Hunter-only vs Hunter+Critic vs Full Debate (agent-decomposition ablation).

Extends the binary Hunter-only/Full-debate ablation (ablation_eval.py) with a third,
purely post-hoc condition: what would the verdict have been with Hunter+Critic only,
no Mediator? This isolates which agent in the debate chain is actually responsible for
FPR suppression and/or recall loss, instead of only knowing the combined effect of both.

No new LLM calls are made — this reads the same already-persisted analysis_trace as
ablation_eval.py. The Hunter+Critic verdict is derived as:
    critic.revised_verdict if the Critic proposed one, else hunter.verdict (Critic upheld)

Hunter-only verdict:       analysis_trace.debate_history[0].hunter.verdict
Hunter+Critic verdict:     analysis_trace.debate_history[0].critic.revised_verdict or hunter.verdict
Full debate verdict:       Finding.met_status  (Mediator output)

Usage:
    python three_way_ablation_eval.py --review-id 106 --ground-truth debate_ground_truth_SISCalendar.json
"""
import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import sdr.apps.standards.models  # noqa: F401 — resolve SQLAlchemy FK
import sdr.apps.designs.models    # noqa: F401
import sdr.apps.reviews.models.finding  # noqa: F401
import sdr.apps.reviews.models.review   # noqa: F401

from sdr.core.database import SessionLocal
from sdr.apps.reviews.models.finding import Finding

from sdr.apps.ai.evaluations.shared import results_path
from sdr.apps.ai.evaluations.shared.metrics import calculate_binary_confusion
from sdr.apps.ai.evaluations.debate.ablation_eval import _load_ground_truth, _extract_hunter_verdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _extract_critic_only_verdict(finding: Finding) -> str | None:
    """Hunter+Critic verdict with no Mediator: the Critic's own revised_verdict
    if it proposed one (OVERTURN/PARTIAL), else the Hunter's verdict stands
    (UPHOLD)."""
    meta = finding.requirement_metadata or {}
    trace = meta.get("analysis_trace", {})
    history = trace.get("debate_history", [])
    if not history:
        return None
    round0 = history[0]
    hunter = round0.get("hunter", {})
    critic = round0.get("critic", {})
    revised = (critic.get("revised_verdict") or "").lower().strip()
    if revised:
        return revised
    verdict = hunter.get("verdict", "")
    return verdict.lower().strip() if verdict else None


def main():
    parser = argparse.ArgumentParser(
        description="Exp 6b: Hunter-only vs Hunter+Critic vs Full debate (agent decomposition)."
    )
    parser.add_argument("--review-id", type=int, required=True)
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to manual ground-truth JSON: {items: [{requirement_id, label}]}"
    )
    parser.add_argument("--output", type=str, default="debate_three_way_ablation_results.json")
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        return

    gt = _load_ground_truth(args.ground_truth)
    logger.info(f"Loaded {len(gt)} ground-truth labels")

    with SessionLocal() as db:
        findings = db.query(Finding).filter_by(review_id=args.review_id).all()
    logger.info(f"Loaded {len(findings)} findings for review_id={args.review_id}")

    if not findings:
        logger.error("No findings found. Run a security design review first.")
        return

    per_item = []
    hunter_labels, hunter_preds = [], []
    critic_labels, critic_preds = [], []
    debate_labels, debate_preds = [], []

    for f in findings:
        req_ref = str(f.requirement_reference or "")
        if not req_ref or req_ref not in gt:
            continue

        true_label = gt[req_ref]
        hunter_verdict = _extract_hunter_verdict(f)
        critic_verdict = _extract_critic_only_verdict(f)
        debate_verdict = (f.met_status or "").lower().strip()

        if hunter_verdict is None or critic_verdict is None:
            logger.warning(f"  [req {req_ref}] missing hunter/critic verdict in analysis_trace — skipping")
            continue

        hunter_labels.append(true_label)
        hunter_preds.append(hunter_verdict)
        critic_labels.append(true_label)
        critic_preds.append(critic_verdict)
        debate_labels.append(true_label)
        debate_preds.append(debate_verdict)

        row = {
            "finding_id": f.id,
            "requirement_reference": f.requirement_reference,
            "true_label": true_label,
            "hunter_verdict": hunter_verdict,
            "hunter_critic_verdict": critic_verdict,
            "debate_verdict": debate_verdict,
            "hunter_correct": hunter_verdict == true_label,
            "hunter_critic_correct": critic_verdict == true_label,
            "debate_correct": debate_verdict == true_label,
        }
        per_item.append(row)

        if hunter_verdict != critic_verdict or critic_verdict != debate_verdict:
            logger.info(
                f"  [finding {f.id}] hunter={hunter_verdict} -> "
                f"hunter+critic={critic_verdict} -> debate={debate_verdict}  (true={true_label})"
            )

    if not per_item:
        logger.error("No findings matched ground-truth labels. Check finding IDs in ground truth file.")
        return

    hunter_cm = calculate_binary_confusion(hunter_labels, hunter_preds)
    critic_cm = calculate_binary_confusion(critic_labels, critic_preds)
    debate_cm = calculate_binary_confusion(debate_labels, debate_preds)

    summary = {
        "review_id": args.review_id,
        "ground_truth_items": len(gt),
        "matched_findings": len(per_item),
        "hunter_only": hunter_cm,
        "hunter_plus_critic": critic_cm,
        "full_debate": debate_cm,
        "delta_fpr_critic_vs_hunter": round(hunter_cm["fpr"] - critic_cm["fpr"], 4),
        "delta_fpr_debate_vs_critic": round(critic_cm["fpr"] - debate_cm["fpr"], 4),
        "delta_recall_critic_vs_hunter": round(critic_cm["recall"] - hunter_cm["recall"], 4),
        "delta_recall_debate_vs_critic": round(debate_cm["recall"] - critic_cm["recall"], 4),
        "per_item_results": per_item,
    }

    output_path = results_path(args.output, subdir="debate")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Exp 6b: Three-Way Agent Decomposition ===")
    logger.info(f"  Matched findings: {len(per_item)}")
    logger.info(f"\n  Hunter-only:     P={hunter_cm['precision']:.3f}  R={hunter_cm['recall']:.3f}  "
                f"F1={hunter_cm['f1']:.3f}  FPR={hunter_cm['fpr']:.3f}  "
                f"(TP={hunter_cm['tp']} FP={hunter_cm['fp']} FN={hunter_cm['fn']} TN={hunter_cm['tn']})")
    logger.info(f"  Hunter+Critic:   P={critic_cm['precision']:.3f}  R={critic_cm['recall']:.3f}  "
                f"F1={critic_cm['f1']:.3f}  FPR={critic_cm['fpr']:.3f}  "
                f"(TP={critic_cm['tp']} FP={critic_cm['fp']} FN={critic_cm['fn']} TN={critic_cm['tn']})")
    logger.info(f"  Full debate:     P={debate_cm['precision']:.3f}  R={debate_cm['recall']:.3f}  "
                f"F1={debate_cm['f1']:.3f}  FPR={debate_cm['fpr']:.3f}  "
                f"(TP={debate_cm['tp']} FP={debate_cm['fp']} FN={debate_cm['fn']} TN={debate_cm['tn']})")
    logger.info(f"\n  FPR reduction from Critic (hunter -> hunter+critic):   {summary['delta_fpr_critic_vs_hunter']:+.4f}")
    logger.info(f"  FPR reduction from Mediator (hunter+critic -> debate): {summary['delta_fpr_debate_vs_critic']:+.4f}")
    logger.info(f"  Recall change from Critic:                             {summary['delta_recall_critic_vs_hunter']:+.4f}")
    logger.info(f"  Recall change from Mediator:                           {summary['delta_recall_debate_vs_critic']:+.4f}")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
