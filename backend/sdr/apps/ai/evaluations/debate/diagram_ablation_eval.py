"""
Diagram debate ablation: Hunter-only vs full 3-agent debate (False Positive Rate),
for diagram findings.

Same thesis as debate/ablation_eval.py (the Critic->Mediator loop suppresses false
positives the Hunter alone would produce) but matched at (diagram_id, requirement_id)
granularity: a single diagram Finding can cover multiple requirements at once
(Finding.requirement_metadata["analysis_trace"]["assessed_requirements"]), unlike text
findings where one Finding == one requirement.

Hunter verdict (per requirement):  analysis_trace.hunter_result.requirement_assessments[*].verdict
Debate final verdict (per requirement): analysis_trace.assessed_requirements[*].verdict
    (Mediator's per-requirement verdict, AFTER the evidence-grounding policy in
    sdr/apps/ai/agents/vision.py::_apply_diagram_evidence_policy has been applied —
    this is NOT the same as Finding.met_status, which is only the finding-level
    worst-case rollup across all assessed requirements.)

Ground truth is the same labeled `diagram_ground_truth_review_<id>.json` file used by
retrieval/diagram_retrieval_eval.py (see evaluations/data/build_diagram_ground_truth_template.py):
for each diagram, candidate_requirements with relevant=true and a filled-in `label` are
used as (diagram_id, requirement_id) -> label ground truth.

Usage:
    python diagram_ablation_eval.py --review-id 12 \\
        --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_review_12.json
"""
import argparse
import json
import logging
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import sdr.apps.standards.models  # noqa: F401 — resolve SQLAlchemy FK
import sdr.apps.designs.models    # noqa: F401
import sdr.apps.reviews.models.finding  # noqa: F401
import sdr.apps.reviews.models.review   # noqa: F401

from sdr.core.database import SessionLocal
from sdr.apps.reviews.models.finding import Finding
from sdr.apps.reviews.models.choices import FindingType

from sdr.apps.ai.evaluations.shared import results_path
from sdr.apps.ai.evaluations.shared.metrics import calculate_binary_confusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_COMPOSITE_REQ_RE = re.compile(r"(job\d+-D-composite-[A-Za-z0-9-]+)")


def _normalize_requirement_id(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    match = _COMPOSITE_REQ_RE.search(raw)
    if match:
        return match.group(1)
    if raw.startswith("job") and "-D-composite-" in raw:
        return raw
    return raw


def _load_ground_truth(gt_path: str) -> dict[tuple[str, str], str]:
    """Load {(diagram_id, requirement_id): label} from a labeled diagram ground-truth JSON."""
    with open(gt_path) as f:
        data = json.load(f)
    gt: dict[tuple[str, str], str] = {}
    for item in data.get("items", []):
        diagram_id = str(item.get("diagram_id", "")).strip()
        for req in item.get("candidate_requirements", []):
            if req.get("relevant") is not True:
                continue
            label = req.get("label")
            requirement_id = _normalize_requirement_id(req.get("requirement_id", ""))
            if not diagram_id or not requirement_id or not label:
                continue
            gt[(diagram_id, requirement_id)] = str(label).lower().strip()
    return gt


def _per_requirement_verdicts(assessments: list) -> dict[str, str]:
    verdicts = {}
    for item in assessments or []:
        if not isinstance(item, dict):
            continue
        requirement_id = _normalize_requirement_id(item.get("requirement_id", ""))
        verdict = str(item.get("verdict", "")).strip().lower()
        if requirement_id and verdict:
            verdicts[requirement_id] = verdict
    return verdicts


def main():
    parser = argparse.ArgumentParser(
        description="Diagram debate ablation: Hunter-only vs multi-agent debate (FPR comparison)."
    )
    parser.add_argument("--review-id", type=int, required=True)
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to a labeled diagram ground-truth JSON (see build_diagram_ground_truth_template.py)"
    )
    parser.add_argument("--output", type=str, default="diagram_debate_ablation_results.json")
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        return

    gt = _load_ground_truth(args.ground_truth)
    logger.info(f"Loaded {len(gt)} ground-truth (diagram_id, requirement_id) labels")

    with SessionLocal() as db:
        findings = (
            db.query(Finding)
            .filter(
                Finding.review_id == args.review_id,
                Finding.finding_type == FindingType.DIAGRAM.value,
            )
            .all()
        )
    logger.info(f"Loaded {len(findings)} diagram findings for review_id={args.review_id}")

    if not findings:
        logger.error("No diagram findings found. Run a security design review with vision enabled first.")
        return

    per_item = []
    hunter_labels, hunter_preds = [], []
    debate_labels, debate_preds = [], []
    disagreements = []

    for f in findings:
        meta = f.requirement_metadata or {}
        trace = meta.get("analysis_trace", {})
        hunter_verdicts = _per_requirement_verdicts(
            trace.get("hunter_result", {}).get("requirement_assessments", [])
        )
        debate_verdicts = _per_requirement_verdicts(trace.get("assessed_requirements", []))

        requirement_ids = set(hunter_verdicts) | set(debate_verdicts)
        for requirement_id in requirement_ids:
            key = (f.diagram_id, requirement_id)
            if key not in gt:
                continue

            true_label = gt[key]
            hunter_verdict = hunter_verdicts.get(requirement_id)
            debate_verdict = debate_verdicts.get(requirement_id)

            if hunter_verdict is None:
                logger.warning(f"  [diagram={f.diagram_id} req={requirement_id}] no hunter verdict — skipping")
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
                "diagram_id": f.diagram_id,
                "requirement_id": requirement_id,
                "true_label": true_label,
                "hunter_verdict": hunter_verdict,
                "debate_verdict": debate_verdict,
                "hunter_correct": hunter_correct,
                "debate_correct": debate_correct,
                "verdict_changed_by_debate": changed,
            }
            per_item.append(row)

            if changed:
                disagreements.append(row)
                direction = "fixed" if (not hunter_correct and debate_correct) else (
                    "broken" if (hunter_correct and not debate_correct) else "still wrong"
                )
                logger.info(
                    f"  [diagram={f.diagram_id} req={requirement_id}] hunter={hunter_verdict} "
                    f"-> debate={debate_verdict} (true={true_label}) {direction}"
                )

    if not per_item:
        logger.error("No (diagram_id, requirement_id) pairs matched ground-truth labels.")
        return

    hunter_cm = calculate_binary_confusion(hunter_labels, hunter_preds)
    debate_cm = calculate_binary_confusion(debate_labels, debate_preds)
    delta_fpr = round(hunter_cm["fpr"] - debate_cm["fpr"], 4)

    # Cases where a single-model (Hunter-only) verdict was wrong and the
    # multi-agent debate corrected it — direct evidence that the debate
    # mitigates single-model (visual) blindness, not just FPR in the aggregate.
    blindness_mitigation_cases = [
        row for row in disagreements if not row["hunter_correct"] and row["debate_correct"]
    ]

    summary = {
        "review_id": args.review_id,
        "ground_truth_items": len(gt),
        "matched_pairs": len(per_item),
        "verdict_changes": len(disagreements),
        "hunter_only": hunter_cm,
        "debate_final": debate_cm,
        "delta_fpr": delta_fpr,
        "fpr_suppression_achieved": delta_fpr >= 0,
        "blindness_mitigation_cases": blindness_mitigation_cases,
        "blindness_mitigation_rate": round(len(blindness_mitigation_cases) / len(per_item), 4),
        "per_item_results": per_item,
        "disagreement_cases": disagreements,
    }

    output_path = results_path(args.output, subdir="debate")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Diagram Debate Ablation Results ===")
    logger.info(f"  Matched (diagram, requirement) pairs: {len(per_item)}")
    logger.info(f"  Verdict changes: {len(disagreements)} ({100 * len(disagreements) / len(per_item):.1f}%)")
    logger.info(
        f"\n  Hunter-only:  P={hunter_cm['precision']:.3f}  R={hunter_cm['recall']:.3f}  "
        f"F1={hunter_cm['f1']:.3f}  FPR={hunter_cm['fpr']:.3f}  "
        f"(TP={hunter_cm['tp']} FP={hunter_cm['fp']} FN={hunter_cm['fn']} TN={hunter_cm['tn']})"
    )
    logger.info(
        f"  Debate final: P={debate_cm['precision']:.3f}  R={debate_cm['recall']:.3f}  "
        f"F1={debate_cm['f1']:.3f}  FPR={debate_cm['fpr']:.3f}  "
        f"(TP={debate_cm['tp']} FP={debate_cm['fp']} FN={debate_cm['fn']} TN={debate_cm['tn']})"
    )
    logger.info(
        f"\n  Delta FPR (hunter - debate): {delta_fpr:+.4f}  "
        f"({'FPR not increased' if delta_fpr >= 0 else 'FPR increased'})"
    )
    logger.info(
        f"  Blindness-mitigation cases (Hunter wrong -> debate correct): "
        f"{len(blindness_mitigation_cases)}/{len(per_item)} "
        f"({100 * summary['blindness_mitigation_rate']:.1f}%)"
    )
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
