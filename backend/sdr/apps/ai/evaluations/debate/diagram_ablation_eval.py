"""
Diagram debate ablation: Hunter-only vs full 3-agent debate (False Positive Rate),
for diagram findings.

Default behavior is a LIVE paired rerun: Hunter-only and full debate are taken
from the same current-code execution over the same diagram/requirement batch, so
the comparison is a real ablation instead of a comparison against historical
stored traces that may have used different prompts, models, or mediator policy.

Ground truth is the same labeled `diagram_ground_truth_design_<id>.json` file used by
retrieval/diagram_retrieval_eval.py (see evaluations/data/build_diagram_ground_truth_template.py):
for each diagram, candidate_requirements with relevant=true and a filled-in `label` are
used as (diagram_id, requirement_id) -> label ground truth.

Usage:
    python diagram_ablation_eval.py --design-id 12 \\
        --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_design_12.json
"""
import argparse
import json
import logging
import os
import re
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
from sdr.apps.reviews.models.review import Review

from sdr.apps.ai.engine.debate.diagram_debate_service import DiagramDebateService
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


def _per_requirement_items(assessments: list) -> dict[str, dict]:
    items: dict[str, dict] = {}
    for item in assessments or []:
        if not isinstance(item, dict):
            continue
        requirement_id = _normalize_requirement_id(item.get("requirement_id", ""))
        if requirement_id:
            items[requirement_id] = dict(item)
    return items


def _critic_reviews_by_requirement_id(critic_result: dict) -> dict[str, dict]:
    reviews: dict[str, dict] = {}
    for review in critic_result.get("requirement_reviews") or []:
        if not isinstance(review, dict):
            continue
        requirement_id = _normalize_requirement_id(review.get("requirement_id", ""))
        if requirement_id:
            reviews[requirement_id] = dict(review)
    for review in critic_result.get("validated_requirements") or []:
        if not isinstance(review, dict):
            continue
        requirement_id = _normalize_requirement_id(review.get("requirement_id", ""))
        if requirement_id and requirement_id not in reviews:
            reviews[requirement_id] = {
                "requirement_id": review.get("requirement_id"),
                "critic_verdict": review.get("verdict"),
                "disposition": "uphold",
                "reason": review.get("reason"),
            }
    for review in critic_result.get("invalidated_requirements") or []:
        if not isinstance(review, dict):
            continue
        requirement_id = _normalize_requirement_id(review.get("requirement_id", ""))
        if requirement_id and requirement_id not in reviews:
            reviews[requirement_id] = {
                "requirement_id": review.get("requirement_id"),
                "critic_verdict": review.get("corrected_verdict") or review.get("verdict"),
                "disposition": "overturn",
                "reason": review.get("reason"),
            }
    return reviews


def _load_ground_truth_labels(gt_path: str) -> dict[tuple[str, str], str]:
    gt_data = load_ground_truth(gt_path)
    samples = load_labeled_samples(gt_data, labels=("met", "not_met", "na"))
    labels: dict[tuple[str, str], str] = {}
    for sample in samples:
        labels[(sample.diagram_id, _normalize_requirement_id(sample.requirement_id))] = sample.label
    return labels


def _build_diag_counters() -> dict[str, int]:
    return {
        "review_rows": 0,
        "met_rejected_by_critic": 0,
        "not_met_promoted_by_critic": 0,
        "partial_compound_flags": 0,
        "verification_rows": 0,
        "hunter_rebuttal_rows": 0,
        "judge_rows": 0,
        "judge_overrode_hunter": 0,
        "judge_overrode_critic": 0,
        "judge_side_took_without_change": 0,
        "judge_outcome_consistency_violations": 0,
        "critic_retry_batches": 0,
        "critic_fallback_rows": 0,
        "mediator_independent_rows": 0,
    }


def _update_diag_counters(
    counters: dict[str, int],
    critic_result: dict,
    hunter_rebuttal_result: dict,
    mediator_result: dict,
    ) -> None:
    for review in critic_result.get("requirement_reviews") or []:
        if not isinstance(review, dict):
            continue
        counters["review_rows"] += 1
        hunter_verdict = str(review.get("hunter_verdict", "")).strip().lower()
        critic_verdict = str(review.get("critic_verdict", "")).strip().lower()
        if hunter_verdict == "met" and critic_verdict != "met":
            counters["met_rejected_by_critic"] += 1
        if hunter_verdict in {"met", "na"} and critic_verdict == "not_met":
            counters["not_met_promoted_by_critic"] += 1
        if str(review.get("failure_mode", "")).strip().lower() == "partial_compound":
            counters["partial_compound_flags"] += 1
        if review.get("verification_checks"):
            counters["verification_rows"] += 1
    for rebuttal in hunter_rebuttal_result.get("rebuttal_requirements") or []:
        if isinstance(rebuttal, dict) and str(rebuttal.get("requirement_id", "")).strip():
            counters["hunter_rebuttal_rows"] += 1
    critic_batch_diagnostics = critic_result.get("batch_diagnostics") or {}
    mediator_batch_diagnostics = mediator_result.get("batch_diagnostics") or {}
    counters["critic_retry_batches"] += int(critic_batch_diagnostics.get("retry_batches", 0) or 0)
    counters["critic_fallback_rows"] += int(critic_batch_diagnostics.get("fallback_rows", 0) or 0)
    counters["mediator_independent_rows"] += int(mediator_batch_diagnostics.get("independent_rows", 0) or 0)


def _update_diag_counters_from_row(counters: dict[str, int], row: dict) -> None:
    counters["judge_rows"] += 1
    final_decision_source = str(row.get("final_decision_source", "")).strip().lower()
    winning_side = str(row.get("winning_side", "")).strip().lower()
    hunter_verdict = str(row.get("hunter_verdict", "")).strip().lower()
    critic_verdict = str(row.get("critic_verdict", "")).strip().lower()
    debate_verdict = str(row.get("debate_verdict", "")).strip().lower()

    if final_decision_source == "mediator_tiebreak_to_critic":
        counters["judge_overrode_hunter"] += 1
    elif final_decision_source == "mediator_tiebreak_to_hunter":
        counters["judge_overrode_critic"] += 1

    if winning_side in {"hunter", "critic"} and debate_verdict == hunter_verdict:
        counters["judge_side_took_without_change"] += 1

    if final_decision_source in {"mediator_tiebreak_to_critic", "critic_corrected"} and debate_verdict == hunter_verdict:
        counters["judge_outcome_consistency_violations"] += 1
    if final_decision_source == "mediator_tiebreak_to_hunter" and critic_verdict and debate_verdict == critic_verdict:
        counters["judge_outcome_consistency_violations"] += 1


def _build_per_item_row(
    *,
    diagram_id: str,
    requirement_id: str,
    true_label: str,
    hunter_verdict: str,
    debate_verdict: str,
    hunter_item: dict | None = None,
    debate_item: dict | None = None,
    critic_review: dict | None = None,
    finding_id: int | None = None,
) -> dict:
    hunter_item = dict(hunter_item or {})
    debate_item = dict(debate_item or {})
    critic_review = dict(critic_review or {})
    final_decision_source = (
        str(debate_item.get("final_decision_source") or "").strip().lower()
        or ("hunter_preserved" if debate_verdict == hunter_verdict else "")
    )
    row = {
        "diagram_id": diagram_id,
        "requirement_id": requirement_id,
        "true_label": true_label,
        "hunter_verdict": hunter_verdict,
        "critic_verdict": str(
            debate_item.get("critic_verdict")
            or critic_review.get("critic_verdict")
            or ""
        ).strip().lower() or None,
        "debate_verdict": debate_verdict,
        "final_verdict": debate_verdict,
        "hunter_correct": hunter_verdict == true_label,
        "debate_correct": debate_verdict == true_label,
        "verdict_changed_by_debate": hunter_verdict != debate_verdict,
        "resolution_basis": debate_item.get("resolution_basis"),
        "winning_side": debate_item.get("winning_side"),
        "judge_reason": debate_item.get("judge_reason"),
        "verdict_policy_source": debate_item.get("verdict_policy_source"),
        "final_decision_source": final_decision_source or None,
        "critic_reason": critic_review.get("reason") or None,
        "critic_scope_evidence": critic_review.get("scope_evidence") or None,
        "critic_absence_evidence": critic_review.get("absence_evidence") or None,
        "critic_failure_mode": critic_review.get("failure_mode") or None,
        "critic_compound_status": critic_review.get("compound_status") or None,
    }
    categories = []
    if row["critic_verdict"] == "not_met" and not row["critic_absence_evidence"]:
        categories.append("critic_not_met_without_absence_evidence")
    if str(critic_review.get("failure_mode", "")).strip().lower() == "partial_compound":
        categories.append("compound_partial_flip")
    if (
        final_decision_source == "mediator_tiebreak_to_critic"
        and debate_verdict == "not_met"
        and not row["critic_absence_evidence"]
    ):
        categories.append("mediator_upheld_critic_without_structured_support")
    row["disagreement_categories"] = categories
    if finding_id is not None:
        row["finding_id"] = finding_id
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Diagram debate ablation: Hunter-only vs multi-agent debate (FPR comparison)."
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
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument(
        "--cheap-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow the debate service to skip the mediator on Critic uphold. Defaults to false for a real full-debate ablation.",
    )
    parser.add_argument(
        "--use-stored-review",
        action="store_true",
        help="Legacy mode: compare against diagram findings already stored on a completed review instead of rerunning the current debate stack live.",
    )
    parser.add_argument("--output", type=str, default="diagram_debate_ablation_results.json")
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
    disagreements = []
    diag_counters = _build_diag_counters()
    comparison_mode = "stored_review" if args.use_stored_review else "paired_live"

    if args.use_stored_review:
        with SessionLocal() as db:
            findings = (
                db.query(Finding)
                .filter(
                    Finding.review_id == review.id,
                    Finding.finding_type == FindingType.DIAGRAM.value,
                )
                .all()
            )
        logger.info(f"Loaded {len(findings)} stored diagram findings for review_id={review.id}")

        if not findings:
            logger.error("No diagram findings found. Run a security design review with vision enabled first.")
            return

        for f in findings:
            meta = f.requirement_metadata or {}
            trace = meta.get("analysis_trace", {})
            hunter_verdicts = _per_requirement_verdicts(
                trace.get("hunter_result", {}).get("requirement_assessments", [])
            )
            debate_verdicts = _per_requirement_verdicts(trace.get("assessed_requirements", []))
            hunter_items = _per_requirement_items(
                trace.get("hunter_result", {}).get("requirement_assessments", [])
            )
            debate_items = _per_requirement_items(trace.get("assessed_requirements", []))
            critic_reviews = _critic_reviews_by_requirement_id(trace.get("critic_result", {}))

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

                row = _build_per_item_row(
                    finding_id=f.id,
                    diagram_id=f.diagram_id,
                    requirement_id=requirement_id,
                    true_label=true_label,
                    hunter_verdict=hunter_verdict,
                    debate_verdict=debate_verdict,
                    hunter_item=hunter_items.get(requirement_id),
                    debate_item=debate_items.get(requirement_id),
                    critic_review=critic_reviews.get(requirement_id),
                )
                per_item.append(row)
                _update_diag_counters_from_row(diag_counters, row)

                if changed:
                    disagreements.append(row)
    else:
        gt_data = load_ground_truth(gt_path)
        samples = load_labeled_samples(gt_data, labels=("met", "not_met", "na"))
        by_diagram: dict[str, list] = defaultdict(list)
        for sample in samples:
            by_diagram[sample.diagram_id].append(sample)

        tsd_doc = load_tsd_document(args.design_id)
        service = DiagramDebateService()

        for diagram_id, diagram_samples in by_diagram.items():
            diagram = build_diagram_input(tsd_doc, diagram_id, f"debate_eval_{diagram_id}", blank_text=False)
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
            output = service.run_diagram_debate_voted(
                diagram=diagram,
                requirements=requirements,
                tsd_context="",
                votes=args.votes,
                skip_mediator_on_uphold=args.cheap_mode,
            )
            hunter_verdicts = _per_requirement_verdicts(
                (output.hunter_result or {}).get("requirement_assessments", [])
            )
            debate_verdicts = _per_requirement_verdicts(
                (output.mediator_result or {}).get("assessed_requirements", [])
            )
            hunter_items = _per_requirement_items(
                (output.hunter_result or {}).get("requirement_assessments", [])
            )
            debate_items = _per_requirement_items(
                (output.mediator_result or {}).get("assessed_requirements", [])
            )
            critic_reviews = _critic_reviews_by_requirement_id(output.critic_result or {})
            _update_diag_counters(
                diag_counters,
                output.critic_result or {},
                output.hunter_rebuttal_result or {},
                output.mediator_result or {},
            )

            for sample in diagram_samples:
                requirement_id = _normalize_requirement_id(sample.requirement_id)
                hunter_verdict = hunter_verdicts.get(requirement_id)
                debate_verdict = debate_verdicts.get(requirement_id)
                if hunter_verdict is None:
                    logger.warning(f"  [diagram={diagram_id} req={requirement_id}] no hunter verdict — skipping")
                    continue

                true_label = sample.label
                hunter_labels.append(true_label)
                hunter_preds.append(hunter_verdict)
                debate_labels.append(true_label)
                debate_preds.append(debate_verdict)

                hunter_correct = hunter_verdict == true_label
                debate_correct = debate_verdict == true_label
                changed = hunter_verdict != debate_verdict
                row = _build_per_item_row(
                    diagram_id=diagram_id,
                    requirement_id=requirement_id,
                    true_label=true_label,
                    hunter_verdict=hunter_verdict,
                    debate_verdict=debate_verdict,
                    hunter_item=hunter_items.get(requirement_id),
                    debate_item=debate_items.get(requirement_id),
                    critic_review=critic_reviews.get(requirement_id),
                )
                per_item.append(row)
                _update_diag_counters_from_row(diag_counters, row)
                if changed:
                    disagreements.append(row)

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
    hunter_met_total = sum(1 for pred in hunter_preds if pred == "met")
    hunter_overclaim_total = sum(
        1 for row in per_item
        if row["hunter_verdict"] == "met" and row["true_label"] == "not_met"
    )

    summary = {
        "design_id": args.design_id,
        "review_id": review.id,
        "comparison_mode": comparison_mode,
        "ground_truth_path": gt_path,
        "votes": args.votes,
        "cheap_mode": args.cheap_mode,
        "ground_truth_items": len(gt),
        "matched_pairs": len(per_item),
        "verdict_changes": len(disagreements),
        "hunter_only": hunter_cm,
        "debate_final": debate_cm,
        "delta_fpr": delta_fpr,
        "critic_retry_batches": diag_counters["critic_retry_batches"],
        "critic_fallback_rows": diag_counters["critic_fallback_rows"],
        "mediator_independent_rows": diag_counters["mediator_independent_rows"],
        "debate_batch_max_concurrency": getattr(service, "_debate_batch_max_concurrency", None) if not args.use_stored_review else None,
        "rebuttal_batch_max_concurrency": getattr(service, "_rebuttal_batch_max_concurrency", None) if not args.use_stored_review else None,
        "fpr_suppression_achieved": delta_fpr >= 0,
        "outcome_override_consistency_ok": (
            diag_counters["judge_overrode_hunter"] <= len(disagreements)
            and diag_counters["judge_overrode_critic"] <= len(disagreements)
            and diag_counters["judge_outcome_consistency_violations"] == 0
        ),
        "critic_diagnostics": {
            **diag_counters,
            "critic_gate_engagement_rate": round(
                diag_counters["verification_rows"] / max(diag_counters["review_rows"], 1),
                4,
            ),
            "hunter_rebuttal_used_rate": round(
                diag_counters["hunter_rebuttal_rows"] / max(diag_counters["review_rows"], 1),
                4,
            ),
            "judge_overrode_hunter_rate": round(
                diag_counters["judge_overrode_hunter"] / max(diag_counters["judge_rows"], 1),
                4,
            ),
            "judge_overrode_critic_rate": round(
                diag_counters["judge_overrode_critic"] / max(diag_counters["judge_rows"], 1),
                4,
            ),
            "hunter_overclaim_rate": round(
                hunter_overclaim_total / max(hunter_met_total, 1),
                4,
            ),
        },
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
