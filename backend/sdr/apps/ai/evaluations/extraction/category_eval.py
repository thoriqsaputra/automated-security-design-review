"""
Extraction Category Accuracy Evaluation.

Measures how accurately the LLM assigned requirement_category tags
(design / code / infrastructure / process) during extraction, compared
to a hand-labeled ground-truth set.

This is critical for thesis evaluation because the category tag gates which
requirements enter the downstream debate pipeline — a design requirement
tagged as 'code' is silently excluded from security design review.

Metrics:
  accuracy            — overall % of labels that match ground truth
  per_category        — precision, recall, F1 per category
  confusion_matrix    — 4×4 matrix: rows=true label, cols=predicted label
  high_impact_errors  — mismatches where true=design (missed debate input)
                        or predicted=design (false debate input)
  match_rate          — % of ground-truth items found in the DB
                        (measures extraction completeness as a side effect)

Two modes:
  --deterministic (default): pure comparison against ground truth JSON, no LLM.
  --explain-mismatches: additionally calls an LLM judge to reason about
    each mismatch — useful for qualitative analysis in the thesis.

Usage:
    python extraction_category_eval.py --category-code web_application
    python extraction_category_eval.py --category-code web_application --active-only
    python extraction_category_eval.py --category-code web_application \\
        --ground-truth extraction_category_ground_truth.json \\
        --explain-mismatches --output eval_category_accuracy.json
"""
import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from sdr.core.database import SessionLocal
from sdr.apps.standards.models import (
    CategoryParameterChild,
    CategoryParameterParent,
    StandardCategory,
    StandardIngestionJob,
)

from sdr.apps.ai.evaluations.shared import results_path, data_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_NUMERIC_ID_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:\.\d+)*)\b")
VALID_CATEGORIES = ["design", "code", "infrastructure", "process"]

_MISMATCH_EXPLAIN_PROMPT = """\
You are evaluating the accuracy of an automated category tag assigned to a security requirement.

Requirement text: {requirement}
True category (hand-labeled): {true_label}
Predicted category (assigned by extraction LLM): {predicted_label}

Category definitions:
- design: TSD-verifiable — architecture, data flows, trust boundaries, protocol/encryption choices, access control model, key management approach
- code: Requires source code inspection — algorithm work factors, cookie flags, parameterized queries, output encoding specifics
- infrastructure: Requires ops/config audit — CI/CD, debug mode, HTTP header values, TLS cipher suite testing, dependency versions
- process: Requires organizational assessment — SDLC policy, threat modeling cadence, developer training, change management

Is the predicted category correct? If not, briefly explain the specific reason the true category is more appropriate.
Respond in JSON: {{"is_correct": true/false, "explanation": "..."}}"""


def _load_ground_truth(gt_path: str) -> list[dict]:
    with open(gt_path) as f:
        data = json.load(f)
    return data.get("items", [])


def _build_id_index(rows: list) -> dict[str, list]:
    """Index DB rows by all X.Y.Z numeric control IDs found in requirement_text."""
    index: dict[str, list] = defaultdict(list)
    for row in rows:
        text = (row.requirement_text or "").strip()
        for cid in set(_NUMERIC_ID_RE.findall(text)):
            index[cid].append(row)
    return dict(index)


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def _explain_mismatch(requirement: str, true_label: str, predicted_label: str) -> str:
    try:
        from sdr.apps.ai.client.manager import ai_service_manager
        prompt = _MISMATCH_EXPLAIN_PROMPT.format(
            requirement=requirement[:400],
            true_label=true_label,
            predicted_label=predicted_label,
        )
        response = ai_service_manager.chat_completion_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            component="eval_judge",
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        if response.error:
            return f"[judge error: {response.error}]"
        parsed = json.loads(response.content or "{}")
        return parsed.get("explanation", "")
    except Exception as exc:
        return f"[exception: {exc}]"


def main():
    parser = argparse.ArgumentParser(
        description="Extraction Category Accuracy Evaluation."
    )
    parser.add_argument("--category-code", type=str, default=None)
    parser.add_argument("--job-id", type=int, default=None)
    parser.add_argument(
        "--active-only", action="store_true", default=False,
        help="Use only the active ingestion job for the category"
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="Path to ground truth JSON (default: bundled extraction_category_ground_truth.json)"
    )
    parser.add_argument(
        "--explain-mismatches", action="store_true", default=False,
        help="Call the LLM eval_judge to explain each mismatch (adds LLM cost)"
    )
    parser.add_argument("--output", type=str, default="eval_extraction_category.json")
    args = parser.parse_args()

    gt_path = args.ground_truth or data_path("extraction_ground_truth.json")
    if args.ground_truth and not os.path.isabs(args.ground_truth) and not os.path.exists(args.ground_truth):
        alt_path = data_path(args.ground_truth)
        if os.path.exists(alt_path):
            gt_path = alt_path

    if not os.path.exists(gt_path):
        logger.error(f"Ground truth file not found: {gt_path}")
        return

    ground_truth = _load_ground_truth(gt_path)
    logger.info(f"Loaded {len(ground_truth)} ground-truth items from {gt_path}")

    with SessionLocal() as db:
        query = db.query(CategoryParameterChild)
        job_label = "all"

        if args.job_id:
            parent_ids = [
                p.id for p in db.query(CategoryParameterParent)
                .filter_by(ingestion_job_id=args.job_id).all()
            ]
            query = query.filter(CategoryParameterChild.parent_id.in_(parent_ids))
            job_label = f"job_{args.job_id}"

        elif args.category_code:
            cat = db.query(StandardCategory).filter_by(code=args.category_code).first()
            if not cat:
                logger.error(f"Category '{args.category_code}' not found.")
                return
            if args.active_only:
                active_job = (
                    db.query(StandardIngestionJob)
                    .filter_by(category_id=cat.id, is_active=True)
                    .order_by(StandardIngestionJob.created_at.desc())
                    .first()
                )
                if not active_job:
                    logger.error(f"No active job for '{args.category_code}'.")
                    return
                parent_ids = [
                    p.id for p in db.query(CategoryParameterParent)
                    .filter_by(ingestion_job_id=active_job.id).all()
                ]
                query = query.filter(CategoryParameterChild.parent_id.in_(parent_ids))
                job_label = f"category_{args.category_code}_job_{active_job.id}_active"
            else:
                job_ids = [j.id for j in db.query(StandardIngestionJob).filter_by(category_id=cat.id).all()]
                parent_ids = [
                    p.id for p in db.query(CategoryParameterParent)
                    .filter(CategoryParameterParent.ingestion_job_id.in_(job_ids)).all()
                ]
                query = query.filter(CategoryParameterChild.parent_id.in_(parent_ids))
                job_label = f"category_{args.category_code}_all_jobs"

        rows = query.all()
        logger.info(f"Loaded {len(rows)} DB rows ({job_label})")

    id_index = _build_id_index(rows)

    # Evaluation loop
    per_item_results = []
    not_found = []

    # confusion_matrix[true][predicted] = count
    confusion: dict[str, dict[str, int]] = {c: {p: 0 for p in VALID_CATEGORIES} for c in VALID_CATEGORIES}

    matched = 0
    correct = 0
    high_impact_errors: list[dict] = []

    for gt_item in ground_truth:
        control_id = gt_item["control_id"]
        true_cat = gt_item["expected_category"].strip().lower()
        db_rows = id_index.get(control_id, [])

        if not db_rows:
            not_found.append({"control_id": control_id, "expected_category": true_cat})
            continue

        # Use the first matching row (most recent by ID if duplicates exist)
        db_row = max(db_rows, key=lambda r: r.id)
        matched += 1
        predicted_cat = (db_row.requirement_category or "unknown").strip().lower()
        if predicted_cat not in VALID_CATEGORIES:
            predicted_cat = "unknown"

        is_correct = (predicted_cat == true_cat)
        if is_correct:
            correct += 1

        if true_cat in VALID_CATEGORIES and predicted_cat in VALID_CATEGORIES:
            confusion[true_cat][predicted_cat] += 1

        # High-impact: true=design was missed (false negative → skips debate)
        #              true≠design but predicted=design (false positive → pollutes debate)
        is_high_impact = (true_cat == "design" and not is_correct) or (
            predicted_cat == "design" and true_cat != "design"
        )

        explanation = ""
        if not is_correct and args.explain_mismatches:
            logger.info(f"  [{control_id}] mismatch true={true_cat} pred={predicted_cat} — calling judge...")
            explanation = _explain_mismatch(db_row.requirement_text, true_cat, predicted_cat)

        row_result = {
            "control_id": control_id,
            "requirement_text": db_row.requirement_text[:120],
            "true_category": true_cat,
            "predicted_category": predicted_cat,
            "is_correct": is_correct,
            "is_high_impact_error": is_high_impact and not is_correct,
            "rationale": gt_item.get("rationale", ""),
            "explanation": explanation,
        }
        per_item_results.append(row_result)

        if not is_correct and is_high_impact:
            high_impact_errors.append(row_result)

        logger.info(
            f"  [{control_id}] true={true_cat} pred={predicted_cat} {'✓' if is_correct else '✗'}"
            + (" [HIGH IMPACT]" if is_high_impact and not is_correct else "")
        )

    # Per-category precision/recall/F1
    per_category_metrics = {}
    for cat in VALID_CATEGORIES:
        tp = confusion[cat][cat]
        fp = sum(confusion[other][cat] for other in VALID_CATEGORIES if other != cat)
        fn = sum(confusion[cat][other] for other in VALID_CATEGORIES if other != cat)
        per_category_metrics[cat] = _precision_recall_f1(tp, fp, fn)

    accuracy = round(correct / matched, 4) if matched else 0.0
    match_rate = round(matched / len(ground_truth), 4) if ground_truth else 0.0

    macro_precision = round(
        sum(m["precision"] for m in per_category_metrics.values()) / len(VALID_CATEGORIES), 4
    )
    macro_recall = round(
        sum(m["recall"] for m in per_category_metrics.values()) / len(VALID_CATEGORIES), 4
    )
    macro_f1 = round(
        sum(m["f1"] for m in per_category_metrics.values()) / len(VALID_CATEGORIES), 4
    )

    summary = {
        "job_label": job_label,
        "ground_truth_items": len(ground_truth),
        "matched_in_db": matched,
        "not_found_in_db": len(not_found),
        "match_rate": match_rate,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "high_impact_error_count": len(high_impact_errors),
        "thresholds": {
            "accuracy_gte_0.80": accuracy >= 0.80,
            "design_recall_gte_0.85": per_category_metrics.get("design", {}).get("recall", 0) >= 0.85,
            "match_rate_gte_0.90": match_rate >= 0.90,
        },
        "confusion_matrix": {
            "rows_are_true_label": True,
            "columns_are_predicted_label": True,
            "categories": VALID_CATEGORIES,
            "matrix": confusion,
        },
        "per_category_metrics": per_category_metrics,
        "high_impact_errors": high_impact_errors,
        "not_found_in_db": not_found,
        "per_item_results": per_item_results,
    }

    output_path = results_path(args.output, subdir="extraction")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Extraction Category Accuracy Results ===")
    logger.info(f"  Scope:             {job_label}")
    logger.info(f"  Ground truth:      {len(ground_truth)} items")
    logger.info(f"  Match rate:        {match_rate:.4f}  ({matched}/{len(ground_truth)} found in DB)")
    logger.info(f"  Accuracy:          {accuracy:.4f}  (threshold ≥0.80: {summary['thresholds']['accuracy_gte_0.80']})")
    logger.info(f"  Macro precision:   {macro_precision:.4f}")
    logger.info(f"  Macro recall:      {macro_recall:.4f}")
    logger.info(f"  Macro F1:          {macro_f1:.4f}")
    logger.info(f"  High-impact errors:{len(high_impact_errors)}  (design FN or design FP)")
    logger.info(f"\n  Per-category metrics:")
    for cat, m in per_category_metrics.items():
        logger.info(f"    {cat:15s}: P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    logger.info(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = "             " + "  ".join(f"{c:12s}" for c in VALID_CATEGORIES)
    logger.info(f"  {header}")
    for true_cat in VALID_CATEGORIES:
        row_vals = "  ".join(f"{confusion[true_cat][pred]:12d}" for pred in VALID_CATEGORIES)
        logger.info(f"  {true_cat:12s} {row_vals}")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
