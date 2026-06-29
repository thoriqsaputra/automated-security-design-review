"""
Extraction Coverage Evaluation — deterministic, no LLM required.

Measures how completely the extraction pipeline captured the ground-truth
requirements of a known security standard, using a curated gold-set of
control IDs (bundled asvs_403_gold_ids.json for OWASP ASVS 4.0.3).

Metrics:
  extraction_recall    — |extracted ∩ gold| / |gold|
                         "What fraction of known requirements did we capture?"
  extraction_precision — |extracted ∩ gold| / |extracted|
                         "What fraction of what we extracted is a real requirement?"
  f1_score             — harmonic mean of recall and precision
  missed_ids           — gold IDs not found in the extracted set (false negatives)
  extra_ids            — extracted IDs not in gold (may be from other standards,
                         or hallucinated/mis-extracted control IDs)

Per-chapter breakdown is included so you can see which ASVS chapters have
poor coverage (e.g. V12 file handling, V14 configuration).

Usage:
    python extraction_coverage_eval.py --category-code web_application
    python extraction_coverage_eval.py --category-code web_application --active-only
    python extraction_coverage_eval.py --category-code web_application --job-id 17 \\
        --gold-set asvs_403_gold_ids.json --chapter V1 --chapter V9
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


def _load_gold_set(gold_path: str, chapter_filter: list[str] | None = None) -> dict:
    """Load and optionally filter the gold ID set by chapter(s)."""
    with open(gold_path) as f:
        gold_data = json.load(f)

    chapters = gold_data.get("chapters", {})
    if chapter_filter:
        chapters = {k: v for k, v in chapters.items() if k in chapter_filter}

    gold_ids_by_chapter: dict[str, set[str]] = {}
    for chapter_key, chapter_info in chapters.items():
        gold_ids_by_chapter[chapter_key] = set(chapter_info.get("ids", []))

    all_gold_ids: set[str] = set()
    for ids in gold_ids_by_chapter.values():
        all_gold_ids |= ids

    return {
        "all_ids": all_gold_ids,
        "by_chapter": gold_ids_by_chapter,
        "meta": {k: v for k, v in gold_data.items() if k != "chapters"},
    }


def _extract_ids_from_rows(rows: list) -> dict[str, list[str]]:
    """
    For each row, extract all X.Y.Z numeric control IDs from requirement_text.
    Returns {control_id: [row_requirement_text, ...]} for dedup analysis.
    """
    id_to_texts: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        text = (row.requirement_text or "").strip()
        for cid in set(_NUMERIC_ID_RE.findall(text)):
            id_to_texts[cid].append(text[:120])
    return dict(id_to_texts)


def _chapter_of(control_id: str) -> str:
    parts = control_id.split(".")
    return f"V{parts[0]}" if parts else "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Extraction Coverage Evaluation — recall/precision vs. gold standard ID set."
    )
    parser.add_argument("--category-code", type=str, default=None)
    parser.add_argument("--job-id", type=int, default=None)
    parser.add_argument(
        "--active-only", action="store_true", default=False,
        help="Use only the active ingestion job for the category"
    )
    parser.add_argument(
        "--gold-set", type=str, default=None,
        help="Path to gold ID JSON (default: bundled asvs_403_gold_ids.json)"
    )
    parser.add_argument(
        "--chapter", action="append", dest="chapters", metavar="CHAPTER",
        help="Restrict gold set to these chapters, e.g. --chapter V1 --chapter V9"
    )
    parser.add_argument("--output", type=str, default="eval_extraction_coverage.json")
    args = parser.parse_args()

    gold_path = args.gold_set or data_path("asvs_403_gold_ids.json")
    if not os.path.exists(gold_path):
        logger.error(f"Gold set file not found: {gold_path}")
        return

    gold = _load_gold_set(gold_path, chapter_filter=args.chapters)
    gold_ids = gold["all_ids"]
    logger.info(f"Gold set: {len(gold_ids)} IDs across {len(gold['by_chapter'])} chapters")

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
        logger.info(f"Loaded {len(rows)} rows ({job_label})")

    extracted_map = _extract_ids_from_rows(rows)
    extracted_ids = set(extracted_map.keys())

    # Core metrics
    tp_ids = gold_ids & extracted_ids
    missed_ids = gold_ids - extracted_ids
    extra_ids = extracted_ids - gold_ids

    recall = round(len(tp_ids) / len(gold_ids), 4) if gold_ids else 0.0
    precision = round(len(tp_ids) / len(extracted_ids), 4) if extracted_ids else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

    # Per-chapter breakdown
    chapter_results = {}
    for chapter, chapter_gold_ids in sorted(gold["by_chapter"].items()):
        chapter_extracted = {cid for cid in extracted_ids if _chapter_of(cid) == chapter}
        c_tp = chapter_gold_ids & chapter_extracted
        c_missed = chapter_gold_ids - chapter_extracted
        c_extra = chapter_extracted - chapter_gold_ids
        c_recall = round(len(c_tp) / len(chapter_gold_ids), 4) if chapter_gold_ids else 0.0
        c_precision = round(len(c_tp) / len(chapter_extracted), 4) if chapter_extracted else 0.0
        chapter_results[chapter] = {
            "gold_count": len(chapter_gold_ids),
            "extracted_count": len(chapter_extracted),
            "matched": len(c_tp),
            "recall": c_recall,
            "precision": c_precision,
            "missed": sorted(c_missed),
            "extra": sorted(c_extra),
        }

    summary = {
        "job_label": job_label,
        "gold_set": gold["meta"].get("standard", gold_path),
        "gold_id_count": len(gold_ids),
        "extracted_id_count": len(extracted_ids),
        "matched_count": len(tp_ids),
        "missed_count": len(missed_ids),
        "extra_count": len(extra_ids),
        "extraction_recall": recall,
        "extraction_precision": precision,
        "f1_score": f1,
        "thresholds": {
            "recall_gte_0.90": recall >= 0.90,
            "precision_gte_0.95": precision >= 0.95,
            "f1_gte_0.90": f1 >= 0.90,
        },
        "by_chapter": chapter_results,
        "missed_ids": sorted(missed_ids),
        "extra_ids": sorted(extra_ids),
    }

    output_path = results_path(args.output, subdir="extraction")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Extraction Coverage Results ===")
    logger.info(f"  Scope:           {job_label}")
    logger.info(f"  Gold IDs:        {len(gold_ids)}")
    logger.info(f"  Extracted IDs:   {len(extracted_ids)}")
    logger.info(f"  Matched:         {len(tp_ids)}")
    logger.info(f"  Recall:          {recall:.4f}  (threshold ≥0.90: {summary['thresholds']['recall_gte_0.90']})")
    logger.info(f"  Precision:       {precision:.4f}  (threshold ≥0.95: {summary['thresholds']['precision_gte_0.95']})")
    logger.info(f"  F1:              {f1:.4f}  (threshold ≥0.90: {summary['thresholds']['f1_gte_0.90']})")
    logger.info(f"  Missed ({len(missed_ids)}):  {sorted(missed_ids)[:10]}{'...' if len(missed_ids) > 10 else ''}")
    logger.info(f"  Extra ({len(extra_ids)}):    {sorted(extra_ids)[:10]}{'...' if len(extra_ids) > 10 else ''}")
    logger.info("\n  Per-chapter recall:")
    for ch, vals in sorted(chapter_results.items()):
        status = "✓" if vals["recall"] >= 0.90 else "✗"
        logger.info(f"    {status} {ch}: recall={vals['recall']:.2f}  matched={vals['matched']}/{vals['gold_count']}")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
