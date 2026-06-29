"""
Extraction Purity Evaluation — deterministic, no LLM required.

Measures the quality of the extraction pipeline's output at the schema level:

  control_id_purity   — fraction of rows that contain a recognized control ID
                        (X.Y.Z numeric or alphanumeric like REQ-INP-01).
                        Garbage entries (section preambles, CIA definitions,
                        advisory paragraphs) lack any such ID and lower this score.

  duplicate_rate      — fraction of extracted requirements that share an identical
                        normalized control ID with another row in the same job.
                        The canonicalization step should keep this near zero.

  category_distribution — count and % per requirement_category tag
                        (design / code / infrastructure / process / unknown).

  empty_rate          — fraction of rows with a blank requirement_text.

Run against a specific ingestion job or all rows for a category:

    python extraction_purity_eval.py --category-code web_application
    python extraction_purity_eval.py --job-id 17
    python extraction_purity_eval.py --category-code web_application --output my_purity.json
"""
import argparse
import json
import logging
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from sdr.core.database import SessionLocal
from sdr.apps.standards.models import (
    CategoryParameterChild,
    CategoryParameterParent,
    StandardCategory,
    StandardIngestionJob,
)

from sdr.apps.ai.evaluations.shared import results_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Same regex used by the extraction normaliser.
_CONTROL_ID_RE = re.compile(r"\b(?:[A-Z]{2,}(?:-[A-Z0-9]+)+|\d+\.\d+\.\d+(?:\.\d+)*)\b")
_NUMERIC_ID_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:\.\d+)*)\b")

VALID_CATEGORIES = {"design", "code", "infrastructure", "process"}


def _extract_control_ids(text: str) -> list[str]:
    return _CONTROL_ID_RE.findall(text or "")


def _primary_numeric_id(text: str) -> str | None:
    """Return the first X.Y.Z numeric control ID found in the text, or None."""
    m = _NUMERIC_ID_RE.search(text or "")
    return m.group(1) if m else None


def _chapter(control_id: str) -> str:
    """Return 'V<major>' chapter label from a numeric control ID."""
    parts = control_id.split(".")
    return f"V{parts[0]}" if parts else "unknown"


def _run_purity_eval(rows: list, job_label: str) -> dict:
    total = len(rows)
    if total == 0:
        return {"total": 0, "job_label": job_label}

    has_control_id = 0
    empty_requirement = 0
    id_counter: dict[str, list[int]] = {}
    category_counts: dict[str, int] = {}
    chapter_counts: dict[str, int] = {}
    garbage_rows: list[dict] = []

    for row in rows:
        req_text = (row.requirement_text or "").strip()

        if not req_text:
            empty_requirement += 1
            continue

        ids = _extract_control_ids(req_text)
        if ids:
            has_control_id += 1
            primary = _primary_numeric_id(req_text)
            if primary:
                id_counter.setdefault(primary, []).append(row.id)
                chapter_counts[_chapter(primary)] = chapter_counts.get(_chapter(primary), 0) + 1
        else:
            garbage_rows.append({"id": row.id, "requirement_text": req_text[:120]})

        cat = (row.requirement_category or "unknown").strip().lower()
        if cat not in VALID_CATEGORIES:
            cat = "unknown"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Duplicates: any control ID that maps to more than one row.
    duplicate_ids = {k: v for k, v in id_counter.items() if len(v) > 1}
    duplicate_rows = sum(len(v) - 1 for v in duplicate_ids.values())

    purity = round(has_control_id / total, 4)
    duplicate_rate = round(duplicate_rows / total, 4)
    empty_rate = round(empty_requirement / total, 4)

    return {
        "job_label": job_label,
        "total_rows": total,
        "control_id_purity": purity,
        "duplicate_rate": duplicate_rate,
        "empty_rate": empty_rate,
        "garbage_count": len(garbage_rows),
        "duplicate_count": duplicate_rows,
        "unique_control_ids": len(id_counter),
        "category_distribution": {
            cat: {"count": cnt, "pct": round(cnt / total * 100, 1)}
            for cat, cnt in sorted(category_counts.items())
        },
        "chapter_coverage": dict(sorted(chapter_counts.items())),
        "garbage_samples": garbage_rows[:20],
        "duplicate_ids": {k: v for k, v in list(duplicate_ids.items())[:20]},
        "thresholds": {
            "control_id_purity_gte_0.99": purity >= 0.99,
            "duplicate_rate_lte_0.01": duplicate_rate <= 0.01,
            "empty_rate_lte_0.01": empty_rate <= 0.01,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extraction Purity Evaluation — deterministic quality check on extracted requirements."
    )
    parser.add_argument("--category-code", type=str, default=None, help="Filter by StandardCategory.code")
    parser.add_argument("--job-id", type=int, default=None, help="Filter by a specific ingestion job ID")
    parser.add_argument(
        "--active-only", action="store_true", default=False,
        help="Restrict to the active ingestion job for the category (ignored if --job-id is set)"
    )
    parser.add_argument("--output", type=str, default="eval_extraction_purity.json")
    args = parser.parse_args()

    with SessionLocal() as db:
        query = db.query(CategoryParameterChild)

        job_label = "all"

        if args.job_id:
            job = db.get(StandardIngestionJob, args.job_id)
            if not job:
                logger.error(f"Job {args.job_id} not found.")
                return
            # Children don't store job_id directly — filter through parent
            parent_ids = [
                p.id for p in db.query(CategoryParameterParent)
                .filter(CategoryParameterParent.ingestion_job_id == args.job_id)
                .all()
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
                    logger.error(f"No active job for category '{args.category_code}'.")
                    return
                parent_ids = [
                    p.id for p in db.query(CategoryParameterParent)
                    .filter(CategoryParameterParent.ingestion_job_id == active_job.id)
                    .all()
                ]
                query = query.filter(CategoryParameterChild.parent_id.in_(parent_ids))
                job_label = f"category_{args.category_code}_job_{active_job.id}_active"
            else:
                # All jobs for this category
                job_ids = [j.id for j in db.query(StandardIngestionJob).filter_by(category_id=cat.id).all()]
                parent_ids = [
                    p.id for p in db.query(CategoryParameterParent)
                    .filter(CategoryParameterParent.ingestion_job_id.in_(job_ids))
                    .all()
                ]
                query = query.filter(CategoryParameterChild.parent_id.in_(parent_ids))
                job_label = f"category_{args.category_code}_all_jobs"

        rows = query.all()
        logger.info(f"Loaded {len(rows)} rows ({job_label})")

        result = _run_purity_eval(rows, job_label)

    output_path = results_path(args.output, subdir="extraction")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=== Extraction Purity Results ===")
    logger.info(f"  Job/Scope:         {result['job_label']}")
    logger.info(f"  Total rows:        {result['total_rows']}")
    logger.info(f"  Control ID purity: {result['control_id_purity']:.4f}  (threshold ≥0.99: {result['thresholds']['control_id_purity_gte_0.99']})")
    logger.info(f"  Duplicate rate:    {result['duplicate_rate']:.4f}  (threshold ≤0.01: {result['thresholds']['duplicate_rate_lte_0.01']})")
    logger.info(f"  Empty rate:        {result['empty_rate']:.4f}  (threshold ≤0.01: {result['thresholds']['empty_rate_lte_0.01']})")
    logger.info(f"  Garbage entries:   {result['garbage_count']}")
    logger.info(f"  Unique control IDs:{result['unique_control_ids']}")
    logger.info(f"  Category dist:     {result['category_distribution']}")
    logger.info(f"  Chapter coverage:  {result['chapter_coverage']}")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
