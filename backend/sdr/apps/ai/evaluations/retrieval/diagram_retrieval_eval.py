"""
Diagram requirement retrieval eval: vector search vs naive fallback.

Compares the production `DiagramRequirementSelector.select_for_diagram` (embeds the
diagram's caption/surrounding_text and does cosine-distance search over
CategoryDiagramRequirementEmbedding) against the same naive fallback path production
falls back to on embedding/search failure (`list_diagram_requirements()[:top_k]`,
ordinal order, no ranking).

Ground truth is a human-labeled `diagram_ground_truth_review_<id>.json` file (see
`evaluations/data/build_diagram_ground_truth_template.py`), where each diagram lists
the FULL candidate pool of diagram requirements for its category with a `relevant`
flag — not just the subset the system happened to retrieve. This is what makes
recall (not just precision) measurable: a requirement the vector search never
retrieved can still be graded as a miss.

Metrics per strategy (averaged across ground-truth diagrams):
  precision      — |retrieved ∩ relevant| / |retrieved|
  recall         — |retrieved ∩ relevant| / |relevant|
  hit_rate       — fraction of diagrams where >=1 relevant requirement was retrieved
  mrr            — 1/rank of the first relevant requirement in the ranked list

Usage:
    python diagram_retrieval_eval.py --design-id 7 \\
        --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_review_12.json
"""
import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import sdr.apps.standards.models  # noqa: F401 — resolve SQLAlchemy FK
import sdr.apps.designs.models    # noqa: F401
import sdr.apps.reviews.models    # noqa: F401

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob
from sdr.apps.ai.engine.config import AnalysisPipelineConfig
from sdr.apps.ai.engine.persistence.workflow_repository import SqlAlchemyReviewWorkflowRepository
from sdr.apps.ai.engine.debate.diagram_requirement_selector import DiagramRequirementSelector

from sdr.apps.ai.evaluations.shared import results_path
from sdr.apps.ai.evaluations.shared.metrics import (
    calculate_context_precision,
    calculate_set_retrieval_precision_recall,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_ground_truth(gt_path: str) -> dict:
    with open(gt_path) as f:
        return json.load(f)


def _expected_ids_for_diagram(item: dict) -> set[str]:
    return {
        str(req["requirement_id"]).strip()
        for req in item.get("candidate_requirements", [])
        if req.get("relevant") is True and str(req.get("requirement_id", "")).strip()
    }


def _diagram_metrics(expected_ids: set[str], ranked: list) -> dict:
    retrieved_ids = [r.stable_key for r in ranked]
    precision, recall = calculate_set_retrieval_precision_recall(expected_ids, retrieved_ids)
    mrr = calculate_context_precision(expected_ids, [[rid] for rid in retrieved_ids])
    hit = 1.0 if (expected_ids & set(retrieved_ids)) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "hit_rate": hit,
        "mrr": round(mrr, 4),
        "retrieved_count": len(retrieved_ids),
        "expected_count": len(expected_ids),
    }


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "precision": 0.0, "recall": 0.0, "hit_rate": 0.0, "mrr": 0.0}
    n = len(rows)
    return {
        "count": n,
        "precision": round(sum(r["precision"] for r in rows) / n, 4),
        "recall": round(sum(r["recall"] for r in rows) / n, 4),
        "hit_rate": round(sum(r["hit_rate"] for r in rows) / n, 4),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagram requirement retrieval eval: vector search vs naive fallback."
    )
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to a labeled diagram ground-truth JSON (see build_diagram_ground_truth_template.py)"
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output", type=str, default="eval_diagram_retrieval.json")
    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        return

    gt = _load_ground_truth(args.ground_truth)
    category_id = gt.get("category_id")
    ingestion_job_id = gt.get("ingestion_job_id")
    if category_id is None or ingestion_job_id is None:
        logger.error(
            "Ground truth file is missing 'category_id' / 'ingestion_job_id' — "
            "regenerate it with build_diagram_ground_truth_template.py."
        )
        return

    with SessionLocal() as db:
        design = db.query(Design).filter(Design.id == args.design_id).first()
        if not design:
            logger.error(f"Design with ID {args.design_id} not found.")
            return

        category = db.query(StandardCategory).filter_by(id=category_id).first()
        ingestion_job = db.query(StandardIngestionJob).filter_by(id=ingestion_job_id).first()
        if not category or not ingestion_job:
            logger.error("category_id/ingestion_job_id from ground truth not found in DB.")
            return

        store = DesignPreparationStore()
        _, tsd_doc, _ = store.load_prepared_assets(db, design)

    config = AnalysisPipelineConfig(vision_diagram_requirements_max_items=args.top_k)
    workflow_repository = SqlAlchemyReviewWorkflowRepository()
    selector = DiagramRequirementSelector(config=config, workflow_repository=workflow_repository)

    vector_rows, naive_rows = [], []
    per_diagram_results = []
    skipped = 0

    for item in gt.get("items", []):
        diagram_id = item.get("diagram_id")
        expected_ids = _expected_ids_for_diagram(item)
        if not expected_ids:
            logger.warning(f"  diagram_id={diagram_id}: no labeled-relevant requirements — skipping")
            skipped += 1
            continue

        diagram = tsd_doc.get_diagram_by_id(diagram_id)
        if diagram is None:
            logger.warning(f"  diagram_id={diagram_id}: not found in prepared TSD document — skipping")
            skipped += 1
            continue

        vector_ranked = selector.select_for_diagram(
            diagram=diagram,
            tsd_document=tsd_doc,
            category=category,
            ingestion_job=ingestion_job,
        )
        naive_ranked = list(
            workflow_repository.list_diagram_requirements(
                category_id=category.id,
                ingestion_job_id=ingestion_job.id,
            )
        )[: args.top_k]

        vector_metrics = _diagram_metrics(expected_ids, vector_ranked)
        naive_metrics = _diagram_metrics(expected_ids, naive_ranked)
        vector_rows.append(vector_metrics)
        naive_rows.append(naive_metrics)

        per_diagram_results.append({
            "diagram_id": diagram_id,
            "expected_ids": sorted(expected_ids),
            "vector": vector_metrics,
            "naive_fallback": naive_metrics,
        })

        logger.info(
            f"  diagram_id={diagram_id} expected={len(expected_ids)} "
            f"vector(P={vector_metrics['precision']:.2f} R={vector_metrics['recall']:.2f}) "
            f"naive(P={naive_metrics['precision']:.2f} R={naive_metrics['recall']:.2f})"
        )

    if not per_diagram_results:
        logger.error("No diagrams with labeled-relevant requirements found. Check the ground truth file.")
        return

    vector_summary = _aggregate(vector_rows)
    naive_summary = _aggregate(naive_rows)
    delta = {
        key: round(vector_summary[key] - naive_summary[key], 4)
        for key in ("precision", "recall", "hit_rate", "mrr")
    }

    summary = {
        "design_id": args.design_id,
        "top_k": args.top_k,
        "diagrams_evaluated": len(per_diagram_results),
        "diagrams_skipped": skipped,
        "vector": vector_summary,
        "naive_fallback": naive_summary,
        "delta_vector_minus_naive": delta,
        "per_diagram_results": per_diagram_results,
    }

    output_path = results_path(args.output, subdir="retrieval")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Diagram Retrieval Eval Results ===")
    logger.info(f"  Diagrams evaluated: {len(per_diagram_results)} (skipped: {skipped})")
    logger.info(
        f"  Vector:  P={vector_summary['precision']:.3f} R={vector_summary['recall']:.3f} "
        f"HitRate={vector_summary['hit_rate']:.3f} MRR={vector_summary['mrr']:.3f}"
    )
    logger.info(
        f"  Naive:   P={naive_summary['precision']:.3f} R={naive_summary['recall']:.3f} "
        f"HitRate={naive_summary['hit_rate']:.3f} MRR={naive_summary['mrr']:.3f}"
    )
    logger.info(f"  Delta (vector - naive): {delta}")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
