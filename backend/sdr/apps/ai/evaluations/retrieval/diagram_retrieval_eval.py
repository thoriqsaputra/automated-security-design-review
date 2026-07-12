"""
Diagram requirement retrieval eval: production default vs explicit gatekeeper
vs explicit hybrid vs naive fallback vs random baseline.

`DiagramRequirementSelector.select_for_diagram` now uses a recall-oriented
hybrid default path in production (hybrid -> naive fallback). To measure both
production behavior and the isolated retrieval arms, this eval records:

  default    — production behavior (`force_strategy=None`)
  gatekeeper — VisionAgent reasons directly over the diagram image
               (batched, confidence-thresholded; see `_gatekeeper_search`)
  hybrid     — BM25 + dense cosine (CategoryDiagramRequirementEmbedding) fused
               via Reciprocal Rank Fusion; no image access
  naive      — `list_diagram_requirements()[:top_k]`, ordinal order, no ranking
  random     — random sample from the same pool (a "no signal" floor)

This file does not rerank hybrid results with a cross-encoder; the selector's
own measured behavior showed that reranking hurt this short-text diagram task.

Ground truth is a human-labeled `diagram_ground_truth_design_<id>.json` file (see
`evaluations/data/build_diagram_ground_truth_template.py`), where each diagram
lists the FULL candidate pool of diagram requirements for its category with a
`relevant` flag — not just the subset the system happened to retrieve. This is
what makes recall (not just precision) measurable: a requirement a strategy
never retrieved can still be graded as a miss.

Metrics per strategy (averaged across ground-truth diagrams):
  precision      — |retrieved ∩ relevant| / |retrieved|
  recall         — |retrieved ∩ relevant| / |relevant|
  hit_rate       — fraction of diagrams where >=1 relevant requirement was retrieved
  mrr            — 1/rank of the first relevant requirement in the ranked list

Usage:
    python diagram_retrieval_eval.py --design-id 7 \\
        --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_design_7.json
"""
import argparse
import json
import logging
import os
import random
import re
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
from sdr.apps.ai.evaluations.shared.diagram_ground_truth import resolve_diagram_ground_truth_path
from sdr.apps.ai.evaluations.shared.metrics import (
    calculate_context_precision,
    calculate_set_retrieval_precision_recall,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_COMPOSITE_REQ_RE = re.compile(r"(job\d+-D-composite-[A-Za-z0-9-]+)")


def _load_ground_truth(gt_path: str) -> dict:
    with open(gt_path) as f:
        return json.load(f)


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


def _expected_ids_for_diagram(item: dict) -> set[str]:
    return {
        normalized_id
        for req in item.get("candidate_requirements", [])
        for normalized_id in [_normalize_requirement_id(req.get("requirement_id", ""))]
        if req.get("relevant") is True and normalized_id
    }


def _diagram_metrics(expected_ids: set[str], ranked: list) -> dict:
    retrieved_ids = [_normalize_requirement_id(r.stable_key) for r in ranked]
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
        "--ground-truth",
        type=str,
        default=None,
        help="Path to a labeled diagram ground-truth JSON (defaults to the canonical design-scoped file)",
    )
    parser.add_argument("--top-k", type=int, default=AnalysisPipelineConfig().vision_diagram_requirements_max_items)
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for the random-baseline draw.")
    parser.add_argument("--output", type=str, default="eval_diagram_retrieval.json")
    args = parser.parse_args()

    gt_path = args.ground_truth or resolve_diagram_ground_truth_path(args.design_id)
    if not os.path.exists(gt_path):
        logger.error(f"Ground truth file not found: {gt_path}")
        return

    gt = _load_ground_truth(gt_path)
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

    default_rows, gatekeeper_rows, hybrid_rows, naive_rows, random_rows = [], [], [], [], []
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

        default_ranked = selector.select_for_diagram(
            diagram=diagram,
            tsd_document=tsd_doc,
            category=category,
            ingestion_job=ingestion_job,
        )
        gatekeeper_ranked = selector.select_for_diagram(
            diagram=diagram,
            tsd_document=tsd_doc,
            category=category,
            ingestion_job=ingestion_job,
            force_strategy="gatekeeper",
        )
        hybrid_ranked = selector.select_for_diagram(
            diagram=diagram,
            tsd_document=tsd_doc,
            category=category,
            ingestion_job=ingestion_job,
            force_strategy="hybrid",
        )
        full_pool = list(
            workflow_repository.list_diagram_requirements(
                category_id=category.id,
                ingestion_job_id=ingestion_job.id,
            )
        )
        naive_ranked = full_pool[: args.top_k]

        # Random-baseline condition: a "no signal" floor. The naive ordinal
        # fallback can look accidentally strong when the requirement pool is
        # small and broadly relevant across diagrams (see threats-to-validity
        # note in README) — random sampling from the same pool answers the
        # more defensible question "does retrieval beat having no signal at
        # all?" Re-seeded per diagram so results are reproducible regardless
        # of iteration order.
        rng = random.Random(args.sample_seed)
        random_ranked = rng.sample(full_pool, min(args.top_k, len(full_pool)))

        default_metrics = _diagram_metrics(expected_ids, default_ranked)
        gatekeeper_metrics = _diagram_metrics(expected_ids, gatekeeper_ranked)
        hybrid_metrics = _diagram_metrics(expected_ids, hybrid_ranked)
        naive_metrics = _diagram_metrics(expected_ids, naive_ranked)
        random_metrics = _diagram_metrics(expected_ids, random_ranked)
        default_rows.append(default_metrics)
        gatekeeper_rows.append(gatekeeper_metrics)
        hybrid_rows.append(hybrid_metrics)
        naive_rows.append(naive_metrics)
        random_rows.append(random_metrics)

        per_diagram_results.append({
            "diagram_id": diagram_id,
            "expected_ids": sorted(expected_ids),
            "production_default": default_metrics,
            "gatekeeper": gatekeeper_metrics,
            "hybrid": hybrid_metrics,
            "naive_fallback": naive_metrics,
            "random_baseline": random_metrics,
        })

        logger.info(
            f"  diagram_id={diagram_id} expected={len(expected_ids)} "
            f"default(P={default_metrics['precision']:.2f} R={default_metrics['recall']:.2f}) "
            f"gatekeeper(P={gatekeeper_metrics['precision']:.2f} R={gatekeeper_metrics['recall']:.2f}) "
            f"hybrid(P={hybrid_metrics['precision']:.2f} R={hybrid_metrics['recall']:.2f}) "
            f"naive(P={naive_metrics['precision']:.2f} R={naive_metrics['recall']:.2f}) "
            f"random(P={random_metrics['precision']:.2f} R={random_metrics['recall']:.2f})"
        )

    if not per_diagram_results:
        logger.error("No diagrams with labeled-relevant requirements found. Check the ground truth file.")
        return

    default_summary = _aggregate(default_rows)
    gatekeeper_summary = _aggregate(gatekeeper_rows)
    hybrid_summary = _aggregate(hybrid_rows)
    naive_summary = _aggregate(naive_rows)
    random_summary = _aggregate(random_rows)

    def _delta(a: dict, b: dict) -> dict:
        return {
            key: round(a[key] - b[key], 4)
            for key in ("precision", "recall", "hit_rate", "mrr")
        }

    delta_hybrid_vs_naive = _delta(hybrid_summary, naive_summary)
    delta_hybrid_vs_random = _delta(hybrid_summary, random_summary)
    delta_hybrid_vs_gatekeeper = _delta(hybrid_summary, gatekeeper_summary)

    summary = {
        "design_id": args.design_id,
        "ground_truth_path": gt_path,
        "top_k": args.top_k,
        "sample_seed": args.sample_seed,
        "diagrams_evaluated": len(per_diagram_results),
        "diagrams_skipped": skipped,
        "production_default": default_summary,
        "gatekeeper": gatekeeper_summary,
        "hybrid": hybrid_summary,
        "naive_fallback": naive_summary,
        "random_baseline": random_summary,
        "delta_hybrid_minus_naive": delta_hybrid_vs_naive,
        "delta_hybrid_minus_random": delta_hybrid_vs_random,
        "delta_hybrid_minus_gatekeeper": delta_hybrid_vs_gatekeeper,
        "per_diagram_results": per_diagram_results,
    }

    output_path = results_path(args.output, subdir="retrieval")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Diagram Retrieval Eval Results ===")
    logger.info(f"  Diagrams evaluated: {len(per_diagram_results)} (skipped: {skipped})")
    logger.info(
        f"  Default:    P={default_summary['precision']:.3f} R={default_summary['recall']:.3f} "
        f"HitRate={default_summary['hit_rate']:.3f} MRR={default_summary['mrr']:.3f}"
    )
    logger.info(
        f"  Gatekeeper: P={gatekeeper_summary['precision']:.3f} R={gatekeeper_summary['recall']:.3f} "
        f"HitRate={gatekeeper_summary['hit_rate']:.3f} MRR={gatekeeper_summary['mrr']:.3f}"
    )
    logger.info(
        f"  Hybrid:     P={hybrid_summary['precision']:.3f} R={hybrid_summary['recall']:.3f} "
        f"HitRate={hybrid_summary['hit_rate']:.3f} MRR={hybrid_summary['mrr']:.3f}"
    )
    logger.info(
        f"  Naive:      P={naive_summary['precision']:.3f} R={naive_summary['recall']:.3f} "
        f"HitRate={naive_summary['hit_rate']:.3f} MRR={naive_summary['mrr']:.3f}"
    )
    logger.info(
        f"  Random:     P={random_summary['precision']:.3f} R={random_summary['recall']:.3f} "
        f"HitRate={random_summary['hit_rate']:.3f} MRR={random_summary['mrr']:.3f}"
    )
    logger.info(f"  Delta (hybrid - naive):      {delta_hybrid_vs_naive}")
    logger.info(f"  Delta (hybrid - random):     {delta_hybrid_vs_random}")
    logger.info(f"  Delta (hybrid - gatekeeper): {delta_hybrid_vs_gatekeeper}")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
