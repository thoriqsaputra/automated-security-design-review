"""
Lost-in-the-Middle Evaluation for RAPTOR retrieval.

Tests whether RAPTOR's hierarchical summarization recovers evidence that flat
dense-vector retrieval loses when it is buried in the middle third of a long TSD.

The "lost in the middle" effect: flat top-k dense retrieval tends to surface
content near the beginning or end of a document, where chunks compete less for
the top-k slots. Evidence in the middle third gets crowded out by many
semantically adjacent chunks. RAPTOR's higher-level summary nodes abstract over
document position, so they should partially recover this lost middle content.

Methodology:
  - Split TSD pages into three equal zones: front (0–⅓), middle (⅓–⅔), back (⅔–1)
  - Sample --samples-per-zone QA pairs deliberately from EACH zone (balanced)
  - Run each question under two retrieval conditions:
      vector_only : raptor_tree=None, force_strategy=VECTOR_ONLY
      hybrid      : default (vector + RAPTOR)
  - Report context_recall and faithfulness per zone × condition
  - Compute thesis metrics:
      middle_deficit_vector  = avg(front_recall, back_recall) − middle_recall  [vector]
      middle_deficit_hybrid  = avg(front_recall, back_recall) − middle_recall  [hybrid]
      raptor_middle_recovery = middle_hybrid_recall − middle_vector_recall
      middle_deficit_reduction_pct = (deficit_vector − deficit_hybrid) / deficit_vector × 100

Usage:
    python lost_in_middle_eval.py --design-id 8 --samples-per-zone 10
    python lost_in_middle_eval.py --design-id 8 --samples-per-zone 10 \\
        --output eval_lost_in_middle.json
"""
import argparse
import json
import logging
import os
import random
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.retrieval.core.types import RetrievalStrategy
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob
from sdr.apps.ai.evaluations import runner as runner_mod
from sdr.apps.ai.evaluations.shared.dataset_generator import (
    _merge_blocks_into_samples,
    generate_qa_pair,
)
from sdr.apps.ai.retrieval.postprocessing.quote_grounding import is_quote_grounded
from sdr.apps.ai.evaluations.shared import results_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PAGE_RE = re.compile(r"^p(\d+)_b\d+$")

ZONES = ("front", "middle", "back")


def _page_of_block(block_id: str) -> int | None:
    m = _PAGE_RE.match(block_id)
    return int(m.group(1)) if m else None


def _zone_of_block(block_id: str, total_pages: int) -> str:
    page = _page_of_block(block_id)
    if page is None or not total_pages:
        return "unknown"
    frac = page / total_pages
    if frac < 1 / 3:
        return "front"
    if frac < 2 / 3:
        return "middle"
    return "back"


def _zone_of_sample(block_ids: list[str], total_pages: int) -> str:
    """Assign zone by the average page of the sample's blocks."""
    pages = [_page_of_block(bid) for bid in block_ids if _page_of_block(bid) is not None]
    if not pages:
        return "unknown"
    avg_page = sum(pages) / len(pages)
    frac = avg_page / total_pages
    if frac < 1 / 3:
        return "front"
    if frac < 2 / 3:
        return "middle"
    return "back"


def _aggregate(results: list[dict]) -> dict:
    if not results:
        return {"count": 0, "context_recall": 0.0, "faithfulness": 0.0,
                "faithfulness_deterministic": 0.0}
    n = len(results)
    return {
        "count": n,
        "context_recall": round(sum(r["context_recall"] for r in results) / n, 4),
        "faithfulness": round(sum(r["faithfulness"] for r in results) / n, 4),
        "faithfulness_deterministic": round(
            sum(r["faithfulness_deterministic"] for r in results) / n, 4
        ),
    }


def _generate_zone_datasets(tsd_doc, total_pages: int, samples_per_zone: int) -> dict[str, list]:
    """Build balanced QA datasets: samples_per_zone items per zone."""
    all_samples = _merge_blocks_into_samples(tsd_doc.all_text_blocks, min_words=30, max_words=150)

    by_zone: dict[str, list] = {z: [] for z in ZONES}
    for s in all_samples:
        zone = _zone_of_sample(s["block_ids"], total_pages)
        if zone in by_zone:
            by_zone[zone].append(s)

    for zone, pool in by_zone.items():
        logger.info(f"Zone '{zone}': {len(pool)} candidate samples")

    datasets: dict[str, list] = {z: [] for z in ZONES}
    for zone in ZONES:
        pool = by_zone[zone]
        if not pool:
            logger.warning(f"No samples found for zone '{zone}' — skipping.")
            continue

        random.shuffle(pool)
        generated = 0
        for sample in pool:
            if generated >= samples_per_zone:
                break
            logger.info(
                f"Generating QA [{zone}] {generated + 1}/{samples_per_zone} "
                f"from blocks {sample['block_ids'][:3]}..."
            )
            qa = None
            for attempt in range(2):
                candidate = generate_qa_pair(sample["text"])
                if not candidate:
                    continue
                if is_quote_grounded(candidate.get("ground_truth_context", ""), sample["text"]):
                    qa = candidate
                    break
                logger.warning(
                    f"[{zone}] ground_truth not grounded (attempt {attempt+1}/2)"
                    + (" — retrying" if attempt == 0 else " — skipping")
                )
            if qa:
                datasets[zone].append({
                    "zone": zone,
                    "block_ids": sample["block_ids"],
                    "original_text": sample["text"],
                    "question": qa.get("question", ""),
                    "ground_truth_context": qa.get("ground_truth_context", ""),
                })
                generated += 1

        logger.info(f"Zone '{zone}': generated {len(datasets[zone])} QA pairs")

    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Lost-in-the-Middle RAPTOR eval — balanced zone-stratified retrieval test."
    )
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument(
        "--samples-per-zone", type=int, default=10,
        help="Number of QA pairs to generate per zone (front/middle/back). Default: 10."
    )
    parser.add_argument("--output", type=str, default="eval_lost_in_middle.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    args = parser.parse_args()

    random.seed(args.seed)

    with SessionLocal() as db:
        design = db.query(Design).filter(Design.id == args.design_id).first()
        if not design:
            logger.error(f"Design {args.design_id} not found.")
            return

        store = DesignPreparationStore()
        prep, tsd_doc, indexes = store.load_prepared_assets(db, design)
        total_pages = len(tsd_doc.pages)
        logger.info(f"Loaded TSD: {tsd_doc.document_name} — {total_pages} pages, {len(tsd_doc.all_text_blocks)} blocks")

        real_category = db.query(StandardCategory).first()
        real_ingestion_job = (
            db.query(StandardIngestionJob)
            .filter_by(is_active=True)
            .order_by(StandardIngestionJob.created_at.desc())
            .first()
        )

        # Generate balanced zone datasets (LLM calls happen here)
        zone_datasets = _generate_zone_datasets(tsd_doc, total_pages, args.samples_per_zone)
        total_items = sum(len(v) for v in zone_datasets.values())
        logger.info(f"Dataset ready: {total_items} QA pairs across {len(ZONES)} zones")

        router = HybridRetrievalRouter()

        vector_overrides = {
            "raptor_tree": None,
            "force_strategy": RetrievalStrategy.VECTOR_ONLY,
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }
        hybrid_overrides = {
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }

        class _Indexes:
            pass
        _Indexes.raptor_tree = indexes.raptor_tree

        all_results = []
        zone_results: dict[str, dict[str, list]] = {
            z: {"vector_only": [], "hybrid": []} for z in ZONES
        }

        for zone in ZONES:
            items = zone_datasets.get(zone, [])
            for i, item in enumerate(items):
                logger.info(f"[{zone} {i+1}/{len(items)}] {item['question'][:70]}")
                row = {"zone": zone, "question": item["question"],
                       "block_ids": item["block_ids"]}
                try:
                    v = runner_mod.evaluate_question(
                        item, router, _Indexes(), db=db,
                        retrieval_overrides=vector_overrides,
                    )
                    h = runner_mod.evaluate_question(
                        item, router, _Indexes(), db=db,
                        retrieval_overrides=hybrid_overrides,
                    )
                    row["vector_only"] = {
                        "context_recall": v["context_recall"],
                        "faithfulness": v["faithfulness"],
                        "faithfulness_deterministic": v["faithfulness_deterministic"],
                    }
                    row["hybrid"] = {
                        "context_recall": h["context_recall"],
                        "faithfulness": h["faithfulness"],
                        "faithfulness_deterministic": h["faithfulness_deterministic"],
                    }
                    row["delta_recall"] = round(
                        h["context_recall"] - v["context_recall"], 4
                    )
                    zone_results[zone]["vector_only"].append(v)
                    zone_results[zone]["hybrid"].append(h)
                except Exception as e:
                    logger.error(f"Failed on [{zone}] item {i+1}: {e}")
                    row["error"] = str(e)

                all_results.append(row)

        # Aggregate by zone
        by_zone_agg = {}
        for zone in ZONES:
            by_zone_agg[zone] = {
                "vector_only": _aggregate(zone_results[zone]["vector_only"]),
                "hybrid": _aggregate(zone_results[zone]["hybrid"]),
            }

        # Overall aggregates
        all_vector = [r for z in ZONES for r in zone_results[z]["vector_only"]]
        all_hybrid = [r for z in ZONES for r in zone_results[z]["hybrid"]]
        overall = {
            "vector_only": _aggregate(all_vector),
            "hybrid": _aggregate(all_hybrid),
        }

        # Thesis metrics
        def _zone_recall(zone: str, condition: str) -> float:
            return by_zone_agg[zone][condition].get("context_recall", 0.0)

        mid_v = _zone_recall("middle", "vector_only")
        mid_h = _zone_recall("middle", "hybrid")
        edge_v = ((_zone_recall("front", "vector_only") + _zone_recall("back", "vector_only")) / 2)
        edge_h = ((_zone_recall("front", "hybrid") + _zone_recall("back", "hybrid")) / 2)

        deficit_v = round(edge_v - mid_v, 4)
        deficit_h = round(edge_h - mid_h, 4)
        recovery = round(mid_h - mid_v, 4)
        reduction_pct = (
            round((deficit_v - deficit_h) / deficit_v * 100, 1) if deficit_v != 0 else None
        )

        thesis_metrics = {
            "middle_deficit_vector": deficit_v,
            "middle_deficit_hybrid": deficit_h,
            "raptor_middle_recovery": recovery,
            "middle_deficit_reduction_pct": reduction_pct,
        }

        summary = {
            "design_id": args.design_id,
            "tsd_name": tsd_doc.document_name,
            "total_pages": total_pages,
            "samples_per_zone": args.samples_per_zone,
            "total_questions": total_items,
            "by_zone": by_zone_agg,
            "overall": overall,
            "thesis_metrics": thesis_metrics,
            "results": all_results,
        }

    output_path = results_path(args.output, subdir="retrieval")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Lost-in-the-Middle Results ===")
    logger.info(f"  TSD: {tsd_doc.document_name} ({total_pages} pages)")
    logger.info(f"  Total QA pairs: {total_items} ({args.samples_per_zone} per zone)")
    logger.info("")
    for zone in ZONES:
        v_r = by_zone_agg[zone]["vector_only"].get("context_recall", 0)
        h_r = by_zone_agg[zone]["hybrid"].get("context_recall", 0)
        logger.info(f"  [{zone:6s}] vector_recall={v_r:.4f}  hybrid_recall={h_r:.4f}  delta={h_r - v_r:+.4f}")
    logger.info("")
    logger.info(f"  Overall vector_recall: {overall['vector_only'].get('context_recall', 0):.4f}")
    logger.info(f"  Overall hybrid_recall: {overall['hybrid'].get('context_recall', 0):.4f}")
    logger.info("")
    logger.info("  Thesis metrics:")
    logger.info(f"    middle_deficit_vector:         {deficit_v:+.4f}  (how much worse middle is vs edges, vector-only)")
    logger.info(f"    middle_deficit_hybrid:         {deficit_h:+.4f}  (how much worse middle is vs edges, hybrid)")
    logger.info(f"    raptor_middle_recovery:        {recovery:+.4f}  (RAPTOR's recall gain specifically for middle zone)")
    logger.info(f"    middle_deficit_reduction_pct:  {reduction_pct}%  (how much RAPTOR closed the middle gap)")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
