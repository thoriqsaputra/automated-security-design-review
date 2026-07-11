"""
Lost-in-the-Middle Evaluation for RAPTOR retrieval.

Tests whether RAPTOR's hierarchical summarization (raptor_high), and then the
full hybrid stack (+ BM25 + query expansion) on top of it, recovers evidence
that flat, single-level dense retrieval (raptor_low) loses when it is buried
in the middle third of a long TSD.

All three arms search the same TSD (unlike a vector-only-vs-hybrid comparison,
where vector-only searches a different corpus entirely — the ASVS standards
catalog, not the TSD — which conflates "any TSD retrieval" with "hybrid
retrieval" instead of isolating what each architectural layer adds).

The "lost in the middle" effect: flat top-k dense retrieval tends to surface
content near the beginning or end of a document, where chunks compete less for
the top-k slots. Evidence in the middle third gets crowded out by many
semantically adjacent chunks. RAPTOR's higher-level summary nodes abstract over
document position, so they should partially recover this lost middle content.

Dataset: loaded from a pre-generated, zone-balanced JSON file (--dataset) rather
than generated inline on every run. Generate it once via:
    python -m sdr.apps.ai.evaluations.shared.dataset_generator \\
        --design-id 8 --source zone_balanced --samples-per-zone 10 \\
        --output eval_dataset_lost_in_middle_design8.json
This keeps every eval run (across hybrid-config A/B combinations, code changes,
re-runs) comparing against the exact same fixed QA set instead of a fresh
LLM-generated one each time (LLM generation is non-deterministic even with a
fixed seed, since temperature > 0) — the same reproducibility guarantee
ablation_eval.py already gets from its fixed --dataset file.

Methodology:
  - Dataset is split into three equal zones by page position: front (0–⅓),
    middle (⅓–⅔), back (⅔–1) — samples_per_zone QA pairs per zone (balanced)
  - Run each question under three retrieval conditions:
      raptor_low  : single leaf-level RAPTOR search only (flat, no hierarchy,
                    no BM25, no query expansion, no reranking)
      raptor_high : multi-level RAPTOR search (leaf + mid + high summaries),
                    still no BM25/query expansion/reranking — isolates the
                    hierarchy's own contribution to middle-zone recovery
      hybrid      : default (multi-level RAPTOR + BM25 + query expansion)
  - Report context_recall and faithfulness per zone × condition
  - Compute thesis metrics (raptor_low as the baseline for all deficits):
      middle_deficit_raptor_low  = avg(front_recall, back_recall) − middle_recall  [raptor_low]
      middle_deficit_raptor_high = avg(front_recall, back_recall) − middle_recall  [raptor_high]
      middle_deficit_hybrid      = avg(front_recall, back_recall) − middle_recall  [hybrid]
      raptor_middle_recovery        = middle_hybrid_recall − middle_raptor_low_recall
      hierarchy_middle_recovery     = middle_raptor_high_recall − middle_raptor_low_recall
      middle_deficit_reduction_pct  = (deficit_raptor_low − deficit_hybrid) / deficit_raptor_low × 100

Usage:
    # One-time (or once per samples-per-zone size): generate the fixed dataset
    python -m sdr.apps.ai.evaluations.shared.dataset_generator \\
        --design-id 8 --source zone_balanced --samples-per-zone 10 \\
        --output eval_dataset_lost_in_middle_design8.json

    # Then run the eval against it, as many times/configs as needed
    python lost_in_middle_eval.py --design-id 8 \\
        --dataset eval_dataset_lost_in_middle_design8.json
"""
import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.retrieval.core.types import AdvancedRetrievalConfig, RetrievalStrategy
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob
from sdr.apps.ai.evaluations import runner as runner_mod
from sdr.apps.ai.evaluations.shared.dataset_generator import ZONES
from sdr.apps.ai.evaluations.shared import results_path, data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def main():
    parser = argparse.ArgumentParser(
        description="Lost-in-the-Middle eval — balanced zone-stratified retrieval test (RAPTOR-low vs RAPTOR-high vs Hybrid)."
    )
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Pre-generated zone-balanced dataset JSON (see module docstring for how to "
            "generate one via dataset_generator.py --source zone_balanced). Looked up under "
            "evaluations/data/ unless an absolute/existing path is given."
        ),
    )
    parser.add_argument("--output", type=str, default="eval_lost_in_middle.json")
    parser.add_argument(
        "--hybrid-rerank",
        choices=("on", "off"),
        default="off",
        help="Enable/disable cross-encoder reranking for the hybrid arm (default: off, matches prior implicit app-default behavior). No effect on raptor_low.",
    )
    parser.add_argument(
        "--hybrid-fusion",
        choices=("agreement_boost", "rrf"),
        default="agreement_boost",
        help="Candidate fusion strategy for the hybrid arm's merge step (default: agreement_boost, today's behavior).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k constant, only used when --hybrid-fusion=rrf (default: 60).",
    )
    args = parser.parse_args()

    if os.path.isabs(args.dataset):
        dataset_path = args.dataset
    elif os.path.exists(args.dataset):
        dataset_path = args.dataset
    else:
        dataset_path = data_path(args.dataset)
    with open(dataset_path, "r") as f:
        full_dataset = json.load(f)

    zone_datasets: dict[str, list] = {z: [] for z in ZONES}
    for item in full_dataset:
        zone = item.get("zone")
        if zone in zone_datasets:
            zone_datasets[zone].append(item)
    total_items = sum(len(v) for v in zone_datasets.values())
    samples_per_zone = max((len(v) for v in zone_datasets.values()), default=0)
    logger.info(f"Loaded {total_items} QA pairs across {len(ZONES)} zones from {dataset_path}")

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

        # Separate router instances per arm: reranking/fusion config is fixed at
        # HybridRetrievalRouter.__init__ time, so an honest per-arm A/B needs
        # dedicated routers rather than one shared instance. raptor_low and
        # raptor_high never reach the reranker/fusion step, so they can share
        # one baseline router.
        raptor_baseline_router = HybridRetrievalRouter(
            advanced_config=AdvancedRetrievalConfig(enable_cross_encoder_rerank=False)
        )
        hybrid_router = HybridRetrievalRouter(
            advanced_config=AdvancedRetrievalConfig(
                enable_cross_encoder_rerank=(args.hybrid_rerank == "on"),
                fusion_method=args.hybrid_fusion,
                rrf_k=args.rrf_k,
            )
        )

        raptor_low_overrides = {
            "raptor_tree": indexes.raptor_tree,
            "force_strategy": RetrievalStrategy.RAPTOR_LOW,
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }
        raptor_high_overrides = {
            "raptor_tree": indexes.raptor_tree,
            "force_strategy": RetrievalStrategy.RAPTOR_HIGH,
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
            z: {"raptor_low": [], "raptor_high": [], "hybrid": []} for z in ZONES
        }

        for zone in ZONES:
            items = zone_datasets.get(zone, [])
            for i, item in enumerate(items):
                logger.info(f"[{zone} {i+1}/{len(items)}] {item['question'][:70]}")
                row = {"zone": zone, "question": item["question"],
                       "block_ids": item["block_ids"]}
                try:
                    rl = runner_mod.evaluate_question(
                        item, raptor_baseline_router, _Indexes(), db=db,
                        retrieval_overrides=raptor_low_overrides,
                    )
                    rh = runner_mod.evaluate_question(
                        item, raptor_baseline_router, _Indexes(), db=db,
                        retrieval_overrides=raptor_high_overrides,
                    )
                    h = runner_mod.evaluate_question(
                        item, hybrid_router, _Indexes(), db=db,
                        retrieval_overrides=hybrid_overrides,
                    )
                    row["raptor_low"] = {
                        "context_recall": rl["context_recall"],
                        "faithfulness": rl["faithfulness"],
                        "faithfulness_deterministic": rl["faithfulness_deterministic"],
                    }
                    row["raptor_high"] = {
                        "context_recall": rh["context_recall"],
                        "faithfulness": rh["faithfulness"],
                        "faithfulness_deterministic": rh["faithfulness_deterministic"],
                    }
                    row["hybrid"] = {
                        "context_recall": h["context_recall"],
                        "faithfulness": h["faithfulness"],
                        "faithfulness_deterministic": h["faithfulness_deterministic"],
                    }
                    row["delta_recall"] = round(
                        h["context_recall"] - rl["context_recall"], 4
                    )
                    row["delta_recall_hierarchy_only"] = round(
                        rh["context_recall"] - rl["context_recall"], 4
                    )
                    zone_results[zone]["raptor_low"].append(rl)
                    zone_results[zone]["raptor_high"].append(rh)
                    zone_results[zone]["hybrid"].append(h)
                except Exception as e:
                    logger.error(f"Failed on [{zone}] item {i+1}: {e}")
                    row["error"] = str(e)

                all_results.append(row)

        # Aggregate by zone
        by_zone_agg = {}
        for zone in ZONES:
            by_zone_agg[zone] = {
                "raptor_low": _aggregate(zone_results[zone]["raptor_low"]),
                "raptor_high": _aggregate(zone_results[zone]["raptor_high"]),
                "hybrid": _aggregate(zone_results[zone]["hybrid"]),
            }

        # Overall aggregates
        all_raptor_low = [r for z in ZONES for r in zone_results[z]["raptor_low"]]
        all_raptor_high = [r for z in ZONES for r in zone_results[z]["raptor_high"]]
        all_hybrid = [r for z in ZONES for r in zone_results[z]["hybrid"]]
        overall = {
            "raptor_low": _aggregate(all_raptor_low),
            "raptor_high": _aggregate(all_raptor_high),
            "hybrid": _aggregate(all_hybrid),
        }

        # Thesis metrics
        def _zone_recall(zone: str, condition: str) -> float:
            return by_zone_agg[zone][condition].get("context_recall", 0.0)

        mid_rl = _zone_recall("middle", "raptor_low")
        mid_rh = _zone_recall("middle", "raptor_high")
        mid_h = _zone_recall("middle", "hybrid")
        edge_rl = ((_zone_recall("front", "raptor_low") + _zone_recall("back", "raptor_low")) / 2)
        edge_rh = ((_zone_recall("front", "raptor_high") + _zone_recall("back", "raptor_high")) / 2)
        edge_h = ((_zone_recall("front", "hybrid") + _zone_recall("back", "hybrid")) / 2)

        deficit_rl = round(edge_rl - mid_rl, 4)
        deficit_rh = round(edge_rh - mid_rh, 4)
        deficit_h = round(edge_h - mid_h, 4)
        recovery = round(mid_h - mid_rl, 4)
        hierarchy_recovery = round(mid_rh - mid_rl, 4)

        # Only report "deficit reduction" when the raptor-low baseline actually
        # exhibits a positive middle-zone deficit. If the baseline deficit is
        # zero or negative, the percentage is not interpretable.
        reduction_pct = (
            round((deficit_rl - deficit_h) / deficit_rl * 100, 1) if deficit_rl > 0 else None
        )
        hierarchy_reduction_pct = (
            round((deficit_rl - deficit_rh) / deficit_rl * 100, 1) if deficit_rl > 0 else None
        )

        thesis_metrics = {
            "middle_deficit_raptor_low": deficit_rl,
            "middle_deficit_raptor_high": deficit_rh,
            "middle_deficit_hybrid": deficit_h,
            "raptor_middle_recovery": recovery,
            "hierarchy_middle_recovery": hierarchy_recovery,
            "middle_deficit_reduction_pct": reduction_pct,
            "middle_deficit_reduction_applicable": deficit_rl > 0,
            "hierarchy_middle_deficit_reduction_pct": hierarchy_reduction_pct,
        }

        summary = {
            "design_id": args.design_id,
            "dataset": dataset_path,
            "tsd_name": tsd_doc.document_name,
            "total_pages": total_pages,
            "samples_per_zone": samples_per_zone,
            "total_questions": total_items,
            "hybrid_config": {
                "rerank": args.hybrid_rerank,
                "fusion_method": args.hybrid_fusion,
                "rrf_k": args.rrf_k,
            },
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
    logger.info(f"  Total QA pairs: {total_items} ({samples_per_zone} per zone)")
    logger.info("")
    for zone in ZONES:
        rl_r = by_zone_agg[zone]["raptor_low"].get("context_recall", 0)
        rh_r = by_zone_agg[zone]["raptor_high"].get("context_recall", 0)
        h_r = by_zone_agg[zone]["hybrid"].get("context_recall", 0)
        logger.info(
            f"  [{zone:6s}] raptor_low_recall={rl_r:.4f}  raptor_high_recall={rh_r:.4f}  "
            f"hybrid_recall={h_r:.4f}  delta(hybrid-low)={h_r - rl_r:+.4f}"
        )
    logger.info("")
    logger.info(f"  Overall raptor_low_recall: {overall['raptor_low'].get('context_recall', 0):.4f}")
    logger.info(f"  Overall raptor_high_recall: {overall['raptor_high'].get('context_recall', 0):.4f}")
    logger.info(f"  Overall hybrid_recall: {overall['hybrid'].get('context_recall', 0):.4f}")
    logger.info("")
    logger.info("  Thesis metrics:")
    logger.info(f"    middle_deficit_raptor_low:     {deficit_rl:+.4f}  (how much worse middle is vs edges, raptor-low)")
    logger.info(f"    middle_deficit_raptor_high:    {deficit_rh:+.4f}  (how much worse middle is vs edges, raptor-high)")
    logger.info(f"    middle_deficit_hybrid:         {deficit_h:+.4f}  (how much worse middle is vs edges, hybrid)")
    logger.info(f"    raptor_middle_recovery:        {recovery:+.4f}  (hybrid's recall gain specifically for middle zone)")
    logger.info(f"    hierarchy_middle_recovery:     {hierarchy_recovery:+.4f}  (raptor-high's recall gain over raptor-low for middle zone, hierarchy alone)")
    if reduction_pct is None:
        logger.info(
            "    middle_deficit_reduction_pct:  n/a  "
            "(baseline raptor-low deficit was <= 0, so percentage reduction is not interpretable)"
        )
    else:
        logger.info(
            f"    middle_deficit_reduction_pct:  {reduction_pct}%  "
            "(how much hybrid closed the middle gap)"
        )
    if hierarchy_reduction_pct is not None:
        logger.info(
            f"    hierarchy_middle_deficit_reduction_pct:  {hierarchy_reduction_pct}%  "
            "(how much hierarchy alone closed the middle gap, before BM25/expansion)"
        )
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
