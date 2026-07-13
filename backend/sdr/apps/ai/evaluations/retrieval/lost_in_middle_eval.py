"""
Lost-in-the-Middle Evaluation for RAPTOR retrieval.

Tests whether RAPTOR's hierarchical summarization (raptor_high), and then the
full hybrid stack (+ BM25 + query expansion) on top of it, recovers evidence
that a vanilla flat top-k retriever (flat_topk) loses when it is buried in
the middle third of a long TSD.

All four arms search the same TSD (unlike a vector-only-vs-hybrid comparison,
where vector-only searches a different corpus entirely — the ASVS standards
catalog, not the TSD — which conflates "any TSD retrieval" with "hybrid
retrieval" instead of isolating what each architectural layer adds).

The "lost in the middle" effect: flat top-k dense retrieval tends to surface
content near the beginning or end of a document, where chunks compete less for
the top-k slots. Evidence in the middle third gets crowded out by many
semantically adjacent chunks. RAPTOR's higher-level summary nodes abstract over
document position, so they should partially recover this lost middle content.

flat_topk vs raptor_low: both search only the RAPTOR tree's leaf (L0) level —
same node granularity, same embeddings, no hierarchy, no BM25, no query
expansion. The one difference is that raptor_low first filters to nodes above
this codebase's min_cosine_similarity threshold and only relaxes to
near-miss/best-available candidates when that filtered set is too small,
whereas flat_topk always returns the k highest-scoring nodes by raw cosine
rank with no threshold gate at all. flat_topk is therefore the "vanilla RAG"
baseline from the lost-in-the-middle literature (Liu et al., 2023) — it
isolates the classic position-bias effect from this codebase's own
threshold-relaxation machinery, which raptor_low is not free of. raptor_low
and raptor_high stay in the eval for comparison against flat_topk and against
each other, but flat_topk is the fair baseline the thesis metrics are
computed against.

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
  - Run each question under four retrieval conditions:
      flat_topk   : vanilla top-k cosine search over leaf nodes only, no
                    threshold gate, no hierarchy, no BM25, no query expansion
                    — the fair "lost in the middle" baseline
      raptor_low  : single leaf-level RAPTOR search only, with this
                    codebase's threshold-relaxation gate (flat, no hierarchy,
                    no BM25, no query expansion, no reranking)
      raptor_high : multi-level RAPTOR search (leaf + mid + high summaries),
                    still no BM25/query expansion/reranking — isolates the
                    hierarchy's own contribution to middle-zone recovery
      hybrid      : default (multi-level RAPTOR + BM25 + query expansion)
  - Report context_recall and faithfulness per zone × condition
  - Compute thesis metrics (flat_topk as the baseline for all deficits):
      middle_deficit_flat_topk   = avg(front_recall, back_recall) − middle_recall  [flat_topk]
      middle_deficit_raptor_low  = avg(front_recall, back_recall) − middle_recall  [raptor_low]
      middle_deficit_raptor_high = avg(front_recall, back_recall) − middle_recall  [raptor_high]
      middle_deficit_hybrid      = avg(front_recall, back_recall) − middle_recall  [hybrid]
      raptor_middle_recovery        = middle_hybrid_recall − middle_flat_topk_recall
      hierarchy_middle_recovery     = middle_raptor_high_recall − middle_flat_topk_recall
      middle_deficit_reduction_pct  = (deficit_flat_topk − deficit_hybrid) / deficit_flat_topk × 100

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

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
    parser.add_argument(
        "--arms",
        type=str,
        default="flat_topk,raptor_low,raptor_high,hybrid",
        help=(
            "Comma-separated subset of {flat_topk,raptor_low,raptor_high,hybrid} to run "
            "per question (default: all four). E.g. --arms flat_topk,hybrid runs only "
            "the vanilla-baseline-vs-production-stack comparison, halving LLM/judge cost "
            "when raptor_low/raptor_high aren't needed for a particular analysis."
        ),
    )
    args = parser.parse_args()

    all_arms = ("flat_topk", "raptor_low", "raptor_high", "hybrid")
    run_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    invalid_arms = [a for a in run_arms if a not in all_arms]
    if invalid_arms:
        raise SystemExit(f"Unknown --arms value(s) {invalid_arms}; must be a subset of {all_arms}")
    if "flat_topk" not in run_arms and "hybrid" not in run_arms:
        raise SystemExit(
            "Thesis metrics need a baseline (flat_topk) and a comparison arm (hybrid) — "
            "--arms must include at least flat_topk and hybrid."
        )

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

        real_category_id = db.query(StandardCategory.id).first()[0]
        real_ingestion_job_id = (
            db.query(StandardIngestionJob.id)
            .filter_by(is_active=True)
            .order_by(StandardIngestionJob.created_at.desc())
            .first()[0]
        )
        raptor_tree = indexes.raptor_tree

        class _Indexes:
            pass
        _Indexes.raptor_tree = raptor_tree

        def _run_zone(zone: str, items: list) -> tuple[str, list, dict[str, list]]:
            # Each zone worker gets its own DB session, ORM instances, and
            # router instances — SQLAlchemy sessions and the routers'
            # per-arm reranker/fusion config are not safe to share across
            # threads, so nothing DB-bound is reused between zones.
            rows: list = []
            zone_local: dict[str, list] = {a: [] for a in run_arms}
            with SessionLocal() as zone_db:
                zone_category = zone_db.get(StandardCategory, real_category_id)
                zone_ingestion_job = zone_db.get(StandardIngestionJob, real_ingestion_job_id)

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
                # (router, retrieval_overrides) per arm — only arms in run_arms
                # get called per question, so an --arms subset directly cuts
                # the number of evaluate_question (and judge) calls made.
                arm_specs = {
                    "flat_topk": (raptor_baseline_router, {
                        "raptor_tree": raptor_tree,
                        "force_strategy": RetrievalStrategy.FLAT_TOPK,
                        "category": zone_category,
                        "ingestion_job": zone_ingestion_job,
                    }),
                    "raptor_low": (raptor_baseline_router, {
                        "raptor_tree": raptor_tree,
                        "force_strategy": RetrievalStrategy.RAPTOR_LOW,
                        "category": zone_category,
                        "ingestion_job": zone_ingestion_job,
                    }),
                    "raptor_high": (raptor_baseline_router, {
                        "raptor_tree": raptor_tree,
                        "force_strategy": RetrievalStrategy.RAPTOR_HIGH,
                        "category": zone_category,
                        "ingestion_job": zone_ingestion_job,
                    }),
                    "hybrid": (hybrid_router, {
                        "category": zone_category,
                        "ingestion_job": zone_ingestion_job,
                    }),
                }

                for i, item in enumerate(items):
                    logger.info(f"[{zone} {i+1}/{len(items)}] {item['question'][:70]}")
                    row = {"zone": zone, "question": item["question"],
                           "block_ids": item["block_ids"]}
                    try:
                        arm_results = {}
                        for arm in run_arms:
                            router, overrides = arm_specs[arm]
                            result = runner_mod.evaluate_question(
                                item, router, _Indexes(), db=zone_db,
                                retrieval_overrides=overrides,
                            )
                            arm_results[arm] = result
                            row[arm] = {
                                "context_recall": result["context_recall"],
                                "faithfulness": result["faithfulness"],
                                "faithfulness_deterministic": result["faithfulness_deterministic"],
                            }
                            zone_local[arm].append(result)

                        if "flat_topk" in arm_results and "hybrid" in arm_results:
                            row["delta_recall"] = round(
                                arm_results["hybrid"]["context_recall"]
                                - arm_results["flat_topk"]["context_recall"], 4
                            )
                        if "flat_topk" in arm_results and "raptor_high" in arm_results:
                            row["delta_recall_hierarchy_only"] = round(
                                arm_results["raptor_high"]["context_recall"]
                                - arm_results["flat_topk"]["context_recall"], 4
                            )
                    except Exception as e:
                        logger.error(f"Failed on [{zone}] item {i+1}: {e}")
                        row["error"] = str(e)

                    rows.append(row)
            return zone, rows, zone_local

        all_results = []
        zone_results: dict[str, dict[str, list]] = {
            z: {a: [] for a in run_arms} for z in ZONES
        }

        with ThreadPoolExecutor(max_workers=len(ZONES)) as pool:
            futures = {
                pool.submit(_run_zone, zone, zone_datasets.get(zone, [])): zone
                for zone in ZONES
            }
            for future in as_completed(futures):
                zone, rows, zone_local = future.result()
                all_results.extend(rows)
                zone_results[zone] = zone_local

        zone_order = {z: i for i, z in enumerate(ZONES)}
        all_results.sort(key=lambda r: zone_order.get(r["zone"], len(ZONES)))

        # Aggregate by zone
        by_zone_agg = {}
        for zone in ZONES:
            by_zone_agg[zone] = {a: _aggregate(zone_results[zone][a]) for a in run_arms}

        # Overall aggregates
        overall = {
            a: _aggregate([r for z in ZONES for r in zone_results[z][a]])
            for a in run_arms
        }

        # Thesis metrics — flat_topk (vanilla top-k, no threshold gate, no
        # hierarchy) is the fair "lost in the middle" baseline; raptor_low/
        # raptor_high metrics are only computed when those arms were run
        # (--arms can restrict the run to just flat_topk + hybrid).
        def _zone_recall(zone: str, condition: str) -> float:
            return by_zone_agg[zone][condition].get("context_recall", 0.0)

        def _deficit(arm: str) -> Optional[float]:
            if arm not in run_arms:
                return None
            mid = _zone_recall("middle", arm)
            edge = (_zone_recall("front", arm) + _zone_recall("back", arm)) / 2
            return round(edge - mid, 4)

        deficit_ft = _deficit("flat_topk")
        deficit_rl = _deficit("raptor_low")
        deficit_rh = _deficit("raptor_high")
        deficit_h = _deficit("hybrid")

        mid_ft = _zone_recall("middle", "flat_topk") if "flat_topk" in run_arms else None
        mid_rh = _zone_recall("middle", "raptor_high") if "raptor_high" in run_arms else None
        mid_h = _zone_recall("middle", "hybrid") if "hybrid" in run_arms else None

        recovery = round(mid_h - mid_ft, 4) if mid_h is not None and mid_ft is not None else None
        hierarchy_recovery = (
            round(mid_rh - mid_ft, 4) if mid_rh is not None and mid_ft is not None else None
        )

        # Only report "deficit reduction" when the flat_topk baseline actually
        # exhibits a positive middle-zone deficit. If the baseline deficit is
        # zero, negative, or wasn't run, the percentage is not interpretable.
        reduction_pct = (
            round((deficit_ft - deficit_h) / deficit_ft * 100, 1)
            if deficit_ft is not None and deficit_h is not None and deficit_ft > 0 else None
        )
        hierarchy_reduction_pct = (
            round((deficit_ft - deficit_rh) / deficit_ft * 100, 1)
            if deficit_ft is not None and deficit_rh is not None and deficit_ft > 0 else None
        )

        thesis_metrics = {
            "arms_run": run_arms,
            "middle_deficit_flat_topk": deficit_ft,
            "middle_deficit_raptor_low": deficit_rl,
            "middle_deficit_raptor_high": deficit_rh,
            "middle_deficit_hybrid": deficit_h,
            "raptor_middle_recovery": recovery,
            "hierarchy_middle_recovery": hierarchy_recovery,
            "middle_deficit_reduction_pct": reduction_pct,
            "middle_deficit_reduction_applicable": bool(deficit_ft is not None and deficit_ft > 0),
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

    def _fmt(v: Optional[float]) -> str:
        return f"{v:+.4f}" if v is not None else "n/a"

    logger.info("\n=== Lost-in-the-Middle Results ===")
    logger.info(f"  TSD: {tsd_doc.document_name} ({total_pages} pages)")
    logger.info(f"  Total QA pairs: {total_items} ({samples_per_zone} per zone)")
    logger.info(f"  Arms run: {', '.join(run_arms)}")
    logger.info("")
    for zone in ZONES:
        parts = [
            f"{a}_recall={by_zone_agg[zone][a].get('context_recall', 0):.4f}"
            for a in run_arms
        ]
        if "flat_topk" in run_arms and "hybrid" in run_arms:
            ft_r = by_zone_agg[zone]["flat_topk"].get("context_recall", 0)
            h_r = by_zone_agg[zone]["hybrid"].get("context_recall", 0)
            parts.append(f"delta(hybrid-flat)={h_r - ft_r:+.4f}")
        logger.info(f"  [{zone:6s}] " + "  ".join(parts))
    logger.info("")
    for a in run_arms:
        logger.info(f"  Overall {a}_recall: {overall[a].get('context_recall', 0):.4f}")
    logger.info("")
    logger.info("  Thesis metrics (baseline = flat_topk, the vanilla top-k retriever):")
    logger.info(f"    middle_deficit_flat_topk:      {_fmt(deficit_ft)}  (how much worse middle is vs edges, flat_topk baseline)")
    logger.info(f"    middle_deficit_raptor_low:     {_fmt(deficit_rl)}  (how much worse middle is vs edges, raptor-low, for reference)")
    logger.info(f"    middle_deficit_raptor_high:    {_fmt(deficit_rh)}  (how much worse middle is vs edges, raptor-high)")
    logger.info(f"    middle_deficit_hybrid:         {_fmt(deficit_h)}  (how much worse middle is vs edges, hybrid)")
    logger.info(f"    raptor_middle_recovery:        {_fmt(recovery)}  (hybrid's recall gain specifically for middle zone, vs flat_topk)")
    logger.info(f"    hierarchy_middle_recovery:     {_fmt(hierarchy_recovery)}  (raptor-high's recall gain over flat_topk for middle zone, hierarchy alone)")
    if reduction_pct is None:
        logger.info(
            "    middle_deficit_reduction_pct:  n/a  "
            "(baseline flat_topk deficit was <= 0 or an arm wasn't run, so percentage reduction is not interpretable)"
        )
    else:
        logger.info(
            f"    middle_deficit_reduction_pct:  {reduction_pct}%  "
            "(how much hybrid closed the middle gap, vs flat_topk)"
        )
    if hierarchy_reduction_pct is not None:
        logger.info(
            f"    hierarchy_middle_deficit_reduction_pct:  {hierarchy_reduction_pct}%  "
            "(how much hierarchy alone closed the middle gap, before BM25/expansion)"
        )
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
