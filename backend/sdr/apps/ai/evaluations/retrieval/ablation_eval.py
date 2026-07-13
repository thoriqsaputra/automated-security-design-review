"""
Ablation: RAPTOR-low vs RAPTOR-high vs Hybrid.

Three retrieval strategies evaluated on the same synthetic QA dataset, all
searching the actual TSD document (unlike a vector-only-vs-hybrid comparison,
where vector-only searches a different corpus entirely — the ASVS standards
catalog, not the TSD — making it a weak baseline for isolating what the
hybrid architecture itself contributes):
  raptor_low   — single leaf-level RAPTOR search only (flat TSD embedding,
                 no hierarchy, no BM25, no query expansion, no reranking)
  raptor_high  — multi-level RAPTOR search (leaf + mid + high summaries),
                 still no BM25, no query expansion, no reranking — isolates
                 what RAPTOR's hierarchy alone contributes over a flat search
  hybrid       — multi-level RAPTOR + BM25 keyword search + query expansion +
                 evidence-tier filtering + configurable reranking/fusion
                 (--hybrid-rerank, --hybrid-fusion, --rrf-k; see CLI flags)

Because all three arms search the same TSD, this isolates the contribution of
each additional component (hierarchy alone via raptor_high, then BM25/query
expansion/evidence filtering/reranking on top via hybrid) rather than
conflating "any TSD retrieval" with "hybrid retrieval."

Key metrics per strategy:
  context_precision      — MRR: 1/rank of first retrieved chunk whose block_ids intersect expected
  retrieved_coverage     — binary: does the expected block appear anywhere in source_block_ids?
  context_recall         — LLM judge: does the retrieved context cover the ground truth?
  faithfulness           — LLM + deterministic judges: is the answer grounded in context?

context_precision vs retrieved_coverage separates two failure modes:
  coverage=0  → retrieval never found the right block (retrieval miss)
  coverage=1, precision<1 → block was retrieved but ranked below rank 1 (ranking miss)
"""
import argparse
import json
import logging
import os
import pickle
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.retrieval.core.types import AdvancedRetrievalConfig, RetrievalStrategy
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob
from sdr.apps.ai.evaluations import runner as runner_mod

from sdr.apps.ai.evaluations.shared import results_path, data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PAGE_RE = re.compile(r"^p(\d+)_b\d+$")


def _expected_pages(expected_block_ids):
    pages = []
    for bid in expected_block_ids:
        m = _PAGE_RE.match(bid)
        if m:
            pages.append(int(m.group(1)))
    return pages


def _position_bucket(expected_block_ids, total_pages):
    pages = _expected_pages(expected_block_ids)
    if not pages or not total_pages:
        return "unknown"
    avg_page = sum(pages) / len(pages)
    frac = avg_page / total_pages
    if frac < 1 / 3:
        return "front"
    if frac < 2 / 3:
        return "middle"
    return "back"


def _aggregate(results):
    if not results:
        return {
            "count": 0,
            "context_precision": 0.0,
            "retrieved_coverage": 0.0,
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "faithfulness_deterministic": 0.0,
        }
    n = len(results)
    return {
        "count": n,
        "context_precision": round(sum(r["context_precision"] for r in results) / n, 4),
        "retrieved_coverage": round(sum(r.get("retrieved_coverage", 0.0) for r in results) / n, 4),
        "context_recall": round(sum(r["context_recall"] for r in results) / n, 4),
        "faithfulness": round(sum(r["faithfulness"] for r in results) / n, 4),
        "faithfulness_deterministic": round(
            sum(r["faithfulness_deterministic"] for r in results) / n, 4
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Retrieval ablation: RAPTOR-low vs RAPTOR-high vs Hybrid."
    )
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument("--dataset", type=str, default="eval_dataset_30.json")
    parser.add_argument("--output", type=str, default="eval_ablation_retrieval.json")
    parser.add_argument(
        "--raptor-tree-pickle",
        type=str,
        default=None,
        help="Optional path to a pickled RAPTORTree (use when persisted tree predates a fix).",
    )
    parser.add_argument(
        "--hybrid-rerank",
        choices=("on", "off"),
        default="on",
        help="Enable/disable cross-encoder reranking for the hybrid arm (default: on, matches prior hardcoded behavior). No effect on raptor_low, which never reaches the reranker.",
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
        help="RRF k constant, only used when --hybrid-fusion=rrf (default: 60, the conventional value).",
    )
    parser.add_argument(
        "--arm",
        choices=("all", "raptor_low", "raptor_high", "hybrid"),
        default="all",
        help="Which arm(s) to evaluate this run (default: all three, the original behavior). "
             "raptor_low/raptor_high never vary with --hybrid-rerank/--hybrid-fusion, so once you have one "
             "baseline run for each of those two arms, re-running them alongside every hybrid config variant "
             "is redundant — use --arm hybrid for the remaining hybrid-config runs, and --arm raptor_low / "
             "--arm raptor_high once each to get a clean 6-runs-of-30-questions matrix instead of "
             "4-runs-of-90-questions-with-raptor-duplicates.",
    )
    args = parser.parse_args()
    run_raptor_low = args.arm in ("all", "raptor_low")
    run_raptor_high = args.arm in ("all", "raptor_high")
    run_hybrid = args.arm in ("all", "hybrid")

    if os.path.isabs(args.dataset):
        dataset_path = args.dataset
    elif os.path.exists(args.dataset):
        dataset_path = args.dataset
    else:
        dataset_path = data_path(args.dataset)
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    with SessionLocal() as db:
        design = db.query(Design).filter(Design.id == args.design_id).first()
        if not design:
            logger.error(f"Design with ID {args.design_id} not found.")
            return

        store = DesignPreparationStore()
        prep, tsd_doc, indexes = store.load_prepared_assets(db, design)
        total_pages = len(tsd_doc.pages)

        if args.raptor_tree_pickle:
            with open(args.raptor_tree_pickle, "rb") as f:
                raptor_tree = pickle.load(f)
        else:
            raptor_tree = indexes.raptor_tree

        class _Indexes:
            pass

        _Indexes.raptor_tree = raptor_tree

        real_category = db.query(StandardCategory).filter_by(id=1).first()
        real_ingestion_job = (
            db.query(StandardIngestionJob)
            .filter_by(is_active=True)
            .order_by(StandardIngestionJob.created_at.desc())
            .first()
        )

        # Flat single-level RAPTOR leaf search — TSD embedding search, no hierarchy,
        # no BM25, no query expansion, no reranking. The same-corpus, single-signal
        # baseline hybrid needs to beat to justify the added components.
        raptor_low_overrides = {
            "raptor_tree": raptor_tree,
            "force_strategy": RetrievalStrategy.RAPTOR_LOW,
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }
        # Multi-level RAPTOR (leaf + mid + high summaries) — still no BM25, no
        # query expansion, no reranking. Isolates the hierarchy's own contribution
        # before hybrid adds BM25/expansion/evidence-filtering/reranking on top.
        raptor_high_overrides = {
            "raptor_tree": raptor_tree,
            "force_strategy": RetrievalStrategy.RAPTOR_HIGH,
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }
        # Full hybrid: multi-level RAPTOR + BM25 keyword + query expansion +
        # evidence-tier filtering + configurable reranking/fusion (see CLI flags).
        hybrid_overrides = {
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }

        # Separate router instances per arm: reranking/fusion config is fixed at
        # HybridRetrievalRouter.__init__ time, so an honest per-arm A/B needs
        # dedicated routers rather than one shared instance. raptor_low and
        # raptor_high never reach the reranker/fusion step (both bypass
        # candidates.py entirely — see execute_raptor_low/execute_raptor_high),
        # so they can safely share one baseline router.
        raptor_baseline_router = HybridRetrievalRouter(
            advanced_config=AdvancedRetrievalConfig(enable_cross_encoder_rerank=False)
        ) if (run_raptor_low or run_raptor_high) else None
        hybrid_router = HybridRetrievalRouter(
            advanced_config=AdvancedRetrievalConfig(
                enable_cross_encoder_rerank=(args.hybrid_rerank == "on"),
                fusion_method=args.hybrid_fusion,
                rrf_k=args.rrf_k,
            )
        ) if run_hybrid else None

        raptor_low_results, raptor_high_results, hybrid_results = [], [], []
        buckets = {"front": [], "middle": [], "back": [], "unknown": []}

        for i, item in enumerate(dataset):
            bucket = _position_bucket(
                item.get("block_ids") or ([item["block_id"]] if "block_id" in item else []),
                total_pages,
            )
            logger.info(f"[{i + 1}/{len(dataset)}] ({bucket}) {item['question'][:70]}")

            try:
                rl = None
                rh = None
                h = None
                if run_raptor_low:
                    rl = runner_mod.evaluate_question(
                        item, raptor_baseline_router, _Indexes(), db=db, retrieval_overrides=raptor_low_overrides
                    )
                if run_raptor_high:
                    rh = runner_mod.evaluate_question(
                        item, raptor_baseline_router, _Indexes(), db=db, retrieval_overrides=raptor_high_overrides
                    )
                if run_hybrid:
                    h = runner_mod.evaluate_question(
                        item, hybrid_router, _Indexes(), db=db, retrieval_overrides=hybrid_overrides
                    )
            except Exception as e:
                logger.error(f"Failed to evaluate question '{item['question']}': {e}")
                continue

            if run_raptor_low:
                raptor_low_results.append(rl)
            if run_raptor_high:
                raptor_high_results.append(rh)
            if run_hybrid:
                hybrid_results.append(h)
            buckets[bucket].append((rl, rh, h))

        summary = {
            "arm": args.arm,
            "total_questions": len(dataset),
            "hybrid_config": {
                "rerank": args.hybrid_rerank,
                "fusion_method": args.hybrid_fusion,
                "rrf_k": args.rrf_k,
            },
        }
        if run_raptor_low:
            summary["raptor_low"] = _aggregate(raptor_low_results)
            summary["raptor_low_results"] = raptor_low_results
        if run_raptor_high:
            summary["raptor_high"] = _aggregate(raptor_high_results)
            summary["raptor_high_results"] = raptor_high_results
        if run_hybrid:
            summary["hybrid"] = _aggregate(hybrid_results)
            summary["hybrid_results"] = hybrid_results

        if args.arm == "all":
            summary["by_position_bucket"] = {
                bucket: {
                    "raptor_low": _aggregate([rl for rl, rh, h in pairs]),
                    "raptor_high": _aggregate([rh for rl, rh, h in pairs]),
                    "hybrid": _aggregate([h for rl, rh, h in pairs]),
                }
                for bucket, pairs in buckets.items()
                if pairs
            }
            metrics = (
                "context_precision",
                "retrieved_coverage",
                "context_recall",
                "faithfulness",
                "faithfulness_deterministic",
            )
            summary["delta_hybrid_minus_raptor_low"] = {
                m: round(summary["hybrid"][m] - summary["raptor_low"][m], 4) for m in metrics
            }
            summary["delta_hybrid_minus_raptor_high"] = {
                m: round(summary["hybrid"][m] - summary["raptor_high"][m], 4) for m in metrics
            }
            summary["delta_raptor_high_minus_raptor_low"] = {
                m: round(summary["raptor_high"][m] - summary["raptor_low"][m], 4) for m in metrics
            }

    output_path = results_path(args.output, subdir="retrieval")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Ablation complete.")
    if run_raptor_low:
        logger.info(f"RAPTOR-low:   {summary['raptor_low']}")
    if run_raptor_high:
        logger.info(f"RAPTOR-high:  {summary['raptor_high']}")
    if run_hybrid:
        logger.info(f"Hybrid:       {summary['hybrid']}")
    if args.arm == "all":
        logger.info(f"Delta Hybrid - RAPTOR-low:       {summary['delta_hybrid_minus_raptor_low']}")
        logger.info(f"Delta Hybrid - RAPTOR-high:      {summary['delta_hybrid_minus_raptor_high']}")
        logger.info(f"Delta RAPTOR-high - RAPTOR-low:  {summary['delta_raptor_high_minus_raptor_low']}")
        for bucket, vals in summary["by_position_bucket"].items():
            logger.info(
                f"  [{bucket}] raptor_low={vals['raptor_low']['context_recall']:.3f} "
                f"raptor_high={vals['raptor_high']['context_recall']:.3f} "
                f"hybrid={vals['hybrid']['context_recall']:.3f}"
            )
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
