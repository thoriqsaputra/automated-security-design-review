"""
Ablation: Vector-only vs RAPTOR-low vs RAPTOR-high vs Hybrid.

Four retrieval strategies evaluated on the same synthetic QA dataset:
  vector_only  — standards DB (CategoryParameterChild) search only; no TSD blocks retrieved
                 (precision=0 by design — included to show the baseline without TSD retrieval)
  raptor_low   — single leaf-level RAPTOR search (flat TSD embedding, no hierarchy)
  raptor_only  — multi-level RAPTOR search across LOW/MID/HIGH levels (hierarchy benefit)
  hybrid       — multi-level RAPTOR + BM25 keyword search + cross-encoder reranking (full stack)

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
    args = parser.parse_args()

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

        # Standards DB search only (CategoryParameterChild). No TSD block_ids are returned,
        # so context_precision is 0 by design. Included as the no-TSD-retrieval baseline.
        vector_only_overrides = {
            "raptor_tree": None,
            "force_strategy": RetrievalStrategy.VECTOR_ONLY,
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }
        # Flat single-level RAPTOR leaf search — TSD embedding search, no hierarchy benefit.
        raptor_low_overrides = {
            "raptor_tree": raptor_tree,
            "force_strategy": RetrievalStrategy.RAPTOR_LOW,
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }
        # Multi-level RAPTOR (leaf + mid + high summaries) — shows hierarchy benefit.
        raptor_only_overrides = {
            "raptor_tree": raptor_tree,
            "force_strategy": RetrievalStrategy.RAPTOR_HIGH,
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }
        # Full hybrid: multi-level RAPTOR + BM25 keyword + cross-encoder reranking.
        # Cross-encoder is enabled explicitly here so hybrid's reranking is active.
        hybrid_overrides = {
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }

        # Cross-encoder reranker enabled so hybrid gets global (cross-tier) reranking.
        router = HybridRetrievalRouter(
            advanced_config=AdvancedRetrievalConfig(enable_cross_encoder_rerank=True)
        )

        vector_results, raptor_low_results, raptor_results, hybrid_results = [], [], [], []
        buckets = {"front": [], "middle": [], "back": [], "unknown": []}

        for i, item in enumerate(dataset):
            bucket = _position_bucket(
                item.get("block_ids") or ([item["block_id"]] if "block_id" in item else []),
                total_pages,
            )
            logger.info(f"[{i + 1}/{len(dataset)}] ({bucket}) {item['question'][:70]}")

            try:
                vo = runner_mod.evaluate_question(
                    item, router, _Indexes(), db=db, retrieval_overrides=vector_only_overrides
                )
                rl = runner_mod.evaluate_question(
                    item, router, _Indexes(), db=db, retrieval_overrides=raptor_low_overrides
                )
                r = runner_mod.evaluate_question(
                    item, router, _Indexes(), db=db, retrieval_overrides=raptor_only_overrides
                )
                h = runner_mod.evaluate_question(
                    item, router, _Indexes(), db=db, retrieval_overrides=hybrid_overrides
                )
            except Exception as e:
                logger.error(f"Failed to evaluate question '{item['question']}': {e}")
                continue

            vector_results.append(vo)
            raptor_low_results.append(rl)
            raptor_results.append(r)
            hybrid_results.append(h)
            buckets[bucket].append((vo, rl, r, h))

        summary = {
            "total_questions": len(hybrid_results),
            "vector_only": _aggregate(vector_results),
            "raptor_low": _aggregate(raptor_low_results),
            "raptor_only": _aggregate(raptor_results),
            "hybrid": _aggregate(hybrid_results),
            "by_position_bucket": {
                bucket: {
                    "vector_only": _aggregate([vo for vo, rl, r, h in pairs]),
                    "raptor_low": _aggregate([rl for vo, rl, r, h in pairs]),
                    "raptor_only": _aggregate([r for vo, rl, r, h in pairs]),
                    "hybrid": _aggregate([h for vo, rl, r, h in pairs]),
                }
                for bucket, pairs in buckets.items()
                if pairs
            },
            "vector_only_results": vector_results,
            "raptor_low_results": raptor_low_results,
            "raptor_only_results": raptor_results,
            "hybrid_results": hybrid_results,
        }

        metrics = (
            "context_precision",
            "retrieved_coverage",
            "context_recall",
            "faithfulness",
            "faithfulness_deterministic",
        )
        summary["delta_raptor_low_minus_vector"] = {
            m: round(summary["raptor_low"][m] - summary["vector_only"][m], 4) for m in metrics
        }
        summary["delta_raptor_minus_low"] = {
            m: round(summary["raptor_only"][m] - summary["raptor_low"][m], 4) for m in metrics
        }
        summary["delta_hybrid_minus_raptor"] = {
            m: round(summary["hybrid"][m] - summary["raptor_only"][m], 4) for m in metrics
        }
        summary["delta_hybrid_minus_vector"] = {
            m: round(summary["hybrid"][m] - summary["vector_only"][m], 4) for m in metrics
        }

    output_path = results_path(args.output, subdir="retrieval")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Ablation complete.")
    logger.info(f"Vector-only:  {summary['vector_only']}")
    logger.info(f"RAPTOR-low:   {summary['raptor_low']}")
    logger.info(f"RAPTOR-high:  {summary['raptor_only']}")
    logger.info(f"Hybrid:       {summary['hybrid']}")
    logger.info(f"Delta RAPTOR-low - Vector:      {summary['delta_raptor_low_minus_vector']}")
    logger.info(f"Delta RAPTOR-high - RAPTOR-low: {summary['delta_raptor_minus_low']}")
    logger.info(f"Delta Hybrid - RAPTOR-high:     {summary['delta_hybrid_minus_raptor']}")
    logger.info(f"Delta Hybrid - Vector:          {summary['delta_hybrid_minus_vector']}")
    for bucket, vals in summary["by_position_bucket"].items():
        logger.info(
            f"  [{bucket}] vector={vals['vector_only']['context_recall']:.3f} "
            f"raptor_low={vals['raptor_low']['context_recall']:.3f} "
            f"raptor_high={vals['raptor_only']['context_recall']:.3f} "
            f"hybrid={vals['hybrid']['context_recall']:.3f}"
        )
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
