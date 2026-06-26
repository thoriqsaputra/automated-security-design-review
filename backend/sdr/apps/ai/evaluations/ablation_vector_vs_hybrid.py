"""
Ablation: Dense Vector-only retrieval vs Hybrid (Vector + RAPTOR + GraphRAG).

Runs the same eval dataset through two retrieval configurations and reports
the standard four metrics for each, plus a breakdown by document position
(front/middle/back third of the TSD) to isolate the lost-in-the-middle
effect: hybrid's advantage over vector-only should be largest for evidence
buried in the middle of the document, where a single flat dense-embedding
top-k search is most likely to lose it among many competing chunks.
"""
import argparse
import json
import logging
import os
import pickle
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.retrieval.core.types import RetrievalStrategy
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob
from sdr.apps.ai.evaluations import runner as runner_mod

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
        return {"count": 0, "context_precision": 0.0, "context_recall": 0.0,
                "faithfulness": 0.0, "faithfulness_deterministic": 0.0}
    n = len(results)
    return {
        "count": n,
        "context_precision": sum(r["context_precision"] for r in results) / n,
        "context_recall": sum(r["context_recall"] for r in results) / n,
        "faithfulness": sum(r["faithfulness"] for r in results) / n,
        "faithfulness_deterministic": sum(r["faithfulness_deterministic"] for r in results) / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Vector-only vs Hybrid retrieval ablation.")
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument("--dataset", type=str, default="eval_dataset_30.json")
    parser.add_argument("--output", type=str, default="eval_ablation_vector_vs_hybrid.json")
    parser.add_argument(
        "--raptor-tree-pickle",
        type=str,
        default=None,
        help="Optional path to a pickled RAPTORTree to use instead of the design's persisted tree "
        "(use this if the persisted tree predates a RAPTOR fix you want reflected in the ablation).",
    )
    args = parser.parse_args()

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
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
            tsd_graph = indexes.tsd_graph

        _Indexes.raptor_tree = raptor_tree

        # The synthetic eval path mocks category with MagicMock, which the vector
        # searcher can't query with (needs a real category.id to look up the
        # active ingestion job). Pass real objects via retrieval_overrides so
        # they override the MagicMock set by evaluate_question's synthetic path.
        real_category = db.query(StandardCategory).filter_by(id=1).first()
        real_ingestion_job = (
            db.query(StandardIngestionJob)
            .filter_by(is_active=True)
            .order_by(StandardIngestionJob.created_at.desc())
            .first()
        )
        vector_only_overrides = {
            "raptor_tree": None,
            "graph": None,
            "force_strategy": RetrievalStrategy.VECTOR_ONLY,
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }
        raptor_only_overrides = {
            "raptor_tree": raptor_tree,
            "graph": None,
            "force_strategy": RetrievalStrategy.RAPTOR_HIGH,
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }
        hybrid_overrides = {
            "category": real_category,
            "ingestion_job": real_ingestion_job,
        }

        router = HybridRetrievalRouter()

        vector_results, raptor_results, hybrid_results = [], [], []
        buckets = {"front": [], "middle": [], "back": [], "unknown": []}

        for i, item in enumerate(dataset):
            bucket = _position_bucket(
                item.get("block_ids") or ([item["block_id"]] if "block_id" in item else []),
                total_pages,
            )
            logger.info(f"[{i + 1}/{len(dataset)}] ({bucket}) {item['question'][:70]}")

            try:
                v = runner_mod.evaluate_question(
                    item, router, _Indexes(), db=db, retrieval_overrides=vector_only_overrides
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

            vector_results.append(v)
            raptor_results.append(r)
            hybrid_results.append(h)
            buckets[bucket].append((v, r, h))

        summary = {
            "total_questions": len(hybrid_results),
            "vector_only": _aggregate(vector_results),
            "raptor_only": _aggregate(raptor_results),
            "hybrid": _aggregate(hybrid_results),
            "by_position_bucket": {
                bucket: {
                    "vector_only": _aggregate([v for v, r, h in pairs]),
                    "raptor_only": _aggregate([r for v, r, h in pairs]),
                    "hybrid": _aggregate([h for v, r, h in pairs]),
                }
                for bucket, pairs in buckets.items()
                if pairs
            },
            "vector_only_results": vector_results,
            "raptor_only_results": raptor_results,
            "hybrid_results": hybrid_results,
        }

        metrics = ("context_precision", "context_recall", "faithfulness", "faithfulness_deterministic")
        summary["delta_raptor_minus_vector"] = {
            m: summary["raptor_only"][m] - summary["vector_only"][m] for m in metrics
        }
        summary["delta_hybrid_minus_raptor"] = {
            m: summary["hybrid"][m] - summary["raptor_only"][m] for m in metrics
        }
        summary["delta_hybrid_minus_vector_only"] = {
            m: summary["hybrid"][m] - summary["vector_only"][m] for m in metrics
        }

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Ablation complete.")
    logger.info(f"Vector-only:  {summary['vector_only']}")
    logger.info(f"RAPTOR-only:  {summary['raptor_only']}")
    logger.info(f"Hybrid:       {summary['hybrid']}")
    logger.info(f"Delta RAPTOR - Vector: {summary['delta_raptor_minus_vector']}")
    logger.info(f"Delta Hybrid - RAPTOR: {summary['delta_hybrid_minus_raptor']}")
    logger.info(f"Delta Hybrid - Vector: {summary['delta_hybrid_minus_vector_only']}")
    for bucket, vals in summary["by_position_bucket"].items():
        logger.info(
            f"  [{bucket}] vector={vals['vector_only']['context_recall']:.3f} "
            f"raptor={vals['raptor_only']['context_recall']:.3f} "
            f"hybrid={vals['hybrid']['context_recall']:.3f}"
        )
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
