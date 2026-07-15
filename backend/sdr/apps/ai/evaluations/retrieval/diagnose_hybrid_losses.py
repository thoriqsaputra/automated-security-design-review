"""
Diagnose WHERE the hybrid retrieval pipeline loses gold blocks.

For every dataset item, runs the hybrid arm with a HybridRetrievalTrace and
classifies each expected block id into exactly one loss bucket, so fixes can
target the stage that actually drops golds instead of guessing:

  HIT_LEAF               — final context contains a literal (level-0) chunk with the block
  SUMMARY_ONLY_COVERAGE  — block only "covered" via a level>0 summary node's block-id
                           union; the literal leaf text never reaches the judge
                           (explains coverage=1/recall=0 eval rows)
  NOT_IN_POOL            — no searcher branch (BM25/dense/multi-level) ever found it
  TOKEN_BUDGET_SKIPPED   — leaf ranks inside the dense branch's ungated cosine top-k but
                           was skipped by search_collapsed_raptor's token-budget loop
  DROPPED_BY_GRADER      — in the fused pool, rejected by the evidence grader
                           (baseline_requirement / empty)
  TIER_CROWDED_OUT       — graded, but evidence-tier ordering alone leaves every
                           candidate carrying the block beyond max_context_chunks
  RERANK_DEMOTED         — inside the cutoff before the cross-encoder rerank,
                           pushed beyond it after
  FILTERED_EMPTY_TEXT    — survived ranking but removed by the empty-text filter
  ROUTED_NON_HYBRID      — the strategy selector sent this query to a RAPTOR arm,
                           so no hybrid trace exists for it

No LLM judges are used — this is retrieval-only and cheap (query-expansion +
embedding calls only, exactly what the hybrid arm itself spends).

Usage (mirrors ablation_eval.py):
  python diagnose_hybrid_losses.py --design-id 15 --dataset eval_dataset_30_carpool.json \
      --hybrid-rerank on --hybrid-fusion agreement_boost \
      --arm-compare raptor_low,raptor_high
"""
import argparse
import json
import logging
import os
import pickle
import sys
from collections import Counter
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.retrieval.core.trace import HybridRetrievalTrace
from sdr.apps.ai.retrieval.core.types import AdvancedRetrievalConfig, RetrievalStrategy
from sdr.apps.standards.models import StandardCategory, StandardIngestionJob
from sdr.apps.ai.evaluations.shared import results_path, data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _ungated_leaf_ranking(raptor_tree, query_embedding, top_k):
    """Plain cosine top-k over leaf nodes, no threshold, no token budget —
    what the dense branch WOULD return if nothing were skipped."""
    leaves = [n for n in raptor_tree.get_leaf_nodes() if getattr(n, "has_embedding", False)]
    scored = sorted(
        ((_cosine(query_embedding, n.embedding), n) for n in leaves),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [n for _, n in scored[:top_k]]


def _keys_with_block(snapshots, block_id, leaf_only=False):
    keys = []
    for snap in snapshots:
        if leaf_only and snap.get("level", 0) != 0:
            continue
        if block_id in (snap.get("block_ids") or []):
            keys.append(snap["key"])
    return keys


def _min_position(order, keys):
    positions = [order.index(k) for k in keys if k in order]
    return min(positions) if positions else None


def _classify_block(block_id, trace, dense_would_find):
    final_leaf = _keys_with_block(trace.final, block_id, leaf_only=True)
    if final_leaf:
        return "HIT_LEAF", {}

    final_any = _keys_with_block(trace.final, block_id)
    if final_any:
        return "SUMMARY_ONLY_COVERAGE", {"summary_keys": final_any}

    pool_keys = set()
    per_list_hits = {}
    for list_name, snapshots in trace.per_list_candidates.items():
        keys = _keys_with_block(snapshots, block_id)
        if keys:
            first_rank = next(
                i for i, snap in enumerate(snapshots) if block_id in (snap.get("block_ids") or [])
            )
            per_list_hits[list_name] = {"keys": keys, "first_rank": first_rank}
            pool_keys.update(keys)

    if not pool_keys:
        if dense_would_find:
            return "TOKEN_BUDGET_SKIPPED", {}
        return "NOT_IN_POOL", {}

    graded_keys = set(_keys_with_block(trace.graded, block_id))
    if not graded_keys:
        fused_keys = set(_keys_with_block(trace.fused, block_id))
        rejected_ids = {r.get("id") for r in trace.rejected}
        detail = {
            "per_list_hits": per_list_hits,
            "rejected_match": bool(fused_keys & {f"id:{rid}" for rid in rejected_ids if rid}),
        }
        return "DROPPED_BY_GRADER", detail

    cutoff = trace.max_context_chunks
    # Prefer literal leaf candidates: a leaf inside the cutoff is what recall needs.
    leaf_graded = set(_keys_with_block(trace.graded, block_id, leaf_only=True)) or graded_keys
    pre_pos = _min_position(trace.pre_rerank_order, leaf_graded)
    post_pos = _min_position(trace.post_rerank_order, leaf_graded)
    detail = {
        "per_list_hits": per_list_hits,
        "pre_rerank_pos": pre_pos,
        "post_rerank_pos": post_pos,
        "cutoff": cutoff,
    }
    if post_pos is not None and post_pos < cutoff:
        return "FILTERED_EMPTY_TEXT", detail
    if pre_pos is not None and pre_pos < cutoff:
        return "RERANK_DEMOTED", detail
    return "TIER_CROWDED_OUT", detail


def main():
    parser = argparse.ArgumentParser(description="Per-stage diagnosis of hybrid retrieval gold-block losses.")
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--raptor-tree-pickle", type=str, default=None)
    parser.add_argument("--hybrid-rerank", choices=("on", "off"), default="on")
    parser.add_argument("--hybrid-fusion", choices=("agreement_boost", "rrf"), default="agreement_boost")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--arm-compare",
        type=str,
        default="raptor_low,raptor_high",
        help="Comma-separated baseline arms to check per lost block (whether the gold is in that arm's return set). Empty string disables.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset if os.path.exists(args.dataset) else data_path(args.dataset)
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    compare_arms = [a.strip() for a in (args.arm_compare or "").split(",") if a.strip()]
    arm_strategy = {
        "raptor_low": RetrievalStrategy.RAPTOR_LOW,
        "raptor_high": RetrievalStrategy.RAPTOR_HIGH,
        "flat_topk": RetrievalStrategy.FLAT_TOPK,
    }

    with SessionLocal() as db:
        design = db.query(Design).filter(Design.id == args.design_id).first()
        if not design:
            logger.error(f"Design with ID {args.design_id} not found.")
            return

        store = DesignPreparationStore()
        prep, tsd_doc, indexes = store.load_prepared_assets(db, design)
        if args.raptor_tree_pickle:
            with open(args.raptor_tree_pickle, "rb") as f:
                raptor_tree = pickle.load(f)
        else:
            raptor_tree = indexes.raptor_tree

        real_category = db.query(StandardCategory).filter_by(id=1).first()
        real_ingestion_job = (
            db.query(StandardIngestionJob)
            .filter_by(is_active=True)
            .order_by(StandardIngestionJob.created_at.desc())
            .first()
        )

        hybrid_router = HybridRetrievalRouter(
            advanced_config=AdvancedRetrievalConfig(
                enable_cross_encoder_rerank=(args.hybrid_rerank == "on"),
                fusion_method=args.hybrid_fusion,
                rrf_k=args.rrf_k,
            )
        )
        baseline_router = HybridRetrievalRouter(
            advanced_config=AdvancedRetrievalConfig(enable_cross_encoder_rerank=False)
        ) if compare_arms else None

        entries = []
        bucket_counter = Counter()
        dense_top_k = max(hybrid_router.vector_top_k, 8)

        for i, item in enumerate(dataset):
            question = item["question"]
            expected = item.get("block_ids") or ([item["block_id"]] if "block_id" in item else [])
            logger.info(f"[{i + 1}/{len(dataset)}] {question[:70]}")

            dummy_param = MagicMock()
            dummy_param.id = 1

            trace = HybridRetrievalTrace()
            result = hybrid_router.retrieve(
                parameter=dummy_param,
                category=real_category,
                raptor_tree=raptor_tree,
                ingestion_job=real_ingestion_job,
                override_query_text=question,
                trace=trace,
            )

            entry = {
                "question": question,
                "zone": item.get("zone"),
                "expected_block_ids": expected,
                "strategy_used": trace.strategy,
                "secondary_search_triggered": trace.secondary_search_triggered,
                "blocks": {},
            }

            if trace.strategy != RetrievalStrategy.HYBRID.value:
                for block_id in expected:
                    entry["blocks"][block_id] = {"bucket": "ROUTED_NON_HYBRID"}
                    bucket_counter["ROUTED_NON_HYBRID"] += 1
                entries.append(entry)
                continue

            for block_id in expected:
                ungated = _ungated_leaf_ranking(raptor_tree, trace.query_embedding, dense_top_k)
                dense_would_find = any(block_id in (n.source_block_ids or []) for n in ungated)
                bucket, detail = _classify_block(block_id, trace, dense_would_find)
                bucket_counter[bucket] += 1
                block_entry = {"bucket": bucket, **detail}

                if bucket not in ("HIT_LEAF",) and compare_arms and baseline_router is not None:
                    arm_hits = {}
                    for arm in compare_arms:
                        strategy = arm_strategy.get(arm)
                        if strategy is None:
                            continue
                        arm_result = baseline_router.retrieve(
                            parameter=dummy_param,
                            category=real_category,
                            raptor_tree=raptor_tree,
                            ingestion_job=real_ingestion_job,
                            override_query_text=question,
                            force_strategy=strategy,
                        )
                        arm_blocks = {
                            bid for group in (arm_result.context_chunk_block_ids or []) for bid in group
                        }
                        arm_hits[arm] = block_id in arm_blocks
                    block_entry["baseline_arm_returns_gold"] = arm_hits

                entry["blocks"][block_id] = block_entry

            entries.append(entry)

    summary = {
        "design_id": args.design_id,
        "dataset": os.path.basename(dataset_path),
        "hybrid_config": {
            "rerank": args.hybrid_rerank,
            "fusion_method": args.hybrid_fusion,
            "rrf_k": args.rrf_k,
        },
        "bucket_histogram": dict(bucket_counter.most_common()),
        "total_expected_blocks": sum(bucket_counter.values()),
        "entries": entries,
    }

    output_name = args.output or (
        f"diagnose_hybrid_design{args.design_id}_"
        f"rerank_{args.hybrid_rerank}_{args.hybrid_fusion}.json"
    )
    output_path = results_path(output_name, subdir="retrieval")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Diagnosis complete. Loss-bucket histogram:")
    for bucket, count in bucket_counter.most_common():
        logger.info(f"  {bucket:24s} {count}")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
