"""
Cross-Boundary Threat Accuracy eval (ASVS V1 Architecture + V9 Communications).

Measures how much GraphRAG contributes to retrieval quality specifically for
architecture and communications security requirements — the categories that
require understanding trust boundaries and inter-component data flows.

For each ASVS V1/V9 requirement:
  - Condition A: full hybrid including graph (current default)
  - Condition B: hybrid WITHOUT graph (raptor+vector only, graph=None)

Metrics: context_recall and faithfulness (LLM) + faithfulness_deterministic,
reported per category (V1, V9, combined) and per condition with delta.

context_precision is omitted: there are no pre-labelled expected_block_ids for
these requirements against the test TSD. Instead context_recall (LLM judge)
assesses whether the retrieved TSD content is sufficient to evaluate each
architecture/communications requirement — a direct measure of retrieval
quality for these cross-boundary categories.
"""
import argparse
import json
import logging
import os
import pickle
import re
import sys
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.standards.models import (
    CategoryParameterChild,
    CategoryParameterParent,
    StandardIngestionJob,
)
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.client.manager import ai_service_manager
from sdr.apps.ai.evaluations.judges import (
    judge_context_recall,
    judge_faithfulness,
    judge_faithfulness_deterministic,
)
from sdr.apps.ai.evaluations.runner import _extract_answer_quotes, JUDGE_DISAGREEMENT_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# V1 Architecture parent_id and V9 Communication parent_id (confirmed in DB)
V1_PARENT_ID = 14
V9_PARENT_ID = 15
SAMPLE_PER_SECTION = 15

ANSWER_PROMPT_TEMPLATE = (
    "Based ONLY on the following context from a Technical Specification Document, "
    "answer this security requirement question as an auditor. "
    "Quote the exact supporting text in double quotes (verbatim, no ellipses). "
    "If the context does not contain relevant information, say 'I cannot determine'.\n\n"
    "Context:\n{context}\n\nRequirement:\n{question}"
)


def _run_condition(router, child, category, ingestion_job, raptor_indexes, use_graph):
    """Run one retrieval+answer+judge cycle for a single requirement + graph condition."""
    retrieve_kwargs = {
        "parameter": child,
        "category": category,
        "ingestion_job": ingestion_job,
        "override_query_text": child.requirement_text,
        "raptor_tree": raptor_indexes.raptor_tree,
        "graph": raptor_indexes.tsd_graph if use_graph else None,
    }
    result = router.retrieve(**retrieve_kwargs)

    chunk_block_ids = result.context_chunk_block_ids or [[] for _ in result.context_chunks]
    retrieved_context = "\n---\n".join(result.context_chunks)
    retrieved_context_blocks = {
        bid: text
        for text, group in zip(result.context_chunks, chunk_block_ids)
        for bid in group
    }

    answer_prompt = ANSWER_PROMPT_TEMPLATE.format(
        context=retrieved_context, question=child.requirement_text
    )
    response = ai_service_manager.chat_completion_with_fallback(
        messages=[{"role": "user", "content": answer_prompt}],
        component="hunter",
        temperature=0.0,
    )
    answer = response.content if not response.error else ""

    # Use requirement text as proxy ground-truth: checks if retrieved TSD
    # content is sufficient to evaluate whether the system meets this
    # architecture/communications requirement.
    recall_eval = judge_context_recall(child.requirement_text, child.requirement_text, retrieved_context)
    faith_eval = judge_faithfulness(child.requirement_text, retrieved_context, answer)
    quotes = _extract_answer_quotes(answer)
    faith_det = judge_faithfulness_deterministic(quotes, retrieved_context_blocks)
    llm_faith = faith_eval.get("faithfulness_score", 0.0)

    return {
        "context_recall": recall_eval.get("context_recall_score", 0.0),
        "faithfulness": llm_faith,
        "faithfulness_deterministic": faith_det,
        "judges_disagree": abs(llm_faith - faith_det) > JUDGE_DISAGREEMENT_THRESHOLD,
        "strategy_used": str(result.strategy_used),
        "chunks_retrieved": len(result.context_chunks),
        "hunter_answer": answer,
    }


def _aggregate(results):
    if not results:
        return {}
    n = len(results)
    return {
        "count": n,
        "context_recall": round(sum(r["context_recall"] for r in results) / n, 4),
        "faithfulness": round(sum(r["faithfulness"] for r in results) / n, 4),
        "faithfulness_deterministic": round(sum(r["faithfulness_deterministic"] for r in results) / n, 4),
        "avg_chunks_retrieved": round(sum(r["chunks_retrieved"] for r in results) / n, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-boundary V1/V9 accuracy eval.")
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument("--output", type=str, default="eval_cross_boundary_v1_v9.json")
    parser.add_argument(
        "--raptor-tree-pickle",
        type=str,
        default=None,
        help="Path to pickled RAPTORTree (use the page-aware-packing fixed tree, "
        "e.g. /tmp/new_raptor_tree.pkl in the container).",
    )
    parser.add_argument(
        "--sample", type=int, default=SAMPLE_PER_SECTION,
        help="Max children per V1/V9 section to evaluate (default 15).",
    )
    args = parser.parse_args()

    with SessionLocal() as db:
        design = db.query(Design).filter(Design.id == args.design_id).first()
        if not design:
            logger.error(f"Design {args.design_id} not found.")
            return

        store = DesignPreparationStore()
        prep, tsd_doc, indexes = store.load_prepared_assets(db, design)

        if args.raptor_tree_pickle:
            with open(args.raptor_tree_pickle, "rb") as f:
                indexes.raptor_tree = pickle.load(f)

        # Load V1 and V9 children plus their ingestion jobs
        sections = {}
        for label, parent_id in [("V1", V1_PARENT_ID), ("V9", V9_PARENT_ID)]:
            parent = db.get(CategoryParameterParent, parent_id)
            if not parent:
                logger.warning(f"{label} parent (id={parent_id}) not found — skipping.")
                continue
            ingestion_job = db.get(StandardIngestionJob, parent.ingestion_job_id)
            if not ingestion_job:
                logger.warning(f"Ingestion job {parent.ingestion_job_id} not found — skipping {label}.")
                continue
            children = (
                db.query(CategoryParameterChild)
                .filter_by(parent_id=parent_id)
                .limit(args.sample)
                .all()
            )
            sections[label] = {"children": children, "ingestion_job": ingestion_job, "parent_title": parent.title}
            logger.info(f"Loaded {label} ({parent.title}): {len(children)} requirements, job_id={ingestion_job.id}")

        # category is web_application (category_id=1) for both V1 and V9
        from sdr.apps.standards.models import StandardCategory
        category = db.query(StandardCategory).filter_by(id=1).first()

        router = HybridRetrievalRouter()
        all_results = []

        for label, section in sections.items():
            children = section["children"]
            ingestion_job = section["ingestion_job"]

            for i, child in enumerate(children):
                logger.info(
                    f"[{label} {i+1}/{len(children)}] {child.requirement_text[:80]}"
                )
                row = {
                    "section": label,
                    "child_id": child.id,
                    "requirement": child.requirement_text,
                }
                try:
                    row["with_graph"] = _run_condition(
                        router, child, category, ingestion_job, indexes, use_graph=True
                    )
                    row["without_graph"] = _run_condition(
                        router, child, category, ingestion_job, indexes, use_graph=False
                    )
                    # Per-metric delta: with_graph minus without_graph
                    row["delta"] = {
                        m: round(row["with_graph"][m] - row["without_graph"][m], 4)
                        for m in ("context_recall", "faithfulness", "faithfulness_deterministic")
                    }
                except Exception as e:
                    logger.error(f"Failed for child {child.id}: {e}")
                    row["error"] = str(e)
                all_results.append(row)

        # Aggregate per section and combined
        def _agg_by_condition(results, condition):
            return _aggregate([r[condition] for r in results if condition in r])

        summary = {"total": len(all_results), "sections": {}, "combined": {}}

        for label in sections:
            sec_results = [r for r in all_results if r["section"] == label]
            wg = _agg_by_condition(sec_results, "with_graph")
            nog = _agg_by_condition(sec_results, "without_graph")
            summary["sections"][label] = {
                "with_graph": wg,
                "without_graph": nog,
                "delta": {
                    m: round(wg.get(m, 0) - nog.get(m, 0), 4)
                    for m in ("context_recall", "faithfulness", "faithfulness_deterministic")
                },
                "parent_title": sections[label]["parent_title"],
            }

        valid = [r for r in all_results if "with_graph" in r]
        summary["combined"] = {
            "with_graph": _agg_by_condition(valid, "with_graph"),
            "without_graph": _agg_by_condition(valid, "without_graph"),
            "delta": {
                m: round(
                    _agg_by_condition(valid, "with_graph").get(m, 0)
                    - _agg_by_condition(valid, "without_graph").get(m, 0),
                    4,
                )
                for m in ("context_recall", "faithfulness", "faithfulness_deterministic")
            },
        }
        summary["results"] = all_results

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Cross-boundary eval complete.")
    for label, sec in summary["sections"].items():
        logger.info(
            f"  {label} ({sec['parent_title']}): "
            f"recall {sec['with_graph'].get('context_recall')} vs {sec['without_graph'].get('context_recall')} "
            f"(delta {sec['delta']['context_recall']:+.4f})"
        )
    logger.info(
        f"  Combined: recall {summary['combined']['with_graph'].get('context_recall')} "
        f"vs {summary['combined']['without_graph'].get('context_recall')} "
        f"(delta {summary['combined']['delta']['context_recall']:+.4f})"
    )
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
