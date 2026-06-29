import json
import logging
import argparse
import re
import sys
import os
from unittest.mock import MagicMock

# Add backend to path to allow running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter
from sdr.apps.ai.evaluations.shared.judges import (
    judge_faithfulness,
    judge_context_recall,
    judge_faithfulness_deterministic,
)
from sdr.apps.ai.evaluations.shared.metrics import calculate_context_precision
from sdr.apps.ai.client.manager import ai_service_manager
from sdr.apps.standards.models import CategoryParameterChild, StandardCategory, StandardIngestionJob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A judge disagreement above this threshold (|llm_faithfulness - deterministic_faithfulness|)
# flags that the LLM judge itself may need recalibration, not just the system under test.
JUDGE_DISAGREEMENT_THRESHOLD = 0.3

THRESHOLDS = {
    "faithfulness": 0.9,
    "faithfulness_deterministic": 0.80,
    "context_recall": 0.80,
    "context_precision": 0.7,
}


def _load_real_review_context(db, item: dict):
    """
    Looks up the real CategoryParameterChild/StandardCategory/StandardIngestionJob
    a "real_review" dataset item refers to, so retrieval runs through the exact
    production query path (build_parameter_analysis_text) instead of a MagicMock.
    """
    parameter = db.get(CategoryParameterChild, item["child_parameter_id"])
    category = db.get(StandardCategory, item["category_id"]) if item.get("category_id") else None
    ingestion_job = None
    if parameter is not None and category is not None:
        ingestion_job = (
            db.query(StandardIngestionJob)
            .filter(StandardIngestionJob.category_id == category.id, StandardIngestionJob.is_active == True)
            .order_by(StandardIngestionJob.created_at.desc())
            .first()
        )
    return parameter, category, ingestion_job


def _extract_answer_quotes(answer: str) -> list:
    """Pulls out evidence quotes from a Hunter-style answer.
    Filters short strings (labels/terms) and strips chunk-separator bleed-in."""
    quotes = re.findall(r'"([^"]+)"', answer or "")
    cleaned = []
    for q in quotes:
        q = re.sub(r'\s*-{2,}\s*', ' ', q).strip()
        if len(q) >= 15:
            cleaned.append(q)
    return cleaned


def evaluate_question(
    question_data: dict,
    router: HybridRetrievalRouter,
    indexes,
    db=None,
    retrieval_overrides: dict = None,
) -> dict:
    question = question_data["question"]
    ground_truth = question_data["ground_truth_context"]
    expected_block_ids = question_data.get("block_ids") or (
        [question_data["block_id"]] if "block_id" in question_data else []
    )
    source = question_data.get("source", "synthetic")

    logger.info(f"Evaluating ({source}): {question}")

    retrieve_kwargs = {
        "raptor_tree": indexes.raptor_tree,
    }

    if source == "real_review":
        if db is None:
            raise ValueError("A db session is required to evaluate real_review items.")
        parameter, category, ingestion_job = _load_real_review_context(db, question_data)
        if parameter is None or category is None:
            raise ValueError(
                f"Could not resolve real parameter/category for finding_id={question_data.get('finding_id')}."
            )
        retrieve_kwargs.update(parameter=parameter, category=category, ingestion_job=ingestion_job)
    else:
        # Synthetic items have no real backing parameter — fall back to a mock
        # target plus an explicit override of the query text.
        dummy_param = MagicMock()
        dummy_param.id = 1
        dummy_category = MagicMock()
        retrieve_kwargs.update(
            parameter=dummy_param, category=dummy_category, override_query_text=question
        )

    # Ablation/cross-boundary eval scripts pass this to force a specific
    # retrieval strategy or null out raptor_tree/graph, isolating the
    # contribution of individual retrieval components.
    if retrieval_overrides:
        retrieve_kwargs.update(retrieval_overrides)

    # 1. Retrieve
    result = router.retrieve(**retrieve_kwargs)

    chunk_block_ids = result.context_chunk_block_ids or [[] for _ in result.context_chunks]
    retrieved_block_ids = sorted({bid for group in chunk_block_ids for bid in group})
    retrieved_context = "\n---\n".join(result.context_chunks)
    # Use chunk-index keys so RAPTOR leaf + summary nodes sharing the same
    # block_ids don't overwrite each other — faithfulness checks need each
    # chunk's own text, not whichever summary happened to be written last.
    retrieved_context_blocks = {i: text for i, text in enumerate(result.context_chunks)}

    # 2. Answer (Hunter)
    answer_prompt = (
        "Based ONLY on the following context, answer this question as a security auditor. "
        "Quote the exact supporting text in double quotes. Each quote must be a character-for-character "
        "verbatim copy — do not paraphrase, abbreviate, or change any word. Each quote must come from "
        "a single contiguous location in the context — do not use ellipses (...) and do not combine "
        "multiple excerpts into one quoted string. If you need to cite multiple separate excerpts, "
        "put each one in its own pair of double quotes. Do NOT include the '---' chunk separator in "
        "any quote. If the context does not contain the answer, say 'I cannot determine'.\n\n"
        f"Context:\n{retrieved_context}\n\nQuestion:\n{question}"
    )

    response = ai_service_manager.chat_completion_with_fallback(
        messages=[{"role": "user", "content": answer_prompt}],
        component="hunter",
        temperature=0.0
    )
    answer = response.content if not response.error else ""

    # 3. Evaluate Metrics
    precision = calculate_context_precision(expected_block_ids, chunk_block_ids)
    recall_eval = judge_context_recall(question, ground_truth, retrieved_context)
    faithfulness_eval = judge_faithfulness(question, retrieved_context, answer)

    answer_quotes = _extract_answer_quotes(answer)
    faithfulness_deterministic = judge_faithfulness_deterministic(answer_quotes, retrieved_context_blocks)
    llm_faithfulness = faithfulness_eval.get("faithfulness_score", 0.0)
    judges_disagree = abs(llm_faithfulness - faithfulness_deterministic) > JUDGE_DISAGREEMENT_THRESHOLD

    return {
        "source": source,
        "question": question,
        "expected_block_ids": expected_block_ids,
        "retrieved_block_ids": retrieved_block_ids,
        "context_precision": precision,
        "context_recall": recall_eval.get("context_recall_score", 0.0),
        "faithfulness": llm_faithfulness,
        "faithfulness_deterministic": faithfulness_deterministic,
        "judges_disagree": judges_disagree,
        "hunter_answer": answer,
        "recall_details": recall_eval,
        "faithfulness_details": faithfulness_eval
    }

def main():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Pipeline.")
    parser.add_argument("--design-id", type=int, required=True, help="ID of the Design in the database")
    parser.add_argument("--dataset", type=str, default="eval_dataset.json", help="Input JSON dataset file")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output results file")
    
    args = parser.parse_args()
    
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
        
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    logger.info(f"Loading Design {args.design_id} and indexes...")
    
    with SessionLocal() as db:
        design = db.query(Design).filter(Design.id == args.design_id).first()
        if not design:
            logger.error(f"Design with ID {args.design_id} not found.")
            return

        store = DesignPreparationStore()
        try:
            prep, tsd_doc, indexes = store.load_prepared_assets(db, design)
        except Exception as e:
            logger.error(f"Failed to load indexes: {e}")
            return

        router = HybridRetrievalRouter()
        results = []

        for item in dataset:
            try:
                eval_result = evaluate_question(item, router, indexes, db=db)
                results.append(eval_result)
            except Exception as e:
                logger.error(f"Failed to evaluate question '{item['question']}': {e}")

    # Calculate averages
    avg_precision = sum(r["context_precision"] for r in results) / len(results) if results else 0
    avg_recall = sum(r["context_recall"] for r in results) / len(results) if results else 0
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results) if results else 0
    avg_faithfulness_deterministic = (
        sum(r["faithfulness_deterministic"] for r in results) / len(results) if results else 0
    )
    judge_agreement = (
        sum(1 for r in results if not r["judges_disagree"]) / len(results) if results else 0
    )

    summary = {
        "total_questions": len(results),
        "average_context_precision": avg_precision,
        "average_context_recall": avg_recall,
        "average_faithfulness": avg_faithfulness,
        "average_faithfulness_deterministic": avg_faithfulness_deterministic,
        "judge_agreement": judge_agreement,
        "results": results
    }

    summary["thresholds_met"] = {
        "faithfulness": avg_faithfulness > THRESHOLDS["faithfulness"],
        "faithfulness_deterministic": avg_faithfulness_deterministic > THRESHOLDS["faithfulness_deterministic"],
        "context_recall": avg_recall > THRESHOLDS["context_recall"],
        "context_precision": avg_precision > THRESHOLDS["context_precision"],
    }
    summary["thresholds_met"]["all_passed"] = all(summary["thresholds_met"].values())

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("Evaluation Complete!")
    logger.info(f"Average Context Precision: {avg_precision:.2f}")
    logger.info(f"Average Context Recall: {avg_recall:.2f}")
    logger.info(f"Average Faithfulness (LLM judge): {avg_faithfulness:.2f}")
    logger.info(f"Average Faithfulness (deterministic): {avg_faithfulness_deterministic:.2f}")
    logger.info(f"Judge agreement rate: {judge_agreement:.2f}")
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
