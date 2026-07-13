"""
Full-Document ("stuff everything") Baseline for Lost-in-the-Middle Evaluation.

The deliberately weak, retrieval-free baseline: instead of retrieving a
targeted context window per question, this dumps the ENTIRE TSD (every text
block, full document) into a single LLM call together with all of a zone's
questions at once, and asks the model to answer each from the full document.
No RAPTOR, no BM25, no vector search, no per-question retrieval at all — the
model gets everything and has to locate the right passage itself.

Cost profile is the opposite of the other retrieval arms: this makes exactly
ONE LLM call per zone (10 questions batched into one prompt), vs. one call
per question for flat_topk/raptor_low/raptor_high/hybrid. It exists as a
cheap "why bother with retrieval at all?" baseline — if this performed as
well as hybrid, targeted retrieval would be hard to justify. The expectation
(and the point of the comparison) is that it performs worse specifically at
locating the correct passage out of ~1000+ blocks, especially for
middle-zone evidence — the same generation-time attention-dilution effect
Liu et al. (2023) documented, just now at the "answer 10 questions against
the whole document at once" scale instead of single-question long-context.

Because the model sees the whole document, the standard context_recall
metric (does the retrieved context contain the ground truth?) is trivially
~1.0 here by construction and not informative — the document IS the ground
truth's source. The metric that matters is whether the model's answer
actually cites the SPECIFIC correct passage (not just any plausible-looking
text), which we check deterministically (fuzzy quote-to-block grounding —
the same 85%-coverage check the Critic's citation validator uses), with the
answer's quoted evidence checked against the exact expected block(s) for
that question:
  block_hit               : did >=1 answer quote fuzzy-match the EXPECTED
                             block's text specifically (not just any block)?
  faithfulness_deterministic : of quotes the answer cites, what fraction are
                             grounded in *some* real block (not hallucinated),
                             regardless of whether it's the right one?
Both are deterministic (no extra LLM judge calls), keeping the whole
per-zone cost at exactly one LLM call.

Usage:
    python full_document_baseline_eval.py --design-id 15 \\
        --dataset eval_dataset_lost_in_middle_carpool.json \\
        --output eval_full_document_baseline_design15.json
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
from sdr.apps.ai.client.manager import ai_service_manager
from sdr.apps.ai.evaluations import runner as runner_mod
from sdr.apps.ai.evaluations.shared.dataset_generator import ZONES
from sdr.apps.ai.evaluations.shared import results_path, data_path
from sdr.apps.ai.retrieval.postprocessing.quote_grounding import is_quote_grounded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ANSWER_SYSTEM_PROMPT = (
    "You are a security auditor answering multiple questions about a technical "
    "design document. You are given the ENTIRE document below, followed by a "
    "numbered list of questions. Answer EVERY question using ONLY the document "
    "content. For each answer, quote the exact supporting text in double quotes — "
    "a character-for-character verbatim copy, from a single contiguous location "
    "in the document. Do not paraphrase, abbreviate, or use ellipses. If you need "
    "multiple separate excerpts for one answer, put each in its own pair of double "
    "quotes. If the document does not contain the answer to a question, say "
    "'I cannot determine' for that question.\n\n"
    "Return ONLY a JSON object of the form "
    '{"answers": {"1": "<answer to question 1>", "2": "<answer to question 2>", ...}} '
    "keyed by question number as a string, with one entry per question."
)


def _build_full_document_text(blocks) -> str:
    return "\n---\n".join(b.text for b in blocks if b.text)


def _build_batch_prompt(document_text: str, questions: list) -> str:
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return f"Document:\n{document_text}\n\nQuestions:\n{numbered}"


def _score_question(answer: str, expected_block_ids: list, all_blocks) -> dict:
    quotes = runner_mod._extract_answer_quotes(answer)
    expected_texts = [b.text for b in all_blocks if b.block_id in expected_block_ids]
    block_hit = bool(quotes) and bool(expected_texts) and any(
        is_quote_grounded(q, t) for q in quotes for t in expected_texts
    )
    if not quotes:
        faithfulness_deterministic = 1.0
    else:
        grounded = sum(
            1 for q in quotes if any(is_quote_grounded(q, b.text) for b in all_blocks)
        )
        faithfulness_deterministic = grounded / len(quotes)
    return {
        "block_hit": block_hit,
        "faithfulness_deterministic": faithfulness_deterministic,
        "n_quotes": len(quotes),
        "answer": answer,
    }


def _aggregate(results: list) -> dict:
    if not results:
        return {"count": 0, "block_hit_rate": 0.0, "faithfulness_deterministic": 0.0}
    n = len(results)
    return {
        "count": n,
        "block_hit_rate": round(sum(1 for r in results if r["block_hit"]) / n, 4),
        "faithfulness_deterministic": round(
            sum(r["faithfulness_deterministic"] for r in results) / n, 4
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full-document (no retrieval) baseline — one LLM call per zone answering all its questions at once."
    )
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_full_document_baseline.json")
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
        all_blocks = tsd_doc.all_text_blocks
        document_text = _build_full_document_text(all_blocks)
        approx_tokens = len(document_text) // 4
        logger.info(
            f"Loaded TSD: {tsd_doc.document_name} — {total_pages} pages, "
            f"{len(all_blocks)} blocks, ~{approx_tokens} tokens of document text"
        )

    all_results = []
    zone_results: dict[str, list] = {z: [] for z in ZONES}

    for zone in ZONES:
        items = zone_datasets.get(zone, [])
        if not items:
            continue
        questions = [item["question"] for item in items]
        logger.info(f"[{zone}] single batched LLM call for {len(questions)} questions")

        prompt = _build_batch_prompt(document_text, questions)
        response = ai_service_manager.chat_completion_with_fallback(
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            component="long_context",
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        if response.error:
            logger.error(f"[{zone}] batch call failed: {response.error}")
            answers_by_number = {}
        else:
            try:
                parsed = json.loads(response.content)
                answers_by_number = parsed.get("answers", {}) if isinstance(parsed, dict) else {}
            except Exception as e:
                logger.error(f"[{zone}] failed to parse batch response: {e}")
                answers_by_number = {}

        for i, item in enumerate(items):
            answer = answers_by_number.get(str(i + 1), "")
            if not answer:
                logger.warning(f"[{zone} {i+1}/{len(items)}] no answer returned for this question")
            scored = _score_question(answer, item["block_ids"], all_blocks)
            row = {
                "zone": zone,
                "question": item["question"],
                "block_ids": item["block_ids"],
                **scored,
            }
            zone_results[zone].append(scored)
            all_results.append(row)

    by_zone_agg = {zone: _aggregate(zone_results[zone]) for zone in ZONES}
    overall = _aggregate([r for z in ZONES for r in zone_results[z]])

    def _zone_rate(zone: str) -> float:
        return by_zone_agg[zone].get("block_hit_rate", 0.0)

    mid = _zone_rate("middle")
    edge = (_zone_rate("front") + _zone_rate("back")) / 2
    middle_deficit = round(edge - mid, 4)

    summary = {
        "design_id": args.design_id,
        "dataset": dataset_path,
        "tsd_name": tsd_doc.document_name,
        "total_pages": total_pages,
        "total_blocks": len(all_blocks),
        "approx_document_tokens": approx_tokens,
        "samples_per_zone": samples_per_zone,
        "total_questions": total_items,
        "llm_calls_made": len([z for z in ZONES if zone_datasets.get(z)]),
        "by_zone": by_zone_agg,
        "overall": overall,
        "middle_deficit_block_hit": middle_deficit,
        "results": all_results,
    }

    output_path = results_path(args.output, subdir="retrieval")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Full-Document Baseline Results ===")
    logger.info(f"  TSD: {tsd_doc.document_name} ({total_pages} pages, {len(all_blocks)} blocks, ~{approx_tokens} tokens)")
    logger.info(f"  Total QA pairs: {total_items} ({samples_per_zone} per zone)")
    logger.info(f"  LLM calls made: {summary['llm_calls_made']} (vs {total_items} for per-question retrieval arms)")
    logger.info("")
    for zone in ZONES:
        agg = by_zone_agg[zone]
        logger.info(
            f"  [{zone:6s}] block_hit_rate={agg.get('block_hit_rate', 0):.4f}  "
            f"faithfulness_deterministic={agg.get('faithfulness_deterministic', 0):.4f}"
        )
    logger.info("")
    logger.info(f"  Overall block_hit_rate: {overall.get('block_hit_rate', 0):.4f}")
    logger.info(f"  Overall faithfulness_deterministic: {overall.get('faithfulness_deterministic', 0):.4f}")
    logger.info(f"  middle_deficit_block_hit: {middle_deficit:+.4f}  (how much worse middle-zone localization is vs edges)")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
