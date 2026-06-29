import json
import logging
import random
import argparse
from typing import Dict, List, Any
import sys
import os

# Add backend to path to allow running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.ai.client.manager import ai_service_manager
from sdr.apps.ai.client.base import AIProvider
from sdr.apps.reviews.models import Finding, CitationAnchor
from sdr.apps.reviews.models.choices import MetStatus
from sdr.apps.ai.retrieval.postprocessing.quote_grounding import is_quote_grounded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Security Architect and Application Security Auditor.
Your task is to generate realistic security requirement questions based on a provided text snippet from a Technical Software Document (TSD).
The question should be formatted similar to an OWASP ASVS requirement (e.g., "Verify that the application encrypts communications...").
The question MUST be perfectly and directly answerable by the provided snippet.

You must also extract the EXACT sentence or paragraph from the snippet that serves as the ground truth answer. Do not rephrase the ground truth.

Return your response ONLY as a valid JSON object with the following structure:
{
    "question": "The generated security requirement question",
    "ground_truth_context": "The exact sentence or phrase from the snippet that answers it"
}
"""

def build_real_review_dataset(db, review_id: int) -> List[Dict[str, Any]]:
    """
    Builds eval items from a real, already-completed review's persisted
    Findings + CitationAnchors instead of LLM-synthesized text. The question
    is the actual ASVS requirement text the system was asked to evaluate, and
    the ground truth is the real cited evidence a Finding was persisted with
    — not an LLM's self-extracted snippet of a random TSD block.

    `na` findings and findings with no citations are skipped: there's no real
    evidence to test retrieval/faithfulness against in either case.
    """
    findings = (
        db.query(Finding)
        .filter(
            Finding.review_id == review_id,
            Finding.met_status.in_([MetStatus.MET.value, MetStatus.NOT_MET.value]),
        )
        .all()
    )

    dataset: List[Dict[str, Any]] = []
    for finding in findings:
        citations = list(finding.citations or [])
        if not citations:
            continue

        dataset.append(
            {
                "source": "real_review",
                "finding_id": finding.id,
                "category_id": finding.category_id,
                "child_parameter_id": finding.child_parameter_id,
                "block_ids": [c.block_id for c in citations],
                "question": finding.requirement_text or "",
                "ground_truth_context": "\n---\n".join(
                    (c.quoted_text or "") for c in citations if c.quoted_text
                ),
            }
        )

    return dataset


def _merge_blocks_into_samples(
    blocks: List[Any], min_words: int = 30, max_words: int = 150
) -> List[Dict[str, Any]]:
    """
    Merges consecutive non-heading text blocks into samples with between
    `min_words` and `max_words` words. Needed because PyMuPDF's layout-based
    block extraction often yields short fragments (table cells, bullets,
    single lines) rather than full paragraphs — sampling single raw blocks
    for QA generation would starve documents made mostly of such fragments
    (e.g. tables, diagrams). A merge resets at section boundaries, headings,
    and the `max_words` cap so a sample's text stays topically coherent and
    small enough for a focused QA pair.
    """
    samples: List[Dict[str, Any]] = []
    current_texts: List[str] = []
    current_block_ids: List[str] = []
    current_words = 0
    current_section = None

    def flush():
        if current_texts and current_words >= min_words:
            samples.append({
                "block_ids": list(current_block_ids),
                "text": "\n".join(current_texts),
            })

    for block in blocks:
        if block.is_heading:
            flush()
            current_texts, current_block_ids, current_words = [], [], 0
            current_section = block.section_heading
            continue
        if block.section_heading != current_section or current_words >= max_words:
            flush()
            current_texts, current_block_ids, current_words = [], [], 0
            current_section = block.section_heading
        current_texts.append(block.text)
        current_block_ids.append(block.block_id)
        current_words += block.word_count

    flush()
    return samples


def generate_qa_pair(text_snippet: str) -> Dict[str, str]:
    user_prompt = f"Generate a QA pair for the following text snippet:\n\n{text_snippet}"
    
    response = ai_service_manager.chat_completion_with_fallback(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        component="hunter",
        response_format={"type": "json_object"},
        temperature=0.3
    )
    
    if response.error:
        logger.error(f"Failed to generate QA: {response.error}")
        return None
        
    try:
        data = json.loads(response.content)
        return data
    except Exception as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}\nContent: {response.content}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate an evaluation dataset from a Design.")
    parser.add_argument("--design-id", type=int, required=True, help="ID of the Design in the database")
    parser.add_argument(
        "--source",
        type=str,
        choices=["real_review", "synthetic"],
        default="synthetic",
        help=(
            "real_review: build the dataset from an already-completed review's persisted "
            "Findings/citations (recommended — grounded in real, human-relevant requirements "
            "and real cited evidence). synthetic: have an LLM invent Q&A pairs from random TSD "
            "text snippets (fallback for designs with no completed review yet)."
        ),
    )
    parser.add_argument("--review-id", type=int, help="Review ID to source Findings from (required for --source real_review)")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of QA pairs to generate (synthetic mode only)")
    parser.add_argument("--output", type=str, default="eval_dataset.json", help="Output JSON file name")

    args = parser.parse_args()

    if args.source == "real_review":
        if not args.review_id:
            logger.error("--review-id is required when --source real_review is used.")
            return

        with SessionLocal() as db:
            dataset = build_real_review_dataset(db, args.review_id)

        if not dataset:
            logger.warning(
                f"No eligible Findings (met/not_met with citations) found for review_id={args.review_id}."
            )

        output_path = os.path.join(os.path.dirname(__file__), args.output)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=4)

        logger.info(f"Successfully generated real-review dataset with {len(dataset)} items and saved to {output_path}")
        return

    logger.info(f"Loading Design {args.design_id} from database...")

    with SessionLocal() as db:
        design = db.query(Design).filter(Design.id == args.design_id).first()
        if not design:
            logger.error(f"Design with ID {args.design_id} not found.")
            return

        store = DesignPreparationStore()
        try:
            prep, tsd_doc, indexes = store.load_prepared_assets(db, design)
        except Exception as e:
            logger.error(f"Failed to load prepared assets for Design {args.design_id}: {e}")
            return

    logger.info(f"Successfully loaded TSD: {tsd_doc.document_name} with {len(tsd_doc.all_text_blocks)} blocks.")

    # Merge consecutive blocks into samples long enough to contain meaningful context.
    valid_samples = _merge_blocks_into_samples(tsd_doc.all_text_blocks, min_words=30)

    if len(valid_samples) < args.num_samples:
        logger.warning(f"Only found {len(valid_samples)} valid samples, less than requested {args.num_samples}.")
        samples = valid_samples
    else:
        samples = random.sample(valid_samples, args.num_samples)

    dataset = []

    for i, sample in enumerate(samples):
        logger.info(f"Generating QA pair {i+1}/{len(samples)} from blocks {sample['block_ids']}...")

        # Retry once if the LLM paraphrases the ground truth instead of quoting
        # it verbatim — an ungrounded ground truth makes perfect recall
        # structurally unattainable later regardless of retrieval quality.
        qa = None
        for attempt in range(2):
            candidate_qa = generate_qa_pair(sample["text"])
            if not candidate_qa:
                continue
            if is_quote_grounded(candidate_qa.get("ground_truth_context", ""), sample["text"]):
                qa = candidate_qa
                break
            logger.warning(
                f"Sample {i+1}: ground_truth_context not grounded in source text "
                f"(attempt {attempt + 1}/2) — {'retrying' if attempt == 0 else 'skipping'}."
            )

        if qa:
            dataset.append({
                "source": "synthetic",
                "block_ids": sample["block_ids"],
                "original_text": sample["text"],
                "question": qa.get("question", ""),
                "ground_truth_context": qa.get("ground_truth_context", "")
            })

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
        
    logger.info(f"Successfully generated dataset with {len(dataset)} items and saved to {output_path}")

if __name__ == "__main__":
    main()
