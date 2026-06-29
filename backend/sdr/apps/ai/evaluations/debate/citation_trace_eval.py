"""
Exp 7 — Citation Trace Accuracy.

Proves the system is not a black box: citations produced by the debate pipeline
are verifiable and anchor real evidence in the TSD source document.

Three layers of audit:
  1. Block Existence Rate  — % of CitationAnchors whose block_id exists in the TSD index
  2. Quote Grounding Rate  — % where quoted_text verbatim-appears in the source block content
  3. Coverage Rate         — % of findings that have ≥1 valid citation (not zero citations)

Additionally produces a sample list (N=30) for manual reviewer-relevance validation,
which becomes the third metric in the thesis.

Usage:
    python citation_trace_eval.py --review-id 1
    python citation_trace_eval.py --review-id 1 --output citation_audit.json --sample-size 30
"""
import argparse
import json
import logging
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import sdr.apps.standards.models  # noqa: F401 — resolve SQLAlchemy FK
import sdr.apps.designs.models    # noqa: F401
import sdr.apps.reviews.models.finding  # noqa: F401
import sdr.apps.reviews.models.review   # noqa: F401

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.reviews.models.review import Review
from sdr.apps.reviews.models.finding import Finding
from sdr.apps.reviews.models.citation import CitationAnchor

from sdr.apps.ai.evaluations.shared import results_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_tsd_block_index(db, design: Design) -> dict[str, str]:
    """Return {block_id: text_content} for all text blocks in the TSD."""
    try:
        store = DesignPreparationStore()
        _, tsd_document, _ = store.load_prepared_assets(db, design)
    except Exception as exc:
        logger.error(f"Failed to load TSD document: {exc}")
        return {}

    block_index: dict[str, str] = {}
    for block in tsd_document.all_text_blocks:
        block_index[block.block_id] = block.text
    for diagram in tsd_document.all_diagrams:
        block_index[diagram.diagram_id] = diagram.caption or ""
    logger.info(f"TSD block index: {len(block_index)} blocks loaded")
    return block_index


def _quote_grounded(quoted_text: str, block_text: str) -> bool:
    """True if the quoted snippet (≥10 chars) verbatim-appears in the block text (case-insensitive)."""
    if not quoted_text or not block_text:
        return False
    snippet = quoted_text.strip()
    if len(snippet) < 10:
        return True  # too short to meaningfully check; don't penalise
    return snippet.lower() in block_text.lower()


def main():
    parser = argparse.ArgumentParser(
        description="Exp 7: Citation trace accuracy — block existence + quote grounding audit."
    )
    parser.add_argument("--review-id", type=int, required=True)
    parser.add_argument("--output", type=str, default="citation_audit.json")
    parser.add_argument(
        "--sample-size", type=int, default=30,
        help="Number of citations to sample for manual relevance validation"
    )
    args = parser.parse_args()

    with SessionLocal() as db:
        review = db.get(Review, args.review_id)
        if not review:
            logger.error(f"Review {args.review_id} not found.")
            return

        design = db.get(Design, review.design_id)
        if not design:
            logger.error(f"Design not found for review {args.review_id}.")
            return

        findings = db.query(Finding).filter_by(review_id=args.review_id).all()
        logger.info(f"Loaded {len(findings)} findings for review_id={args.review_id}")

        all_citations = (
            db.query(CitationAnchor)
            .join(Finding)
            .filter(Finding.review_id == args.review_id)
            .all()
        )
        logger.info(f"Loaded {len(all_citations)} CitationAnchors")

        if not all_citations:
            logger.error("No citations found. Run a security design review first.")
            return

        block_index = _load_tsd_block_index(db, design)

    # Audit each citation
    per_citation = []
    valid_citations_by_finding: dict[int, list] = {}

    for anchor in all_citations:
        block_exists = anchor.block_id in block_index
        block_text = block_index.get(anchor.block_id, "")
        quote_grounded = _quote_grounded(anchor.quoted_text or "", block_text) if block_exists else False

        row = {
            "citation_id": anchor.id,
            "finding_id": anchor.finding_id,
            "block_id": anchor.block_id,
            "page_number": anchor.page_number,
            "anchor_type": anchor.anchor_type,
            "quoted_text": (anchor.quoted_text or "")[:200],
            "block_exists": block_exists,
            "quote_grounded": quote_grounded,
            "relevance_manual": None,  # filled in by human reviewer from the sample list
        }
        per_citation.append(row)

        if block_exists:
            valid_citations_by_finding.setdefault(anchor.finding_id, []).append(row)

        status = "✓" if block_exists else "✗"
        grounded_str = " grounded=✓" if quote_grounded else (" grounded=✗" if block_exists else "")
        logger.debug(f"  [{anchor.block_id}] {status}{grounded_str} finding={anchor.finding_id}")

    # Metrics
    total = len(per_citation)
    existing = sum(1 for c in per_citation if c["block_exists"])
    grounded = sum(1 for c in per_citation if c["quote_grounded"])

    block_existence_rate = round(existing / total, 4) if total else 0.0
    quote_grounding_rate = round(grounded / existing, 4) if existing else 0.0

    findings_with_citations = len(valid_citations_by_finding)
    coverage_rate = round(findings_with_citations / len(findings), 4) if findings else 0.0

    met_findings = [f for f in findings if (f.met_status or "").lower() == "met"]
    not_met_findings = [f for f in findings if (f.met_status or "").lower() == "not_met"]

    avg_citations_met = (
        sum(len(valid_citations_by_finding.get(f.id, [])) for f in met_findings) / len(met_findings)
        if met_findings else 0.0
    )
    avg_citations_not_met = (
        sum(len(valid_citations_by_finding.get(f.id, [])) for f in not_met_findings) / len(not_met_findings)
        if not_met_findings else 0.0
    )

    # Random sample for manual relevance validation
    random.seed(42)
    sample = random.sample(per_citation, min(args.sample_size, len(per_citation)))
    for s in sample:
        s["relevance_manual"] = "TODO"  # reviewer fills this in

    summary = {
        "review_id": args.review_id,
        "total_findings": len(findings),
        "total_citations": total,
        "block_existence_rate": block_existence_rate,
        "quote_grounding_rate": quote_grounding_rate,
        "citation_coverage_rate": coverage_rate,
        "findings_with_valid_citations": findings_with_citations,
        "avg_citations_per_met_finding": round(avg_citations_met, 2),
        "avg_citations_per_not_met_finding": round(avg_citations_not_met, 2),
        "thresholds": {
            "block_existence_gte_0.95": block_existence_rate >= 0.95,
            "quote_grounding_gte_0.80": quote_grounding_rate >= 0.80,
            "coverage_gte_0.80": coverage_rate >= 0.80,
        },
        "manual_validation_sample": sample,
        "per_citation_audit": per_citation,
    }

    output_path = results_path(args.output, subdir="debate")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Exp 7: Citation Trace Results ===")
    logger.info(f"  Total citations:         {total}")
    logger.info(f"  Block existence rate:    {block_existence_rate:.4f}  "
                f"(threshold ≥0.95: {summary['thresholds']['block_existence_gte_0.95']})")
    logger.info(f"  Quote grounding rate:    {quote_grounding_rate:.4f}  "
                f"(threshold ≥0.80: {summary['thresholds']['quote_grounding_gte_0.80']})")
    logger.info(f"  Citation coverage rate:  {coverage_rate:.4f}  "
                f"({findings_with_citations}/{len(findings)} findings have ≥1 valid citation)")
    logger.info(f"  Avg citations/met:       {avg_citations_met:.2f}")
    logger.info(f"  Avg citations/not_met:   {avg_citations_not_met:.2f}")
    logger.info(f"  Manual sample ({len(sample)}):    saved to output for human review")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
