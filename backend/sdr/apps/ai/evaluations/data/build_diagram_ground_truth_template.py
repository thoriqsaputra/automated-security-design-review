"""
Ground-truth-authoring scaffold for diagram evaluations.

Produces ONE labeled-template file that feeds BOTH `retrieval/diagram_retrieval_eval.py`
(relevance labels) and `debate/diagram_ablation_eval.py` (verdict labels), built to avoid
the two biases that would otherwise creep into a hand-written ground truth file:

  1. Selection bias — if a human only ever sees the requirements the system already
     retrieved/assessed, a retrieval MISS can never be caught or labeled. So every
     diagram in the template lists the FULL candidate pool of diagram requirements for
     the review's category (via `list_diagram_requirements`), not just the subset the
     system happened to select.
  2. Grounding bias — a human labeling from a caption/summary alone is guessing at what
     the model actually saw. So this script downloads each diagram's persisted MARKED
     image (the Set-of-Mark-annotated image the Hunter/Critic/Mediator actually looked
     at) from MinIO to local disk, next to the template JSON.

Usage:
    python build_diagram_ground_truth_template.py --review-id 12
    python build_diagram_ground_truth_template.py --review-id 12 --output my_gt.json

    # Auto-populate "relevant" via a vision LLM judge (component=eval_judge,
    # a different model family from Hunter/Critic/Mediator) instead of manual
    # review. Writes to diagram_ground_truth_review_<id>_llm_judged.json by
    # default, so it never overwrites a hand-labeled file:
    python build_diagram_ground_truth_template.py --review-id 12 --llm-judge

Without --llm-judge, open each image under
`evaluations/results/vision/ground_truth_images/review_<id>/` and, for every diagram's
`candidate_requirements` entries in the output JSON:
  - set "relevant": true/false — is this requirement genuinely checkable from this diagram?
  - for "relevant": true rows only, set "label": "met" | "not_met" | "na"
  - optionally fill "notes" with your reasoning

With --llm-judge, "relevant" (plus "judge_reasoning") is filled in automatically;
"label" still requires manual review either way (out of scope for the judge).

The filled-in file is then passed as --ground-truth to both:
    retrieval/diagram_retrieval_eval.py   (uses "relevant" flags)
    debate/diagram_ablation_eval.py       (uses "relevant" + "label")
"""
import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import sdr.apps.standards.models  # noqa: F401 — resolve SQLAlchemy FK
import sdr.apps.designs.models    # noqa: F401
import sdr.apps.reviews.models.finding  # noqa: F401
import sdr.apps.reviews.models.review   # noqa: F401

from sdr.core.database import SessionLocal
from sdr.apps.reviews.models.review import Review
from sdr.apps.reviews.models.finding import Finding
from sdr.apps.reviews.models.choices import FindingType
from sdr.apps.ai.engine.persistence.workflow_repository import SqlAlchemyReviewWorkflowRepository
from sdr.apps.workspace.services.storage import storage_service

from sdr.apps.ai.evaluations.shared import results_path, data_path
from sdr.apps.ai.evaluations.shared.judges import judge_diagram_requirement_relevance

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _assessed_requirement_ids(finding: Finding) -> list[str]:
    meta = finding.requirement_metadata or {}
    trace = meta.get("analysis_trace", {})
    assessed = trace.get("assessed_requirements") or meta.get("assessed_requirements") or []
    return [
        str(item.get("requirement_id", "")).strip()
        for item in assessed
        if isinstance(item, dict) and str(item.get("requirement_id", "")).strip()
    ]


def _download_diagram_image(finding: Finding, images_dir: str) -> str | None:
    meta = finding.requirement_metadata or {}
    diagram_image = meta.get("diagram_image") or {}
    object_name = diagram_image.get("object_name")
    if not object_name:
        logger.warning(f"  finding_id={finding.id} diagram_id={finding.diagram_id}: no diagram_image.object_name")
        return None
    try:
        image_bytes = storage_service.download_bytes(object_name)
    except Exception as exc:
        logger.warning(f"  finding_id={finding.id} diagram_id={finding.diagram_id}: download failed: {exc}")
        return None

    ext = os.path.splitext(object_name)[1] or ".png"
    filename = f"{finding.diagram_id}{ext}"
    local_path = os.path.join(images_dir, filename)
    with open(local_path, "wb") as f:
        f.write(image_bytes)
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Build a labeled ground-truth template for diagram retrieval/debate evals."
    )
    parser.add_argument("--review-id", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Auto-populate 'relevant' via a vision LLM judge (component=eval_judge) "
             "instead of leaving it null for manual review. Costs one vision LLM call "
             "per (diagram, candidate requirement) row. Does not touch 'label' "
             "(met/not_met/na), which stays manual either way.",
    )
    args = parser.parse_args()

    with SessionLocal() as db:
        review = db.query(Review).filter(Review.id == args.review_id).first()
        if not review:
            logger.error(f"Review with ID {args.review_id} not found.")
            return
        if review.category_id is None or review.ingestion_job_id is None:
            logger.error(
                f"Review {args.review_id} has no category_id/ingestion_job_id — "
                "cannot resolve its diagram requirement pool."
            )
            return

        findings = (
            db.query(Finding)
            .filter(
                Finding.review_id == args.review_id,
                Finding.finding_type == FindingType.DIAGRAM.value,
            )
            .order_by(Finding.diagram_id)
            .all()
        )
        if not findings:
            logger.error(f"No diagram findings for review_id={args.review_id}. Run a vision-enabled review first.")
            return

        workflow_repository = SqlAlchemyReviewWorkflowRepository()
        candidate_pool = workflow_repository.list_diagram_requirements(
            category_id=review.category_id,
            ingestion_job_id=review.ingestion_job_id,
        )
        if not candidate_pool:
            logger.error(
                f"No diagram requirements found for category_id={review.category_id} "
                f"ingestion_job_id={review.ingestion_job_id}."
            )
            return

        images_subdir = f"ground_truth_images/review_{args.review_id}"
        images_dir = os.path.join(results_path("", subdir="vision"), images_subdir)
        os.makedirs(images_dir, exist_ok=True)

        items = []
        for finding in findings:
            meta = finding.requirement_metadata or {}
            assessed_ids = set(_assessed_requirement_ids(finding))
            image_filename = _download_diagram_image(finding, images_dir)
            image_path = f"vision/{images_subdir}/{image_filename}" if image_filename else None

            image_bytes = None
            if args.llm_judge and image_filename:
                with open(os.path.join(images_dir, image_filename), "rb") as f:
                    image_bytes = f.read()
                image_format = os.path.splitext(image_filename)[1].lstrip(".") or "png"

            candidate_requirements = []
            for i, req in enumerate(candidate_pool):
                row = {
                    "requirement_id": req.stable_key,
                    "requirement_text": req.requirement_text,
                    "verification_hint": req.verification_hint,
                    "was_assessed_by_system": req.stable_key in assessed_ids,
                    "relevant": None,
                    "label": None,
                    "notes": "",
                }
                if args.llm_judge and image_bytes:
                    logger.info(
                        f"  [{i + 1}/{len(candidate_pool)}] judging diagram_id={finding.diagram_id} "
                        f"requirement_id={req.stable_key}"
                    )
                    judged = judge_diagram_requirement_relevance(
                        image_bytes=image_bytes,
                        requirement_text=req.requirement_text,
                        verification_hint=req.verification_hint,
                        image_format=image_format,
                    )
                    row["relevant"] = judged.get("relevant")
                    row["judge_reasoning"] = judged.get("reasoning", "")
                    if judged.get("error"):
                        row["judge_error"] = judged["error"]
                candidate_requirements.append(row)

            items.append({
                "diagram_id": finding.diagram_id,
                "finding_id": finding.id,
                "page_number": meta.get("diagram_page_number"),
                "caption": meta.get("diagram_caption") or finding.diagram_caption,
                "image_path": image_path,
                "system_assessed_requirement_ids": sorted(assessed_ids),
                "candidate_requirements": candidate_requirements,
            })

    template = {
        "review_id": review.id,
        "design_id": review.design_id,
        "category_id": review.category_id,
        "ingestion_job_id": review.ingestion_job_id,
        "items": items,
    }

    if args.output:
        output_filename = args.output
    elif args.llm_judge:
        # Never collide with (and silently overwrite) a hand-labeled file —
        # LLM-judged output always goes to its own distinctly-named file.
        output_filename = f"diagram_ground_truth_review_{args.review_id}_llm_judged.json"
    else:
        output_filename = f"diagram_ground_truth_review_{args.review_id}.json"
    output_path = data_path(output_filename)
    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)

    total_candidates = sum(len(item["candidate_requirements"]) for item in items)
    logger.info("\n=== Diagram Ground Truth Template Built ===")
    logger.info(f"  Diagrams: {len(items)}")
    logger.info(f"  Candidate requirements per diagram: {len(candidate_pool)} (full category pool)")
    logger.info(f"  Total (diagram, requirement) rows to label: {total_candidates}")
    logger.info(f"  Images saved under: {images_dir}/")
    logger.info(f"  Template written to: {output_path}")
    logger.info(
        "\nNext steps: open each diagram's image_path, then for every candidate_requirements "
        "entry set \"relevant\": true/false, and — for relevant:true rows only — "
        "\"label\": \"met\"|\"not_met\"|\"na\". Then pass this file as --ground-truth to "
        "retrieval/diagram_retrieval_eval.py and debate/diagram_ablation_eval.py."
    )


if __name__ == "__main__":
    main()
