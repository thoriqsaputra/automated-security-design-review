"""
Visual Grounding & Marker Utilization Eval (real diagrams).

Measures whether the Vision Agent actually uses Set-of-Mark marker references
when citing visual evidence, how reliably the Critic catches bad citations,
and how well-calibrated the final confidence scores are — run against REAL
diagrams from a design's parsed TSD document, using the same labeled
`diagram_ground_truth_review_<id>.json` file consumed by
retrieval/diagram_retrieval_eval.py and debate/diagram_ablation_eval.py (see
evaluations/data/build_diagram_ground_truth_template.py). design_id is read
directly from the ground-truth file.

Metrics:
  1. marker_utilization_rate       — fraction of Hunter assessments that cite [N]
  2. scope_accuracy                — diagram scope classification accuracy
  3. critic_overturn_rate          — how often Critic overturns Hunter
  4. invalid_marker_citation_rate  — fraction of Hunter's cited markers that
                                      Critic flagged as invalid
  5. confidence_calibration        — per-bucket accuracy vs. confidence score

Scope-accuracy needs both an "architecture_relevant" and a "non_architecture"
class. The former comes from any labeled diagram with >=1 relevant=true
requirement; the latter from any labeled diagram where EVERY candidate
requirement is relevant=false (a real non-architectural diagram, e.g. a
screenshot). If the labeled ground truth set has no such diagram, one
synthetic blank-image scenario is used as a documented fallback so the
metric remains computable.

Usage:
    python grounding_eval.py --ground-truth /app/sdr/apps/ai/evaluations/data/diagram_ground_truth_review_48.json [--output grounding_results.json]
    python grounding_eval.py --from-blindness-results eval_vision_blindness.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from PIL import Image

from sdr.apps.ai.agents.vision import DiagramInput
from sdr.apps.ai.engine.debate.diagram_debate_service import DiagramDebateService
from sdr.apps.ai.evaluations.shared import results_path
from sdr.apps.ai.evaluations.vision.real_diagram_source import (
    build_diagram_input,
    diagrams_with_no_relevant_requirements,
    load_ground_truth,
    load_labeled_samples,
    load_tsd_document,
    save_image_b64,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Regex that matches [N] marker references in free-text fields
_MARKER_RE = re.compile(r"\[\d+\]")

# Kept ONLY as a fallback for the non-architecture scope-negative class, used
# if the labeled ground truth set has no real diagram with zero relevant
# requirements (see run_grounding_eval).
_FALLBACK_NON_ARCH_CAPTION = "Screenshot of login form"
_FALLBACK_REQUIREMENT = SimpleNamespace(
    ordinal=1,
    stable_key="D-SCOPE-1",
    requirement_text="Verify that authentication mechanisms are explicitly depicted in the architecture diagram.",
    verification_hint="Look for any authentication or auth service component.",
)


def _count_marker_refs(text: str) -> int:
    return len(_MARKER_RE.findall(text or ""))


def _has_any_marker_ref(*texts: str) -> bool:
    return any(_MARKER_RE.search(t or "") for t in texts)


def _compute_marker_utilization(debate_output) -> dict:
    """Returns per-output marker utilization stats from Hunter result."""
    hunter = debate_output.hunter_result or {}
    assessments = hunter.get("requirement_assessments") or []
    if not assessments:
        return {"total_assessments": 0, "assessments_with_marker": 0, "marker_ids_cited": []}

    assessments_with_marker = sum(
        1
        for a in assessments
        if _has_any_marker_ref(a.get("visual_evidence"), a.get("reasoning"))
    )
    top_level_cited = hunter.get("marker_ids_cited") or []
    return {
        "total_assessments": len(assessments),
        "assessments_with_marker": assessments_with_marker,
        "marker_ids_cited": top_level_cited,
        "utilization_rate": assessments_with_marker / len(assessments),
    }


def _blank_white_image_b64() -> str:
    img = Image.new("RGB", (900, 300), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _run_debate_and_record(service, diagram, requirement, scope_label, images_dir=None, control_tag=None):
    if images_dir:
        save_image_b64(diagram.image_b64, os.path.join(images_dir, f"{diagram.diagram_id}_input.png"))

    output = service.run_diagram_debate(diagram=diagram, requirements=[requirement], tsd_context="")

    if images_dir:
        save_image_b64(diagram.image_b64, os.path.join(images_dir, f"{diagram.diagram_id}_marked.png"))

    marker_stats = _compute_marker_utilization(output)
    critic = output.critic_result or {}
    mediator = output.mediator_result or {}

    return {
        "diagram_id": diagram.diagram_id,
        "control": control_tag or "requirement_check",
        "scope_verdict": mediator.get("diagram_scope_verdict"),
        "scope_label": scope_label,
        "scope_correct": mediator.get("diagram_scope_verdict") == scope_label,
        "final_verdict": mediator.get("final_verdict"),
        "confidence": mediator.get("confidence"),
        "critic_outcome": critic.get("outcome"),
        "invalid_marker_citations": critic.get("invalid_marker_citations") or [],
        **marker_stats,
        "error": output.error,
    }


def run_grounding_eval(service: DiagramDebateService, tsd_doc, gt_data: dict, images_dir: str | None = None) -> list[dict]:
    """Runs debates on real, labeled diagrams and records grounding metrics."""
    results = []

    for sample in load_labeled_samples(gt_data, labels=("met", "not_met")):
        diagram_id_tag = f"grounding_{sample.diagram_id}_{sample.requirement_id}"
        diagram = build_diagram_input(tsd_doc, sample.diagram_id, diagram_id_tag, blank_text=False)
        if diagram is None:
            logger.warning(f"  {diagram_id_tag}: diagram not found or invalid — skipping")
            continue
        requirement = SimpleNamespace(
            ordinal=1,
            stable_key=sample.requirement_id,
            requirement_text=sample.requirement_text,
            verification_hint=sample.verification_hint,
        )
        logger.info("Grounding eval: %s", diagram_id_tag)
        results.append(
            _run_debate_and_record(
                service, diagram, requirement, scope_label="architecture_relevant", images_dir=images_dir
            )
        )

    non_arch = diagrams_with_no_relevant_requirements(gt_data)
    if non_arch:
        for diagram_id, req_row in non_arch:
            diagram_id_tag = f"scope_test_{diagram_id}"
            diagram = build_diagram_input(tsd_doc, diagram_id, diagram_id_tag, blank_text=False)
            if diagram is None:
                logger.warning(f"  {diagram_id_tag}: diagram not found or invalid — skipping")
                continue
            requirement = SimpleNamespace(
                ordinal=1,
                stable_key=str(req_row.get("requirement_id", "")),
                requirement_text=req_row.get("requirement_text") or "",
                verification_hint=req_row.get("verification_hint") or "",
            )
            logger.info("Scope eval (real non-architecture diagram): %s", diagram_id_tag)
            results.append(
                _run_debate_and_record(
                    service, diagram, requirement, scope_label="non_architecture",
                    images_dir=images_dir, control_tag="scope_test",
                )
            )
    else:
        logger.warning(
            "No real diagram in the ground truth is labeled fully non-architecture "
            "(all candidate_requirements relevant:false) — falling back to one "
            "synthetic blank-image scope-negative scenario."
        )
        diagram = DiagramInput(
            diagram_id="scope_test_synthetic_blank",
            image_b64=_blank_white_image_b64(),
            page_number=1,
            caption=_FALLBACK_NON_ARCH_CAPTION,
            surrounding_text="",
        )
        logger.info("Scope eval (synthetic fallback): %s", diagram.diagram_id)
        row = _run_debate_and_record(
            service, diagram, _FALLBACK_REQUIREMENT, scope_label="non_architecture",
            images_dir=images_dir, control_tag="scope_test",
        )
        row["synthetic_fallback"] = True
        results.append(row)

    return results


def compute_summary(results: list[dict]) -> dict:
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return {"error": "No valid results"}

    total_assessments = sum(r.get("total_assessments", 0) for r in valid)
    assessments_with_marker = sum(r.get("assessments_with_marker", 0) for r in valid)
    marker_utilization = assessments_with_marker / total_assessments if total_assessments else 0.0

    runs_with_cited = sum(1 for r in valid if r.get("marker_ids_cited"))
    structured_utilization = runs_with_cited / len(valid) if valid else 0.0

    scope_results = [r for r in valid if "scope_label" in r]
    scope_accuracy = sum(1 for r in scope_results if r.get("scope_correct")) / len(scope_results) if scope_results else 0.0

    critic_results = [r for r in valid if r.get("critic_outcome")]
    overturn_rate = sum(1 for r in critic_results if r.get("critic_outcome") == "overturn") / len(critic_results) if critic_results else 0.0

    total_invalid = sum(len(r.get("invalid_marker_citations") or []) for r in valid)
    total_cited_markers = sum(len(r.get("marker_ids_cited") or []) for r in valid)
    invalid_citation_rate = total_invalid / total_cited_markers if total_cited_markers else 0.0

    calibration_buckets: dict[str, dict] = {}
    for r in valid:
        conf = r.get("confidence")
        verdict = r.get("final_verdict")
        scope_correct = r.get("scope_correct")
        if conf is None or verdict is None:
            continue
        bucket = f"{int(conf * 10) / 10:.1f}"
        if bucket not in calibration_buckets:
            calibration_buckets[bucket] = {"count": 0, "scope_correct": 0}
        calibration_buckets[bucket]["count"] += 1
        if scope_correct:
            calibration_buckets[bucket]["scope_correct"] += 1
    calibration = {
        bucket: {
            "count": v["count"],
            "scope_accuracy": v["scope_correct"] / v["count"] if v["count"] else 0.0,
        }
        for bucket, v in sorted(calibration_buckets.items())
    }

    return {
        "total_samples": len(valid),
        "marker_utilization_rate": round(marker_utilization, 3),
        "structured_marker_utilization_rate": round(structured_utilization, 3),
        "scope_accuracy": round(scope_accuracy, 3),
        "critic_overturn_rate": round(overturn_rate, 3),
        "invalid_marker_citation_rate": round(invalid_citation_rate, 3),
        "confidence_calibration": calibration,
        "thresholds": {
            "marker_utilization_above_0.7": marker_utilization >= 0.7,
            "scope_accuracy_above_0.8": scope_accuracy >= 0.8,
            "critic_overturn_rate_below_0.4": overturn_rate <= 0.4,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Vision grounding and marker utilization eval (real diagrams).")
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="Path to a labeled diagram ground-truth JSON (see build_diagram_ground_truth_template.py)"
    )
    parser.add_argument("--output", type=str, default="grounding_results.json")
    parser.add_argument(
        "--from-blindness-results",
        type=str,
        default=None,
        help="Re-use a previous blindness_eval JSON output instead of running live LLM calls.",
    )
    args = parser.parse_args()

    images_dir = results_path("images", subdir="vision")
    os.makedirs(images_dir, exist_ok=True)

    if args.from_blindness_results:
        with open(args.from_blindness_results) as f:
            blindness_data = json.load(f)
        results = blindness_data.get("results", [])
        logger.info("Loaded %d results from %s (marker stats not available in this mode)", len(results), args.from_blindness_results)
        summary = {
            "source": args.from_blindness_results,
            "critic_overturn_rate": sum(1 for r in results if r.get("critic_outcome") == "overturn") / len(results) if results else 0.0,
            "accuracy": blindness_data.get("accuracy"),
            "precision": blindness_data.get("precision"),
            "recall": blindness_data.get("recall"),
            "f1": blindness_data.get("f1"),
        }
    else:
        if not args.ground_truth:
            logger.error("--ground-truth is required unless --from-blindness-results is used.")
            return
        if not os.path.exists(args.ground_truth):
            logger.error(f"Ground truth file not found: {args.ground_truth}")
            return

        gt_data = load_ground_truth(args.ground_truth)
        design_id = gt_data.get("design_id")
        if design_id is None:
            logger.error("Ground truth file is missing 'design_id' — regenerate it with build_diagram_ground_truth_template.py.")
            return

        tsd_doc = load_tsd_document(design_id)
        service = DiagramDebateService()
        results = run_grounding_eval(service, tsd_doc, gt_data, images_dir=images_dir)
        summary = compute_summary(results)
        summary["design_id"] = design_id
        summary["results"] = results

    output_path = results_path(args.output, subdir="vision")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Grounding eval complete.")
    logger.info("Marker utilization rate:     %.2f", summary.get("marker_utilization_rate", 0))
    logger.info("Scope accuracy:              %.2f", summary.get("scope_accuracy", 0))
    logger.info("Critic overturn rate:        %.2f", summary.get("critic_overturn_rate", 0))
    logger.info("Invalid marker citation rate:%.2f", summary.get("invalid_marker_citation_rate", 0))
    logger.info("Marked images saved to %s/", images_dir)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
