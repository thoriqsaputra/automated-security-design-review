"""
Visual Grounding & Marker Utilization Eval.

Measures whether the Vision Agent actually uses Set-of-Mark marker references
when citing visual evidence, how reliably the Critic catches bad citations, and
how well-calibrated the final confidence scores are.

Metrics:
  1. marker_utilization_rate  — fraction of Hunter assessments that cite [N]
  2. scope_accuracy           — diagram scope classification accuracy
  3. critic_overturn_rate     — how often Critic overturns Hunter
  4. invalid_marker_citation_rate — fraction of Hunter's cited markers that
                                    Critic flagged as invalid
  5. confidence_calibration   — per-bucket accuracy vs. confidence score

Run against the synthetic scenarios in blindness_eval.py by default (no DB
needed) or supply a JSON results file from a previous blindness_eval run.

Usage:
    python grounding_eval.py [--output grounding_results.json]
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

from PIL import Image, ImageDraw, ImageFont

from sdr.apps.ai.agents.vision import DiagramInput
from sdr.apps.ai.engine.debate.diagram_debate_service import DiagramDebateService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Regex that matches [N] marker references in free-text fields
_MARKER_RE = re.compile(r"\[\d+\]")

CANVAS_SIZE = (900, 300)
BOX_W, BOX_H = 150, 80
GAP = 40
START_X, Y = 30, 110

# Labeled scope dataset — these supplement the architecture scenarios with
# non-architecture examples to test scope classification accuracy.
SCOPE_SCENARIOS = [
    # Architecture-relevant diagrams (synthetic network/component diagrams)
    {
        "label": "architecture_relevant",
        "stages": ["Client", "API Gateway", "Auth Service", "Database"],
        "caption": "System architecture overview",
    },
    {
        "label": "architecture_relevant",
        "stages": ["Internet", "WAF", "Load Balancer", "App Server"],
        "caption": "Network ingress flow",
    },
    {
        "label": "architecture_relevant",
        "stages": ["User", "MFA Gateway", "Auth Service", "App Server"],
        "caption": "Authentication sequence",
    },
    # Non-architecture diagrams: blank white images (simulating screenshots/photos)
    # A fully white image with no structural boxes represents a "screenshot" or blank page.
    {
        "label": "non_architecture",
        "stages": [],
        "caption": "Screenshot of login form",
    },
]


def _font():
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, 14)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_box(draw, x, y, label, font):
    draw.rectangle([x, y, x + BOX_W, y + BOX_H], outline="black", width=2, fill="white")
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x + (BOX_W - tw) / 2, y + (BOX_H - th) / 2), label, fill="black", font=font)


def _draw_arrow(draw, x1, x2, y):
    mid = y + BOX_H / 2
    draw.line([x1, mid, x2, mid], fill="black", width=2)
    draw.polygon([(x2, mid - 6), (x2, mid + 6), (x2 + 10, mid)], fill="black")


def _generate_diagram_b64(stages: list[str]) -> str:
    img = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(img)
    font = _font()

    x = START_X
    positions = []
    for label in stages:
        positions.append(x)
        _draw_box(draw, x, Y, label, font)
        x += BOX_W + GAP
    for i in range(len(positions) - 1):
        _draw_arrow(draw, positions[i] + BOX_W, positions[i + 1], Y)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
    # Also check top-level fields
    top_level_cited = hunter.get("marker_ids_cited") or []
    return {
        "total_assessments": len(assessments),
        "assessments_with_marker": assessments_with_marker,
        "marker_ids_cited": top_level_cited,
        "utilization_rate": assessments_with_marker / len(assessments),
    }


def run_grounding_eval(service: DiagramDebateService) -> list[dict]:
    """Runs debates on architecture scenarios and records grounding metrics."""
    from sdr.apps.ai.evaluations.vision.blindness_eval import CONTROL_SCENARIOS

    results = []

    # Run on architecture scenarios (all stages present = "met" expected)
    for scenario in CONTROL_SCENARIOS:
        image_b64 = _generate_diagram_b64(scenario["stages"])
        diagram_id = f"grounding_{scenario['control'].replace(' ', '_')}"
        diagram = DiagramInput(
            diagram_id=diagram_id,
            image_b64=image_b64,
            page_number=1,
            caption=f"System architecture: {scenario['control']} present",
            surrounding_text="",
        )
        req = SimpleNamespace(
            ordinal=1,
            stable_key=f"D-{scenario['control'].upper().replace(' ', '-')}-1",
            requirement_text=scenario["requirement_text"],
            verification_hint=scenario["verification_hint"],
        )

        logger.info("Grounding eval: %s", diagram_id)
        output = service.run_diagram_debate(diagram=diagram, requirements=[req], tsd_context="")

        marker_stats = _compute_marker_utilization(output)
        critic = output.critic_result or {}
        mediator = output.mediator_result or {}

        results.append({
            "diagram_id": diagram_id,
            "control": scenario["control"],
            "scope_verdict": mediator.get("diagram_scope_verdict"),
            "scope_label": "architecture_relevant",
            "scope_correct": mediator.get("diagram_scope_verdict") == "architecture_relevant",
            "final_verdict": mediator.get("final_verdict"),
            "confidence": mediator.get("confidence"),
            "critic_outcome": critic.get("outcome"),
            "invalid_marker_citations": critic.get("invalid_marker_citations") or [],
            **marker_stats,
            "error": output.error,
        })

    # Run on scope scenarios (for scope classification accuracy only)
    for i, scope_s in enumerate(SCOPE_SCENARIOS):
        image_b64 = _generate_diagram_b64(scope_s["stages"])
        diagram_id = f"scope_test_{i}"
        diagram = DiagramInput(
            diagram_id=diagram_id,
            image_b64=image_b64,
            page_number=1,
            caption=scope_s["caption"],
            surrounding_text="",
        )
        # Use a generic requirement to force the scope classification step
        req = SimpleNamespace(
            ordinal=1,
            stable_key="D-SCOPE-1",
            requirement_text="Verify that authentication mechanisms are explicitly depicted in the architecture diagram.",
            verification_hint="Look for any authentication or auth service component.",
        )

        logger.info("Scope eval: %s (expected=%s)", diagram_id, scope_s["label"])
        output = service.run_diagram_debate(diagram=diagram, requirements=[req], tsd_context="")

        marker_stats = _compute_marker_utilization(output)
        critic = output.critic_result or {}
        mediator = output.mediator_result or {}

        results.append({
            "diagram_id": diagram_id,
            "control": "scope_test",
            "scope_verdict": mediator.get("diagram_scope_verdict"),
            "scope_label": scope_s["label"],
            "scope_correct": mediator.get("diagram_scope_verdict") == scope_s["label"],
            "final_verdict": mediator.get("final_verdict"),
            "confidence": mediator.get("confidence"),
            "critic_outcome": critic.get("outcome"),
            "invalid_marker_citations": critic.get("invalid_marker_citations") or [],
            **marker_stats,
            "error": output.error,
        })

    return results


def compute_summary(results: list[dict]) -> dict:
    valid = [r for r in results if not r.get("error")]
    if not valid:
        return {"error": "No valid results"}

    # Marker utilization
    total_assessments = sum(r.get("total_assessments", 0) for r in valid)
    assessments_with_marker = sum(r.get("assessments_with_marker", 0) for r in valid)
    marker_utilization = assessments_with_marker / total_assessments if total_assessments else 0.0

    # Structured cited marker rate (explicit marker_ids_cited field populated)
    runs_with_cited = sum(1 for r in valid if r.get("marker_ids_cited"))
    structured_utilization = runs_with_cited / len(valid) if valid else 0.0

    # Scope classification
    scope_results = [r for r in valid if "scope_label" in r]
    scope_accuracy = sum(1 for r in scope_results if r.get("scope_correct")) / len(scope_results) if scope_results else 0.0

    # Critic overturn rate
    critic_results = [r for r in valid if r.get("critic_outcome")]
    overturn_rate = sum(1 for r in critic_results if r.get("critic_outcome") == "overturn") / len(critic_results) if critic_results else 0.0

    # Invalid marker citation rate
    total_invalid = sum(len(r.get("invalid_marker_citations") or []) for r in valid)
    total_cited_markers = sum(len(r.get("marker_ids_cited") or []) for r in valid)
    invalid_citation_rate = total_invalid / total_cited_markers if total_cited_markers else 0.0

    # Confidence calibration (bucket accuracy)
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
    parser = argparse.ArgumentParser(description="Vision grounding and marker utilization eval.")
    parser.add_argument("--output", type=str, default="grounding_results.json")
    parser.add_argument(
        "--from-blindness-results",
        type=str,
        default=None,
        help="Re-use a previous blindness_eval JSON output instead of running live LLM calls.",
    )
    args = parser.parse_args()

    if args.from_blindness_results:
        with open(args.from_blindness_results) as f:
            blindness_data = json.load(f)
        # blindness_eval results don't include marker stats — report what we can
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
        service = DiagramDebateService()
        results = run_grounding_eval(service)
        summary = compute_summary(results)
        summary["results"] = results

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Grounding eval complete.")
    logger.info("Marker utilization rate:     %.2f", summary.get("marker_utilization_rate", 0))
    logger.info("Scope accuracy:              %.2f", summary.get("scope_accuracy", 0))
    logger.info("Critic overturn rate:        %.2f", summary.get("critic_overturn_rate", 0))
    logger.info("Invalid marker citation rate:%.2f", summary.get("invalid_marker_citation_rate", 0))
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
