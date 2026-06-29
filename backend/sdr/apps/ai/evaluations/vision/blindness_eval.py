"""
Visual Blindness Mitigation eval.

Tests whether the Vision Agent (Hunter -> Critic -> Mediator) can detect a
security control's absence purely from a synthetic architecture diagram
image, with zero textual description anywhere (caption/surrounding_text/
tsd_context are all either empty or deliberately silent about the control
under test). This isolates the vision channel from the text channel.

For each control type, two diagrams are generated: one with the control
box/label present, one with it omitted. Ground truth: present -> "met",
omitted -> "not_met". Runs the real Hunter/Critic/Mediator debate (no
mocked LLM calls) via DiagramDebateService.run_diagram_debate.
"""
import argparse
import base64
import io
import json
import logging
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from PIL import Image, ImageDraw, ImageFont

from sdr.apps.ai.agents.vision import DiagramInput
from sdr.apps.ai.engine.debate.diagram_debate_service import DiagramDebateService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANVAS_SIZE = (900, 300)
BOX_W, BOX_H = 150, 80
GAP = 40
START_X, Y = 30, 110


def _font():
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
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


def _draw_network_diagram(stages, omit=None):
    """stages: list of box labels left-to-right. omit: label to drop (vulnerable variant)."""
    img = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(img)
    font = _font()
    visible_stages = [s for s in stages if s != omit]

    x = START_X
    positions = []
    for label in visible_stages:
        positions.append(x)
        _draw_box(draw, x, Y, label, font)
        x += BOX_W + GAP

    for i in range(len(positions) - 1):
        _draw_arrow(draw, positions[i] + BOX_W, positions[i + 1], Y)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


CONTROL_SCENARIOS = [
    {
        "control": "WAF",
        "stages": ["Internet", "WAF", "Load Balancer", "App Server", "Database"],
        "requirement_text": "Verify that internet-facing traffic passes through a Web Application Firewall (WAF) before reaching application servers.",
        "verification_hint": "Look for a box labeled 'WAF' between the Internet/ingress box and the Load Balancer/App Server box.",
    },
    {
        "control": "MFA Gateway",
        "stages": ["User", "MFA Gateway", "Auth Service", "App Server"],
        "requirement_text": "Verify that user authentication is routed through a multi-factor authentication (MFA) gateway before reaching the auth service.",
        "verification_hint": "Look for a box labeled 'MFA Gateway' between the User box and the Auth Service box.",
    },
    {
        "control": "TLS Proxy",
        "stages": ["Client", "TLS Proxy", "API Gateway", "Backend Service"],
        "requirement_text": "Verify that client connections are terminated/encrypted at a TLS proxy before reaching the API gateway.",
        "verification_hint": "Look for a box labeled 'TLS Proxy' between the Client box and the API Gateway box.",
    },
    {
        "control": "DMZ Boundary",
        "stages": ["Internet", "DMZ Boundary", "Internal Network", "Database"],
        "requirement_text": "Verify that a DMZ network-segmentation boundary separates internet-facing components from the internal network.",
        "verification_hint": "Look for a box labeled 'DMZ Boundary' between the Internet box and the Internal Network box.",
    },
]


def _build_requirement(scenario):
    return SimpleNamespace(
        ordinal=1,
        stable_key=f"D-{scenario['control'].upper().replace(' ', '-')}-1",
        requirement_text=scenario["requirement_text"],
        verification_hint=scenario["verification_hint"],
    )


def main():
    parser = argparse.ArgumentParser(description="Visual blindness mitigation eval for the Vision Agent.")
    parser.add_argument("--output", type=str, default="eval_vision_blindness.json")
    parser.add_argument("--save-images-dir", type=str, default=None, help="Optional dir to dump generated PNGs for manual inspection.")
    args = parser.parse_args()

    service = DiagramDebateService()
    samples = []
    for scenario in CONTROL_SCENARIOS:
        for condition, omit in (("present", None), ("absent", scenario["control"])):
            samples.append((scenario, condition, omit))

    results = []
    for i, (scenario, condition, omit) in enumerate(samples):
        image_b64 = _draw_network_diagram(scenario["stages"], omit=omit)
        diagram_id = f"vb_{scenario['control'].replace(' ', '_')}_{condition}"

        if args.save_images_dir:
            os.makedirs(args.save_images_dir, exist_ok=True)
            with open(os.path.join(args.save_images_dir, f"{diagram_id}.png"), "wb") as f:
                f.write(base64.b64decode(image_b64))

        diagram = DiagramInput(
            diagram_id=diagram_id,
            image_b64=image_b64,
            page_number=1,
            caption="Figure 1: System Architecture",
            surrounding_text="",
        )
        requirement = _build_requirement(scenario)

        logger.info(f"[{i + 1}/{len(samples)}] {diagram_id} (expecting {'met' if omit is None else 'not_met'})")
        output = service.run_diagram_debate(diagram=diagram, requirements=[requirement], tsd_context="")

        expected_verdict = "met" if omit is None else "not_met"
        mediator = output.mediator_result or {}
        actual_verdict = mediator.get("final_verdict")
        correct = actual_verdict == expected_verdict

        results.append({
            "control": scenario["control"],
            "condition": condition,
            "diagram_id": diagram_id,
            "expected_verdict": expected_verdict,
            "actual_verdict": actual_verdict,
            "correct": correct,
            "confidence": mediator.get("confidence"),
            "diagram_scope_verdict": mediator.get("diagram_scope_verdict"),
            "finding_description": mediator.get("finding_description"),
            "error": output.error,
            "hunter_verdict": (output.hunter_result or {}).get("overall_verdict"),
            "critic_outcome": (output.critic_result or {}).get("outcome"),
        })

    # Detection accuracy/precision/recall: positive class = correctly flagging
    # an absent control as not_met (the "blindness mitigation" signal).
    tp = sum(1 for r in results if r["condition"] == "absent" and r["actual_verdict"] == "not_met")
    fn = sum(1 for r in results if r["condition"] == "absent" and r["actual_verdict"] != "not_met")
    fp = sum(1 for r in results if r["condition"] == "present" and r["actual_verdict"] == "not_met")
    tn = sum(1 for r in results if r["condition"] == "present" and r["actual_verdict"] != "not_met")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = sum(1 for r in results if r["correct"]) / len(results) if results else 0.0
    confidences = [r["confidence"] for r in results if isinstance(r["confidence"], (int, float))]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    in_scope_rate = sum(
        1 for r in results if r["diagram_scope_verdict"] == "architecture_relevant"
    ) / len(results) if results else 0.0

    summary = {
        "total_samples": len(results),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_confidence": avg_confidence,
        "architecture_relevant_rate": in_scope_rate,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "results": results,
    }

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Vision blindness eval complete.")
    logger.info(f"Accuracy={accuracy:.2f} Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f}")
    logger.info(f"Architecture-relevant scope rate: {in_scope_rate:.2f}")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
