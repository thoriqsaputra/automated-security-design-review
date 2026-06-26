"""
Exp 5: GraphRAG-Enhanced Vision Context eval.

Compares two tsd_context conditions for the Vision Agent debate:
  - Baseline:       full_text[:3000] (current flat-slice behaviour)
  - Graph-enhanced: GraphSearcher.search(caption+surrounding_text) formatted as
                    component-relationship text specific to the diagram

For each real diagram in Design 5's TSD (via DesignPreparationStore), both
conditions are run against the live V4 diagram requirements. Synthetic ambiguous
diagrams are added when the real set is small (<4 eligible), to ensure coverage
of the edge case where the graph context carries security evidence not visible
in the image.

Metrics per condition:
  - faithfulness_deterministic: fraction of Hunter quotes that appear verbatim
    in the tsd_context provided — measures grounding in the specific context given
  - verdict_consistency: fraction of diagrams where both conditions agree on
    final_verdict — disagreements mark cases where context materially changes output
  - architecture_relevant_rate: fraction of diagrams classified as in-scope
"""
import argparse
import base64
import io
import json
import logging
import os
import pickle
import re
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from PIL import Image, ImageDraw, ImageFont

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.standards.models import StandardIngestionJob
from sdr.apps.ai.agents.vision import DiagramInput
from sdr.apps.ai.engine.debate.diagram_debate_service import DiagramDebateService
from sdr.apps.ai.retrieval.searchers.graph import GraphSearcher
from sdr.apps.ai.evaluations.runner import _extract_answer_quotes
from sdr.apps.ai.evaluations.judges import judge_faithfulness_deterministic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Graph context builder (mirrors DiagramAnalysisCoordinator._build_graph_context)
# --------------------------------------------------------------------------- #

def build_graph_context(diagram: DiagramInput, graph, top_k: int = 5) -> str:
    query = " ".join(filter(None, [diagram.caption, diagram.surrounding_text])).strip()
    if not query or graph is None:
        return ""
    try:
        response = GraphSearcher().search(parameter_text=query, graph=graph)
        if response.error or response.is_empty:
            return ""
        lines = ["Component relationships relevant to this diagram:"]
        for result in response.results[:top_k]:
            entity_line = f"- {result.entity_name} ({result.entity_type})"
            if result.entity.description:
                entity_line += f": {result.entity.description}"
            lines.append(entity_line)
            for rel in result.relevant_relations[:3]:
                rel_line = f"    → {rel.relation_type} → {rel.target_entity_id}"
                parts = []
                if rel.protocol:
                    parts.append(f"protocol={rel.protocol}")
                if rel.is_encrypted is not None:
                    parts.append(f"encrypted={'yes' if rel.is_encrypted else 'no'}")
                if rel.requires_auth is not None:
                    parts.append(f"auth={'yes' if rel.requires_auth else 'no'}")
                if rel.description:
                    parts.append(rel.description)
                if parts:
                    rel_line += f" [{', '.join(parts)}]"
                lines.append(rel_line)
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("build_graph_context failed: %s", exc)
        return ""


# --------------------------------------------------------------------------- #
# Synthetic diagram helpers (reused from vision_blindness_eval.py)
# --------------------------------------------------------------------------- #

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


def _draw_arrow(draw, x1, x2, y_base, font, label=None):
    mid = y_base + BOX_H / 2
    draw.line([x1, mid, x2, mid], fill="black", width=2)
    draw.polygon([(x2, mid - 6), (x2, mid + 6), (x2 + 10, mid)], fill="black")
    if label:
        lx = (x1 + x2) / 2
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((lx - tw / 2, mid - 18), label, fill="#cc0000", font=font)


def _draw_network_diagram(stages, arrow_labels=None):
    """arrow_labels: dict mapping (src_idx, dst_idx) → label string."""
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
        arrow_label = (arrow_labels or {}).get(i)
        _draw_arrow(draw, positions[i] + BOX_W, positions[i + 1], Y, font, label=arrow_label)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Synthetic scenarios: images have protocol labels ON the arrows so the Hunter
# has visual evidence to cite. Graph context is consistent with those labels and
# adds semantic enrichment (component descriptions, relation types). Baseline
# uses empty tsd_context — forces the Hunter to work from image alone.
# This isolates graph context's contribution to grounding quality.
SYNTHETIC_SCENARIOS = [
    {
        "diagram_id": "syn_http_unencrypted",
        "stages": ["Client App", "API Gateway", "Backend"],
        # Arrow 0: Client App→API Gateway labeled "HTTP"; Arrow 1: API Gateway→Backend labeled "HTTP"
        "arrow_labels": {0: "HTTP", 1: "HTTP"},
        "caption": "Figure 1: API Communication Flow",
        "surrounding_text": "",
        "baseline_context": "",
        "graph_context": (
            "Component relationships relevant to this diagram:\n"
            "- API Gateway (service): Routes client requests to backend services\n"
            "    → communicates_with → Backend [protocol=HTTP, encrypted=no, auth=yes]\n"
            "- Client App (client): Mobile application\n"
            "    → connects_to → API Gateway [protocol=HTTP, encrypted=no]"
        ),
        "requirement": SimpleNamespace(
            ordinal=1,
            stable_key="SYN-TLS-1",
            requirement_text="Verify that all communication between components uses encrypted channels (TLS/HTTPS). Unencrypted HTTP must not be used for any data in transit.",
            verification_hint="Look for protocol labels on arrows. HTTP (non-TLS) connections are a finding.",
        ),
        "expected_verdict": "not_met",
    },
    {
        "diagram_id": "syn_auth_missing",
        "stages": ["Mobile App", "User Service", "Database"],
        # Arrow 0: Mobile→UserService labeled "HTTPS no-auth"; Arrow 1: UserService→Database labeled "TCP no-auth"
        "arrow_labels": {0: "HTTPS no-auth", 1: "TCP no-auth"},
        "caption": "Figure 2: User Data Flow",
        "surrounding_text": "",
        "baseline_context": "",
        "graph_context": (
            "Component relationships relevant to this diagram:\n"
            "- User Service (service): Handles user account operations\n"
            "    → reads_from → Database [protocol=TCP, encrypted=yes, auth=no]\n"
            "    → writes_to → Database [protocol=TCP, encrypted=yes, auth=no]\n"
            "- Mobile App (client): End-user application\n"
            "    → connects_to → User Service [protocol=HTTPS, encrypted=yes, auth=no]"
        ),
        "requirement": SimpleNamespace(
            ordinal=2,
            stable_key="SYN-AUTH-1",
            requirement_text="Verify that all inter-service connections require mutual authentication. Unauthenticated service-to-service calls must not be permitted.",
            verification_hint="Look for auth labels on arrows. 'no-auth' connections are a finding.",
        ),
        "expected_verdict": "not_met",
    },
    {
        "diagram_id": "syn_fully_secured",
        "stages": ["Client", "TLS Proxy", "App Server", "DB"],
        # All arrows labeled HTTPS/TLS
        "arrow_labels": {0: "HTTPS", 1: "TLS 1.3", 2: "TLS 1.2"},
        "caption": "Figure 3: Secured Service Topology",
        "surrounding_text": "",
        "baseline_context": "",
        "graph_context": (
            "Component relationships relevant to this diagram:\n"
            "- TLS Proxy (service): Terminates inbound TLS, forwards to app tier\n"
            "    → proxies_to → App Server [protocol=TLS 1.3, encrypted=yes, auth=yes]\n"
            "- App Server (service): Core application logic\n"
            "    → reads_from → DB [protocol=TLS 1.2, encrypted=yes, auth=yes]\n"
            "- Client (client): External end user\n"
            "    → connects_to → TLS Proxy [protocol=HTTPS, encrypted=yes]"
        ),
        "requirement": SimpleNamespace(
            ordinal=3,
            stable_key="SYN-TLS-2",
            requirement_text="Verify that all communication between components uses encrypted channels (TLS/HTTPS). Unencrypted HTTP must not be used for any data in transit.",
            verification_hint="Look for protocol labels on arrows. HTTPS/TLS on all connections means the requirement is met.",
        ),
        "expected_verdict": "met",
    },
]


# --------------------------------------------------------------------------- #
# Per-diagram evaluation
# --------------------------------------------------------------------------- #

def _run_one(service, diagram, requirements, context_label, tsd_context):
    output = service.run_diagram_debate(
        diagram=diagram, requirements=requirements, tsd_context=tsd_context
    )
    mediator = output.mediator_result or {}
    hunter = output.hunter_result or {}
    verdict = mediator.get("final_verdict")
    hunter_answer = hunter.get("evidence_description") or hunter.get("hunter_answer") or ""
    quotes = _extract_answer_quotes(hunter_answer)

    # Faithfulness deterministic: were quotes grounded in the provided context?
    context_blocks = {"ctx": tsd_context} if tsd_context else {}
    faith_det = judge_faithfulness_deterministic(quotes, context_blocks)

    return {
        "context_label": context_label,
        "verdict": verdict,
        "confidence": mediator.get("confidence"),
        "diagram_scope_verdict": mediator.get("diagram_scope_verdict"),
        "faithfulness_deterministic": faith_det,
        "hunter_answer": hunter_answer,
        "error": output.error,
    }


def main():
    parser = argparse.ArgumentParser(description="GraphRAG-Enhanced Vision Context eval (Exp 5).")
    parser.add_argument("--design-id", type=int, required=True)
    parser.add_argument("--output", type=str, default="eval_vision_graph_context.json")
    parser.add_argument("--raptor-tree-pickle", type=str, default=None)
    parser.add_argument("--graph-top-k", type=int, default=5)
    args = parser.parse_args()

    service = DiagramDebateService()
    results = []

    with SessionLocal() as db:
        design = db.query(Design).filter(Design.id == args.design_id).first()
        if not design:
            logger.error(f"Design {args.design_id} not found.")
            return

        store = DesignPreparationStore()
        prep, tsd_doc, indexes = store.load_prepared_assets(db, design)
        tsd_graph = getattr(indexes, "tsd_graph", None)
        fallback_context = tsd_doc.full_text[:3000] if hasattr(tsd_doc, "full_text") else ""

        # Load real diagram requirements from V4 category
        from sdr.apps.standards.models import CategoryDiagramRequirement, StandardIngestionJob
        active_job = (
            db.query(StandardIngestionJob)
            .filter_by(is_active=True)
            .order_by(StandardIngestionJob.created_at.desc())
            .first()
        )
        real_requirements = []
        if active_job:
            real_requirements = (
                db.query(CategoryDiagramRequirement)
                .filter_by(ingestion_job_id=active_job.id)
                .limit(10)
                .all()
            )
            logger.info(f"Loaded {len(real_requirements)} real diagram requirements (job_id={active_job.id})")

        # --- Real TSD diagrams ---
        all_diagrams = getattr(tsd_doc, "all_diagrams", []) or []
        eligible_real = []
        for dblock in all_diagrams:
            dblock.ensure_image_loaded(512)
            if dblock.is_valid():
                eligible_real.append(dblock)

        logger.info(f"Real eligible diagrams: {len(eligible_real)}")

        for i, dblock in enumerate(eligible_real):
            diagram = DiagramInput(
                diagram_id=dblock.diagram_id,
                image_b64=dblock.image_b64,
                page_number=dblock.page_number,
                caption=dblock.caption,
                surrounding_text=dblock.surrounding_text,
                image_format=getattr(dblock, "image_format", "png"),
            )
            graph_ctx = build_graph_context(diagram, tsd_graph, top_k=args.graph_top_k)
            reqs = real_requirements if real_requirements else []
            if not reqs:
                logger.warning(f"No requirements for diagram {diagram.diagram_id} — skipping")
                continue

            logger.info(f"[Real {i+1}/{len(eligible_real)}] {diagram.diagram_id}")
            baseline = _run_one(service, diagram, reqs, "baseline", fallback_context)
            enhanced = _run_one(service, diagram, reqs, "graph_enhanced", graph_ctx)

            results.append({
                "type": "real",
                "diagram_id": diagram.diagram_id,
                "page_number": diagram.page_number,
                "baseline": baseline,
                "graph_enhanced": enhanced,
                "verdict_agreement": baseline["verdict"] == enhanced["verdict"],
                "graph_context_chars": len(graph_ctx),
            })

    # --- Synthetic diagrams with labeled arrows (always run, no DB needed) ---
    for scenario in SYNTHETIC_SCENARIOS:
        image_b64 = _draw_network_diagram(scenario["stages"], arrow_labels=scenario.get("arrow_labels"))
        diagram = DiagramInput(
            diagram_id=scenario["diagram_id"],
            image_b64=image_b64,
            page_number=1,
            caption=scenario["caption"],
            surrounding_text=scenario["surrounding_text"],
        )
        reqs = [scenario["requirement"]]
        logger.info(f"[Synthetic] {scenario['diagram_id']} (expected={scenario['expected_verdict']})")
        baseline = _run_one(service, diagram, reqs, "baseline", scenario["baseline_context"])
        enhanced = _run_one(service, diagram, reqs, "graph_enhanced", scenario["graph_context"])
        results.append({
            "type": "synthetic",
            "diagram_id": scenario["diagram_id"],
            "expected_verdict": scenario["expected_verdict"],
            "baseline": baseline,
            "graph_enhanced": enhanced,
            "verdict_agreement": baseline["verdict"] == enhanced["verdict"],
            "graph_context_chars": len(scenario["graph_context"]),
        })

    # --- Aggregate ---
    def _agg(cond_key):
        cond_results = [r[cond_key] for r in results if cond_key in r]
        if not cond_results:
            return {}
        n = len(cond_results)
        arch_relevant = sum(1 for r in cond_results if r.get("diagram_scope_verdict") == "architecture_relevant")
        faithfulness_vals = [r["faithfulness_deterministic"] for r in cond_results]
        return {
            "count": n,
            "faithfulness_deterministic": round(sum(faithfulness_vals) / n, 4),
            "architecture_relevant_rate": round(arch_relevant / n, 4),
        }

    verdict_agreement_rate = sum(1 for r in results if r.get("verdict_agreement")) / len(results) if results else 0

    # For synthetic: check how often each condition matches the expected verdict
    synthetic = [r for r in results if r["type"] == "synthetic" and "expected_verdict" in r]
    baseline_accuracy_on_synthetic = (
        sum(1 for r in synthetic if r["baseline"]["verdict"] == r["expected_verdict"])
        / len(synthetic) if synthetic else None
    )
    graph_accuracy_on_synthetic = (
        sum(1 for r in synthetic if r["graph_enhanced"]["verdict"] == r["expected_verdict"])
        / len(synthetic) if synthetic else None
    )

    summary = {
        "total_diagrams": len(results),
        "real_diagrams": sum(1 for r in results if r["type"] == "real"),
        "synthetic_diagrams": sum(1 for r in results if r["type"] == "synthetic"),
        "verdict_agreement_rate": round(verdict_agreement_rate, 4),
        "baseline_accuracy_on_synthetic": baseline_accuracy_on_synthetic,
        "graph_accuracy_on_synthetic": graph_accuracy_on_synthetic,
        "baseline": _agg("baseline"),
        "graph_enhanced": _agg("graph_enhanced"),
        "results": results,
    }
    summary["delta_graph_minus_baseline"] = {
        m: round(summary["graph_enhanced"].get(m, 0) - summary["baseline"].get(m, 0), 4)
        for m in ("faithfulness_deterministic", "architecture_relevant_rate")
        if m in summary["baseline"] and m in summary["graph_enhanced"]
    }

    output_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Vision graph context eval complete.")
    logger.info(f"  Baseline faithfulness_det:       {summary['baseline'].get('faithfulness_deterministic')}")
    logger.info(f"  Graph-enhanced faithfulness_det: {summary['graph_enhanced'].get('faithfulness_deterministic')}")
    logger.info(f"  Verdict agreement rate: {verdict_agreement_rate:.2f}")
    if baseline_accuracy_on_synthetic is not None:
        logger.info(f"  Baseline accuracy on synthetic: {baseline_accuracy_on_synthetic:.2f}")
    if graph_accuracy_on_synthetic is not None:
        logger.info(f"  Graph accuracy on synthetic:    {graph_accuracy_on_synthetic:.2f}")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
