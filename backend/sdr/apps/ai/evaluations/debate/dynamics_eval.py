"""
Exp 8 — Debate Dynamics Analysis (Critic Effectiveness).

Proves the Critic is not a rubber stamp: it actively challenges Hunter claims,
revises verdicts, and filters hallucinated citations.

Fully automated — reads from existing analysis_trace JSONB data, no LLM calls,
no manual ground truth needed.

Metrics:
  critic_intervention_rate  — % of debates where Critic outcome ≠ UPHOLD
  verdict_revision_rate     — % where Critic revised_verdict ≠ Hunter initial verdict
  citation_rejection_rate   — avg invalid_citation_ids per debate / total Hunter citations
  escalation_rate           — % of debates that ran >1 round
  confidence_delta          — Mediator final confidence − Hunter initial confidence

Usage:
    python debate_dynamics_eval.py --review-id 1
    python debate_dynamics_eval.py --review-id 1 --output debate_dynamics.json
"""
import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from statistics import mean, stdev

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import sdr.apps.standards.models  # noqa: F401 — resolve SQLAlchemy FK
import sdr.apps.designs.models    # noqa: F401
import sdr.apps.reviews.models.finding  # noqa: F401
import sdr.apps.reviews.models.review   # noqa: F401

from sdr.core.database import SessionLocal
from sdr.apps.reviews.models.finding import Finding

from sdr.apps.ai.evaluations.shared import results_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _extract_debate_stats(finding: Finding) -> dict | None:
    """
    Extract per-finding debate statistics from analysis_trace.
    Returns None if no valid debate history is found.
    """
    meta = finding.requirement_metadata or {}
    if not isinstance(meta, dict):
        return None
    trace = meta.get("analysis_trace", {})
    if not isinstance(trace, dict):
        return None
    history = trace.get("debate_history", [])
    if not history or not isinstance(history, list):
        return None

    round0 = history[0]
    if not isinstance(round0, dict):
        return None

    hunter = round0.get("hunter", {}) or {}
    critic = round0.get("critic", {}) or {}

    hunter_verdict = (hunter.get("verdict") or "").lower().strip()
    hunter_conf = hunter.get("confidence") or 0.0
    hunter_citations = hunter.get("citation_ids") or []

    critic_outcome = (critic.get("outcome") or "").upper().strip()
    critic_revised = (critic.get("revised_verdict") or "").lower().strip()
    invalid_citations = critic.get("invalid_citation_ids") or []

    final_verdict = (finding.met_status or "").lower().strip()
    final_conf = finding.confidence_score or 0.0

    num_rounds = len(history)
    multi_round = num_rounds > 1

    # Did the Critic actually intervene (not just rubber-stamp)?
    critic_intervened = critic_outcome in ("OVERTURN", "PARTIAL")

    # Did the Critic's revised verdict differ from the Hunter's initial verdict?
    verdict_revised = bool(critic_revised) and critic_revised != hunter_verdict

    # Verdict changed end-to-end (Hunter initial → Mediator final)
    verdict_changed_final = bool(hunter_verdict) and bool(final_verdict) and hunter_verdict != final_verdict

    total_hunter_citations = len(hunter_citations)
    invalid_count = len(invalid_citations)
    citation_rejection_ratio = invalid_count / total_hunter_citations if total_hunter_citations else 0.0

    confidence_delta = final_conf - hunter_conf

    return {
        "finding_id": finding.id,
        "requirement_reference": finding.requirement_reference,
        "finding_title": finding.title[:80] if finding.title else "",
        "hunter_verdict": hunter_verdict,
        "hunter_confidence": hunter_conf,
        "hunter_citation_count": total_hunter_citations,
        "critic_outcome": critic_outcome,
        "critic_revised_verdict": critic_revised,
        "invalid_citation_count": invalid_count,
        "citation_rejection_ratio": round(citation_rejection_ratio, 4),
        "final_verdict": final_verdict,
        "final_confidence": final_conf,
        "num_rounds": num_rounds,
        "multi_round": multi_round,
        "critic_intervened": critic_intervened,
        "verdict_revised_by_critic": verdict_revised,
        "verdict_changed_final": verdict_changed_final,
        "confidence_delta": round(confidence_delta, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exp 8: Debate dynamics — Critic intervention rate, verdict revision, citation rejection."
    )
    parser.add_argument("--review-id", type=int, required=True)
    parser.add_argument("--output", type=str, default="debate_dynamics.json")
    args = parser.parse_args()

    with SessionLocal() as db:
        findings = db.query(Finding).filter_by(review_id=args.review_id).all()
    logger.info(f"Loaded {len(findings)} findings for review_id={args.review_id}")

    if not findings:
        logger.error("No findings found. Run a security design review first.")
        return

    per_finding = []
    skipped = 0

    for f in findings:
        stats = _extract_debate_stats(f)
        if stats is None:
            skipped += 1
            continue
        per_finding.append(stats)

    logger.info(f"Parsed {len(per_finding)} findings with debate traces ({skipped} skipped — no trace)")

    if not per_finding:
        logger.error("No debate traces found in finding metadata.")
        return

    n = len(per_finding)

    # Core metrics
    interventions = [s for s in per_finding if s["critic_intervened"]]
    revisions = [s for s in per_finding if s["verdict_revised_by_critic"]]
    multi_round = [s for s in per_finding if s["multi_round"]]
    verdict_changes = [s for s in per_finding if s["verdict_changed_final"]]

    critic_intervention_rate = round(len(interventions) / n, 4)
    verdict_revision_rate = round(len(revisions) / n, 4)
    escalation_rate = round(len(multi_round) / n, 4)
    final_change_rate = round(len(verdict_changes) / n, 4)

    rejection_ratios = [s["citation_rejection_ratio"] for s in per_finding]
    avg_citation_rejection_rate = round(mean(rejection_ratios), 4) if rejection_ratios else 0.0

    conf_deltas = [s["confidence_delta"] for s in per_finding]
    avg_confidence_delta = round(mean(conf_deltas), 4) if conf_deltas else 0.0
    std_confidence_delta = round(stdev(conf_deltas), 4) if len(conf_deltas) > 1 else 0.0

    # Critic outcome distribution
    outcome_counts = Counter(s["critic_outcome"] for s in per_finding)

    # Round distribution
    round_counts = Counter(s["num_rounds"] for s in per_finding)

    # Verdict change breakdown: cases where verdict flipped (hunter → mediator)
    flip_cases = [s for s in per_finding if s["verdict_changed_final"]]
    flip_directions: Counter = Counter()
    for s in flip_cases:
        flip_directions[f"{s['hunter_verdict']} → {s['final_verdict']}"] += 1

    # Most impactful Critic interventions (verdict changed AND Critic intervened)
    impactful = sorted(
        [s for s in per_finding if s["critic_intervened"] and s["verdict_changed_final"]],
        key=lambda x: abs(x["confidence_delta"]),
        reverse=True,
    )[:10]

    summary = {
        "review_id": args.review_id,
        "total_findings": len(findings),
        "findings_with_trace": n,
        "critic_intervention_rate": critic_intervention_rate,
        "verdict_revision_rate": verdict_revision_rate,
        "avg_citation_rejection_rate": avg_citation_rejection_rate,
        "escalation_rate": escalation_rate,
        "final_verdict_change_rate": final_change_rate,
        "avg_confidence_delta": avg_confidence_delta,
        "std_confidence_delta": std_confidence_delta,
        "thresholds": {
            "critic_intervention_rate_gt_0.30": critic_intervention_rate > 0.30,
            "verdict_revision_rate_gt_0.10": verdict_revision_rate > 0.10,
            "escalation_rate_gt_0": escalation_rate > 0,
        },
        "critic_outcome_distribution": dict(outcome_counts),
        "rounds_distribution": {str(k): v for k, v in sorted(round_counts.items())},
        "verdict_flip_directions": dict(flip_directions),
        "most_impactful_interventions": impactful,
        "per_finding_stats": per_finding,
    }

    output_path = results_path(args.output, subdir="debate")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n=== Exp 8: Debate Dynamics Results ===")
    logger.info(f"  Findings with trace:        {n} / {len(findings)}")
    logger.info(f"  Critic intervention rate:   {critic_intervention_rate:.4f}  "
                f"(threshold >0.30: {summary['thresholds']['critic_intervention_rate_gt_0.30']})")
    logger.info(f"  Verdict revision rate:      {verdict_revision_rate:.4f}  "
                f"(Critic changed verdict; threshold uses final_change_rate={final_change_rate:.4f}: "
                f"{summary['thresholds']['verdict_revision_rate_gt_0.10']})")
    logger.info(f"  Avg citation rejection rate:{avg_citation_rejection_rate:.4f}  "
                f"(invalid citations / hunter citations per debate)")
    logger.info(f"  Escalation rate:            {escalation_rate:.4f}  "
                f"(debates running >1 round)")
    logger.info(f"  Final verdict change rate:  {final_change_rate:.4f}  "
                f"(Hunter initial ≠ Mediator final)")
    logger.info(f"  Avg confidence delta:       {avg_confidence_delta:+.4f} ± {std_confidence_delta:.4f}  "
                f"(Mediator − Hunter confidence)")
    logger.info(f"\n  Critic outcome distribution: {dict(outcome_counts)}")
    logger.info(f"  Rounds distribution:         {dict(sorted(round_counts.items()))}")
    logger.info(f"  Verdict flip directions:     {dict(flip_directions)}")
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
