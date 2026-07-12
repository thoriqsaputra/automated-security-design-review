from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# TSD Relevance Screening
# ---------------------------------------------------------------------------

TSD_SCREENING_SYSTEM_PROMPT = """\
You are a security document classifier. Your job is to determine whether
a provided document is a Technical Software Document (TSD) that describes
a software system's architecture, components, and design decisions.

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""


def build_tsd_screening_prompt(document_text_sample: str) -> str:
    return f"""\
## DOCUMENT SAMPLE (representative excerpts)

{document_text_sample}

## YOUR TASK

Determine whether the document above is a Technical Software Document (TSD)
describing a software system — including architecture diagrams, component
descriptions, API designs, data flows, infrastructure, or security controls.

Respond with a single JSON object:

{{
  "is_tsd": <true | false>,
  "confidence": <float 0.0–1.0>,
  "document_type": "<your best guess at the document type>",
  "reasoning": "<one sentence explanation>"
}}

Rules:
- is_tsd = true  → document describes a software system's design or architecture
- is_tsd = false → document is a contract, policy, standard, marketing material, etc.
- confidence     → your certainty (1.0 = certain, 0.5 = ambiguous)
- Do not classify a document as non-TSD only because the excerpt contains
  table-of-contents, revision-history, approval, glossary, or front-matter text.
- If any excerpt describes architecture, components, APIs, data flows,
  infrastructure, databases, security controls, authentication, authorization,
  or deployment of a software system, classify it as a TSD.
- If the evidence is mixed or ambiguous, prefer is_tsd = true with lower
  confidence instead of blocking analysis.
"""

# ---------------------------------------------------------------------------
# Severity Justification
# ---------------------------------------------------------------------------

SEVERITY_JUSTIFICATION_SYSTEM_PROMPT = """\
You are a security risk analyst justifying severity ratings for security
compliance findings. You produce structured, evidence-based severity
assessments aligned with CVSS-style risk reasoning.

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""


def build_severity_justification_prompt(
    parameter_text: str,
    parameter_section: str,
    mediator_reasoning: str,
    proposed_severity: str,
    tsd_context: Optional[str] = None,
) -> str:
    """
    Generates a detailed severity justification for a not_met Finding.

    Called optionally by TSDAnalysisOrchestrator._persist_finding()
    for CRITICAL and HIGH findings where a detailed severity_analysis
    JSON is needed — stored in Finding.severity_analysis [3].

    The Mediator already assigns a severity label in mediator.py, but
    this prompt produces the structured breakdown (impact, likelihood,
    affected components) that goes into Finding.severity_analysis.

    Args:
        parameter_text:     The full security requirement text.
        parameter_section:  The parent section title.
        mediator_reasoning: The Mediator's final verdict reasoning.
        proposed_severity:  The severity already assigned by the Mediator
                            ("critical" | "high" | "medium" | "low" | "info").
        tsd_context:        Optional relevant TSD excerpt for additional context.

    Returns:
        A fully formed prompt string.
    """
    context_section = (
        f"\n\n## RELEVANT TSD CONTEXT\n{tsd_context}"
        if tsd_context
        else ""
    )

    return f"""\
## SECURITY PARAMETER

Section:     {parameter_section}
Requirement: {parameter_text}

## MEDIATOR VERDICT

Severity:  {proposed_severity.upper()}
Reasoning: {mediator_reasoning}{context_section}

## YOUR TASK

Produce a structured severity justification for this not_met finding.

Respond with a single JSON object:

{{
  "severity": "{proposed_severity}",
  "severity_score": <float 0.0–10.0>,
  "impact": {{
    "confidentiality": "none" | "low" | "high",
    "integrity":       "none" | "low" | "high",
    "availability":    "none" | "low" | "high"
  }},
  "likelihood": "low" | "medium" | "high",
  "affected_components": ["<component name>"],
  "attack_vector": "network" | "adjacent" | "local" | "physical",
  "justification": "<two sentence evidence-based justification>"
}}

Rules:
- severity_score: 0.0–3.9 = low, 4.0–6.9 = medium, 7.0–8.9 = high, 9.0–10.0 = critical
- affected_components: list the specific TSD components that are at risk
- justification: reference the specific gap identified by the Mediator
"""


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # System prompts
    "TSD_SCREENING_SYSTEM_PROMPT",
    "SEVERITY_JUSTIFICATION_SYSTEM_PROMPT",
    # Prompt builders
    "build_tsd_screening_prompt",
    "build_severity_justification_prompt",
]

