# apps/ai/prompts/analysis_prompts.py

"""
Analysis prompts for the TSD security review pipeline.

These prompts are distinct from agent_prompts.py in scope:
    agent_prompts.py     → per-parameter debate prompts (Hunter/Critic/Mediator/Vision)
    analysis_prompts.py  → pre-analysis and post-analysis prompts that operate
                           at the document level, not the per-parameter level.

Currently houses:
    - TSD relevance screening prompt
    - Parameter applicability pre-filter prompt
    - Severity justification prompt
"""

from __future__ import annotations

from typing import List, Optional


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
    """
    Screens whether the uploaded document is actually a TSD before
    running the full expensive analysis pipeline.

    Called by TSDAnalysisOrchestrator._ingest_tsd() before committing
    to RAPTOR tree and GraphRAG index construction — if the document
    is not a TSD (e.g. a legal contract or a marketing PDF was uploaded
    by mistake), we fail fast with a clear error rather than wasting
    N×3 agent API calls.

    Args:
        document_text_sample: First ~3000 chars of the TSD document text
                              from TSDIngestor.ingest() [ingestor.py].

    Returns:
        A fully formed prompt string.
    """
    return f"""\
## DOCUMENT SAMPLE (first 3000 characters)

{document_text_sample[:3000]}

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
"""


# ---------------------------------------------------------------------------
# Parameter Applicability Pre-Filter
# ---------------------------------------------------------------------------

PARAMETER_APPLICABILITY_SYSTEM_PROMPT = """\
You are a security analyst pre-filtering security parameters before a full
compliance review. Your job is to quickly determine which parameters are
clearly not applicable to a given TSD based on a high-level document summary.

OUTPUT: Strict JSON only. No prose outside the JSON object.
"""


def build_parameter_applicability_prompt(
    document_summary: str,
    parameters: List[dict],
    category_code: str = "",
) -> str:
    """
    Pre-filters a batch of security parameters against a TSD summary
    to identify parameters that are clearly N/A before running the
    full Hunter/Critic/Mediator debate.

    This is an optional optimisation step called by
    TSDAnalysisOrchestrator.run() before the per-parameter loop.
    By identifying clearly out-of-scope parameters upfront, it avoids
    running 3 LLM calls (Hunter + Critic + Mediator) per N/A parameter.

    For example — if the TSD describes a mobile app, all web
    parameters can be pre-filtered as N/A
    without running the full debate on each one.

    Args:
        document_summary:  The root-level RAPTORTree summary (Level-3 node text)
                           which describes the overall TSD scope [raptor.py].
        parameters:        List of dicts with 'id' and 'requirement_text' keys
                           from CategoryParameterChild [3].
        category_code:     The category identifier.

    Returns:
        A fully formed prompt string.
    """
    import json

    params_text = json.dumps(
        [
            {
                "id": str(p.get("id", "")),
                "text": (p.get("requirement_text") or "")[:200],
                "parent_title": (p.get("parent_title") or "")[:120],
                "domain_keywords": list(p.get("domain_keywords") or []),
                "contract_summary": (p.get("contract_summary") or "")[:240],
            }
            for p in parameters
        ],
        indent=2,
    )

    return f"""\
## TSD DOCUMENT SUMMARY

{document_summary}

## CATEGORY CONTEXT

Category code: {category_code or "unknown"}

## SECURITY PARAMETERS TO PRE-FILTER

{params_text}

## YOUR TASK

For each parameter, determine whether it is CLEARLY not applicable to this
TSD based on the document summary above.

Only mark a parameter as not applicable if the document summary explicitly
rules out the technology, platform, or control domain named in the parameter
context, and you are highly confident that the TSD's scope makes it irrelevant.
When in doubt, mark it as applicable — the full debate pipeline will
make the final determination.

Respond with a single JSON object:

{{
  "results": [
    {{
      "id": "<parameter id>",
      "applicable": <true | false>,
      "confidence": <float 0.0–1.0>,
      "reason": "<one sentence if not applicable, else null>"
    }}
  ]
}}

Rules:
- applicable = true  → run the full Hunter/Critic/Mediator debate
- applicable = false → skip debate, verdict is automatically "na"
- Only set applicable=false when the summary explicitly excludes the parameter's technology or control domain
- Do not infer "not applicable" from missing detail, weak mention, or incomplete summaries
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
    "PARAMETER_APPLICABILITY_SYSTEM_PROMPT",
    "SEVERITY_JUSTIFICATION_SYSTEM_PROMPT",
    # Prompt builders
    "build_tsd_screening_prompt",
    "build_parameter_applicability_prompt",
    "build_severity_justification_prompt",
]
