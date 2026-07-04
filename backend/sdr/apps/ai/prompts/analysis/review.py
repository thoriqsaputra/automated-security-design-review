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
# Parent Applicability
# ---------------------------------------------------------------------------

PARENT_APPLICABILITY_SYSTEM_PROMPT = (
    "You are a senior application security analyst deciding whether a control family "
    "is in scope for the documented design. Return strict JSON only."
)


def build_parent_applicability_prompt(
    category_code: str,
    version_label: str,
    parent_title: str,
    parent_description: str,
    child_block: str,
    context_text: str,
    scope_terms: Optional[list[str]] = None,
) -> str:
    scope_term_block = ", ".join([str(item).strip() for item in (scope_terms or []) if str(item).strip()]) or "none"
    return f"""\
Decide whether this parent control family from a security standard is applicable to the documented TSD scope.

Return only valid JSON:
{{
  "applicable": false,
  "confidence": 0.42,
  "decision_mode": "unclear",
  "reasoning": "The retrieved context mentions authentication generally but does not clearly establish whether this subsystem is in scope.",
  "evidence": ["mentions authentication", "no explicit subsystem boundary"]
}}

Rules:
- applicable=true only when the TSD explicitly describes the subsystem, capability, or scope that this parent control family governs. Use decision_mode="positive_match".
- applicable=false when the TSD clearly indicates the design does not use that subsystem/capability, or when the retrieved context is generic and does not directly match this family. This is the common case for an out-of-scope family — concluding "not described anywhere in this TSD" is a confident, decisive judgment, not an uncertain one. Use decision_mode="negative_match" with a correspondingly high confidence.
- Reserve decision_mode="unclear" for genuine indecision only — cases where the evidence could reasonably support either applicable or not-applicable and you cannot tell which. In that case set applicable=false, but confidence must be low (<=0.5) to reflect the real ambiguity. Do not use "unclear" just because the TSD never explicitly rules the family out — inferring absence from silence is still a "negative_match", not "unclear".
- Do not treat missing implementation detail as out of scope. This step is only about scope/applicability.
- Do not infer applicability from broad security language unless it directly matches the family-specific scope terms.
- FAMILY SCOPE TERMS are heuristic hints to help you focus your reading; the TSD may describe
  the same capability using different wording. Do not conclude inapplicable solely because
  none of these literal terms appear in the context — judge applicability by meaning.
- confidence → your certainty in this specific applicable/not-applicable call (1.0 = certain,
  0.5 = genuinely ambiguous). This value is used verbatim to decide whether to skip debate for
  this family, so it must reflect your true certainty, independent of decision_mode.
- reasoning must be exactly one sentence on a single line.
- evidence must be a JSON array with 0 to 3 short single-line strings.
- Do not use markdown, bullets, code fences, or multiline string values.
- If you include quotes inside a JSON string value, they must be properly escaped.

STANDARD CATEGORY: {category_code}
STANDARD VERSION: {version_label}

PARENT TITLE: {parent_title}
PARENT DESCRIPTION: {parent_description}
FAMILY SCOPE TERMS: {scope_term_block}
CHILD REQUIREMENTS:
{child_block}

RETRIEVED TSD CONTEXT:
{context_text[:8000]}
"""


# ---------------------------------------------------------------------------
# Contract Synthesis
# ---------------------------------------------------------------------------

CONTRACT_SYNTHESIS_SYSTEM_PROMPT = (
    "Generate a strict JSON contract with keys: given, when, then, not_sufficient, "
    "in_scope, specific_enough, domain, confidence, synth_mode."
)


def build_contract_synthesis_prompt(
    parameter_section: str,
    parent_description: str,
    child_context: str,
    parameter_text: str,
    domain: str,
) -> str:
    return f"""\
Section: {parameter_section}
Parent description: {parent_description}
Child context: {child_context}
Requirement: {parameter_text}
Domain hint: {domain}
Return JSON only."""


def build_contract_repair_prompt(content: str) -> str:
    return f"""\
The following JSON has syntax errors. Fix it and return ONLY valid JSON.

{content}"""


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # System prompts
    "TSD_SCREENING_SYSTEM_PROMPT",
    "SEVERITY_JUSTIFICATION_SYSTEM_PROMPT",
    "PARENT_APPLICABILITY_SYSTEM_PROMPT",
    "CONTRACT_SYNTHESIS_SYSTEM_PROMPT",
    # Prompt builders
    "build_tsd_screening_prompt",
    "build_severity_justification_prompt",
    "build_parent_applicability_prompt",
    "build_contract_synthesis_prompt",
    "build_contract_repair_prompt",
]

