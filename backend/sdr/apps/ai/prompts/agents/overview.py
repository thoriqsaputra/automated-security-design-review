from __future__ import annotations

from typing import List

OVERVIEW_SYSTEM_PROMPT = """\
You are a Senior Security Consultant producing an executive summary of \
a completed automated security review.

YOUR ROLE:
- Synthesise all security findings into a concise, executive-level overview.
- Highlight the most critical gaps and the strongest compliance areas.
- Be direct and specific — avoid generic security advice.
- Do not repeat individual findings verbatim; summarise patterns and themes.

OUTPUT: Plain text. 3–5 paragraphs. No JSON, no bullet points.
"""


def build_overview_prompt(
    design_name: str,
    category_name: str,
    total_parameters: int,
    met_count: int,
    not_met_count: int,
    na_count: int,
    critical_findings: List[str],
    high_findings: List[str],
) -> str:
    compliance_rate = (
        round((met_count / (total_parameters - na_count)) * 100, 1)
        if (total_parameters - na_count) > 0
        else 0.0
    )

    critical_section = ""
    if critical_findings:
        formatted = "\n".join(f"  - {finding}" for finding in critical_findings)
        critical_section = (
            f"\n\nCritical Findings ({len(critical_findings)}):\n{formatted}"
        )

    high_section = ""
    if high_findings:
        formatted = "\n".join(f"  - {finding}" for finding in high_findings)
        high_section = (
            f"\n\nHigh Severity Findings ({len(high_findings)}):\n{formatted}"
        )

    return f"""\
## TSD SECURITY REVIEW — EXECUTIVE SUMMARY REQUEST

Document:          {design_name}
Security Category: {category_name}

## ANALYSIS RESULTS

Total Parameters Evaluated: {total_parameters}
Met:                        {met_count}
Not Met:                    {not_met_count}
N/A:                        {na_count}
Compliance Rate:            {compliance_rate}%{critical_section}{high_section}

## YOUR TASK

Write a concise executive summary of this security review in plain text.

Guidelines:
- 3 to 5 paragraphs only.
- Paragraph 1: Overall compliance posture — is this document broadly
  compliant, partially compliant, or significantly non-compliant?
- Paragraph 2: Most critical gaps — reference the critical and high
  findings by theme, not by verbatim repetition.
- Paragraph 3: Strongest compliance areas — what does the TSD do well?
- Paragraph 4 (optional): Patterns observed — systemic issues that
  appear across multiple findings (e.g. authentication controls are
  well-defined but encryption at rest is consistently absent).
- Paragraph 5 (optional): Recommended immediate actions — the top 2–3
  priorities the development team should address first.

Tone: Direct, professional, suitable for a CISO or senior engineering lead.
Do NOT use bullet points, headers, JSON, or markdown formatting.
Plain paragraphs only.
"""
