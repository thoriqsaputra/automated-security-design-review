def build_hierarchical_extraction_prompt(standard_text: str) -> str:
    return f"""
Extract every security parameter from this standard document, grouping them by their respective section headers.

RULES:
1. Extract ONLY actionable, technical security requirements or design constraints from the document body. These are rules meant to be implemented in the system architecture.
2. Do not limit extraction to mandatory words like must/shall/required, but ensure the item describes a system property or security behavior (e.g., input validation, cryptography, access control).
3. Keep verbatim_quote in the original source language as the exact quote from the document. Do not translate verbatim_quote.
4. Group child parameters under their corresponding parent section header, translated to English when needed.
5. Output all section header keys, requirement fields, and details fields in English. If the source text is not English, translate them into clear technical English.
6. For each child parameter heading, put only the heading/name/control label in the requirement field. Put the concise English meaning distilled from the paragraphs, bullets, sub-points, and clauses that follow that heading in the details field.
7. For each child heading, capture the clauses immediately following it until the next child or parent section heading begins. Do not drop those clauses.
8. Ignore DAFTAR ISI, Table of Contents, Contents, page indexes, and outline-only entries.
9. DO NOT invent or generate arbitrary numbers or bullet points (e.g., do not prepend "1.", "2.", "a.").
10. PRESERVE official compliance IDs, parameter numbers, section numbers, and labels ONLY IF they exist in the text (e.g., "REQ-INP-01", "PCI-3.4", "5.1.2"). Include the official ID or number at the start of the English requirement string exactly as written.
11. Do not deduplicate. Preserve duplicates if the document repeats them.
12. PROVENANCE (CRITICAL): For every extracted parameter, you MUST provide an exact, verbatim quote from the text, and identify the closest structural marker (like a subsection number, page, or paragraph) so we can trace it back to its exact origin.
13. ASVS LEVELS: When the source is OWASP ASVS or includes ASVS level columns/markers, set asvs_level to the official level integer for that child requirement: 1, 2, or 3. If no level is visible for a child requirement, set asvs_level to null. Do not infer a level from requirement wording.
14. REASONING: Analyze step-by-step internally to decide what qualifies as a real security parameter and where it belongs.
15. OUTPUT: Do not reveal reasoning. Do not include analysis, explanation, markdown fences, or XML-style tags.

HIERARCHY RULES (CRITICAL - READ CAREFULLY):
- Some standards (e.g., OWASP ASVS) use a two-level numbering system: top-level chapters ("V1 Architecture") and sub-sections ("V1.1 SDLC", "V1.2 Auth").
- The JSON key MUST always be the TOP-LEVEL chapter (e.g. "V1 Architecture, Design and Threat Modeling").
- Sub-section headings like "V1.1 Secure Software Development Lifecycle" are NOT separate parent keys. Instead, they are the start of a group of requirements that belong UNDER the top-level key.
- When you encounter a sub-section heading (e.g. "V1.1", "V1.2", "2.1", "2.2"), treat each individual numbered control (e.g. "1.1.1 Verify...", "1.2.3 Verify...") within it as a child requirement item. Prefix the requirement with the sub-section label for traceability (e.g., requirement: "V1.1 - 1.1.1 Verify the use of a secure SDLC").
- For standards without this two-level structure, use the main section headers as keys directly.

EXCLUSIONS (CRITICAL - DO NOT EXTRACT THESE):
- DO NOT extract Introductions, Document Overviews, or Executive Summaries.
- DO NOT extract Scope, Applicability, or Target Audience definitions (e.g., "Standard Applicability", "This standard applies to...").
- DO NOT extract generic descriptions or lists of security domains (e.g., "Security domains include ARCH, AUTH...").
- DO NOT extract Compliance, Audit, Verification, or Testing processes (e.g., SAST, DAST, Manual Penetration Testing procedures).
- DO NOT extract glossaries, definitions of basic terms, or administrative document metadata.
- DO NOT extract items marked as [DELETED], [BLANK], or [RESERVED]. Skip them entirely.
- If you skip a deleted item, DO NOT duplicate the surrounding items to fill the gap. Maintain the original IDs (e.g. if 4.1.4 is deleted, the next item remains 4.1.5). DO NOT renumber subsequent items.

Format:
Output a valid JSON dictionary where keys are TOP-LEVEL parent section headers, and values are lists of requirement objects.

Example Expected Output for OWASP-style documents:
{{
  "V1 Architecture, Design and Threat Modeling": [
    {{
      "requirement": "V1.1 - 1.1.1 Verify the use of a secure software development lifecycle that addresses security in all stages of development",
      "details": "Applications must be developed using a documented SDLC that includes security in all stages, from design through testing and deployment.",
      "verbatim_quote": "1.1.1 Verify the use of a secure software development lifecycle that addresses security in all stages of development. (C1)",
      "context_marker": "V1.1",
      "asvs_level": 1
    }},
    {{
      "requirement": "V1.2 - 1.2.1 Verify the use of unique or low-privilege OS accounts for all application components",
      "details": "All application components, services, and servers must run under unique or special low-privilege OS accounts.",
      "verbatim_quote": "1.2.1 Verify the use of unique or special low-privilege operating system accounts for all application components, services, and servers. (C3)",
      "context_marker": "V1.2",
      "asvs_level": 3
    }}
  ]
}}

Example for non-OWASP (flat) standards:
{{
  "5.4 API and Web Service": [
    {{
      "requirement": "5.4.1 Generic Web Service Security",
      "details": "All APIs must consistently define and document their interfaces and security behavior.",
      "verbatim_quote": "a. semua API mendefinisikan dan mendokumentasikan antarmuka...",
      "context_marker": "Section 5.4.1",
      "asvs_level": null
    }}
  ]
}}

--- DOCUMENT ---
{standard_text}
"""
