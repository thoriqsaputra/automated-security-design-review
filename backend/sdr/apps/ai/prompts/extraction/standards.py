def build_hierarchical_extraction_prompt(standard_text: str) -> str:
    return f"""
Extract every security parameter from this standard document, grouping them by their respective section headers.

RULES:
1. Extract ONLY actionable, technical security requirements or design constraints from the document body. These are rules meant to be implemented in the system architecture.
2. Do not limit extraction to mandatory words like must/shall/required, but ensure the item describes a system property or security behavior (e.g., input validation, cryptography, access control).
3. Group child parameters under their corresponding parent section header, translated to English when needed.
4. Output all section header keys, requirement fields, and details fields in English. If the source text is not English, translate them into clear technical English.
5. For each child parameter, choose based on how the control is written in the source:
   - Inline numbered controls ("2.1.1 Verify that the application..."): put the FULL text — section ID plus the entire "Verify that..." statement — together in the requirement field. Use details only for supplementary clause text that expands on the inline control beyond the main sentence.
   - Heading-only entries (a short label followed by separate paragraphs or bullets): put the heading/label in requirement and the concise English meaning distilled from the following paragraphs in details.
   - Never put a bare section number (e.g., "11.1.8") alone in requirement while the actual control text sits in details.
6. For each child heading, capture the clauses immediately following it until the next child or parent section heading begins. Do not drop those clauses.
7. Ignore DAFTAR ISI, Table of Contents, Contents, page indexes, and outline-only entries.
8. DO NOT invent or generate arbitrary numbers or bullet points (e.g., do not prepend "1.", "2.", "a.").
9. PRESERVE official compliance IDs, parameter numbers, section numbers, and labels ONLY IF they exist in the text (e.g., "REQ-INP-01", "PCI-3.4", "5.1.2"). Include the official ID or number at the start of the English requirement string exactly as written.
10. Do not invent duplicates. Extract controls exactly as they appear.
11. PROVENANCE (CRITICAL): Identify the closest structural marker (like a subsection number, page, or paragraph) in context_marker so we can trace it back to its exact origin.
12. REASONING: Analyze step-by-step internally to decide what qualifies as a real security parameter and where it belongs.
13. OUTPUT: Do not reveal reasoning. Do not include analysis, explanation, markdown fences, or XML-style tags.

HIERARCHY RULES (CRITICAL - READ CAREFULLY):
- Some standards use a two-level numbering system: top-level chapters ("V1 Architecture") and sub-sections ("V1.1 SDLC", "V1.2 Auth").
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
- DO NOT extract explanatory notes, caution boxes, advisory "Note:" paragraphs, examples, or side commentary unless the note itself is explicitly written as a normative requirement.
- DO NOT extract items marked as [DELETED], [BLANK], or [RESERVED]. Skip them entirely.
- DO NOT extract sub-section headings that do not contain an actual numbered control. For example, do NOT extract "V2.1 Password Security" or "V1.3 Session Management Architecture - This is a placeholder for future architectural requirements." Only extract items that have a three-level control ID (like "2.1.1" or "1.3.2") embedded in the requirement text.
- If you skip a deleted item, DO NOT duplicate the surrounding items to fill the gap. Maintain the original IDs (e.g. if 4.1.4 is deleted, the next item remains 4.1.5). DO NOT renumber subsequent items.

Format:
Output a valid JSON dictionary where keys are TOP-LEVEL parent section headers, and values are lists of requirement objects.

Example Expected Output for OWASP-style documents:
{{
  "V1 Architecture, Design and Threat Modeling": [
    {{
      "requirement": "V1.1 - 1.1.1 Verify the use of a secure software development lifecycle that addresses security in all stages of development",
      "details": "Applications must be developed using a documented SDLC that includes security in all stages, from design through testing and deployment.",
      "context_marker": "V1.1"
    }},
    {{
      "requirement": "V1.2 - 1.2.1 Verify the use of unique or low-privilege OS accounts for all application components",
      "details": "All application components, services, and servers must run under unique or special low-privilege OS accounts.",
      "context_marker": "V1.2"
    }}
  ]
}}

Example for non-OWASP (flat) standards:
{{
  "5.4 API and Web Service": [
    {{
      "requirement": "5.4.1 Generic Web Service Security",
      "details": "All APIs must consistently define and document their interfaces and security behavior.",
      "context_marker": "Section 5.4.1"
    }}
  ]
}}

--- DOCUMENT ---
{standard_text}
"""

# ---------------------------------------------------------------------------
# JSON Repair Prompt
# ---------------------------------------------------------------------------

def build_json_repair_prompt(content: str) -> str:
    return (
        "The following JSON has syntax errors (e.g. missing commas, unescaped quotes). "
        "Fix it and return ONLY valid JSON without any markdown or conversational text.\n\n"
        f"{content}"
    )


# ---------------------------------------------------------------------------
# Diagram Requirements Extraction
# ---------------------------------------------------------------------------

DIAGRAM_REQ_EXTRACTION_SYSTEM_PROMPT = """\
You are a security standards expert. Your task is to convert text-based \
security requirements into diagram-verifiable equivalents — requirements that \
can be assessed by LOOKING AT an architecture or data-flow diagram.

Apply your per-requirement and merge rules mechanically and consistently — given \
the same input, you must always produce the same requirements, grouping, and \
stable_key assignment. Do not introduce arbitrary variation between runs.

Output strict JSON only.
"""


def build_diagram_req_extraction_prompt(requirements_text: str) -> str:
    return f"""\
Given the following security requirements, generate diagram-verifiable \
equivalents — requirements that can be assessed by looking at an architecture \
or data-flow diagram.

Each requirement must be assessable by LOOKING AT an architecture or data-flow diagram.
Do not target a specific count of requirements — the per-requirement and merge rules below
determine the count, and they must be applied mechanically so the same input always yields
the same output.

For each requirement provide:
- stable_key: deterministic ID, NOT freely chosen. If source_requirement_id is a single id,
  use "D-{{source_requirement_id}}". If merged (composite), use
  "D-composite-{{first source_requirement_id in the merge, in input order}}".
- source_requirement_id: The original text requirement's stable_key (or "composite" if merged from multiple)
- requirement_text: ONE LINE, max 100 chars, written as "what must be visible in the diagram"
- verification_hint: 1-2 sentences describing what specific visual elements to look for
- parent_section: The section this belongs to (e.g. "V1 Architecture")

Level 1: Basic architecture controls (topology, boundaries, flows, component labels)
Level 2: + Authentication, encryption, logging, session management, API gateway
Level 3: + Defense-in-depth, key management, mTLS, secrets, zero-trust

PROCEDURE (apply mechanically, in the order the source requirements are given):
- SKIP any requirement that cannot be verified visually (code-level, process-level,
  configuration-level controls are NOT diagram-verifiable).
- For every remaining requirement, emit exactly ONE diagram requirement, UNLESS it verifies
  the IDENTICAL diagram element via the IDENTICAL visual check as another remaining requirement
  (e.g. both are satisfied by the same trust-boundary line, or the same TLS/padlock icon on the
  same connection) — only then merge them into one diagram requirement with
  source_requirement_id="composite". A shared theme or related topic is NOT sufficient grounds
  to merge; the visual check itself must be identical.
- Emit diagram requirements in the same order as their (first, for composites) source requirement.

## SOURCE REQUIREMENTS

{requirements_text}

Respond with a single JSON object:
{{
  "diagram_requirements": [
    {{
      "stable_key": "D-V1.1",
      "source_requirement_id": "V1.1",
      "requirement_text": "Layered architecture: labeled trust boundaries between layers",
      "verification_hint": "Look for boxes/regions labeled with layer names separated by boundary lines or colored zones",
      "parent_section": "V1 Architecture"
    }}
  ]
}}
"""
