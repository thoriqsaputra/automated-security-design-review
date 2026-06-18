def build_hierarchical_extraction_prompt(standard_text: str) -> str:
    return f"""
Extract every security parameter from this standard document, grouping them by their respective section headers.

RULES:
1. Extract ONLY actionable, technical security requirements or design constraints from the document body. These are rules meant to be implemented in the system architecture.
2. Do not limit extraction to mandatory words like must/shall/required, but ensure the item describes a system property or security behavior (e.g., input validation, cryptography, access control).
3. Group child parameters under their corresponding parent section header, translated to English when needed.
4. Output all section header keys, requirement fields, and details fields in English. If the source text is not English, translate them into clear technical English.
5. For each child parameter, choose based on how the control is written in the source:
   - ASVS-style inline controls ("2.1.1 Verify that the application..."): put the FULL text — section ID plus the entire "Verify that..." statement — together in the requirement field. Use details only for supplementary clause text that expands on the inline control beyond the main sentence.
   - Heading-only entries (a short label followed by separate paragraphs or bullets): put the heading/label in requirement and the concise English meaning distilled from the following paragraphs in details.
   - Never put a bare section number (e.g., "11.1.8") alone in requirement while the actual control text sits in details.
6. For each child heading, capture the clauses immediately following it until the next child or parent section heading begins. Do not drop those clauses.
7. Ignore DAFTAR ISI, Table of Contents, Contents, page indexes, and outline-only entries.
8. DO NOT invent or generate arbitrary numbers or bullet points (e.g., do not prepend "1.", "2.", "a.").
9. PRESERVE official compliance IDs, parameter numbers, section numbers, and labels ONLY IF they exist in the text (e.g., "REQ-INP-01", "PCI-3.4", "5.1.2"). Include the official ID or number at the start of the English requirement string exactly as written.
10. Do not invent duplicates. Extract controls exactly as they appear.
11. PROVENANCE (CRITICAL): Identify the closest structural marker (like a subsection number, page, or paragraph) in context_marker so we can trace it back to its exact origin.
12. ASVS LEVELS: When the source is OWASP ASVS or includes ASVS level columns/markers, set asvs_level to the official level integer for that child requirement: 1, 2, or 3. If no level is visible for a child requirement, set asvs_level to null. Do not infer a level from requirement wording.
13. REASONING: Analyze step-by-step internally to decide what qualifies as a real security parameter and where it belongs.
14. OUTPUT: Do not reveal reasoning. Do not include analysis, explanation, markdown fences, or XML-style tags.

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
      "context_marker": "V1.1",
      "asvs_level": 1
    }},
    {{
      "requirement": "V1.2 - 1.2.1 Verify the use of unique or low-privilege OS accounts for all application components",
      "details": "All application components, services, and servers must run under unique or special low-privilege OS accounts.",
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
      "context_marker": "Section 5.4.1",
      "asvs_level": null
    }}
  ]
}}

--- DOCUMENT ---
{standard_text}
"""

# ---------------------------------------------------------------------------
# ASVS Level Definitions Extraction
# ---------------------------------------------------------------------------

ASVS_LEVEL_DEFINITIONS_EXTRACTION_SYSTEM_PROMPT = (
    "You extract OWASP ASVS verification level definitions from standards. "
    "Return strict JSON only."
)


def build_asvs_level_definitions_extraction_prompt(source_doc_text: str) -> str:
    return f"""\
Extract the OWASP ASVS verification level definitions from this standard text.

Return ONLY valid JSON with this shape:
{{
  "levels": [
    {{
      "level": 1,
      "code": "L1",
      "name": "<official level name if present>",
      "description": "<what this ASVS level means>",
      "classification_guidance": "<how to decide that an application/TSD belongs to this level>",
      "source_quote": "<exact quote from the standard text>",
      "context_marker": "<nearest heading/page/section marker>"
    }}
  ]
}}

Rules:
- Extract only ASVS L1, L2, and L3 definitions.
- Use the document's own wording and version-specific meaning.
- Do not invent a level definition if it is absent from the text.
- If a field is not explicitly named, infer concise English from the surrounding definition text.

--- STANDARD TEXT ---
{source_doc_text[:12000]}
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
You are a security standards expert. Your task is to convert text-based ASVS \
security requirements into diagram-verifiable equivalents — requirements that \
can be assessed by LOOKING AT an architecture or data-flow diagram.

Output strict JSON only.
"""


# ---------------------------------------------------------------------------
# Control Family Summary Requirements (CFSR) Extraction
# ---------------------------------------------------------------------------

CFSR_EXTRACTION_SYSTEM_PROMPT = """\
You are a senior security architect. Your task is to synthesize detailed ASVS \
security requirements into a small set of actionable, text-verifiable summary \
requirements that can be evaluated against a Technical System Design document.

Output strict JSON only.
"""


def build_cfsr_extraction_prompt(
    requirements_text: str,
    parent_section: str,
    max_count: int = 5,
) -> str:
    return f"""\
Given the following ASVS security requirements from the control family \
"{parent_section}", synthesize a SMALL set of concise summary requirements \
that capture the essential security intent for evaluating a Technical System Design document.

BUDGET: Generate at most {max_count} summary requirements TOTAL for this entire control family \
(NOT per ASVS level — the entire family gets at most {max_count} across all levels combined).

Each summary requirement MUST:
- Be evaluable by reading a Technical System Design document (architecture descriptions,
  data-flow diagrams, API contracts, component specifications — NOT source code or config).
- Combine related child requirements across all levels into a single assertable statement.
- Be specific enough that a "not met" verdict indicates a real architectural gap.

For each summary requirement provide:
- stable_key: short ID like "CFSR-V2-1" (section-ordinal, no level suffix)
- requirement_text: ONE concise sentence, max 150 chars, stating what the TSD must demonstrate
- analysis_hint: 2-3 sentences describing what specific TSD content would satisfy this requirement
- asvs_level: the MINIMUM level at which this control first becomes mandatory (1, 2, or 3).
  Use 1 for fundamental baseline controls, 2 for standard assurance, 3 for high-assurance only.
  If a child has no level set, treat it as L1.
- covered_child_keys: list of the child IDs (e.g. "child-001") from the SOURCE REQUIREMENTS section that this summary covers

STRATEGY:
- Scan ALL children across L1, L2, L3 together.
- Identify the 3–{max_count} most architecturally distinct security assertions for this family.
- Assign each CFSR its minimum applicable level (the lowest-level child it covers).
- Merge children that share the same architectural implication into one CFSR.
- Each CFSR must express a UNIQUELY DISTINCT architectural assertion. Before emitting, scan all
  your planned CFSRs and ensure no two share the same or overlapping security concept. If two
  planned CFSRs cover the same idea, merge them into one and combine their covered_child_keys.
- Each child ID must appear in EXACTLY ONE CFSR's covered_child_keys — not zero, not more than one.
- Every child ID in SOURCE REQUIREMENTS MUST appear in at least one CFSR's covered_child_keys.
  Do NOT skip any child. If a requirement is code/config/process-level, merge it into the most
  thematically similar CFSR rather than omitting it.
- After writing all CFSRs, verify that the union of all covered_child_keys equals the full set
  of child IDs provided (child-001 through child-NNN). Add any missing child IDs to the closest CFSR.

## SOURCE REQUIREMENTS (with stable_keys and ASVS levels)

{requirements_text}

Respond with a single JSON object:
{{
  "summary_requirements": [
    {{
      "stable_key": "CFSR-V2-1",
      "requirement_text": "The TSD must define how user credentials are stored and protected at rest",
      "analysis_hint": "Look for descriptions of password hashing algorithms and salting in the data-at-rest section. The TSD should name the algorithm (e.g., Argon2, bcrypt) and minimum work factor.",
      "asvs_level": 1,
      "covered_child_keys": ["child-001", "child-002", "child-005"]
    }}
  ]
}}
"""


def build_diagram_req_extraction_prompt(requirements_text: str) -> str:
    return f"""\
Given the following ASVS security requirements, generate diagram-verifiable \
equivalents — requirements that can be assessed by looking at an architecture \
or data-flow diagram.

BUDGET: Generate exactly 5-7 diagram-verifiable requirements per ASVS level.
Each requirement must be assessable by LOOKING AT an architecture or data-flow diagram.

For each requirement provide:
- stable_key: A short unique ID like "D-V1.1" or "D-V2.3" based on the section prefix
- source_requirement_id: The original text requirement's stable_key (or "composite" if merged from multiple)
- requirement_text: ONE LINE, max 100 chars, written as "what must be visible in the diagram"
- verification_hint: 1-2 sentences describing what specific visual elements to look for
- asvs_level: 1, 2, or 3
- parent_section: The section this belongs to (e.g. "V1 Architecture")

Level 1: Basic architecture controls (topology, boundaries, flows, component labels)
Level 2: + Authentication, encryption, logging, session management, API gateway
Level 3: + Defense-in-depth, key management, mTLS, secrets, zero-trust

SKIP any requirement that cannot be verified visually (code-level, process-level,
configuration-level controls are NOT diagram-verifiable).

If multiple text requirements map to the same visual check, merge them into one
diagram requirement with source_requirement_id="composite".

## SOURCE REQUIREMENTS

{requirements_text}

Respond with a single JSON object:
{{
  "diagram_requirements": [
    {{
      "stable_key": "D-V1.1",
      "source_requirement_id": "...",
      "requirement_text": "Layered architecture: labeled trust boundaries between layers",
      "verification_hint": "Look for boxes/regions labeled with layer names separated by boundary lines or colored zones",
      "asvs_level": 1,
      "parent_section": "V1 Architecture"
    }}
  ]
}}
"""
