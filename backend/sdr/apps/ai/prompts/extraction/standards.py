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
- DO NOT extract sub-section headings that do not contain an actual numbered control. For example, do NOT extract "V2.1 Password Security" or "V1.3 Session Management Architecture - This is a placeholder for future architectural requirements."
- If you skip a deleted item, DO NOT duplicate the surrounding items to fill the gap. Maintain the original IDs (e.g. if 4.1.4 is deleted, the next item remains 4.1.5). DO NOT renumber subsequent items.
- DO NOT extract release notes, versioning text, or migration guidance. Examples to skip:
  "Major release - Full reorganization, almost everything may have changed, including requirement numbers."
  "Minor release - Requirements may be added or removed, but overall numbering will stay the same."
- DO NOT extract advisory paragraphs that reference external standards without themselves being a numbered requirement. Example to skip:
  "Post-quantum cryptography (PQC) implementations should follow FIPS-203, FIPS-204, and FIPS-205."
  These are guidance sentences, not verifiable requirements — they have no X.Y.Z control ID of their own.

CRITICAL: CONTROL ID REQUIRED — Every extracted requirement MUST contain a numeric control
ID in the format X.Y.Z (e.g. "1.1.1", "2.4.4", "13.2.3") or a recognized standard ID
(e.g. "REQ-INP-01", "PCI-3.4") in the requirement text itself. If the item has no such ID,
DO NOT extract it — no exceptions.

The following kinds of text NEVER have a control ID and must ALWAYS be skipped:
- Section preamble bullets: "Does not phone home...", "Does not have back doors...",
  "Malicious activity is handled securely and properly...", "Does not have time bombs..."
- CIA triad or security goal definitions: "Availability: Data should be available...",
  "Confidentiality: Data should be protected...", "Integrity: Data should be protected..."
- Advisory/best-practice guidance: "Avoid weak or soon to be deprecated algorithms...",
  "Check your configuration periodically...",
  "Disable deprecated or known insecure algorithms, ciphers, and protocols...",
  "Follow the latest guidance on updating TLS configuration...",
  "Require TLS or strong encryption, independent of sensitivity of the content.",
  "Stay current with recommended industry advice on secure TLS configuration.",
  "Use the most recent versions of TLS configuration recommendations."

These are context paragraphs, not verifiable requirements. Extracting them pollutes the knowledge base.

Format:
Output a valid JSON dictionary where keys are TOP-LEVEL parent section headers, and values are lists of requirement objects.

Each requirement object must include a "requirement_category" field — one of: "design", "code", "infrastructure", "process".

CATEGORY ASSIGNMENT — follow this decision tree in order and stop at the FIRST matching step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 ▸ DOCUMENTATION RULE (HIGHEST PRIORITY — CHECK FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Does the requirement say the application's TSD or design documentation must DEFINE, STATE,
SPECIFY, or otherwise record a specific security control, parameter, or policy?
→ category = "design" (these requirements are verified by reading the TSD, not by observing org activity)

Exact examples — every one of these is "design":
  "documentation states the expected security features that browsers must support" → design
  "documentation defines the permitted file types and maximum size for each upload feature" → design
  "documentation defines how rate limiting and anti-automation controls are used" → design
  "a list of context-specific words is documented to prevent their use in passwords" → design
  "authentication pathways are all documented with their security controls" → design
  "session inactivity timeout and maximum session lifetime are documented" → design
  "documentation defines how many concurrent sessions are allowed" → design
  "authorization documentation defines rules for restricting function-level access" → design
  "documented policy for management of cryptographic keys and key lifecycle" → design
  "cryptographic discovery mechanisms are employed to identify all cryptography in the system" → design
  "inventory exists documenting the logging performed at each layer of the technology stack" → design
  "algorithm allowlist is documented as part of the system security architecture" → design

EXCEPTION — even when the word "documented" appears, use "process" if the requirement describes
an ongoing organizational ACTIVITY rather than a TSD property:
  "documented change management process" → process (org activity)
  "developer security training is documented" → process (HR activity)
  "secure SDLC is followed and documented" → process (development methodology)
  "documentation defines risk-based remediation timeframes for 3rd party components" → process (vuln management)
  "cryptographic inventory is performed, maintained, and regularly updated" → process (ongoing audit activity)
  "user stories and features contain functional security constraints" → process (SDLC practice)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 ▸ INFRASTRUCTURE RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Does compliance require auditing server/deployment/OS/network configuration that is NOT
visible in a design document?
→ category = "infrastructure"

Exact examples (these are commonly mis-tagged as design or code):
  "unique or special low-privilege OS accounts for all components, services, and servers" → infrastructure
  "intra-service secrets do not rely on unchanging credentials such as passwords or API keys" → infrastructure
  "only user-facing endpoints automatically redirect HTTP to HTTPS, other endpoints do not" → infrastructure
  "HTTP Strict Transport Security (HSTS), Content-Security-Policy header values set" → infrastructure
  "TLS cipher suites enabled on the server are restricted to strong suites" → infrastructure
  "debug mode is disabled in production" → infrastructure
  "HTTP response Content-Type header specifies a safe charset" → infrastructure
  "all components are up to date; dependency checker used during build" → infrastructure (CI/CD tooling)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 ▸ CODE RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Does compliance require reading source code to verify a specific low-level IMPLEMENTATION DETAIL
(exact algorithm work factor, exact parsing order, where a variable is placed in a request)?
→ category = "code"

Exact examples (these are genuinely code-level, do NOT use code for high-level architectural rules):
  "input is decoded or unescaped into a canonical form only once" → code (exact parsing order)
  "application performs output encoding as a final step before being used by the interpreter" → code
  "sensitive data is sent in the HTTP message body or headers, NOT in query string parameters" → code
  "application protects sensitive data from being cached in server components (load balancers, app caches)" → code
  "bcrypt work factor is at least 10" → code (exact parameter value)
  "cookie attributes Secure, HttpOnly, SameSite are set" → code
  "session token is regenerated after successful authentication" → code
  "per-object authorization checks prevent IDOR" → code
  "log entries are encoded to prevent log injection" → code

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 ▸ DESIGN RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Does compliance require verifying an architectural PROPERTY or POLICY that should appear in a TSD
(architecture diagrams, data flows, trust boundaries, protocol choices, component interaction specs)?
→ category = "design"

The key question: is this about WHAT/WHETHER the system does something (design), or HOW the code
does it precisely (code)? Architecture, protocols, data classification, and security policies → design.

Exact examples (these are commonly mis-tagged as code):
  "only algorithms on an allowlist can be used to create and verify self-contained tokens" → design
  "application employs integrity protections such as code signing or subresource integrity" → design
  "inter-component mTLS or mutual TLS — authenticated communication between services" → design
  "application components verify the authenticity of each side in a communication link" → design
  "TLS is used for all client connectivity and does not fall back to insecure communications" → design
  "centralized, well-vetted access control mechanism used for all access decisions" → design
  "key management: secrets management solution such as a key vault is used" → design
  "PII encrypted at rest with a defined encryption strategy" → design
  "trust boundaries between network zones are defined" → design
  "THAT all authentication events are logged" → design (WHETHER it exists, not HOW)
  "THAT the system has rate limiting / CAPTCHA / account lockout" → design (WHETHER, not HOW)
  "data sensitivity policy: application does not log credentials or payment details" → design
  "serialized objects use integrity checks or are encrypted to prevent tampering" → design
  "business logic flows are processed in sequential step order" → design (workflow architecture)
  "application will not accept large files that could cause denial of service" → design (capacity policy)
  "all application components use the same encodings and parsers to avoid parsing attacks" → design
  "security controls prevent browsers from rendering content in an incorrect context" → design (CSP policy)
  "system-generated activation or recovery secrets are not sent in clear text" → design (protocol security)
  "only supported HTTP methods can be used; unused methods are blocked" → design (API policy)
  "application performs all session token verification using a trusted backend service" → design
  "input is validated to enforce business or functional expectations" → design (validation policy)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 ▸ PROCESS RULE (use only if none above match)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Does compliance require assessing an ORGANIZATIONAL ACTIVITY (people, procedures, governance)?
→ category = "process"

Examples: secure SDLC maturity, threat modeling cadence, developer security training program,
change management procedures, vulnerability remediation timeframes for third-party dependencies,
cryptographic inventory that is regularly reviewed and updated,
requiring security constraints to appear in user stories or change tickets.

Example Expected Output for OWASP-style documents:
{{
  "V1 Architecture, Design and Threat Modeling": [
    {{
      "requirement": "V1.1 - 1.1.1 Verify the use of a secure software development lifecycle that addresses security in all stages of development",
      "details": "Applications must be developed using a documented SDLC that includes security in all stages, from design through testing and deployment.",
      "context_marker": "V1.1",
      "requirement_category": "process"
    }},
    {{
      "requirement": "V1.2 - 1.2.2 Verify that communications between application components, including APIs, middleware and data layers, are authenticated",
      "details": "Components should have the least necessary privileges needed. Authentication between services must be verifiable from the system architecture.",
      "context_marker": "V1.2",
      "requirement_category": "design"
    }},
    {{
      "requirement": "V2.4 - 2.4.4 Verify that if bcrypt is used, the work factor SHOULD be as large as verification server performance will allow, with a minimum of 10",
      "details": "The bcrypt cost factor must be set to at least 10 in the application configuration.",
      "context_marker": "V2.4",
      "requirement_category": "code"
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
      "requirement_category": "design"
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

_DIAGRAM_TYPE_DESCRIPTIONS = {
    "data_flow": (
        "Data Flow Diagram (DFD)",
        "external entities, processes (circles/rectangles), data stores (open rectangles), "
        "labeled data-flow arrows, and trust boundaries (dashed lines separating zones)",
    ),
    "sequence": (
        "Sequence Diagram",
        "actor/system lifelines, labeled message arrows between lifelines, "
        "activation boxes, and the temporal order of interactions",
    ),
    "architecture": (
        "System Architecture Diagram",
        "components/services as labeled boxes, network zones or deployment boundaries, "
        "connections annotated with protocols (HTTPS, TLS, gRPC), load balancers, "
        "databases, caches, and external integrations",
    ),
}

_DEFAULT_DIAGRAM_TYPES = ["data_flow", "sequence", "architecture"]

DIAGRAM_REQ_EXTRACTION_SYSTEM_PROMPT = """\
You are a security standards expert. Your task is to convert text-based \
security requirements into diagram-verifiable equivalents — requirements \
that can be assessed by LOOKING AT one of the supported diagram types.

Apply your filtering and merge rules mechanically and consistently — given \
the same input you must always produce the same output. Do not introduce \
arbitrary variation between runs.

Output strict JSON only. Do not output analysis, reasoning, or markdown fences.
"""


def build_diagram_req_extraction_prompt(
    requirements_text: str,
    diagram_types: list[str] | None = None,
) -> str:
    types = diagram_types or _DEFAULT_DIAGRAM_TYPES
    valid_types = [t for t in types if t in _DIAGRAM_TYPE_DESCRIPTIONS]
    if not valid_types:
        valid_types = _DEFAULT_DIAGRAM_TYPES

    diagram_type_block = "\n".join(
        f"- **{_DIAGRAM_TYPE_DESCRIPTIONS[t][0]}** (`{t}`): {_DIAGRAM_TYPE_DESCRIPTIONS[t][1]}"
        for t in valid_types
    )
    valid_type_list = ", ".join(f'"{t}"' for t in valid_types)

    return f"""\
Convert the security requirements below into diagram-verifiable equivalents.
Only extract requirements that can be assessed by LOOKING AT one of these diagram types:

{diagram_type_block}

A requirement is diagram-verifiable ONLY if its visual evidence appears directly in one of
the diagram types above. Skip code-level, configuration-level, and process-level controls.

MERGE RULES (apply before emitting):
- Merge two or more requirements into ONE if they verify the SAME architectural concern
  on the SAME diagram type (e.g. all TLS/encryption checks on data-flow arrows → one item;
  all trust-boundary checks on a DFD → one item).
- A shared theme alone is NOT enough to merge; the diagram type and the visual check target
  must both be the same.
- After merging, aim for roughly 1 output requirement per 4–5 source requirements.
  Quality over quantity — fewer, sharper requirements are better.

For each output item provide:
- stable_key: If from a single source use "D-{{source_requirement_id}}".
  If merged use "D-composite-{{first source_requirement_id in the merge}}".
- source_requirement_id: Original stable_key, or "composite" if merged.
- diagram_type: One of {valid_type_list}
- requirement_text: ONE LINE, max 100 chars — "what must be visible in the diagram".
- verification_hint: 1–2 sentences on which specific visual elements confirm this.
- parent_section: The section this belongs to (e.g. "V1 Architecture").

Emit items in the same order as their first source requirement.

## SOURCE REQUIREMENTS

{requirements_text}

Respond with a single JSON object:
{{
  "diagram_requirements": [
    {{
      "stable_key": "D-V1.1",
      "source_requirement_id": "V1.1",
      "diagram_type": "architecture",
      "requirement_text": "Layered architecture: labeled trust boundaries between layers",
      "verification_hint": "Look for regions separated by boundary lines or colored zones with layer labels.",
      "parent_section": "V1 Architecture"
    }}
  ]
}}
"""
