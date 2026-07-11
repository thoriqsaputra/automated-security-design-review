def build_hierarchical_extraction_prompt(standard_text: str) -> str:
    return f"""
Extract security requirements from this standard document, grouped by top-level section headers.

CONTROL ID REQUIRED: Every extracted requirement MUST contain a numeric ID (e.g. "1.1.1", "2.4.4") or
recognized standard ID (e.g. "REQ-INP-01", "PCI-3.4") in the requirement text. No ID = skip, no exceptions.

EXTRACTION RULES:
- Extract only actionable security requirements describing a system property or security behavior.
- Include the full control text — never a bare section number alone.
- All keys and requirement text must be in English. Translate non-English source text.
- For inline numbered controls: put the full text (ID + "Verify that..." statement) in the requirement field.
- For heading-only entries: combine the heading with a concise English summary of the following paragraphs.

HIERARCHY: Use only the TOP-LEVEL chapter as the JSON key (e.g. "V1 Architecture, Design and Threat Modeling").
Sub-sections (V1.1, V1.2, 2.1, 2.2) are NOT separate keys — prefix their child requirements for traceability
(e.g. requirement: "V1.1 - 1.1.1 Verify the use of a secure SDLC").

SKIP ENTIRELY:
- Table of contents, scope/applicability, glossaries, document metadata, introductions
- Items marked [DELETED], [BLANK], or [RESERVED]
- Advisory sentences, "Note:" paragraphs, or any guidance text without a control ID
- Preamble bullets (e.g. "Does not phone home...", "Availability: Data should be available...")
- Release notes, version text, migration guidance, SAST/DAST procedure descriptions

CATEGORY — assign one of: "design", "code", "infrastructure", "process". Apply steps in order, stop at first match.

STEP 1 — DESIGN (documentation): TSD/design doc must DEFINE, STATE, or SPECIFY a security control or policy.
  design: "documentation defines permitted file types", "algorithm allowlist is documented", "session timeout is documented"
  Exception → process if it describes an ongoing org activity even when "documented" appears:
    "documented change management process", "secure SDLC is followed and documented", "developer security training is documented",
    "security requirements/constraints captured in user stories or backlog items"

STEP 2 — INFRASTRUCTURE: Server/OS/network/CI configuration not visible in a design doc.
  infra: OS service accounts, HTTP security headers (HSTS, CSP, Content-Type), TLS cipher suites on server,
         debug mode disabled in production, dependency checker in CI/CD, secrets management service / key vault storage,
         network segmentation or trust-boundary enforcement via firewall rules, API gateways, reverse proxies,
         or cloud security groups (naming these mechanisms means infra even when "trust boundary" is mentioned)

STEP 3 — CODE: Verifiable only by reading source code — exact algorithm parameters, exact parsing order,
  specific placement in a request.
  code: bcrypt work factor value, cookie attribute flags (Secure/HttpOnly/SameSite), session token regeneration,
        output encoding as final step, per-object IDOR checks, log entry encoding, TLS certificate/chain validation logic,
        file/upload size limits enforced by the application, business-logic workflow step-order enforcement (no skipping steps)

STEP 4 — DESIGN (architecture): Architectural property, policy, or protocol choice visible in a TSD.
  Key question: "WHAT/WHETHER the system does it" → design; "HOW the code implements it precisely" → code.
  design: mTLS between services, PII encrypted at rest,
          trust boundaries defined at the architectural/policy level (not the specific network mechanism enforcing them — see STEP 2),
          rate limiting policy, centralized access control, authentication flow architecture, data sensitivity policy

STEP 5 — PROCESS (fallback): Organizational activity — people, procedures, governance.
  process: SDLC maturity, threat modeling cadence, developer training program, vuln remediation timeframes

OUTPUT: Valid JSON only. No markdown fences, no analysis, no reasoning.

{{
  "V1 Architecture, Design and Threat Modeling": [
    {{
      "requirement": "V1.1 - 1.1.1 Verify the use of a secure SDLC that includes security in all stages.",
      "context_marker": "V1.1",
      "requirement_category": "process"
    }},
    {{
      "requirement": "V2.4 - 2.4.4 Verify that bcrypt work factor is at least 10.",
      "context_marker": "V2.4",
      "requirement_category": "code"
    }}
  ],
  "5.4 API and Web Service": [
    {{
      "requirement": "5.4.1 All APIs must define and document their interfaces and security behavior.",
      "context_marker": "Section 5.4.1",
      "requirement_category": "design"
    }}
  ]
}}

--- DOCUMENT ---
{standard_text}
"""


REQUIREMENT_CATEGORY_VALIDATION_SYSTEM_PROMPT = """\
You are a deterministic security-requirement classifier. Classify each item
using the TSD-verifiability rubric exactly as written. Return strict JSON only.
Do not add, remove, reorder, or rewrite items.
"""


def build_requirement_category_validation_prompt(items_json: str) -> str:
    return f"""\
Classify every requirement below into exactly one category. The category means
the evidence needed to verify the requirement, not the technology words it contains.

Apply this precedence and stop at the first matching rule:
1. process: a recurring organizational activity, governance artifact, or lifecycle
   practice (for example SDLC, training, threat-model cadence, inventory/SBOM upkeep,
   remediation timeframes, discovery/inventory programs, retention-classification schedules).
2. infrastructure: server, gateway, OS, deployment, CI/CD, or runtime configuration
   (for example HTTP response headers, TLS versions/ciphers, redirects, default OS/DB
   accounts, reverse proxies, load-balancer caches, application-cache/server-cache behavior,
   dependency currency, automated build/deploy pipelines, key vault deployment).
3. code: source-level implementation behavior or exact algorithm/validation logic
   (for example encoding order, cookie flags, parameterized queries, per-object checks,
   token generation/validation, GraphQL depth limits, rate-limit counters, per-user action limits,
   URL-vs-body placement enforcement, password hashing, cryptographic RNG usage, log encoding).
4. design: a TSD-visible architecture, documented policy/rule, data flow, trust boundary,
   protocol choice, or access-control model.

Important boundaries:
- A requirement explicitly about documentation, defining rules, or architectural placement
  is design unless it is an ongoing organizational process.
- A runtime HTTP header or server/cache/deployment setting is infrastructure even when an
  application can emit it dynamically.
- A requirement to prove enforcement, exact limits, or implementation flow is code unless
  it explicitly asks for the policy/documentation itself.
- If the requirement is about classifying data, documented timeout/risk decisions, tenant-isolation
  model, or controls defined according to security documentation, classify it as design.
- If the requirement is about discovering all cryptography, keeping an inventory, or deleting data
  on a defined governance schedule, classify it as process.

Anchor examples:
- "cookie-based session tokens have the Secure/HttpOnly attribute set" => code
- "build and deployment processes are secure and repeatable" => infrastructure
- "all components are up to date using a dependency checker" => infrastructure
- "anti-caching headers" / "HSTS" / "CORS allowlist header" / "HTTP to HTTPS redirect behavior" => infrastructure
- "query allowlist, depth limiting, or query cost analysis" => code
- "session token verification uses a trusted backend service" => code
- "terminate all other active sessions after auth-factor change" => code
- "sensitive data only in body/header, not URL/query string" => code
- "cryptographic discovery mechanisms identify all cryptography" => process
- "sensitive information is subject to data retention classification and deletion schedule" => process
- "classified into protection levels" / "cross-tenant controls model" / "inactivity timeout according to risk analysis" => design

Input JSON array (preserve every index):
{items_json}

Respond with exactly:
{{"items":[{{"index":0,"requirement_category":"design"}}]}}
"""

# ---------------------------------------------------------------------------
# Standard Screening Prompt
# ---------------------------------------------------------------------------

STANDARD_SCREENING_SYSTEM_PROMPT = """\
You are a security compliance analyst. Determine whether a document is a Security Standard — \
a document containing NUMBERED, VERIFIABLE security controls structured for compliance or audit purposes.

A security standard MUST have:
- Numbered controls in formats like "1.1.1 Verify that...", "REQ-INP-01", "PCI-3.4", "CTRL-5"
- Imperative/compliance language: "shall", "must", "verify that", "is required to"
- Sections organized for audit or implementation (e.g. authentication, cryptography, access control)

Examples of security standards: OWASP ASVS, NIST 800-53, ISO 27001, PCI-DSS, CIS Benchmarks, \
internal corporate security policies with numbered controls.

NOT a security standard (reject these):
- Vendor whitepapers or product brochures that discuss security without numbered controls
- Security advisories or CVE reports
- Blog posts, training materials, or awareness content
- Policy overview documents listing security domains without verifiable controls
- Legal contracts, financial reports, or unrelated documents
"""


def build_standard_screening_prompt(document_sample: str) -> str:
    return f"""Does this document sample come from a Security Standard with numbered, verifiable controls?

Look for: control IDs (X.Y.Z format, REQ-X-XX, PCI-X.X), compliance language ("Verify that", "shall", "must"), \
structured sections for audit/implementation.

<sample>
{document_sample}
</sample>

Respond with a JSON object:
{{
    "is_security_standard": bool,
    "confidence": float,
    "document_type": str,
    "reasoning": str
}}
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

VALID_DIAGRAM_TYPES = ["data_flow", "sequence", "architecture"]

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
) -> str:
    valid_types = VALID_DIAGRAM_TYPES

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
