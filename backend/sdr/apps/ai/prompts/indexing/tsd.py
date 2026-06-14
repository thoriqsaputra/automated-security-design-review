from __future__ import annotations


# ---------------------------------------------------------------------------
# RAPTOR Document Summarisation
# ---------------------------------------------------------------------------

RAPTOR_SUMMARISATION_SYSTEM_PROMPT = (
    "You are a technical writer summarising security "
    "documentation for a compliance review pipeline. "
    "Be precise and preserve all security-relevant details."
)


def build_raptor_summarisation_prompt(
    instruction: str,
    token_budget: int,
    combined_text: str,
) -> str:
    return (
        f"You are summarising sections of a Technical Software Document "
        f"(TSD) for a security compliance review pipeline.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Target length: approximately {token_budget} tokens.\n\n"
        f"Content to summarise:\n\n{combined_text}\n\n"
        f"Output the summary as plain text only. "
        f"No JSON, no bullet points, no markdown headers."
    )


# ---------------------------------------------------------------------------
# GraphRAG Entity/Relation Extraction
# ---------------------------------------------------------------------------

GRAPH_EXTRACTION_SYSTEM_PROMPT = (
    "You are a security architecture analyst extracting "
    "entities and relationships from Technical Software "
    "Documents for a security compliance review pipeline. "
    "Output strictly valid JSON only."
)


def build_graph_extraction_prompt(
    page_text: str,
    page_number: int,
    entity_types: str,
    relation_types: str,
) -> str:
    return f"""\
## TSD PAGE {page_number} — ENTITY AND RELATION EXTRACTION

Extract all named architectural entities and directed security-relevant
relationships from the text below.

### VALID ENTITY TYPES
{entity_types}

### VALID RELATION TYPES
{relation_types}

### OUTPUT SCHEMA
Return a single JSON object with exactly two keys:

{{
  "entities": [
    {{
      "entity_id": "<lowercase_slug_no_spaces>",
      "name": "<original name as it appears in the text>",
      "entity_type": "<one of the valid entity types above>",
      "description": "<short phrase or null>"
    }}
  ],
  "relations": [
    {{
      "source_entity_id": "<entity_id of the source>",
      "target_entity_id": "<entity_id of the target>",
      "relation_type": "<one of the valid relation types above>",
      "description": "<short phrase or null>",
      "is_encrypted": <true | false | null>,
      "requires_auth": <true | false | null>,
      "protocol": "<e.g. TLS 1.2, mTLS, HTTPS, JWT or null>",
      "confidence": <float 0.0-1.0>
    }}
  ]
}}

### RULES
- entity_id must be a lowercase slug: "api_gateway", "user_db", "auth_service"
- Only extract entities explicitly named in the text — do not infer
- Only include relations between entities you have extracted above
- Return minified JSON only, with no prose, comments, markdown, or code fences
- Keep descriptions short phrases, not sentences
- Set is_encrypted / requires_auth to null if the text does not state it
- Set confidence to 1.0 only if the relation is explicitly stated
- If no entities or relations are found, return empty lists

### TEXT
{page_text}
"""


def build_graph_retry_extraction_prompt(
    page_text: str,
    page_number: int,
    entity_types: str,
    relation_types: str,
    failure_hint: str,
) -> str:
    base_prompt = build_graph_extraction_prompt(page_text, page_number, entity_types, relation_types)
    return (
        f"{base_prompt}\n"
        "### RETRY REQUIREMENTS\n"
        f"{failure_hint}"
        "- Return one minified JSON object only.\n"
        "- Do not include markdown, backticks, commentary, or explanations.\n"
        "- Use null, true, and false JSON literals only.\n"
        "- Do not include trailing commas.\n"
        "- Keep each description to a short phrase or null.\n"
        "- Avoid newline characters inside string values.\n"
    )
