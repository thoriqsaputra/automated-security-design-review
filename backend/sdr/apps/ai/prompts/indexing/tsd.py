from __future__ import annotations


# ---------------------------------------------------------------------------
# RAPTOR Document Summarisation
# ---------------------------------------------------------------------------

RAPTOR_SUMMARISATION_SYSTEM_PROMPT = (
    "You are a technical writer summarising security "
    "documentation for a compliance review pipeline. "
    "Be precise and preserve all security-relevant details."
)


def build_raptor_leaf_context_prompt(
    doc_title: str,
    section_heading: str,
    chunk_text: str,
) -> str:
    return (
        f"Document: {doc_title}\n"
        f"Section: {section_heading}\n\n"
        f"Here is a chunk from this section:\n{chunk_text}\n\n"
        "Write one or two sentences (maximum 60 words) describing what security topic this chunk covers "
        "and which section it belongs to. This prefix will be prepended to the chunk to improve retrieval. "
        "Do not copy the chunk text verbatim. Be specific about the security mechanism or control described."
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

