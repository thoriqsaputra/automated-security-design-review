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

