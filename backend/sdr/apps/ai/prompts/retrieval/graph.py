from __future__ import annotations


# ---------------------------------------------------------------------------
# Graph Communities Retrieval Summary
# ---------------------------------------------------------------------------

GRAPH_COMMUNITY_SUMMARY_SYSTEM_PROMPT = (
    "You are a software security architecture summarizer. "
    "Output strict JSON."
)


def build_graph_community_summary_prompt(
    key_entities: list,
    key_relationships: list,
    block_ids: list,
) -> str:
    return (
        "Summarize this architecture community in concise report style JSON. "
        "Return keys: title, summary.\n"
        f"Entities: {key_entities}\n"
        f"Relationships: {key_relationships}\n"
        f"Blocks: {block_ids}"
    )
