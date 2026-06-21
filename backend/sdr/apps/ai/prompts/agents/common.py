from __future__ import annotations

from typing import List, Optional


_CITABLE_CHUNK_RULE = (
    "\nCITATION ELIGIBILITY: Each CONTEXT_CHUNK has a citable attribute. "
    "Only cite block_ids from CONTEXT_CHUNK elements where citable=\"true\" — "
    "these are the ids listed below. CONTEXT_CHUNK elements with citable=\"false\" "
    "are background/retrieval context only and have no precise page or location; "
    "you may use them to understand the document, but if evidence appears only in a "
    "citable=\"false\" chunk, check whether a citable=\"true\" chunk corroborates the "
    "same evidence. If none does, set evidence_found=false and omit the citation rather "
    "than citing a citable=\"false\" chunk's id."
)


def _build_block_ids_block(available_block_ids: Optional[List[str]]) -> str:
    if not available_block_ids:
        return ""
    return (
        _CITABLE_CHUNK_RULE
        + "\n\nVALID CITATION BLOCK IDS (only cite block_ids from this list — "
        "do not invent or guess IDs):\n"
        + ", ".join(sorted(available_block_ids))
    )


_CITATION_SCHEMA = """
{
  "block_id": "p3_b12",          // MUST exactly match the 'id' attribute of the CONTEXT_CHUNK
  "page_number": 3,              // MUST exactly match the 'page_number' attribute of the CONTEXT_CHUNK (or null if missing)
  "quoted_text": "...",          // verbatim excerpt supporting this citation
  "bbox": {                      // MUST exactly match the 'bbox_x0', 'bbox_y0' etc. attributes from the CONTEXT_CHUNK (or null if missing)
    "x0": 72.0,
    "y0": 540.0,
    "x1": 540.0,
    "y1": 560.0
  }
}
"""

_VERDICT_VALUES = '"met" | "not_met" | "na"'

_REASONING_SCHEMA = """
  "assumptions": [
    "<explicit assumption grounded in the supplied context>"
  ],
  "logic_summary": "<concise reasoning summary, <= 5 short sentences>",
  "cot_trace": "<audit-only stepwise reasoning, <= 12 short lines>"
"""

_ASSUMPTIONS_FIRST_RULES = """
- Start with assumptions derived from the supplied material only.
- Keep logic_summary concise and decision-oriented (maximum 4 short sentences).
- Keep cot_trace short and factual for audit; do not repeat the prompt.
- Return strict JSON only.
"""
