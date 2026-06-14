from __future__ import annotations

_CITATION_SCHEMA = """
{
  "block_id": "p3_b12",          // TextBlock or DiagramBlock ID from TSD ingestor
  "page_number": 3,              // 1-based page number
  "quoted_text": "...",          // verbatim excerpt supporting this citation
  "bbox": {                      // bounding box in PDF coordinate space (nullable)
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
