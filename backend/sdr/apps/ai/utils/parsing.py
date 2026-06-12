import logging
import re
logger = logging.getLogger(__name__)


def strip_markdown_code_blocks(text: str) -> str:
    """Removes markdown formatting if the LLM wraps the JSON in ```json ... ``` """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def strip_thinking_block(text: str) -> str:
    """Removes an optional <thinking>...</thinking> or <think>...</think> block from model output."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()