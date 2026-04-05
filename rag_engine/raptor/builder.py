import logging
from typing import List
from llama_index.core import Settings

logger = logging.getLogger(__name__)

def summarize_cluster(cluster_texts: List[str]) -> str:
    """Summarizes a group of nodes to form the next RAPTOR layer."""
    combined_text = "\n\n".join(f"Chunk {i+1}: {t}" for i, t in enumerate(cluster_texts))
    prompt = (
        "Synthesize and summarize the following architectural context into a cohesive "
        "high-level system description. Focus on data flows, components, and security boundaries:\n\n"
        f"{combined_text}"
    )
    
    try:
        response = Settings.llm.complete(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Failed to summarize cluster: {e}")
        raise RuntimeError("RAPTOR summarization failed") from e