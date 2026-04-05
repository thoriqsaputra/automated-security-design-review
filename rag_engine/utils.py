from typing import List

def embed_text(text: str) -> List[float]:
    """Generates pgvector compatible embeddings via LlamaIndex globally configured LiteLLM."""
    from llama_index.core import Settings
    return Settings.embed_model.get_text_embedding(text)
