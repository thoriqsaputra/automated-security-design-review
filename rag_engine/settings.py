# rag_engine/settings.py
from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding

def configure_rag_settings():
    Settings.llm = LiteLLM(
        model="openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        # model="openrouter/qwen/qwen3.6-plus:free",
        temperature=0.1,
    )
    
    Settings.embed_model = LiteLLMEmbedding(
        # model_name="gemini/gemini-embedding-2-preview",
        model_name="openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free",
        dimensions=768
    )
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 64
