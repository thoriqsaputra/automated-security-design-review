from typing import List, Optional, Union, Generator, Any, Dict
from sdr.apps.ai.client.base import AIServiceInterface, AIResponse
from sdr.apps.ai.client.manager import ai_service_manager

def get_ai_service() -> Optional[AIServiceInterface]:
    if ai_service_manager.openrouter_service:
        return ai_service_manager.openrouter_service
    return ai_service_manager.nvidia_service

def chat_completion(*args, **kwargs) -> Union[AIResponse, Generator[str, None, None]]:
    return ai_service_manager.chat_completion_with_fallback(*args, **kwargs)

def get_embedding(text: str, model: Optional[str] = None, dimensions: int = 1024) -> List[float]:
    return ai_service_manager.get_embedding_with_fallback(text=text, model=model, dimensions=dimensions)

def get_embeddings(texts: List[str], model: Optional[str] = None, dimensions: int = 1024) -> List[List[float]]:
    return ai_service_manager.get_embeddings_with_fallback(texts=texts, model=model, dimensions=dimensions)

def get_model_for_component(component: str) -> str:
    # Just ask manager to resolve it for OpenRouter by default as fallback
    from sdr.apps.ai.client.base import AIProvider
    provider = ai_service_manager._get_provider_for_component(component)
    return ai_service_manager.get_model_for_component(component, provider)
