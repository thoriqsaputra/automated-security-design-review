import logging
from typing import Dict, List, Optional, Any, Generator, Union

import time
import json
from billiard.exceptions import SoftTimeLimitExceeded
from sdr.core.config import settings
from openai import OpenAI, APIError, APIConnectionError

from sdr.apps.ai.client.base import (
    AIServiceInterface, AIResponse, AIProvider, AIModel,
    convert_to_openai_messages
)
from sdr.apps.ai.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)
_DEFAULT_MAX_TOKENS = 4000

class OpenRouterAIService(AIServiceInterface):
    def __init__(self):
        self.api_key = getattr(settings, 'OPENROUTER_API_KEY', None)
        self.default_model = getattr(settings, 'OPENROUTER_DEFAULT_MODEL', 'meta-llama/llama-3.1-70b-instruct')
        self.fast_model = getattr(settings, 'OPENROUTER_FAST_MODEL', 'meta-llama/llama-3.1-8b-instruct')
        
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
        self.rate_limiter = get_rate_limiter("openrouter")

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        image_bytes: Optional[bytes] = None,
        image_format: str = "png",
        stream: bool = False,
        **kwargs
    ) -> Union[AIResponse, Generator[str, None, None]]:
        
        model_to_use = model or self.default_model
        
        if not self.client:
            logger.error("OpenRouter API key not configured.")
            if stream:
                def err(): yield "Error: OpenRouter API key missing"
                return err()
            return AIResponse(content="", model=model_to_use, provider=AIProvider.OPENROUTER, error="API key missing")

        self.rate_limiter.acquire()
        
        try:
            # Convert messages to OpenAI format (handling images if any)
            oa_messages = convert_to_openai_messages(messages, image_bytes, image_format)
            
            request_kwargs = {
                "model": model_to_use,
                "messages": oa_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
                "stream": stream,
            }

            metadata = kwargs.get("metadata")
            if metadata:
                request_kwargs["extra_body"] = {"metadata": metadata}
            
            if "top_p" in kwargs:
                request_kwargs["top_p"] = kwargs["top_p"]
            
            if response_format:
                request_kwargs["response_format"] = response_format
                
            if stream:
                def generate():
                    try:
                        response = self.client.chat.completions.create(**request_kwargs)
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content is not None:
                                yield chunk.choices[0].delta.content
                    except Exception as e:
                        logger.error(f"OpenRouter streaming error: {e}")
                        raise
                return generate()
            
            # Standard request with custom retry for JSONDecodeError
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(**request_kwargs)
                    content_text = response.choices[0].message.content or ""
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                    
                    # Extract usage if available
                    usage = None
                    if response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                        
                    return AIResponse(
                        content=content_text,
                        model=model_to_use,
                        provider=AIProvider.OPENROUTER,
                        usage=usage,
                        raw_usage=usage,
                        finish_reason=finish_reason,
                    )
                except (json.JSONDecodeError, APIConnectionError, APIError) as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"OpenRouter API glitch (attempt {attempt+1}/3): {e}")
                    time.sleep(2 ** attempt)

        except SoftTimeLimitExceeded:
            logger.warning("Completion interrupted by SoftTimeLimitExceeded; propagating.")
            raise
        except Exception as e:
            logger.error(f"OpenRouter chat completion error: {e}")
            if stream:
                def error_generator(): yield f"Error: {str(e)}"
                return error_generator()
            return AIResponse(
                content="",
                model=model_to_use,
                provider=AIProvider.OPENROUTER,
                error=str(e),
                status_code=getattr(e, "status_code", None),
            )
    
    def get_embedding(self, text: str, model: Optional[str] = None, dimensions: int = 1024) -> List[float]:
        model_to_use = model or getattr(settings, 'AI_MODEL_EMBEDDING', 'mxbai-embed-large-v1')
        try:
            client = OpenAI(base_url="http://embeddings:80/v1", api_key="tei")
            request_kwargs = {"model": model_to_use, "input": text}
            if dimensions and model_to_use not in ["text-embedding-ada-002"]:
                request_kwargs["dimensions"] = dimensions
            resp = client.embeddings.create(**request_kwargs)
            return resp.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding via local endpoint with {model_to_use}: {e}", exc_info=True)
            return []

    def get_embeddings(self, texts: List[str], model: Optional[str] = None, dimensions: int = 1024) -> List[List[float]]:
        if not texts:
            return []
            
        model_to_use = model or getattr(settings, 'AI_MODEL_EMBEDDING', 'mxbai-embed-large-v1')
        embedding_batch_size = max(1, int(getattr(settings, "AI_EMBEDDING_BATCH_SIZE", 32)))

        try:
            client = OpenAI(base_url="http://embeddings:80/v1", api_key="tei")
            vectors = []
            for batch_start in range(0, len(texts), embedding_batch_size):
                text_batch = texts[batch_start:batch_start + embedding_batch_size]
                request_kwargs = {"model": model_to_use, "input": text_batch}
                if dimensions and model_to_use not in ["text-embedding-ada-002"]:
                    request_kwargs["dimensions"] = dimensions
                resp = client.embeddings.create(**request_kwargs)
                vectors.extend([item.embedding for item in sorted(resp.data, key=lambda i: getattr(i, 'index', 0))])
            return vectors
        except Exception as exc:
            logger.error(f"Batch embedding request failed: {exc}", exc_info=True)
            return []
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_available_models(self) -> List[AIModel]:
        provider = AIProvider.OPENROUTER
        return [
            AIModel(name=self.default_model, provider=provider, max_tokens=8192),
            AIModel(name=self.fast_model, provider=provider, max_tokens=8192),
        ]
