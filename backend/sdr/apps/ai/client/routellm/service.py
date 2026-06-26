import logging
import re
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
_MARKDOWN_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_markdown_json_fence(content: str) -> str:
    """
    Some models (e.g. Kimi/Moonshot via RouteLLM) ignore response_format
    json_object and wrap their JSON in a markdown code fence anyway.
    """
    match = _MARKDOWN_JSON_FENCE_RE.match(content.strip())
    return match.group(1) if match else content

class RouteLLMAIService(AIServiceInterface):
    def __init__(self):
        self.api_key = getattr(settings, 'ROUTELLM_API_KEY', None)
        self.default_model = getattr(settings, 'ROUTELLM_DEFAULT_MODEL', 'gpt-4o')
        self.fast_model = getattr(settings, 'ROUTELLM_FAST_MODEL', 'gpt-4o-mini')
        
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                base_url="https://routellm.abacus.ai/v1",
                api_key=self.api_key
            )
        self.rate_limiter = get_rate_limiter("routellm")

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
            logger.error("RouteLLM API key not configured.")
            if stream:
                def err(): yield "Error: RouteLLM API key missing"
                return err()
            return AIResponse(content="", model=model_to_use, provider=AIProvider.ROUTELLM, error="API key missing")

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
                        logger.error(f"RouteLLM streaming error: {e}")
                        raise
                return generate()
            
            # Standard request with custom retry for JSONDecodeError
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(**request_kwargs)
                    content = response.choices[0].message.content or ""
                    
                    if response_format and response_format.get("type") == "json_object":
                        content = _strip_markdown_json_fence(content)
                        try:
                            json.loads(content)
                        except json.JSONDecodeError:
                            if attempt < 2:
                                logger.warning(f"RouteLLM JSON parse failed, retrying ({attempt+1}/3)")
                                time.sleep(1)
                                continue
                            else:
                                logger.error("RouteLLM final JSON parse failure")
                                return AIResponse(
                                    content=content,
                                    model=model_to_use,
                                    provider=AIProvider.ROUTELLM,
                                    error="Failed to return valid JSON after 3 attempts."
                                )
                                
                    usage_dict = {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0
                    }
                    
                    return AIResponse(
                        content=content,
                        model=model_to_use,
                        provider=AIProvider.ROUTELLM,
                        usage=usage_dict,
                        raw_usage=dict(response.usage) if response.usage else None,
                        status_code=200,
                        finish_reason=response.choices[0].finish_reason if response.choices else None
                    )
                except (APIError, APIConnectionError) as ae:
                    logger.error(f"RouteLLM API Error: {ae}")
                    if attempt == 2:
                        return AIResponse(content="", model=model_to_use, provider=AIProvider.ROUTELLM, error=str(ae))
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"RouteLLM unexpected error: {e}")
                    return AIResponse(content="", model=model_to_use, provider=AIProvider.ROUTELLM, error=str(e))
                    
        except SoftTimeLimitExceeded:
            logger.error("RouteLLM task timed out.")
            return AIResponse(content="", model=model_to_use, provider=AIProvider.ROUTELLM, error="Task timeout")
        except Exception as e:
            logger.error(f"RouteLLM failed: {e}")
            return AIResponse(content="", model=model_to_use, provider=AIProvider.ROUTELLM, error=str(e))

    def get_embedding(self, text: str, model: str, dimensions: int = 1024) -> List[float]:
        raise NotImplementedError("Embedding not supported by RouteLLM")

    def get_embeddings(self, texts: List[str], model: str, dimensions: int = 1024) -> List[List[float]]:
        raise NotImplementedError("Embeddings not supported by RouteLLM")
        
    def is_available(self) -> bool:
        return bool(self.api_key)
        
    def get_available_models(self) -> List[AIModel]:
        return [
            AIModel(name=self.default_model, provider=AIProvider.ROUTELLM, max_tokens=4000),
            AIModel(name=self.fast_model, provider=AIProvider.ROUTELLM, max_tokens=4000)
        ]
