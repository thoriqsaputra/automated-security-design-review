import json
import logging
from typing import Any, Dict, Generator, List, Optional, Union

import requests

from billiard.exceptions import SoftTimeLimitExceeded
from sdr.core.config import settings
from openai import OpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from sdr.apps.ai.client.base import (
    AIModel,
    AIProvider,
    AIResponse,
    AIServiceInterface,
    convert_to_langchain_messages,
)
from sdr.apps.ai.rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)
_DEFAULT_MAX_TOKENS = 4000
_DEFAULT_EMBEDDING_BATCH_SIZE = 32
_DEFAULT_MAX_RETRIES = 5

class NVIDIAAIService(AIServiceInterface):
    def __init__(self):
        self.nvidia_api_key = getattr(settings, 'NVIDIA_API_KEY', None)
        self.default_model = getattr(settings, 'DEFAULT_LLM_MODEL', 'meta/llama-3.1-70b-instruct')
        
        # Component-specific models
        self.model_standard_extraction = getattr(settings, 'AI_MODEL_STANDARD_EXTRACTION', 'meta/llama-3.1-8b-instruct')
        self.model_tsd_ingestion = getattr(settings, 'AI_MODEL_TSD_INGESTION', 'meta/llama-3.1-8b-instruct')
        self.model_vision = getattr(settings, 'AI_MODEL_VISION', 'meta/llama-3.2-90b-vision-instruct')
        self.model_embedding = getattr(settings, 'AI_MODEL_EMBEDDING', 'nvidia/nv-embedqa-e5-v5')
        
        self.embedding_batch_size = max(1, int(getattr(settings, "AI_EMBEDDING_BATCH_SIZE", _DEFAULT_EMBEDDING_BATCH_SIZE)))
        self.rate_limiter = get_rate_limiter("nvidia")

    def _parse_retry_after(self, response: requests.Response) -> Optional[float]:
        header_value = response.headers.get("Retry-After")
        if not header_value:
            return None
        try:
            return max(0.0, float(header_value))
        except (TypeError, ValueError):
            return None

    def _build_throttle_delay(self, attempt: int, response: requests.Response) -> float:
        retry_after = self._parse_retry_after(response)
        if retry_after is not None:
            return retry_after
        base_cooldown = max(
            0.0,
            float(getattr(settings, "AI_NVIDIA_429_COOLDOWN_SECONDS", 5.0)),
        )
        return base_cooldown + (2 ** attempt)

    def _request_with_retries(
        self,
        *,
        invoke_url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        stream: bool,
    ) -> requests.Response:
        last_response: Optional[requests.Response] = None

        for attempt in range(_DEFAULT_MAX_RETRIES):
            self.rate_limiter.acquire()
            response = requests.post(
                invoke_url,
                headers=headers,
                json=payload,
                stream=stream,
            )
            last_response = response

            if response.status_code == 429:
                retry_delay = self._build_throttle_delay(attempt, response)
                self.rate_limiter.register_throttle(retry_delay)
                if attempt < _DEFAULT_MAX_RETRIES - 1:
                    logger.warning(
                        "NVIDIA API 429 Too Many Requests. Body: %s. Retrying in %.2fs (attempt %d/%d)",
                        response.text,
                        retry_delay,
                        attempt + 1,
                        _DEFAULT_MAX_RETRIES,
                    )
                    continue

                logger.error("NVIDIA API Error 429: %s", response.text)
                response.raise_for_status()

            if response.status_code != 200:
                logger.error(
                    "NVIDIA API Error %s: %s",
                    response.status_code,
                    response.text,
                )
                response.raise_for_status()

            return response

        if last_response is None:
            raise RuntimeError("NVIDIA request loop exited without a response")
        return last_response

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

        try:
            # `messages` is modified by `convert_to_langchain_messages` for image support if needed
            convert_to_langchain_messages(messages, image_bytes, image_format)
            
            invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.nvidia_api_key}",
                "Accept": "text/event-stream" if stream else "application/json"
            }
            
            payload = {
                "model": model_to_use,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
                "stream": stream,
            }
            
            if "top_p" in kwargs:
                payload["top_p"] = kwargs["top_p"]
            
            if response_format:
                payload["response_format"] = response_format
                
            if stream:
                def generate():
                    try:
                        resp = self._request_with_retries(
                            invoke_url=invoke_url,
                            headers=headers,
                            payload=payload,
                            stream=True,
                        )
                        for line in resp.iter_lines():
                            if line:
                                line_str = line.decode("utf-8")
                                if line_str.startswith("data: "):
                                    data_str = line_str[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        if data_json.get("choices") and data_json["choices"][0].get("delta", {}).get("content"):
                                            yield data_json["choices"][0]["delta"]["content"]
                                    except json.JSONDecodeError:
                                        pass
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                        raise
                return generate()
            
            resp = self._request_with_retries(
                invoke_url=invoke_url,
                headers=headers,
                payload=payload,
                stream=False,
            )
            resp_json = resp.json()
            content_text = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract usage if available
            usage = resp_json.get("usage", {})
                
            return AIResponse(
                content=content_text,
                model=model_to_use,
                provider=AIProvider.NVIDIA,
                usage=usage,
                raw_usage=usage,
                status_code=resp.status_code,
            )

        except SoftTimeLimitExceeded:
            logger.warning("Completion interrupted by SoftTimeLimitExceeded; propagating.")
            raise
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            error_code = "rate_limit_exhausted" if status_code == 429 else None
            logger.error(f"Chat completion error: {exc}", exc_info=True)
            if stream:
                def error_generator():
                    yield f"Error: {str(exc)}"
                return error_generator()
            return AIResponse(
                content="",
                model=model_to_use,
                provider=AIProvider.NVIDIA,
                error=str(exc),
                status_code=status_code,
                error_code=error_code,
            )
        except Exception as e:
            logger.error(f"Chat completion error: {e}", exc_info=True)
            if stream:
                def error_generator(): yield f"Error: {str(e)}"
                return error_generator()
            return AIResponse(
                content="",
                model=model_to_use,
                provider=AIProvider.NVIDIA,
                error=str(e),
            )
    
    def get_embedding(self, text: str, model: Optional[str] = None, dimensions: int = 1024) -> List[float]:
        model_to_use = model or self.model_embedding
        
        try:
            # Check if we should use local TEI endpoint
            if "mxbai" in model_to_use or "bge" in model_to_use:
                client = OpenAI(base_url="http://embeddings:80/v1", api_key="tei")
                request_kwargs = {"model": model_to_use, "input": text}
                if dimensions and model_to_use not in ["text-embedding-ada-002"]:
                    request_kwargs["dimensions"] = dimensions
                resp = client.embeddings.create(**request_kwargs)
                return resp.data[0].embedding
            
            # Otherwise use NVIDIA Embeddings
            self.rate_limiter.acquire()
            embedder = NVIDIAEmbeddings(model=model_to_use, nvidia_api_key=self.nvidia_api_key, base_url="https://integrate.api.nvidia.com/v1")
            return embedder.embed_query(text)
            
        except Exception as e:
            logger.error(f"Failed to generate embedding with {model_to_use}: {e}", exc_info=True)
            return []

    def get_embeddings(self, texts: List[str], model: Optional[str] = None, dimensions: int = 1024) -> List[List[float]]:
        if not texts:
            return []

        model_to_use = model or self.model_embedding

        try:
            if "mxbai" in model_to_use or "bge" in model_to_use:
                client = OpenAI(base_url="http://embeddings:80/v1", api_key="tei")
                vectors = []
                for batch_start in range(0, len(texts), self.embedding_batch_size):
                    text_batch = texts[batch_start:batch_start + self.embedding_batch_size]
                    request_kwargs = {"model": model_to_use, "input": text_batch}
                    if dimensions and model_to_use not in ["text-embedding-ada-002"]:
                        request_kwargs["dimensions"] = dimensions
                    resp = client.embeddings.create(**request_kwargs)
                    vectors.extend([item.embedding for item in sorted(resp.data, key=lambda i: getattr(i, 'index', 0))])
                return vectors
            
            # NVIDIA Embeddings (batch)
            self.rate_limiter.acquire()
            embedder = NVIDIAEmbeddings(model=model_to_use, nvidia_api_key=self.nvidia_api_key)
            return embedder.embed_documents(texts)
            
        except Exception as exc:
            logger.warning(f"Batch embedding request failed: {exc}. Falling back to per-text requests.")
            return [self.get_embedding(text=text, model=model_to_use, dimensions=dimensions) for text in texts]
    
    def is_available(self) -> bool:
        return bool(self.nvidia_api_key)
    
    def get_available_models(self) -> List[AIModel]:
        provider = AIProvider.NVIDIA
        return [
            AIModel(name=self.model_standard_extraction, provider=provider, max_tokens=4096),
            AIModel(name=self.model_vision, provider=provider, max_tokens=4096),
            AIModel(name=self.model_embedding, provider=provider, max_tokens=2048, is_embedding_model=True),
        ]
