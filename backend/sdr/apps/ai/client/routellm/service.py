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
# Broad object-span match used as a fallback when the model wraps its JSON in
# explanatory prose (e.g. "Here is the JSON:\n\n{...}\n\nLet me know...").
# OpenAI's response_format=json_object is a hard constraint on OpenAI's own
# API, but RouteLLM's pass-through to other model families (Claude, etc.)
# doesn't enforce it the same way, so those models can add commentary around
# an otherwise-valid JSON object despite the "Return ONLY a JSON object"
# instruction. Greedy first-'{'-to-last-'}' is safe here because these judge
# prompts never legitimately contain other brace-delimited text.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_markdown_json_fence(content: str) -> str:
    match = _MARKDOWN_JSON_FENCE_RE.match(content.strip())
    return match.group(1) if match else content


def _extract_json_object(content: str) -> str:
    """Best-effort extraction of a JSON object from a possibly prose-wrapped
    LLM response. Tries the content as-is (after fence-stripping) first, then
    falls back to pulling the outermost {...} span out of surrounding text."""
    stripped = _strip_markdown_json_fence(content)
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(content)
    if match:
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return stripped


def _has_image_payload(
    image_bytes: Optional[bytes] = None,
    image_payloads: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    if image_bytes:
        return True
    return any(payload.get("image_bytes") for payload in (image_payloads or []))


def _annotate_routellm_error(error: Exception, *, model: str, has_image: bool) -> str:
    message = str(error)
    lowered = message.lower()
    if has_image and "does not support image uploads" in lowered:
        return (
            f"RouteLLM model '{model}' does not support image uploads. "
            "Use an image-capable model/provider for multimodal requests."
        )
    if "invalid model" in lowered:
        return (
            f"RouteLLM rejected model '{model}' as unsupported. "
            "Use a RouteLLM-supported model name for this provider."
        )
    return message

class RouteLLMAIService(AIServiceInterface):
    def __init__(self):
        self.api_key = getattr(settings, 'ROUTELLM_API_KEY', None)
        self.default_model = getattr(settings, 'ROUTELLM_DEFAULT_MODEL', 'gpt-4o')
        self.fast_model = getattr(settings, 'ROUTELLM_FAST_MODEL', 'gpt-4o-mini')
        self.timeout_seconds = max(1, int(getattr(settings, 'ROUTELLM_TIMEOUT_SECONDS', 180)))
        
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                base_url="https://routellm.abacus.ai/v1",
                api_key=self.api_key,
                timeout=self.timeout_seconds,
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
        image_payloads: Optional[List[Dict[str, Any]]] = None,
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
        has_image = _has_image_payload(image_bytes=image_bytes, image_payloads=image_payloads)
        
        try:
            # Convert messages to OpenAI format (handling images if any)
            oa_messages = convert_to_openai_messages(
                messages,
                image_bytes,
                image_format,
                image_payloads=image_payloads,
            )
            
            request_kwargs = {
                "model": model_to_use,
                "messages": oa_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
                "stream": stream,
            }

            if "top_p" in kwargs:
                request_kwargs["top_p"] = kwargs["top_p"]

            reasoning = kwargs.get("reasoning")
            metadata = kwargs.get("metadata")
            if reasoning or metadata:
                extra_body: Dict[str, Any] = {}
                if reasoning:
                    extra_body["reasoning"] = reasoning
                if metadata:
                    extra_body["metadata"] = metadata
                request_kwargs["extra_body"] = extra_body

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
            
            request_attempts = max(1, int(kwargs.get("request_attempts", 3)))
            request_timeout_seconds = max(
                1.0, float(kwargs.get("request_timeout_seconds", self.timeout_seconds))
            )
            transport_retries = max(0, int(kwargs.get("transport_retries", 2)))
            request_client = self.client
            if hasattr(request_client, "with_options"):
                request_client = request_client.with_options(
                    timeout=request_timeout_seconds,
                    max_retries=transport_retries,
                )

            # Standard request with custom retry for JSONDecodeError
            for attempt in range(request_attempts):
                try:
                    response = request_client.chat.completions.create(**request_kwargs)
                    content = response.choices[0].message.content or ""
                    
                    if response_format and response_format.get("type") == "json_object":
                        content = _extract_json_object(content)
                        try:
                            json.loads(content)
                        except json.JSONDecodeError:
                            if attempt < request_attempts - 1:
                                logger.warning(
                                    "RouteLLM JSON parse failed, retrying (%d/%d)",
                                    attempt + 1,
                                    request_attempts,
                                )
                                time.sleep(1)
                                continue
                            else:
                                logger.error("RouteLLM final JSON parse failure")
                                return AIResponse(
                                    content=content,
                                    model=model_to_use,
                                    provider=AIProvider.ROUTELLM,
                                    error=f"Failed to return valid JSON after {request_attempts} attempt(s)."
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
                    annotated_error = _annotate_routellm_error(ae, model=model_to_use, has_image=has_image)
                    logger.error(
                        "RouteLLM API Error model=%s has_image=%s: %s",
                        model_to_use,
                        has_image,
                        annotated_error,
                    )
                    if attempt == request_attempts - 1:
                        return AIResponse(
                            content="",
                            model=model_to_use,
                            provider=AIProvider.ROUTELLM,
                            error=annotated_error,
                        )
                    time.sleep(2)
                except Exception as e:
                    annotated_error = _annotate_routellm_error(e, model=model_to_use, has_image=has_image)
                    logger.error(
                        "RouteLLM unexpected error model=%s has_image=%s: %s",
                        model_to_use,
                        has_image,
                        annotated_error,
                    )
                    return AIResponse(
                        content="",
                        model=model_to_use,
                        provider=AIProvider.ROUTELLM,
                        error=annotated_error,
                    )
                    
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
