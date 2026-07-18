import logging
import time
from typing import Dict, List, Optional, Any, Generator, Union

from sdr.core.config import settings
from sdr.apps.ai.client.base import AIResponse, AIProvider
from sdr.apps.ai.client.llm_logger import log_llm_interaction
from sdr.apps.ai.client.nvidia.service import NVIDIAAIService
from sdr.apps.ai.client.openrouter.service import OpenRouterAIService
from sdr.apps.ai.client.routellm.service import RouteLLMAIService
from sdr.apps.ai.client import usage_tracker
from sdr.apps.ai.client.session import get_current_request_metadata, merge_request_metadata

logger = logging.getLogger(__name__)

class AIServiceManager:
    def __init__(self):
        self.nvidia_service = None
        self.openrouter_service = None
        self.routellm_service = None
        self._initialize_services()
        

    
    def _initialize_services(self):
        nv_service = NVIDIAAIService()
        if nv_service.is_available():
            self.nvidia_service = nv_service
            logger.info("AIServiceManager: Initialized NVIDIA service")
        else:
            logger.warning("AIServiceManager: NVIDIA API Key missing. Service unavailable.")
            
        or_service = OpenRouterAIService()
        if or_service.is_available():
            self.openrouter_service = or_service
            logger.info("AIServiceManager: Initialized OpenRouter service")
        else:
            logger.warning("AIServiceManager: OpenRouter API Key missing. Service unavailable.")
            
        routellm = RouteLLMAIService()
        if routellm.is_available():
            self.routellm_service = routellm
            logger.info("AIServiceManager: Initialized RouteLLM service")
        else:
            logger.warning("AIServiceManager: RouteLLM API Key missing. Service unavailable.")

    def _get_model_setting(self, component: str) -> Optional[str]:
        models = {
            'standard_extraction': getattr(settings, 'AI_MODEL_STANDARD_EXTRACTION', 'meta/llama-3.1-8b-instruct'),
            'standard_category_validation': getattr(
                settings,
                'AI_MODEL_STANDARD_CATEGORY_VALIDATION',
                getattr(settings, 'AI_MODEL_EVAL_JUDGE', 'meta/llama-3.1-70b-instruct'),
            ),
            'diagram_requirement_extraction': getattr(settings, 'AI_MODEL_DIAGRAM_REQUIREMENT_EXTRACTION', 'meta/llama-3.1-8b-instruct'),
            'tsd_ingestion': getattr(settings, 'AI_MODEL_TSD_INGESTION', 'meta/llama-3.1-8b-instruct'),
            'vision': getattr(settings, 'AI_MODEL_VISION', 'meta/llama-3.2-90b-vision-instruct'),
            'vision_hunter': getattr(settings, 'AI_MODEL_VISION_HUNTER', None) or getattr(settings, 'AI_MODEL_VISION', 'meta/llama-3.2-90b-vision-instruct'),
            'vision_critic': getattr(settings, 'AI_MODEL_VISION_CRITIC', None) or getattr(settings, 'AI_MODEL_VISION', 'meta/llama-3.2-90b-vision-instruct'),
            'vision_mediator': getattr(settings, 'AI_MODEL_VISION_MEDIATOR', None) or getattr(settings, 'AI_MODEL_VISION', 'meta/llama-3.2-90b-vision-instruct'),
            'vision_extractor': getattr(settings, 'AI_MODEL_VISION_EXTRACTOR', None) or getattr(settings, 'AI_MODEL_VISION', 'meta/llama-3.2-90b-vision-instruct'),
            'vision_reasoner': getattr(settings, 'AI_MODEL_VISION_REASONER', None) or getattr(settings, 'AI_MODEL_VISION', 'meta/llama-3.2-90b-vision-instruct'),
            'hunter': getattr(settings, 'AI_MODEL_HUNTER', 'deepseek-ai/DeepSeek-V4-Flash|routellm'),
            'critic': getattr(settings, 'AI_MODEL_CRITIC', 'deepseek-ai/DeepSeek-V4-Flash|routellm'),
            'mediator': getattr(settings, 'AI_MODEL_MEDIATOR', 'deepseek-ai/DeepSeek-V4-Flash|routellm'),
            'embedding': getattr(settings, 'AI_MODEL_EMBEDDING', 'nvidia/nv-embedqa-e5-v5'),
            'fallback': getattr(settings, 'AI_MODEL_FALLBACK', 'meta/llama-3.1-8b-instruct'),
            'long_context': getattr(settings, 'AI_MODEL_LONG_CONTEXT', 'meta/llama-3.1-70b-instruct'),
            'query_expansion': getattr(settings, 'AI_MODEL_QUERY_EXPANSION', 'meta/llama-3.1-8b-instruct'),
            'eval_judge': getattr(settings, 'AI_MODEL_EVAL_JUDGE', 'meta/llama-3.1-70b-instruct'),
        }
        return models.get(component)

    def _get_provider_for_component(self, component: str) -> AIProvider:
        if not component:
            return AIProvider.NVIDIA
            
        setting_val = self._get_model_setting(component)
        if setting_val and '|' in setting_val:
            _, provider_str = setting_val.split('|', 1)
            provider_str = provider_str.strip().lower()
            if provider_str == 'openrouter':
                return AIProvider.OPENROUTER
            elif provider_str == 'routellm':
                return AIProvider.ROUTELLM
            elif provider_str == 'nvidia':
                return AIProvider.NVIDIA

        return AIProvider.NVIDIA

    def _get_service_for_provider(self, provider: AIProvider):
        if provider == AIProvider.ROUTELLM and self.routellm_service:
            return self.routellm_service
        if provider == AIProvider.OPENROUTER and self.openrouter_service:
            return self.openrouter_service
        if provider == AIProvider.NVIDIA and self.nvidia_service:
            return self.nvidia_service
            
        # Fallback to whichever is available
        if self.openrouter_service:
            return self.openrouter_service
        if self.routellm_service:
            return self.routellm_service
        if self.nvidia_service:
            return self.nvidia_service
        return None

    def get_model_for_component(self, component: str, provider: AIProvider) -> str:
        setting_val = self._get_model_setting(component)
        if setting_val:
            if '|' in setting_val:
                model_str, provider_str = setting_val.split('|', 1)
                provider_str = provider_str.strip().lower()
                if (
                    (provider == AIProvider.OPENROUTER and provider_str == "openrouter")
                    or (provider == AIProvider.ROUTELLM and provider_str == "routellm")
                    or (provider == AIProvider.NVIDIA and provider_str == "nvidia")
                ):
                    return model_str.strip()
            elif provider == AIProvider.NVIDIA:
                return setting_val.strip()
            
        # Global fallback based on provider
        if provider == AIProvider.OPENROUTER:
            if component in {"standard_extraction"}:
                return getattr(settings, 'OPENROUTER_FAST_MODEL', 'meta-llama/llama-3.1-8b-instruct')
            return getattr(settings, 'OPENROUTER_DEFAULT_MODEL', 'meta-llama/llama-3.1-70b-instruct')
        if provider == AIProvider.ROUTELLM:
            if component in {"standard_extraction"}:
                return getattr(settings, 'ROUTELLM_FAST_MODEL', 'gpt-4o-mini')
            return getattr(settings, 'ROUTELLM_DEFAULT_MODEL', 'gpt-4o')
        
        return getattr(settings, 'DEFAULT_LLM_MODEL', 'meta/llama-3.1-70b-instruct')

    def _build_error_response(self, provider: AIProvider, error: str) -> AIResponse:
        return AIResponse(
            content="",
            model="unknown",
            provider=provider,
            error=error,
        )

    def _invoke_service(
        self,
        service,
        provider: AIProvider,
        component: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Union[AIResponse, Generator[str, None, None]]:
        request_kwargs = dict(kwargs)
        resolved_metadata = merge_request_metadata(request_kwargs.get("metadata"))
        if resolved_metadata:
            request_kwargs["metadata"] = resolved_metadata
        else:
            request_kwargs.pop("metadata", None)
        if 'model' not in request_kwargs or not request_kwargs['model']:
            if component:
                request_kwargs['model'] = self.get_model_for_component(component, provider)
            else:
                request_kwargs['model'] = service.default_model
        request_kwargs.pop('component', None)

        started_at = time.perf_counter()
        result = service.chat_completion(**request_kwargs)

        if isinstance(result, AIResponse):
            duration_seconds = time.perf_counter() - started_at
            log_llm_interaction(
                component=component,
                provider=provider,
                request_kwargs=request_kwargs,
                response=result,
                duration_seconds=duration_seconds,
            )
            usage_tracker.record(
                get_current_request_metadata().get("session_id"),
                result.usage,
                duration_seconds,
                result.error,
            )
            return result

        return self._wrap_streaming_response(
            result,
            component=component,
            provider=provider,
            request_kwargs=request_kwargs,
            started_at=started_at,
        )

    def _wrap_streaming_response(
        self,
        stream: Generator[str, None, None],
        *,
        component: Optional[str],
        provider: AIProvider,
        request_kwargs: Dict[str, Any],
        started_at: float,
    ) -> Generator[str, None, None]:
        chunks: List[str] = []
        try:
            for chunk in stream:
                if chunk:
                    chunks.append(chunk)
                yield chunk
        finally:
            duration_seconds = time.perf_counter() - started_at
            log_llm_interaction(
                component=component,
                provider=provider,
                request_kwargs=request_kwargs,
                streamed_content="".join(chunks),
                duration_seconds=duration_seconds,
            )
            # Streamed responses don't carry a parsed usage object today — still
            # record the duration so streamed calls count toward pipeline time,
            # even though their token contribution is unknown.
            usage_tracker.record(
                get_current_request_metadata().get("session_id"),
                None,
                duration_seconds,
            )

    def chat_completion_with_fallback(self, *args, **kwargs) -> Union[AIResponse, Generator[str, None, None]]:
        component = kwargs.get('component', None)
        target_provider = self._get_provider_for_component(component)
        service = self._get_service_for_provider(target_provider)
        
        if not service:
            if kwargs.get('stream'):
                def err(): yield "No AI providers available"
                return err()
            return self._build_error_response(target_provider, "No AI providers available")

        if isinstance(service, OpenRouterAIService):
            actual_provider = AIProvider.OPENROUTER
        elif isinstance(service, RouteLLMAIService): # Need to import RouteLLMAIService if not already, but it's at the top
            actual_provider = AIProvider.ROUTELLM
        else:
            actual_provider = AIProvider.NVIDIA

        request_kwargs: Dict[str, Any] = dict(kwargs)
        if args:
            request_kwargs["messages"] = args[0]

        return self._invoke_service(
            service,
            actual_provider,
            component,
            request_kwargs,
        )
    
    def get_embedding_with_fallback(self, text: str, model: Optional[str] = None, dimensions: int = 1024) -> List[float]:
        target_provider = self._get_provider_for_component('embedding')
        service = self._get_service_for_provider(target_provider)
        if not service:
            logger.error(f"No AI provider available for embedding (target: {target_provider}).")
            return []
        
        if not model:
            model = self.get_model_for_component('embedding', target_provider)
            
        return service.get_embedding(text=text, model=model, dimensions=dimensions)

    def get_embeddings_with_fallback(self, texts: List[str], model: Optional[str] = None, dimensions: int = 1024) -> List[List[float]]:
        target_provider = self._get_provider_for_component('embedding')
        service = self._get_service_for_provider(target_provider)
        if not service:
            logger.error(f"No AI provider available for embeddings (target: {target_provider}).")
            return [[] for _ in texts]

        if not model:
            model = self.get_model_for_component('embedding', target_provider)

        return service.get_embeddings(texts=texts, model=model, dimensions=dimensions)

ai_service_manager = AIServiceManager()
