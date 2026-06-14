import logging
import os
from pathlib import Path
from typing import Literal, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    ENVIRONMENT: Literal["dev", "prod", "test"] = Field(default="dev")
    DEBUG: bool = Field(default=False)
    PROJECT_NAME: str = Field(default="Automated SDR API")
    PROJECT_DESCRIPTION: str = Field(default="API for the Security Design Review application.")
    PROJECT_VERSION: str = Field(default="1.0.0")
    API_V1_STR: str = Field(default="/api/v1")
    SECRET_KEY: str | None = Field(default=None)

    ALLOWED_HOSTS: list[str] = Field(default=["localhost", "127.0.0.1"])
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOWED_ORIGINS: list[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])
    CSRF_TRUSTED_ORIGINS: list[str] = Field(default=["http://localhost:3000", "http://localhost", "https://localhost"])
    
    DATABASE_URL: str = Field(default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}")
    DATABASE_CONNECTION_POOL_SIZE: int = Field(default=20)
    DATABASE_CONNECTION_MAX_AGE: int = Field(default=600)

    REDIS_URL: str = Field(default="redis://redis:6379/0")
    CELERY_BROKER_URL: str | None = Field(default=None)
    CELERY_RESULT_BACKEND: str | None = Field(default=None)
    CELERY_TASK_TIME_LIMIT: int = Field(default=3600)
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(default=3300)

    CACHE_TTL: Dict[str, int] = Field(default={'project_list': 180, 'standard_list': 600})

    DEFAULT_LLM_MODEL: str = Field(default="meta/llama-3.1-70b-instruct")
    NVIDIA_API_KEY: str | None = Field(default=None)
    LITELLM_TIMEOUT: int = Field(default=300)
    LITELLM_RETRIES: int = Field(default=3)
    LITELLM_ENABLE_CACHING: bool = Field(default=False)
    LITELLM_REDIS_URL: str | None = Field(default=None)

    OPENROUTER_API_KEY: str | None = Field(default=None)
    OPENROUTER_DEFAULT_MODEL: str = Field(default="meta-llama/llama-3.1-70b-instruct")
    OPENROUTER_FAST_MODEL: str = Field(default="meta-llama/llama-3.1-8b-instruct")

    AI_MODEL_STANDARD_EXTRACTION: str = Field(default="meta/llama-3.1-8b-instruct")
    AI_MODEL_DIAGRAM_REQUIREMENT_EXTRACTION: str = Field(default="meta/llama-3.1-8b-instruct")
    AI_STANDARD_EXTRACTION_MAX_WORKERS: int = Field(default=3)
    AI_DIAGRAM_REQUIREMENT_EXTRACTION_MAX_CONCURRENCY: int = Field(default=3)
    AI_MODEL_TSD_INGESTION: str = Field(default="meta/llama-3.1-8b-instruct")
    AI_MODEL_ORCHESTRATOR: str = Field(default="meta/llama-3.1-70b-instruct")
    AI_MODEL_CONTRACT_SYNTHESIZER: str = Field(default="meta/llama-3.1-70b-instruct")
    AI_MODEL_HUNTER: str = Field(default="meta/llama-3.1-70b-instruct")
    AI_MODEL_CRITIC: str = Field(default="meta/llama-3.1-70b-instruct")
    AI_MODEL_MEDIATOR: str = Field(default="meta/llama-3.1-70b-instruct")
    AI_MODEL_VISION: str = Field(default="meta/llama-3.2-90b-vision-instruct")
    AI_MODEL_CODING_GRAPH: str = Field(default="meta/llama-3.1-70b-instruct")
    AI_MODEL_EMBEDDING: str = Field(default="nvidia/nv-embedqa-e5-v5")
    AI_MODEL_FALLBACK: str = Field(default="meta/llama-3.1-8b-instruct")
    AI_MODEL_LONG_CONTEXT: str = Field(default="meta/llama-3.1-70b-instruct")
    AI_MODEL_TSD_ASVS_LEVEL_CLASSIFICATION: str = Field(default="meta/llama-3.1-8b-instruct")
    AI_MODEL_PARENT_APPLICABILITY: str = Field(default="meta/llama-3.1-8b-instruct")

    AI_LLM_MAX_RETRIES: int = Field(default=3)
    AI_LLM_RETRY_INITIAL_DELAY_SECONDS: float = Field(default=2.0)
    AI_LLM_RETRY_BACKOFF_MULTIPLIER: float = Field(default=2.0)
    AI_LLM_RETRYABLE_STATUS_CODES: list[int] = Field(default=[408, 429, 500, 502, 503, 504, 524])
    AI_LLM_FALLBACK_ON_RETRY_EXHAUSTED: bool = Field(default=True)
    AI_STANDARD_EXTRACTION_FALLBACK_PROVIDER: str = Field(default="none")
    AI_NVIDIA_RPM_LIMIT: int = Field(default=4)
    AI_NVIDIA_429_COOLDOWN_SECONDS: float = Field(default=5.0)
    AI_OPENROUTER_RPM_LIMIT: int = Field(default=12)

    AI_DEBATE_SCOPE_STRATIFIED_HUNTING_ENABLED: bool = Field(default=False)
    AI_DEBATE_SCOPE_CHUNK_THRESHOLD: int = Field(default=15)
    AI_DEBATE_SCOPE_TOKEN_THRESHOLD: int = Field(default=7000)
    AI_DEBATE_SCOPE_MAX_GROUPS: int = Field(default=4)
    AI_DEBATE_MAX_HUNTER_CALLS_PER_PARAMETER: int = Field(default=8)
    AI_DEBATE_MAX_DEBATE_ROUNDS: int = Field(default=1)
    AI_DEBATE_CRITIC_AUTO_UPHOLD_STRONG_NOT_MET: bool = Field(default=True)
    AI_DEBATE_PARALLEL_TIMEOUT_SECONDS: int = Field(default=180)

    AI_BATCH_DEBATE_ENABLED: bool = Field(default=True)
    AI_BATCH_DEBATE_BATCH_SIZE: int = Field(default=3)
    AI_BATCH_DEBATE_MAX_CONCURRENCY: int = Field(default=3)
    AI_BATCH_DEBATE_CONFIDENCE_THRESHOLD: float = Field(default=0.75)
    AI_BATCH_DEBATE_SOFT_CONFIDENCE_THRESHOLD: float = Field(default=0.65)
    AI_BATCH_DEBATE_FALLBACK_ENABLED: bool = Field(default=True)
    AI_BATCH_DEBATE_PARENT_CONTEXT_CACHE_ENABLED: bool = Field(default=True)
    AI_BATCH_DEBATE_REQUIRE_CITATIONS_FOR_NOT_MET: bool = Field(default=False)
    AI_BATCH_DEBATE_UNGROUNDED_NOT_MET_POLICY: str = Field(default="selective_fallback")
    AI_PARENT_APPLICABILITY_ENABLED: bool = Field(default=True)
    AI_PARENT_APPLICABILITY_MAX_CHILD_TEXTS: int = Field(default=8)
    AI_PARENT_APPLICABILITY_CONFIDENCE_THRESHOLD: float = Field(default=0.7)
    AI_PARENT_APPLICABILITY_FALLBACK_MODE: str = Field(default="assume_applicable")

    AI_VISION_ENABLED: bool = Field(default=True)
    AI_VISION_DIAGRAM_ANALYSIS_ENABLED: bool = Field(default=True)
    AI_VISION_MAX_CONCURRENCY: int = Field(default=2)
    AI_VISION_MIN_DIAGRAM_BYTES: int = Field(default=512)
    AI_VISION_DIAGRAM_REQUIREMENTS_MAX_ITEMS: int = Field(default=15)

    AI_RAPTOR_SUMMARY_MAX_CONCURRENCY: int = Field(default=3)
    AI_RAPTOR_EMBED_MAX_CONCURRENCY: int = Field(default=4)
    AI_GRAPH_EXTRACTION_MAX_CONCURRENCY: int = Field(default=4)
    AI_GRAPH_EMBED_MAX_CONCURRENCY: int = Field(default=4)
    AI_EMBEDDING_BATCH_SIZE: int = Field(default=32)
    AI_GRAPH_EMBED_BATCH_SIZE: int = Field(default=32)
    AI_RAPTOR_EMBED_BATCH_SIZE: int = Field(default=32)
    AI_TSD_CONTENT_FILTER_ENABLED: bool = Field(default=True)
    AI_TSD_CONTENT_FILTER_MODE: str = Field(default="conservative")
    AI_TSD_CONTENT_FILTER_MIN_SCORE: int = Field(default=1)

    STATIC_ROOT: str = Field(default=str(BASE_DIR / 'staticfiles'))
    MEDIA_ROOT: str = Field(default=str(BASE_DIR / 'media'))
    
    MINIO_ENDPOINT: str = Field(default="minio:9000")
    MINIO_ACCESS_KEY: str = Field(default="admin")
    MINIO_SECRET_KEY: str = Field(default="password123")
    MINIO_SECURE: bool = Field(default=False)
    MINIO_BUCKET_NAME: str = Field(default="sdr-media")

    PRINT_SETTINGS_ENV: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=f".env.{ENVIRONMENT}" if Path(f".env.{ENVIRONMENT}").exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True
    )

    def print_active_settings(self) -> None:
        if not self.PRINT_SETTINGS_ENV:
            return

        print('\n' + '=' * 50)
        print('ACTIVE FASTAPI SETTINGS REPORT')
        print('=' * 50)

        sensitive_keywords = {'SECRET', 'KEY', 'PASSWORD', 'URL', 'TOKEN', 'ARN'}
        settings_dict = self.model_dump()

        for key, value in settings_dict.items():
            if any(secret in key.upper() for secret in sensitive_keywords) and value:
                display_value = "********"
            else:
                display_value = value
            print(f"{key}: {display_value}")
        
        print('=' * 50 + '\n')

settings = Settings()
