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

    ROUTELLM_API_KEY: str | None = Field(default=None)
    ROUTELLM_DEFAULT_MODEL: str = Field(default="gpt-4o")
    ROUTELLM_FAST_MODEL: str = Field(default="gpt-4o-mini")

    AI_MODEL_STANDARD_EXTRACTION: str = Field(default="meta/llama-3.1-8b-instruct")
    AI_MODEL_STANDARD_CATEGORY_VALIDATION: str = Field(
        default="gpt-4o|routellm"
    )
    AI_MODEL_DIAGRAM_REQUIREMENT_EXTRACTION: str = Field(default="meta/llama-3.1-8b-instruct")
    AI_STANDARD_EXTRACTION_MAX_WORKERS: int = Field(default=3)
    AI_STANDARD_EXTRACTION_CHUNK_TOKEN_TARGET: int = Field(default=4500)
    AI_STANDARD_CATEGORY_VALIDATION_BATCH_SIZE: int = Field(default=25)
    AI_STANDARD_EXTRACTION_CACHE_TTL_SECONDS: int = Field(default=86400)
    AI_DIAGRAM_REQUIREMENT_EXTRACTION_MAX_CONCURRENCY: int = Field(default=3)
    AI_MODEL_TSD_INGESTION: str = Field(default="meta/llama-3.1-8b-instruct")
    AI_MODEL_HUNTER: str = Field(default="meta/llama-3.1-70b-instruct")
    AI_MODEL_CRITIC: str = Field(default="meta/llama-3.1-70b-instruct")
    AI_MODEL_MEDIATOR: str = Field(default="meta/llama-3.1-70b-instruct")
    # Deliberately a different model family from Hunter/Critic/Mediator so the eval
    # harness doesn't grade answers with the same model that produced them.
    AI_MODEL_EVAL_JUDGE: str = Field(default="anthropic/claude-sonnet-4.5")
    AI_MODEL_VISION: str = Field(default="meta/llama-3.2-90b-vision-instruct")
    AI_MODEL_EMBEDDING: str = Field(default="nvidia/nv-embedqa-e5-v5")
    AI_MODEL_FALLBACK: str = Field(default="meta/llama-3.1-8b-instruct")
    AI_MODEL_LONG_CONTEXT: str = Field(default="meta/llama-3.1-70b-instruct")

    AI_LLM_MAX_RETRIES: int = Field(default=3)
    AI_LLM_RETRY_INITIAL_DELAY_SECONDS: float = Field(default=2.0)
    AI_LLM_RETRY_BACKOFF_MULTIPLIER: float = Field(default=2.0)
    AI_LLM_RETRYABLE_STATUS_CODES: list[int] = Field(default=[408, 429, 500, 502, 503, 504, 524])
    AI_LLM_FALLBACK_ON_RETRY_EXHAUSTED: bool = Field(default=True)
    AI_NVIDIA_RPM_LIMIT: int = Field(default=4)
    AI_NVIDIA_429_COOLDOWN_SECONDS: float = Field(default=5.0)
    AI_OPENROUTER_RPM_LIMIT: int = Field(default=12)
    AI_OPENROUTER_REASONING_EFFORT: str = Field(default="low")
    AI_ROUTELLM_RPM_LIMIT: int = Field(default=12)


    AI_DEBATE_MAX_HUNTER_CALLS_PER_PARAMETER: int = Field(default=8)
    AI_DEBATE_MAX_DEBATE_ROUNDS: int = Field(default=2)
    AI_DEBATE_CRITIC_AUTO_UPHOLD_STRONG_NOT_MET: bool = Field(default=False)
    AI_DEBATE_MAX_COT_TRACE_CHARS_FOR_HANDOFF: int = Field(default=4000)

    AI_BATCH_DEBATE_MAX_CONCURRENCY: int = Field(default=3)
    AI_DEBATE_CONTEXT_SUPPLEMENTAL_BLOCK_LIMIT: int = Field(default=12)
    AI_DEBATE_SUPPLEMENTAL_BLOCK_CANDIDATE_LIMIT: int = Field(default=32)
    AI_DEBATE_SUPPLEMENTAL_BLOCK_WINDOW_BEFORE: int = Field(default=2)
    AI_DEBATE_SUPPLEMENTAL_BLOCK_WINDOW_AFTER: int = Field(default=2)
    AI_DEBATE_SUPPLEMENTAL_BLOCK_CHAR_BUDGET: int = Field(default=3000)
    AI_DEBATE_WARN_CONTEXT_CHUNK_THRESHOLD: int = Field(default=40)
    AI_LLM_LOG_ENABLED: bool = Field(default=False)
    AI_LLM_LOG_DIR: str = Field(default=str(BASE_DIR / "llm_logs"))

    AI_RETRIEVAL_QUERY_EXPANSION_ENABLED: bool = Field(default=True)
    AI_RETRIEVAL_QUERY_EXPANSION_VARIANT_COUNT: int = Field(default=3)
    AI_MODEL_QUERY_EXPANSION: str = Field(default="meta/llama-3.1-8b-instruct")
    AI_RETRIEVAL_SECONDARY_SEARCH_ENABLED: bool = Field(default=True)

    AI_VISION_ENABLED: bool = Field(default=True)
    AI_VISION_MAX_CONCURRENCY: int = Field(default=2)
    AI_VISION_MIN_DIAGRAM_BYTES: int = Field(default=512)
    AI_VISION_DIAGRAM_REQUIREMENTS_MAX_ITEMS: int = Field(default=16)
    AI_VISION_SKIP_MEDIATOR_ON_UPHOLD: bool = Field(default=True)
    AI_VISION_DEBATE_REQUIREMENT_BATCH_SIZE: int = Field(default=10)
    AI_VISION_DEBATE_BATCH_RETRY_LIMIT: int = Field(default=1)
    AI_VISION_DEBATE_BATCH_MAX_CONCURRENCY: int = Field(default=6)
    AI_VISION_DEBATE_REBUTTAL_MAX_CONCURRENCY: int = Field(default=6)
    # Self-consistency voting: run each diagram's full debate this many times
    # and take the majority verdict (DiagramDebateService.run_diagram_debate_voted).
    # Literature-backed lever for stabilizing debate outputs against the kind
    # of batch-dependent instability observed empirically (design 18: identical
    # Hunter hallucination pattern corrected on one diagram, missed entirely on
    # another in the same run). 1 = disabled (single run, current behavior).
    AI_VISION_DEBATE_VOTES: int = Field(default=1)
    # Diagram requirement selection (hybrid BM25+dense over short requirement
    # texts): query segmentation, weighted RRF, priors, and adaptive cutoff.
    AI_VISION_DIAGRAM_QUERY_PAGE_WINDOW_CHARS: int = Field(default=1800)
    AI_VISION_DIAGRAM_RRF_K: int = Field(default=60)
    AI_VISION_DIAGRAM_RRF_VECTOR_WEIGHT: float = Field(default=1.0)
    AI_VISION_DIAGRAM_RRF_BM25_WEIGHT: float = Field(default=1.0)
    AI_VISION_DIAGRAM_TYPE_MATCH_BONUS: float = Field(default=1.0 / 60.0)
    AI_VISION_DIAGRAM_CHAPTER_PRIOR_BONUS: float = Field(default=0.0)
    AI_VISION_DIAGRAM_SCORE_FLOOR_RATIO: float = Field(default=0.0)

    # Extract-then-reason pipeline (DiagramExtractReasonService): decouples
    # perception (Stage 1, vision-in-the-loop structured extraction, voted
    # via self-consistency) from reasoning (Stage 2, text-only verdicts over
    # the extraction) — an alternative to the debate pipeline above.
    AI_VISION_EXTRACTION_VOTES: int = Field(default=3)
    AI_VISION_EXTRACTION_MERGE_THRESHOLD: float = Field(default=0.5)
    AI_VISION_EXTRACTION_FUZZY_MATCH_THRESHOLD: float = Field(default=0.75)
    AI_VISION_EXTRACTION_MAX_CONCURRENCY: int = Field(default=3)
    AI_VISION_REASONER_BATCH_SIZE: int = Field(default=10)
    AI_VISION_REASONER_BATCH_MAX_CONCURRENCY: int = Field(default=6)
    AI_VISION_REASONER_CITATION_RETRY_LIMIT: int = Field(default=1)
    # Batches where the reasoner's response failed entirely (empty/unparseable,
    # nothing answered at all) get a larger retry budget than partial-omission
    # batches — a total failure is a cheap, likely-transient generation hiccup
    # worth retrying harder, not a persistent completeness/reasoning gap.
    AI_VISION_REASONER_FULL_FAILURE_RETRY_LIMIT: int = Field(default=2)
    AI_MODEL_VISION_EXTRACTOR: str = Field(default="")
    AI_MODEL_VISION_REASONER: str = Field(default="")

    AI_RAPTOR_SUMMARY_MAX_CONCURRENCY: int = Field(default=3)
    AI_RAPTOR_EMBED_MAX_CONCURRENCY: int = Field(default=4)
    AI_RAPTOR_LEAF_TOKEN_BUDGET: int = Field(default=800)
    AI_RAPTOR_LEAF_MAX_PAGES: int = Field(default=1)
    AI_RAPTOR_CONTEXTUAL_ENRICHMENT_ENABLED: bool = Field(default=True)
    AI_RAPTOR_CONTEXT_CONCURRENCY: int = Field(default=8)
    AI_RETRIEVAL_HYBRID_MAX_WORKERS: int = Field(default=3)
    AI_RETRIEVAL_MANY_MAX_CONCURRENCY: int = Field(default=2)
    AI_RETRIEVAL_ENABLE_CROSS_ENCODER_RERANK: bool = Field(default=True)
    # Primary fusion step in execute_hybrid: "agreement_boost" (dedupe + additive
    # per-source score bump) or "rrf" (Reciprocal Rank Fusion over each searcher's
    # ranked list). Previously read by AdvancedRetrievalConfig.from_settings() but
    # never declared here, which silently pinned production to agreement_boost.
    AI_RETRIEVAL_FUSION_METHOD: str = Field(default="agreement_boost")
    AI_RETRIEVAL_RRF_K: int = Field(default=60)
    AI_RETRIEVAL_MAX_CONTEXT_CHUNKS: int = Field(default=16)
    # Recall-safety floor: the final hybrid context always includes the top-N
    # candidates from each constituent signal's PRIMARY-query ranking (dense
    # leaf cosine / BM25 / multi-level RAPTOR), inserted at the tail if fusion,
    # tier ordering, reranking, or truncation would otherwise drop them. A
    # hybrid ensemble should never return a context strictly worse than any of
    # its single signals. The dense floor deliberately equals the raptor_low
    # arm's top-k (7): hybrid's final context is then a superset of what the
    # flat leaf-cosine searcher alone would have returned, so hybrid recall is
    # bounded below by raptor_low recall by construction. The raptor floor
    # protects deep enough into hybrid's own multi-level branch (up to 9
    # distinct nodes from search_multi_level's 3-levels*top_k_per_level=3) to
    # rescue the specific raptor_high-only wins observed in practice (ranked
    # up to position 7) — set to 7, not the full 9, because protected_dense +
    # protected_bm25 + protected_raptor must stay comfortably under
    # max_context_chunks=16 or the floors themselves start evicting genuinely
    # good fused candidates (measured regression: raising this to 9 dropped
    # HIT_LEAF from 264->228 in diagnosis). 0 disables a floor.
    AI_RETRIEVAL_PROTECTED_DENSE_TOP_N: int = Field(default=7)
    AI_RETRIEVAL_PROTECTED_BM25_TOP_N: int = Field(default=2)
    AI_RETRIEVAL_PROTECTED_RAPTOR_TOP_N: int = Field(default=7)
    # How many distinct literal leaf descendants to ground per surviving
    # hierarchical_summary candidate — a summary can union many blocks, one
    # leaf isn't enough coverage for a multi-block-expected query, but more
    # than 1 measurably crowds the budget (see note above). Default 1.
    AI_RETRIEVAL_SUMMARY_LEAVES_PER_GROUNDING: int = Field(default=1)
    # Candidate-pool width for the hybrid arm. Dense and BM25 intentionally
    # collect deeper than the final context size so the agreement/rerank stage
    # can rescue literal leaves that were previously absent from the pool.
    AI_RETRIEVAL_HYBRID_DENSE_TOP_K: int = Field(default=20)
    AI_RETRIEVAL_HYBRID_BM25_TOP_K: int = Field(default=20)
    # Cross-encoder rerank is a strong semantic signal, but it must not erase
    # hybrid agreement/fusion. Final rerank score blends normalized cross-
    # encoder relevance with the pre-rerank hybrid score.
    AI_RETRIEVAL_RERANK_SCORE_WEIGHT: float = Field(default=0.72)
    AI_EMBEDDING_BATCH_SIZE: int = Field(default=32)
    AI_RAPTOR_EMBED_BATCH_SIZE: int = Field(default=32)
    AI_TSD_CONTENT_FILTER_ENABLED: bool = Field(default=True)
    AI_TSD_CONTENT_FILTER_MODE: str = Field(default="conservative")
    AI_TSD_CONTENT_FILTER_MIN_SCORE: int = Field(default=1)
    AI_TSD_MIN_BLOCK_TEXT_LENGTH: int = Field(default=10)
    AI_TSD_CONTENT_FILTER_EMBEDDING_GATE_ENABLED: bool = Field(default=True)
    AI_TSD_CONTENT_FILTER_EMBEDDING_SIMILARITY_THRESHOLD: float = Field(default=0.55)
    AI_TSD_CONTENT_FILTER_EMBEDDING_BATCH_SIZE: int = Field(default=32)

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
