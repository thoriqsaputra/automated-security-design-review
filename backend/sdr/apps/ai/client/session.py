from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Callable, Dict, Iterator, Mapping, Optional, TypeVar

logger = logging.getLogger(__name__)

_request_metadata_ctx: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "ai_request_metadata",
    default=None,
)

_R = TypeVar("_R")


def build_standard_ingestion_session_id(job_id: int | str) -> str:
    return f"standard_ingestion_job_{job_id}"


def build_tsd_analysis_session_id(review_id: int | str) -> str:
    return f"tsd_analysis_review_{review_id}"


def build_tsd_ingestion_session_id(preparation_id: int | str) -> str:
    return f"tsd_ingestion_preparation_{preparation_id}"


def capture_current_context(fn: Callable[..., _R]) -> Callable[..., _R]:
    """Wrap fn so it runs inside a snapshot of the CALLING thread's contextvars.

    contextvars (including the request-metadata session_id set by
    job_session_context) do not propagate across real OS threads the way they
    do across asyncio tasks — a plain `executor.submit(fn, ...)` runs fn with a
    fresh/empty context on the worker thread. Wrap any function handed to a
    ThreadPoolExecutor with this so session/job metadata (and therefore
    per-pipeline token/duration attribution) survives into the worker thread.
    """
    ctx = contextvars.copy_context()

    def _wrapped(*args: object, **kwargs: object) -> _R:
        return ctx.run(fn, *args, **kwargs)

    return _wrapped


def get_current_request_metadata() -> Dict[str, str]:
    current = _request_metadata_ctx.get()
    return dict(current or {})


def merge_request_metadata(
    explicit_metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    merged = get_current_request_metadata()
    active_session_id = merged.get("session_id")

    if explicit_metadata:
        for key, value in explicit_metadata.items():
            if value is None:
                continue
            merged[str(key)] = str(value)

    if active_session_id:
        merged["session_id"] = active_session_id

    return merged


@contextmanager
def request_metadata_context(metadata: Optional[Mapping[str, object]]) -> Iterator[None]:
    current = get_current_request_metadata()
    next_metadata = merge_request_metadata(metadata)
    if not next_metadata and not current:
        yield
        return

    token: Token = _request_metadata_ctx.set(next_metadata or None)
    try:
        yield
    finally:
        _request_metadata_ctx.reset(token)


@contextmanager
def job_session_context(*, session_id: str, job_type: str, job_id: int | str) -> Iterator[None]:
    metadata = {
        "session_id": session_id,
        "job_type": str(job_type),
        "job_id": str(job_id),
    }
    logger.info(
        "job_session_context: job_type=%s job_id=%s session_id=%s",
        job_type,
        job_id,
        session_id,
    )
    with request_metadata_context(metadata):
        yield
