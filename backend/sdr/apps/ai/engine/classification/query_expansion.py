from __future__ import annotations

import hashlib
import logging
import threading
from typing import Dict, List

from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.engine.classification.json_utils import parse_json_with_repair

logger = logging.getLogger(__name__)

_QUERY_EXPANSION_SYSTEM_PROMPT = (
    "You are a retrieval query rewriter for a security design review system. "
    "Given an abstract security requirement (written in standards/compliance language), "
    "produce concrete search-query rephrasings that use the kind of language a technical "
    "specification document (TSD) would use to describe satisfying or violating that "
    "requirement — concrete mechanisms, components, protocols, and verbs (e.g. 'session "
    "validation', 'server-side authorization check', 'access control middleware') instead "
    "of abstract control-objective wording. If the requirement concerns protecting data, "
    "attributes, or state from end-user tampering or manipulation, make sure at least one "
    "variant names the underlying integrity/state-management mechanism that would actually "
    "enforce it (e.g. 'server-side session validation', 'signed or encrypted session "
    "token', 'session state integrity check') rather than only access-control-flavored "
    "phrasing — the enforcement detail often lives in a different section of the TSD than "
    "the access-control model it protects. Respond ONLY with a JSON object: "
    '{"variants": ["...", "...", "..."]}.'
)

_cache_lock = threading.Lock()
_variant_cache: Dict[str, List[str]] = {}


def _cache_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def expand_retrieval_query_variants(
    requirement_text: str,
    *,
    cache_key: str,
    variant_count: int = 3,
    enabled: bool = True,
) -> List[str]:
    """Generate concrete, TSD-implementation-flavored rephrasings of an abstract
    requirement sentence to bridge the vocabulary gap between standards language
    and technical specification language. Cached per cache_key (typically
    parameter + ingestion job) so this costs at most one LLM call per parameter
    per TSD ingestion.
    """
    requirement_text = (requirement_text or "").strip()
    if not enabled or not requirement_text or variant_count <= 0:
        return []

    key = _cache_key(f"{cache_key}:{requirement_text}")
    with _cache_lock:
        cached = _variant_cache.get(key)
    if cached is not None:
        return cached

    variants = _generate_variants(requirement_text, variant_count=variant_count, cache_key=cache_key)

    if variants:
        with _cache_lock:
            _variant_cache[key] = variants
    return variants


def _generate_variants(requirement_text: str, *, variant_count: int, cache_key: str = "") -> List[str]:
    prompt = (
        f"Abstract security requirement:\n{requirement_text}\n\n"
        f"Produce exactly {variant_count} concrete, implementation-flavored rephrasings "
        "of this requirement, each on its own line within the JSON array. Keep each "
        "variant under 30 words."
    )
    messages = [
        {"role": "system", "content": _QUERY_EXPANSION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        for attempt in range(2):
            response = chat_completion(
                messages=messages,
                component="query_expansion",
                temperature=0.0,
                max_tokens=2000,
                # First attempt uses low-effort reasoning for latency; if that
                # comes back empty (observed failure mode: reasoning tokens
                # consume the whole budget, leaving nothing for content), the
                # retry drops reasoning entirely rather than repeating the
                # same failure.
                reasoning=({"effort": "low"} if attempt == 0 else None),
                response_format={"type": "json_object"},
            )
            if response.error or not response.content:
                logger.warning(
                    "expand_retrieval_query_variants: empty/error response (attempt %d/2, cache_key=%s): %s",
                    attempt + 1,
                    cache_key,
                    response.error,
                )
                continue

            parsed, parse_error = parse_json_with_repair(
                response.content,
                component="query_expansion",
                max_tokens=400,
                chat_completion_fn=chat_completion,
            )
            if not isinstance(parsed, dict):
                logger.warning(
                    "expand_retrieval_query_variants: could not parse variants (attempt %d/2, cache_key=%s): %s",
                    attempt + 1,
                    cache_key,
                    parse_error,
                )
                continue

            raw_variants = parsed.get("variants") or []
            if not isinstance(raw_variants, list):
                continue
            variants = [str(item).strip() for item in raw_variants if str(item).strip()]
            if variants:
                return variants[:variant_count]
        return []
    except Exception:
        logger.exception("expand_retrieval_query_variants: failed (cache_key=%s)", cache_key)
        return []


__all__ = ["expand_retrieval_query_variants"]
