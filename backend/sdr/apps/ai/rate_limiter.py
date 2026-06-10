import logging
import threading
import time
from typing import Dict, Optional

import redis

from sdr.core.config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Provider-scoped limiter with shared cooldown support.
    """

    def __init__(
        self,
        *,
        provider_key: str,
        rpm_limit: int,
        throttle_cooldown_seconds: float = 5.0,
    ) -> None:
        self.provider_key = provider_key
        self.rpm_limit = max(1, int(rpm_limit))
        self.throttle_cooldown_seconds = max(0.0, float(throttle_cooldown_seconds))
        self._min_interval = 60.0 / self.rpm_limit
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0
        self._local_cooldown_until: float = 0.0

        self.redis_client = None
        if settings.REDIS_URL:
            try:
                self.redis_client = redis.Redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                )
            except Exception as exc:
                logger.error(
                    "Failed to connect to Redis for RateLimiter(%s): %s",
                    self.provider_key,
                    exc,
                )

    def _window_key(self, window_id: int) -> str:
        return f"rate_limit:{self.provider_key}:{window_id}"

    def _cooldown_key(self) -> str:
        return f"rate_limit:{self.provider_key}:cooldown_until"

    def _read_shared_cooldown_until(self) -> float:
        if not self.redis_client:
            return 0.0
        try:
            value = self.redis_client.get(self._cooldown_key())
            return float(value) if value else 0.0
        except Exception as exc:
            logger.error(
                "RateLimiter(%s): failed to read cooldown: %s",
                self.provider_key,
                exc,
            )
            return 0.0

    def _write_shared_cooldown_until(self, cooldown_until: float, ttl_seconds: int) -> None:
        if not self.redis_client:
            return
        try:
            self.redis_client.setex(
                self._cooldown_key(),
                max(1, int(ttl_seconds)),
                f"{cooldown_until:.6f}",
            )
        except Exception as exc:
            logger.error(
                "RateLimiter(%s): failed to write cooldown: %s",
                self.provider_key,
                exc,
            )

    def _wait_for_cooldown(self) -> None:
        cooldown_until = max(
            self._local_cooldown_until,
            self._read_shared_cooldown_until(),
        )
        now = time.time()
        if cooldown_until > now:
            wait_seconds = cooldown_until - now
            logger.warning(
                "RateLimiter(%s): cooling down for %.2fs after provider throttle.",
                self.provider_key,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    def acquire(self) -> None:
        """
        Blocks until the provider cooldown, spacing, and window budget allow a call.
        """
        with self._lock:
            while True:
                self._wait_for_cooldown()

                elapsed = time.monotonic() - self._last_call_time
                if elapsed < self._min_interval:
                    wait = self._min_interval - elapsed
                    logger.debug(
                        "RateLimiter(%s): spacing wait %.2fs",
                        self.provider_key,
                        wait,
                    )
                    time.sleep(wait)

                if not self.redis_client:
                    self._last_call_time = time.monotonic()
                    return

                window = 60
                current_time = time.time()
                window_id = int(current_time / window)
                key = self._window_key(window_id)
                try:
                    current_count = self.redis_client.incr(key)
                    if current_count == 1:
                        self.redis_client.expire(key, window * 2)

                    if current_count <= self.rpm_limit:
                        self._last_call_time = time.monotonic()
                        return

                    sleep_time = window - (current_time % window) + 0.1
                    logger.warning(
                        "RateLimiter(%s): window full (%d/%d RPM). Waiting %.2fs.",
                        self.provider_key,
                        current_count,
                        self.rpm_limit,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                except Exception as exc:
                    logger.error(
                        "RateLimiter(%s): Redis error %s. Falling back to spacing-only.",
                        self.provider_key,
                        exc,
                    )
                    self._last_call_time = time.monotonic()
                    return

    def register_throttle(self, retry_after_seconds: Optional[float] = None) -> float:
        cooldown_seconds = retry_after_seconds
        if cooldown_seconds is None:
            cooldown_seconds = self.throttle_cooldown_seconds
        cooldown_seconds = max(0.0, float(cooldown_seconds))
        cooldown_until = time.time() + cooldown_seconds
        self._local_cooldown_until = max(self._local_cooldown_until, cooldown_until)
        self._write_shared_cooldown_until(
            cooldown_until,
            ttl_seconds=max(1, int(cooldown_seconds) + 1),
        )
        return cooldown_seconds


_limiter_registry: Dict[str, RateLimiter] = {}
_registry_lock = threading.Lock()


def get_rate_limiter(
    provider_key: str,
    *,
    rpm_limit: Optional[int] = None,
    throttle_cooldown_seconds: Optional[float] = None,
) -> RateLimiter:
    with _registry_lock:
        limiter = _limiter_registry.get(provider_key)
        if limiter is None:
            if provider_key == "nvidia":
                default_rpm = settings.AI_NVIDIA_RPM_LIMIT
                default_cooldown = settings.AI_NVIDIA_429_COOLDOWN_SECONDS
            elif provider_key == "openrouter":
                default_rpm = settings.AI_OPENROUTER_RPM_LIMIT
                default_cooldown = settings.AI_NVIDIA_429_COOLDOWN_SECONDS
            else:
                default_rpm = settings.AI_NVIDIA_RPM_LIMIT
                default_cooldown = settings.AI_NVIDIA_429_COOLDOWN_SECONDS

            limiter = RateLimiter(
                provider_key=provider_key,
                rpm_limit=rpm_limit or default_rpm,
                throttle_cooldown_seconds=(
                    default_cooldown
                    if throttle_cooldown_seconds is None
                    else throttle_cooldown_seconds
                ),
            )
            _limiter_registry[provider_key] = limiter
        return limiter
