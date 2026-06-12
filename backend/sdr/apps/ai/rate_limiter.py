import logging
import threading
import time
from typing import Dict, Optional, Tuple

import redis

from sdr.core.config import settings

logger = logging.getLogger(__name__)
_ROLLING_WINDOW_SECONDS = 60.0
_WINDOW_GRACE_SECONDS = 0.05
_ROLLING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
local reserve = tonumber(ARGV[5])
redis.call("ZREMRANGEBYSCORE", key, 0, now - window)
local count = redis.call("ZCARD", key)
if count < limit then
  if reserve == 1 then
    redis.call("ZADD", key, now, member)
    redis.call("EXPIRE", key, math.ceil(window * 2))
  end
  return {1, count, 0}
end
local oldest = redis.call("ZRANGE", key, 0, 0, "WITHSCORES")
local oldest_score = 0
if oldest[2] then
  oldest_score = tonumber(oldest[2])
end
return {0, count, oldest_score}
"""


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

    def _window_key(self) -> str:
        return f"rate_limit:{self.provider_key}:events"

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

    def _get_cooldown_wait(self) -> float:
        cooldown_until = max(
            self._local_cooldown_until,
            self._read_shared_cooldown_until(),
        )
        now = time.time()
        if cooldown_until > now:
            return cooldown_until - now
        return 0.0

    def _reserve_rolling_window_slot(self, *, reserve: bool) -> Tuple[bool, float, int]:
        if not self.redis_client:
            return True, 0.0, 0

        now = time.time()
        member = f"{time.time_ns()}:{threading.get_ident()}"
        try:
            allowed_flag, current_count, oldest_score = self.redis_client.eval(
                _ROLLING_WINDOW_LUA,
                1,
                self._window_key(),
                now,
                _ROLLING_WINDOW_SECONDS,
                self.rpm_limit,
                member,
                1 if reserve else 0,
            )
        except Exception as exc:
            logger.error(
                "RateLimiter(%s): Redis error %s. Falling back to spacing-only.",
                self.provider_key,
                exc,
            )
            return True, 0.0, 0

        allowed = bool(int(allowed_flag or 0))
        count = int(current_count or 0)
        oldest = float(oldest_score or 0.0)
        if allowed:
            return True, 0.0, count
        if oldest <= 0.0:
            return False, max(self._min_interval, 1.0), count
        wait_seconds = max(
            _WINDOW_GRACE_SECONDS,
            (oldest + _ROLLING_WINDOW_SECONDS) - now + _WINDOW_GRACE_SECONDS,
        )
        return False, wait_seconds, count

    def wait_for_availability(self) -> None:
        """
        Blocks until the shared provider window has room, without reserving a slot.
        Useful as a preflight before a new phase starts issuing requests.
        """
        while True:
            wait_for = 0.0
            with self._lock:
                cooldown_wait = self._get_cooldown_wait()
                if cooldown_wait > 0:
                    wait_for = cooldown_wait
                    logger.warning("RateLimiter(%s): cooling down for %.2fs.", self.provider_key, wait_for)
                else:
                    if not self.redis_client:
                        return
                    allowed, wait_seconds, current_count = self._reserve_rolling_window_slot(reserve=False)
                    if allowed:
                        return
                    wait_for = wait_seconds
                    logger.warning(
                        "RateLimiter(%s): preflight window full (%d/%d RPM). Waiting %.2fs.",
                        self.provider_key, current_count, self.rpm_limit, wait_for,
                    )
            
            if wait_for > 0:
                time.sleep(wait_for)

    def acquire(self) -> None:
        """
        Blocks until the provider cooldown, spacing, and window budget allow a call.
        """
        while True:
            wait_for = 0.0
            with self._lock:
                cooldown_wait = self._get_cooldown_wait()
                if cooldown_wait > 0:
                    wait_for = cooldown_wait
                    logger.warning("RateLimiter(%s): cooling down for %.2fs.", self.provider_key, wait_for)
                else:
                    elapsed = time.monotonic() - self._last_call_time
                    if elapsed < self._min_interval:
                        wait_for = self._min_interval - elapsed
                        logger.debug("RateLimiter(%s): spacing wait %.2fs", self.provider_key, wait_for)
                    else:
                        if not self.redis_client:
                            self._last_call_time = time.monotonic()
                            return

                        allowed, wait_seconds, current_count = self._reserve_rolling_window_slot(reserve=True)
                        if allowed:
                            self._last_call_time = time.monotonic()
                            return
                        wait_for = wait_seconds
                        logger.warning(
                            "RateLimiter(%s): window full (%d/%d RPM). Waiting %.2fs.",
                            self.provider_key, current_count, self.rpm_limit, wait_for,
                        )

            if wait_for > 0:
                time.sleep(wait_for)

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
