from __future__ import annotations

import requests

from sdr.apps.ai.client.base import AIProvider, AIResponse
from sdr.apps.ai.client.manager import AIServiceManager
from sdr.apps.ai.client.nvidia.service import NVIDIAAIService
from sdr.apps.ai.rate_limiter import RateLimiter


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start
        self.sleep_calls: list[float] = []

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


class _CountingLimiter:
    def __init__(self) -> None:
        self.acquire_calls = 0
        self.cooldowns: list[float] = []

    def acquire(self) -> None:
        self.acquire_calls += 1

    def register_throttle(self, retry_after_seconds=None) -> float:
        self.cooldowns.append(retry_after_seconds)
        return float(retry_after_seconds or 0.0)


class _FakeResponse:
    def __init__(self, status_code: int, *, body=None, text: str = "", headers=None) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} error",
                response=self,
            )


def test_rate_limiter_cooldown_blocks_same_provider(monkeypatch):
    clock = _FakeClock()
    limiter = RateLimiter(
        provider_key="nvidia-test",
        rpm_limit=60000,
        throttle_cooldown_seconds=5.0,
    )
    limiter.redis_client = None

    monkeypatch.setattr("sdr.apps.ai.rate_limiter.time.time", clock.time)
    monkeypatch.setattr("sdr.apps.ai.rate_limiter.time.monotonic", clock.monotonic)
    monkeypatch.setattr("sdr.apps.ai.rate_limiter.time.sleep", clock.sleep)

    limiter.register_throttle(4.0)
    limiter.acquire()

    assert clock.sleep_calls == [4.0]


def test_rate_limiter_cooldown_does_not_block_other_provider(monkeypatch):
    clock = _FakeClock()
    nvidia_limiter = RateLimiter(
        provider_key="nvidia-test-a",
        rpm_limit=60000,
        throttle_cooldown_seconds=5.0,
    )
    openrouter_limiter = RateLimiter(
        provider_key="openrouter-test-a",
        rpm_limit=60000,
        throttle_cooldown_seconds=5.0,
    )
    nvidia_limiter.redis_client = None
    openrouter_limiter.redis_client = None

    monkeypatch.setattr("sdr.apps.ai.rate_limiter.time.time", clock.time)
    monkeypatch.setattr("sdr.apps.ai.rate_limiter.time.monotonic", clock.monotonic)
    monkeypatch.setattr("sdr.apps.ai.rate_limiter.time.sleep", clock.sleep)

    nvidia_limiter.register_throttle(4.0)
    openrouter_limiter.acquire()

    assert clock.sleep_calls == []


def test_nvidia_retries_reacquire_limiter(monkeypatch):
    limiter = _CountingLimiter()
    service = NVIDIAAIService()
    service.rate_limiter = limiter

    responses = iter(
        [
            _FakeResponse(
                429,
                text='{"status":429,"title":"Too Many Requests"}',
                headers={"Retry-After": "3"},
            ),
            _FakeResponse(
                200,
                body={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 12}},
            ),
        ]
    )

    monkeypatch.setattr(
        "sdr.apps.ai.client.nvidia.service.requests.post",
        lambda *args, **kwargs: next(responses),
    )

    response = service.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        model="meta/test",
    )

    assert response.content == "ok"
    assert response.status_code == 200
    assert limiter.acquire_calls == 2
    assert limiter.cooldowns == [3.0]


def test_standard_extraction_falls_back_to_openrouter(settings_override):
    settings_override(
        AI_LLM_FALLBACK_ON_RETRY_EXHAUSTED=True,
        AI_STANDARD_EXTRACTION_FALLBACK_PROVIDER="openrouter",
        OPENROUTER_FAST_MODEL="openrouter/fast-model",
    )
    manager = AIServiceManager()
    nvidia_calls = []
    openrouter_calls = []

    class _FakeService:
        def __init__(self, default_model: str, provider: AIProvider, response: AIResponse, sink: list):
            self.default_model = default_model
            self.provider = provider
            self.response = response
            self.sink = sink

        def chat_completion(self, **kwargs):
            self.sink.append(kwargs)
            return self.response

    manager.nvidia_service = _FakeService(
        "meta/nvidia-model",
        AIProvider.NVIDIA,
        AIResponse(
            content="",
            model="meta/nvidia-model",
            provider=AIProvider.NVIDIA,
            error="429",
            status_code=429,
            error_code="rate_limit_exhausted",
        ),
        nvidia_calls,
    )
    manager.openrouter_service = _FakeService(
        "openrouter/default-model",
        AIProvider.OPENROUTER,
        AIResponse(
            content="fallback-ok",
            model="openrouter/fast-model",
            provider=AIProvider.OPENROUTER,
        ),
        openrouter_calls,
    )

    response = manager.chat_completion_with_fallback(
        messages=[{"role": "user", "content": "extract"}],
        component="standard_extraction",
    )

    assert response.content == "fallback-ok"
    assert len(nvidia_calls) == 1
    assert len(openrouter_calls) == 1
    assert openrouter_calls[0]["model"] == "openrouter/fast-model"
