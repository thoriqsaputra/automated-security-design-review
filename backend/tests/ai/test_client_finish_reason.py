from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.client.base import AIProvider
from sdr.apps.ai.client.nvidia.service import NVIDIAAIService
from sdr.apps.ai.client.openrouter.service import OpenRouterAIService


class _NoOpLimiter:
    def acquire(self) -> None:
        return None


def test_openrouter_chat_completion_preserves_finish_reason(monkeypatch):
    service = OpenRouterAIService()
    service.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"ok": true}'),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                )
            )
        )
    )
    service.rate_limiter = _NoOpLimiter()

    response = service.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        model="test-model",
    )

    assert response.provider == AIProvider.OPENROUTER
    assert response.finish_reason == "stop"
    assert response.usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def test_nvidia_chat_completion_preserves_finish_reason(monkeypatch):
    service = NVIDIAAIService()
    service.rate_limiter = _NoOpLimiter()
    service._request_with_retries = lambda **_kwargs: SimpleNamespace(  # type: ignore[assignment]
        status_code=200,
        json=lambda: {
            "choices": [
                {
                    "message": {"content": '{"ok": true}'},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    )

    response = service.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        model="test-model",
    )

    assert response.provider == AIProvider.NVIDIA
    assert response.finish_reason == "length"
    assert response.usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
