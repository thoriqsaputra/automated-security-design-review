from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class ExtractionLLMClient:
    def __init__(self, *, chat_completion: Callable[..., Any]) -> None:
        self._chat_completion = chat_completion

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        component: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "component": component,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        return self._chat_completion(**kwargs)

    def repair_json(self, *, user_prompt: str, max_tokens: int) -> Any:
        return self._chat_completion(
            messages=[{"role": "user", "content": user_prompt}],
            component="fallback",
            temperature=0.0,
            max_tokens=max_tokens,
        )
