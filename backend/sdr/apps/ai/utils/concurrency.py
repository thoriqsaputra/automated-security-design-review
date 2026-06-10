from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class ConcurrencySnapshot:
    submitted: int
    started: int
    completed: int
    failed: int
    current_in_flight: int
    peak_in_flight: int
    max_concurrency: int
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "submitted": self.submitted,
            "started": self.started,
            "completed": self.completed,
            "failed": self.failed,
            "current_in_flight": self.current_in_flight,
            "peak_in_flight": self.peak_in_flight,
            "max_concurrency": self.max_concurrency,
            "elapsed_seconds": self.elapsed_seconds,
        }


class ConcurrencyProbe:
    """Tracks configured and observed concurrency for a worker pool."""

    def __init__(self, *, max_concurrency: int) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self._submitted = 0
        self._started = 0
        self._completed = 0
        self._failed = 0
        self._current_in_flight = 0
        self._peak_in_flight = 0
        self._started_at = time.monotonic()
        self._lock = threading.Lock()

    def mark_submitted(self, count: int = 1) -> None:
        with self._lock:
            self._submitted += max(0, int(count))

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self._started += 1
                self._current_in_flight += 1
                if self._current_in_flight > self._peak_in_flight:
                    self._peak_in_flight = self._current_in_flight
            try:
                result = fn(*args, **kwargs)
            except Exception:
                with self._lock:
                    self._failed += 1
                raise
            else:
                with self._lock:
                    self._completed += 1
                return result
            finally:
                with self._lock:
                    self._current_in_flight = max(
                        0,
                        self._current_in_flight - 1,
                    )

        return wrapped

    def snapshot(self) -> ConcurrencySnapshot:
        with self._lock:
            return ConcurrencySnapshot(
                submitted=self._submitted,
                started=self._started,
                completed=self._completed,
                failed=self._failed,
                current_in_flight=self._current_in_flight,
                peak_in_flight=self._peak_in_flight,
                max_concurrency=self.max_concurrency,
                elapsed_seconds=round(time.monotonic() - self._started_at, 4),
            )
