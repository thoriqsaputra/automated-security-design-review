from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MINIO_ENDPOINT", "127.0.0.1:9")

from sdr.core.config import settings


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture
def settings_override(monkeypatch):
    def _override(**values):
        for key, value in values.items():
            monkeypatch.setattr(settings, key, value)

    return _override
