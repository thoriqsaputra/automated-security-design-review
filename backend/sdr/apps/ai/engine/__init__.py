from __future__ import annotations

from typing import Any

__all__ = ["TSDAnalysisPipeline", "run_tsd_analysis"]


def __getattr__(name: str) -> Any:
    if name in {"TSDAnalysisPipeline", "run_tsd_analysis"}:
        from .pipeline import TSDAnalysisPipeline, run_tsd_analysis

        exports = {
            "TSDAnalysisPipeline": TSDAnalysisPipeline,
            "run_tsd_analysis": run_tsd_analysis,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
