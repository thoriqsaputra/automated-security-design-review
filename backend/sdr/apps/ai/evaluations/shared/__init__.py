"""
Shared utilities for evaluation scripts.
"""
import os

_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # evaluations/
_RESULTS_DIR = os.path.join(_PACKAGE_ROOT, "results")


def results_path(filename: str, subdir: str | None = None) -> str:
    """Return the full path for an output file inside the results/ directory.

    Pass subdir (e.g. "extraction", "retrieval", "debate", "vision") to write
    into a section-specific subdirectory under results/.
    """
    target = os.path.join(_RESULTS_DIR, subdir) if subdir else _RESULTS_DIR
    os.makedirs(target, exist_ok=True)
    return os.path.join(target, filename)


def data_path(filename: str) -> str:
    """Return the full path for a bundled data file (ground truth, gold sets)."""
    return os.path.join(_PACKAGE_ROOT, "data", filename)
