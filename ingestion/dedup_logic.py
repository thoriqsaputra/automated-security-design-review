from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

DedupAction = Literal["INSERT", "MERGE", "SKIP"]


@dataclass(slots=True)
class DedupDecision:
    action: DedupAction
    confidence: float
    similarity_score: float | None = None
    reason: str | None = None


def normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def cosine_similarity_from_distance(distance: float | None) -> float:
    if distance is None:
        return 0.0
    similarity = 1.0 - float(distance)
    if similarity < 0.0:
        return 0.0
    if similarity > 1.0:
        return 1.0
    return similarity


def fuzzy_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def is_exact_duplicate(
    title_a: str,
    rule_a: str,
    title_b: str,
    rule_b: str,
) -> bool:
    return normalize_text(title_a) == normalize_text(title_b) and normalize_text(rule_a) == normalize_text(rule_b)


def pick_parent(
    current_parent_id,
    current_confidence: float,
    candidate_parent_id,
    candidate_confidence: float,
) -> tuple[object | None, float]:
    if candidate_parent_id is None:
        return current_parent_id, current_confidence

    if current_parent_id is None or candidate_confidence > current_confidence:
        return candidate_parent_id, candidate_confidence

    return current_parent_id, current_confidence
