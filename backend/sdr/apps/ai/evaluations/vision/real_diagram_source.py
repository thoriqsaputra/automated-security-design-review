"""
Shared real-diagram sourcing for vision evaluation scripts.

Loads the same labeled `diagram_ground_truth_design_<id>.json` produced by
`evaluations/data/build_diagram_ground_truth_template.py` and consumed by
`retrieval/diagram_retrieval_eval.py` / `debate/diagram_ablation_eval.py`, and resolves each labeled
(diagram_id, requirement_id) row to a real `DiagramInput` built from the
design's parsed TSD document — no synthetic image generation. `design_id` is
read directly from the ground-truth file (written there by
build_diagram_ground_truth_template.py), so callers only need to pass
`--ground-truth`.
"""
from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass
from typing import Optional

import sdr.apps.designs.models  # noqa: F401 — resolve SQLAlchemy FK
import sdr.apps.standards.models  # noqa: F401

from sdr.core.database import SessionLocal
from sdr.apps.designs.models import Design
from sdr.apps.designs.preparation_store import DesignPreparationStore
from sdr.apps.ai.agents.vision import DiagramInput
from sdr.apps.standards.models.diagram_requirement import CategoryDiagramRequirement

_MIN_DIAGRAM_BYTES = 512


@dataclass
class RealLabeledSample:
    diagram_id: str
    requirement_id: str
    requirement_text: str
    verification_hint: str
    label: str  # "met" | "not_met" | "na"


def load_ground_truth(gt_path: str) -> dict:
    with open(gt_path) as f:
        return json.load(f)


def load_labeled_samples(
    gt_data: dict, labels: tuple[str, ...] = ("met", "not_met")
) -> list[RealLabeledSample]:
    """Flatten the ground-truth JSON into one row per relevant, labeled (diagram, requirement) pair."""
    candidate_rows = [
        req
        for item in gt_data.get("items", [])
        for req in item.get("candidate_requirements", [])
        if req.get("relevant") is True
        and str(req.get("label") or "").lower().strip() in labels
    ]
    hint_map = (
        _load_requirement_hint_map(gt_data)
        if any(not str(req.get("verification_hint") or "").strip() for req in candidate_rows)
        else {}
    )
    samples: list[RealLabeledSample] = []
    for item in gt_data.get("items", []):
        diagram_id = str(item.get("diagram_id", "")).strip()
        if not diagram_id:
            continue
        for req in item.get("candidate_requirements", []):
            if req.get("relevant") is not True:
                continue
            label = str(req.get("label") or "").lower().strip()
            if label not in labels:
                continue
            requirement_id = str(req.get("requirement_id", "")).strip()
            if not requirement_id:
                continue
            samples.append(
                RealLabeledSample(
                    diagram_id=diagram_id,
                    requirement_id=requirement_id,
                    requirement_text=req.get("requirement_text") or "",
                    verification_hint=(
                        req.get("verification_hint")
                        or hint_map.get(requirement_id)
                        or ""
                    ),
                    label=label,
                )
            )
    return samples


def _load_requirement_hint_map(gt_data: dict) -> dict[str, str]:
    category_id = gt_data.get("category_id")
    ingestion_job_id = gt_data.get("ingestion_job_id")
    if not category_id or not ingestion_job_id:
        return {}

    with SessionLocal() as db:
        rows = (
            db.query(CategoryDiagramRequirement.stable_key, CategoryDiagramRequirement.verification_hint)
            .filter(
                CategoryDiagramRequirement.category_id == category_id,
                CategoryDiagramRequirement.ingestion_job_id == ingestion_job_id,
            )
            .all()
        )
    return {
        str(stable_key).strip(): str(verification_hint or "").strip()
        for stable_key, verification_hint in rows
        if str(stable_key).strip()
    }


def select_balanced_samples(
    samples: list[RealLabeledSample],
    *,
    max_samples: int,
    seed: int = 42,
) -> list[RealLabeledSample]:
    """Deterministically sample up to `max_samples`, roughly balancing labels."""
    if max_samples <= 0 or len(samples) <= max_samples:
        return list(samples)

    rng = random.Random(seed)
    by_label: dict[str, list[RealLabeledSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    for group in by_label.values():
        rng.shuffle(group)

    selected: list[RealLabeledSample] = []
    labels = sorted(by_label.keys())

    # First pass: reserve one sample per label when possible.
    for label in labels:
        if len(selected) >= max_samples:
            break
        group = by_label.get(label) or []
        if group:
            selected.append(group.pop())

    # Second pass: fill the remainder round-robin for rough balance.
    while len(selected) < max_samples:
        progressed = False
        for label in labels:
            group = by_label.get(label) or []
            if not group:
                continue
            selected.append(group.pop())
            progressed = True
            if len(selected) >= max_samples:
                break
        if not progressed:
            break

    # Stable output order keeps logs/diffs predictable.
    return sorted(
        selected,
        key=lambda sample: (sample.label, sample.diagram_id, sample.requirement_id),
    )


def diagrams_with_no_relevant_requirements(gt_data: dict) -> list[tuple[str, dict]]:
    """(diagram_id, first_candidate_row) pairs where every candidate requirement was
    labeled relevant:false — a natural real "non-architecture" example for
    scope-accuracy testing. first_candidate_row supplies requirement text/hint
    needed to force the scope-classification step even though it's irrelevant."""
    result = []
    for item in gt_data.get("items", []):
        candidates = item.get("candidate_requirements", [])
        if candidates and all(req.get("relevant") is False for req in candidates):
            diagram_id = str(item.get("diagram_id", "")).strip()
            if diagram_id:
                result.append((diagram_id, candidates[0]))
    return result


def load_tsd_document(design_id: int):
    """Loads and fully materializes the design's parsed TSD document (diagrams included)."""
    with SessionLocal() as db:
        design = db.query(Design).filter(Design.id == design_id).first()
        if design is None:
            raise ValueError(f"Design with ID {design_id} not found.")
        store = DesignPreparationStore()
        _, tsd_doc, _ = store.load_prepared_assets(db, design)
    return tsd_doc


def build_diagram_input(
    tsd_doc, diagram_id: str, diagram_input_id: str, *, blank_text: bool = False
) -> Optional[DiagramInput]:
    """Fetches a real DiagramBlock by id and converts it to a DiagramInput.
    If blank_text, caption/surrounding_text are wiped to isolate the vision
    channel (a real diagram's caption may otherwise describe the control
    under test, leaking the answer to a vision-blindness test)."""
    dblock = tsd_doc.get_diagram_by_id(diagram_id)
    if dblock is None:
        return None
    dblock.ensure_image_loaded(_MIN_DIAGRAM_BYTES)
    if not dblock.is_valid():
        return None
    return DiagramInput(
        diagram_id=diagram_input_id,
        image_b64=dblock.image_b64,
        page_number=dblock.page_number,
        caption="" if blank_text else (dblock.caption or ""),
        surrounding_text="" if blank_text else (dblock.surrounding_text or ""),
        image_format=dblock.image_format or "png",
    )


def save_image_b64(b64: str, path: str) -> None:
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
