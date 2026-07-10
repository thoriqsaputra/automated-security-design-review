from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import sdr.apps.designs.models  # noqa: F401 - resolve SQLAlchemy FK
import sdr.apps.reviews.models.finding  # noqa: F401 - resolve SQLAlchemy FK
import sdr.apps.reviews.models.review  # noqa: F401 - resolve SQLAlchemy FK

from sdr.apps.reviews.models.choices import FindingType, ReviewStatus
from sdr.apps.reviews.models.finding import Finding
from sdr.apps.reviews.models.review import Review
from sdr.core.database import SessionLocal

from . import data_path


@dataclass(frozen=True)
class DiagramGroundTruthFiles:
    design_id: int
    filename: str
    llm_judged_filename: str


def diagram_ground_truth_filename(design_id: int, *, llm_judged: bool = False) -> str:
    suffix = "_llm_judged" if llm_judged else ""
    return f"diagram_ground_truth_design_{design_id}{suffix}.json"


def diagram_ground_truth_files(design_id: int) -> DiagramGroundTruthFiles:
    return DiagramGroundTruthFiles(
        design_id=design_id,
        filename=diagram_ground_truth_filename(design_id),
        llm_judged_filename=diagram_ground_truth_filename(design_id, llm_judged=True),
    )


def diagram_ground_truth_path(design_id: int, *, llm_judged: bool = False) -> str:
    return data_path(diagram_ground_truth_filename(design_id, llm_judged=llm_judged))


def resolve_diagram_ground_truth_path(design_id: int) -> str:
    canonical = diagram_ground_truth_path(design_id)
    if os.path.exists(canonical):
        return canonical
    llm_judged = diagram_ground_truth_path(design_id, llm_judged=True)
    if os.path.exists(llm_judged):
        return llm_judged
    return canonical


def get_latest_diagram_review_for_design(design_id: int, db=None) -> Optional[Review]:
    """Return the newest completed review for a design that has at least one diagram finding."""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    try:
        review = (
            db.query(Review)
            .join(Finding, Finding.review_id == Review.id)
            .filter(
                Review.design_id == design_id,
                Review.completed_at.isnot(None),
                Review.status.in_(
                    [
                        ReviewStatus.COMPLETED_CLEAN.value,
                        ReviewStatus.COMPLETED_WITH_FINDINGS.value,
                        ReviewStatus.APPROVED.value,
                        ReviewStatus.REJECTED.value,
                    ]
                ),
                Finding.finding_type == FindingType.DIAGRAM.value,
            )
            .order_by(Review.completed_at.desc(), Review.created_at.desc(), Review.id.desc())
            .first()
        )
        return review
    finally:
        if close_db:
            db.close()
