from __future__ import annotations

import logging
from datetime import datetime, timezone

from sdr.apps.ai.engine.dto import AnalysisSummary
from sdr.apps.reviews.models import Review
from sdr.apps.reviews.models.choices import ReviewStatus
from sdr.apps.reviews.services import review_debate_event_store


class AnalysisCancelledError(RuntimeError):
    """Raised when the user has cancelled the review."""


class ReviewRunStateService:
    def __init__(self, *, workflow_repository) -> None:
        self.workflow_repository = workflow_repository
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def mark_running(self, review: Review) -> None:
        new_status = ReviewStatus.RUNNING.value
        now = datetime.now(timezone.utc)
        review_debate_event_store.reset_review(review.id)
        self.workflow_repository.mark_review_running(
            review.id,
            status=new_status,
            started_at=now,
        )
        review.status = new_status
        review.started_at = now
        review_debate_event_store.publish_review_status(review.id, review_status=new_status)

    def save_overview(self, review: Review, overview: str) -> None:
        self.workflow_repository.save_review_overview(review.id, overview=overview)
        review.overview = overview

    def persist_summary_snapshot(self, review: Review, summary: AnalysisSummary) -> None:
        try:
            with summary.lock:
                summary_dict = summary.to_dict()
            self.workflow_repository.save_summary_snapshot(review.id, summary=summary_dict)
            review.summary_json = summary_dict
        except Exception as exc:
            self.logger.exception(
                "ReviewRunStateService.persist_summary_snapshot: failed for review_id=%s: %s",
                review.id,
                exc,
            )

    def update_stage(self, review: Review, summary: AnalysisSummary, stage: str) -> None:
        with summary.lock:
            summary.current_stage = stage
        self.persist_summary_snapshot(review, summary)

    def complete_review(self, review: Review, summary: AnalysisSummary) -> None:
        try:
            if self.is_cancelled(review):
                self.logger.warning(
                    "ReviewRunStateService.complete_review: cancellation detected, skipping completion for review_id=%s",
                    review.id,
                )
                return
            if int(summary.not_met_count or 0) > 0:
                new_status = ReviewStatus.COMPLETED_WITH_FINDINGS.value
            else:
                new_status = ReviewStatus.COMPLETED_CLEAN.value
            now = datetime.now(timezone.utc)
            summary_dict = summary.to_dict()
            self.workflow_repository.mark_review_completed(
                review.id,
                status=new_status,
                completed_at=now,
                summary=summary_dict,
            )
            review.status = new_status
            review.completed_at = now
            review.summary_json = summary_dict
            review_debate_event_store.publish_review_status(review.id, review_status=new_status)
        except Exception as exc:
            self.logger.exception(
                "ReviewRunStateService.complete_review: failed for review_id=%s: %s",
                review.id,
                exc,
            )

    def fail_review(self, review: Review, error_message: str) -> None:
        try:
            if self.is_cancelled(review):
                self.logger.warning(
                    "ReviewRunStateService.fail_review: skipping failed status because review_id=%s is cancelled",
                    review.id,
                )
                return
            new_status = ReviewStatus.FAILED.value
            now = datetime.now(timezone.utc)
            self.workflow_repository.mark_review_failed(
                review.id,
                status=new_status,
                completed_at=now,
                error_message=error_message,
            )
            review.status = new_status
            review.completed_at = now
            review.error_message = error_message
            review_debate_event_store.publish_review_status(
                review.id,
                review_status=new_status,
                error_message=error_message,
            )
        except Exception as exc:
            self.logger.exception(
                "ReviewRunStateService.fail_review: could not mark review failed id=%s: %s",
                review.id,
                exc,
            )

    def is_cancelled(self, review: Review) -> bool:
        latest = self.workflow_repository.get_latest_review(review.id)
        if not latest:
            return False
        if latest.status == ReviewStatus.CANCELLED.value:
            return True
        return (
            latest.status == ReviewStatus.FAILED.value
            and (latest.error_message or "").strip().lower().startswith("analysis was cancelled")
        )

    def raise_if_cancelled(self, review: Review, *, phase: str) -> None:
        if not self.is_cancelled(review):
            return
        self.logger.warning(
            "ReviewRunStateService.raise_if_cancelled: cancellation detected review_id=%s phase=%s",
            review.id,
            phase,
        )
        raise AnalysisCancelledError("Analysis was cancelled by user.")
