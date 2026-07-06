from __future__ import annotations

from sdr.apps.standards.models.ingestion import StandardIngestionJob


def test_progress_defaults_to_zero_percent_before_any_detailed_progress():
    # Regression: a job freshly marked RUNNING (before the extraction pipeline
    # has emitted its first progress event) previously fell back to a
    # hardcoded 50%, making the UI progress bar jump backward once the first
    # real percentage (5%) arrived. It must start at 0, not halfway.
    job = StandardIngestionJob(status=StandardIngestionJob.STATUS_RUNNING, summary_json={})

    assert job.progress["percentage"] == 0


def test_progress_completed_with_no_detailed_progress_is_100_percent():
    job = StandardIngestionJob(status=StandardIngestionJob.STATUS_COMPLETED, summary_json={})

    assert job.progress["percentage"] == 100


def test_progress_failed_with_no_detailed_progress_is_100_percent():
    job = StandardIngestionJob(status=StandardIngestionJob.STATUS_FAILED, summary_json={})

    assert job.progress["percentage"] == 100


def test_progress_respects_an_explicit_zero_percentage():
    # Regression for the `or`-vs-falsy-zero bug: an explicitly stored
    # percentage of 0 must not be treated as "missing" and overridden.
    job = StandardIngestionJob(
        status=StandardIngestionJob.STATUS_RUNNING,
        summary_json={"detailed_progress": {"percentage": 0, "label": "Starting"}},
    )

    assert job.progress["percentage"] == 0
    assert job.progress["status_label"] == "Starting"


def test_progress_uses_stored_percentage_when_present():
    job = StandardIngestionJob(
        status=StandardIngestionJob.STATUS_RUNNING,
        summary_json={"detailed_progress": {"percentage": 42, "label": "Extracting"}},
    )

    assert job.progress["percentage"] == 42
    assert job.progress["status_label"] == "Extracting"
