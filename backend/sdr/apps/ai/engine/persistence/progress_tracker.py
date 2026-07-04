from __future__ import annotations

from typing import Any, Dict, List, Optional

from sdr.apps.ai.engine.dto import AnalysisSummary


class SummaryProgressService:
    def prepare_category_stats(
        self,
        *,
        summary: AnalysisSummary,
        parameters: List[Any],
        category_code: str,
    ) -> None:
        with summary.lock:
            category_stats = summary.category_stats.setdefault(category_code, {})
            category_stats.setdefault("debate_total_count", 0)
            category_stats.setdefault("debate_completed_count", 0)
            category_stats.setdefault("debate_remaining_count", 0)
            category_stats.setdefault("persistence_total_count", 0)
            category_stats.setdefault("persistence_completed_count", 0)
            category_stats.setdefault("persistence_remaining_count", 0)
            category_stats.setdefault("analysis_total_count", 0)
            category_stats.setdefault("analysis_processed_count", 0)
            category_stats.setdefault("analysis_remaining_count", 0)

    def initialize_category_progress(
        self,
        *,
        summary: AnalysisSummary,
        category_code: str,
        total_count: int,
    ) -> None:
        with summary.lock:
            category_stats = summary.category_stats.setdefault(category_code, {})
            total_count = int(total_count)
            category_stats["debate_total_count"] = total_count
            category_stats["debate_completed_count"] = 0
            category_stats["debate_remaining_count"] = total_count
            category_stats["persistence_total_count"] = total_count
            category_stats["persistence_completed_count"] = 0
            category_stats["persistence_remaining_count"] = total_count
            category_stats["analysis_total_count"] = total_count
            category_stats["analysis_processed_count"] = 0
            category_stats["analysis_remaining_count"] = total_count

    def sync_analysis_aliases(
        self,
        *,
        summary: AnalysisSummary,
        category_code: Optional[str] = None,
    ) -> None:
        with summary.lock:
            summary.analysis_total_parameters = summary.debate_total_parameters
            summary.analysis_processed_parameters = summary.debate_completed_parameters
            summary.analysis_remaining_parameters = summary.debate_remaining_parameters
            if category_code is None:
                return
            category_stats = summary.category_stats.setdefault(category_code, {})
            category_stats["analysis_total_count"] = int(category_stats.get("debate_total_count") or 0)
            category_stats["analysis_processed_count"] = int(category_stats.get("debate_completed_count") or 0)
            category_stats["analysis_remaining_count"] = int(category_stats.get("debate_remaining_count") or 0)

    def record_debate_progress(
        self,
        *,
        summary: AnalysisSummary,
        category_code: str,
        completed_count: int,
    ) -> Optional[Dict[str, int]]:
        if completed_count <= 0:
            return None
        with summary.lock:
            category_stats = summary.category_stats.setdefault(category_code, {})
            total_count = int(category_stats.get("debate_total_count") or 0)
            processed_count = min(
                int(category_stats.get("debate_completed_count") or 0) + int(completed_count),
                total_count,
            )
            remaining_count = max(total_count - processed_count, 0)
            category_stats["debate_completed_count"] = processed_count
            category_stats["debate_remaining_count"] = remaining_count
            summary.debate_completed_parameters = min(
                int(summary.debate_completed_parameters or 0) + int(completed_count),
                int(summary.debate_total_parameters or 0),
            )
            summary.debate_remaining_parameters = max(
                int(summary.debate_total_parameters or 0) - int(summary.debate_completed_parameters or 0),
                0,
            )
            self.sync_analysis_aliases(summary=summary, category_code=category_code)
            return {
                "processed_count": processed_count,
                "remaining_count": remaining_count,
                "total_count": total_count,
            }

    def record_persistence_progress(
        self,
        *,
        summary: AnalysisSummary,
        category_code: str,
    ) -> Dict[str, int]:
        with summary.lock:
            category_stats = summary.category_stats.setdefault(category_code, {})
            total_count = int(category_stats.get("persistence_total_count") or 0)
            processed_count = min(int(category_stats.get("persistence_completed_count") or 0) + 1, total_count)
            remaining_count = max(total_count - processed_count, 0)
            category_stats["persistence_completed_count"] = processed_count
            category_stats["persistence_remaining_count"] = remaining_count
            summary.persistence_completed_parameters = min(
                int(summary.persistence_completed_parameters or 0) + 1,
                int(summary.persistence_total_parameters or 0),
            )
            summary.persistence_remaining_parameters = max(
                int(summary.persistence_total_parameters or 0) - int(summary.persistence_completed_parameters or 0),
                0,
            )
            return {
                "processed_count": processed_count,
                "remaining_count": remaining_count,
                "total_count": total_count,
            }
