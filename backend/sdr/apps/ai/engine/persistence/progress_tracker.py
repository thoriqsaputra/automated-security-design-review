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
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
        parent_ids = set()
        parent_titles = []
        seen_parent_titles = set()
        level_counts = {"L1": 0, "L2": 0, "L3": 0, "unknown": 0}

        for parameter in parameters or []:
            parent_id = getattr(parameter, "parent_id", None)
            if parent_id is not None:
                parent_ids.add(parent_id)
            parent = getattr(parameter, "parent", None)
            parent_title = str(getattr(parent, "title", "") or "").strip()
            if parent_title and parent_title not in seen_parent_titles:
                seen_parent_titles.add(parent_title)
                if len(parent_titles) < 10:
                    parent_titles.append(parent_title)
            level = getattr(parameter, "asvs_level", None)
            if level == 1:
                level_counts["L1"] += 1
            elif level == 2:
                level_counts["L2"] += 1
            elif level == 3:
                level_counts["L3"] += 1
            else:
                level_counts["unknown"] += 1

        category_stats["parameter_count_before_parent_applicability"] = len(parameters or [])
        category_stats["parent_count_before_parent_applicability"] = len(parent_ids)
        category_stats["asvs_level_counts"] = level_counts
        category_stats["parent_titles_sample"] = parent_titles
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
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
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
        summary.analysis_total_parameters = summary.debate_total_parameters
        summary.analysis_processed_parameters = summary.debate_completed_parameters
        summary.analysis_remaining_parameters = summary.debate_remaining_parameters
        if category_code is None:
            return
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
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
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
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
        category_stats = summary.asvs.setdefault("categories", {}).setdefault(category_code, {})
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
