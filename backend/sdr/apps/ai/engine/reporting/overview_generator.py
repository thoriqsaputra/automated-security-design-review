from __future__ import annotations

import logging
from typing import Optional

from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.prompts.agents import (
    OVERVIEW_SYSTEM_PROMPT,
    build_overview_prompt,
)

logger = logging.getLogger(__name__)


class OverviewGenerator:
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate(self, review, summary) -> Optional[str]:
        try:
            category_name = "Unknown"
            selected_categories = list(review.selected_categories)

            if selected_categories:
                category_name = selected_categories[0].name
            elif review.ingestion_job and review.ingestion_job.category:
                category_name = review.ingestion_job.category.name

            prompt = build_overview_prompt(
                design_name=review.design.name,
                category_name=category_name,
                total_parameters=summary.total_parameters,
                met_count=summary.met_count,
                not_met_count=summary.not_met_count,
                na_count=summary.na_count,
                critical_findings=summary.critical_findings[:10],
                high_findings=summary.high_findings[:10],
            )

            response = chat_completion(
                messages=[
                    {"role": "system", "content": OVERVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                component="orchestrator",
                temperature=0.2,
                max_tokens=1024,
            )
            if response.error or not response.content:
                return None
            return (response.content or "").strip()
        except Exception as exc:
            self.logger.exception("OverviewGenerator.generate: failed: %s", exc)
            return None
