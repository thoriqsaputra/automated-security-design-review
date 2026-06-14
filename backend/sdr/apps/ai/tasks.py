import json
import logging

from celery import shared_task
from sqlalchemy import select, update
from sqlalchemy.orm import joinedload

from sdr.core.database import SessionLocal
from sdr.apps.ai.client import chat_completion
from sdr.apps.ai.prompts.analysis import (
    SEVERITY_JUSTIFICATION_SYSTEM_PROMPT,
    build_severity_justification_prompt,
)
from sdr.apps.reviews.models import Finding
from sdr.apps.standards.utils import build_parameter_analysis_text

logger = logging.getLogger(__name__)

_SEVERITY_JUSTIFICATION_TEMPERATURE = 0.0
_SEVERITY_JUSTIFICATION_MAX_TOKENS = 512
_SEVERITY_JUSTIFICATION_TRIGGERS = {"critical", "high"}


@shared_task(
    bind=True,
    name="ai.generate_finding_severity_analysis_task",
    max_retries=2,
    default_retry_delay=30,
    retry_backoff=True,
    retry_jitter=True,
)
def generate_finding_severity_analysis_task(self, finding_id: str) -> None:
    with SessionLocal() as db:
        finding = db.execute(
            select(Finding)
            .options(joinedload(Finding.child_parameter)) # the child_parameter__parent will be lazy loaded or we could joinedload it
            .where(Finding.id == finding_id)
        ).scalars().first()
        
        if not finding:
            return
        if finding.severity_analysis:
            return
        if (finding.severity or "").lower() not in _SEVERITY_JUSTIFICATION_TRIGGERS:
            return

        parameter = finding.child_parameter
        if parameter is None:
            return

        top_anchor = db.execute(
            select(finding.citations.property.mapper.class_)
            .where(finding.citations.property.mapper.class_.finding_id == finding.id)
            .order_by(finding.citations.property.mapper.class_.id)
        ).scalars().first()
        
        tsd_context = (top_anchor.quoted_text or "")[:500] if top_anchor else None
        
        # Access parent to trigger lazy load or use it directly
        parameter_section = parameter.parent.title if parameter.parent else "General"
        
        prompt = build_severity_justification_prompt(
            parameter_text=build_parameter_analysis_text(parameter),
            parameter_section=parameter_section,
            mediator_reasoning=finding.mediator_reasoning or finding.reason or "",
            proposed_severity=finding.severity or "high",
            tsd_context=tsd_context,
        )
        
    response = chat_completion(
        messages=[
            {"role": "system", "content": SEVERITY_JUSTIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        component="orchestrator",
        temperature=_SEVERITY_JUSTIFICATION_TEMPERATURE,
        max_tokens=_SEVERITY_JUSTIFICATION_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    if response.error or not response.content:
        raise RuntimeError(f"severity_analysis_llm_error: {response.error or 'empty_response'}")
    result = json.loads(response.content.strip())
    if not isinstance(result, dict):
        raise RuntimeError("severity_analysis_non_dict_response")

    with SessionLocal() as db:
        stmt = (
            update(Finding)
            .where(Finding.id == finding_id, Finding.severity_analysis == None)
            .values(severity_analysis=result)
        )
        res = db.execute(stmt)
        db.commit()
        
        logger.info(
            "generate_finding_severity_analysis_task: finding_id=%s updated=%s",
            finding_id,
            res.rowcount,
        )
