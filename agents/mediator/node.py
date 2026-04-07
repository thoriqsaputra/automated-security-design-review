"""
Mediator Agent Node - Final report synthesis with citations.

The Mediator Agent compiles validated threats into a comprehensive report,
ensuring every threat has proper citations to NIST/OWASP or architecture context.
"""

import json
import logging
from typing import Any

import litellm

from agents.state import AgentState, FinalThreatReport, ValidationFeedback, ThreadThreat
from core.prompts import MEDIATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def mediator_node(state: AgentState) -> dict[str, Any]:
    """
    Mediator Agent Node: Synthesize final report from validated threats.
    """
    logger.info("[Mediator] Synthesizing final report from validated threats")
    
    if not state.get("validation_feedbacks"):
        logger.warning("[Mediator] No feedback available to process.")
        return {"final_report": [], "analysis_complete": True}

    final_loop = max(f.get("recursion_iteration", 0) for f in state["validation_feedbacks"])
    
    valid_feedback_map = {
        f["threat_id"]: f 
        for f in state["validation_feedbacks"] 
        if f.get("recursion_iteration") == final_loop and f.get("is_valid")
    }

    if not valid_feedback_map:
        logger.warning("[Mediator] No valid threats in final loop; falling back to all validated feedbacks.")
        valid_feedback_map = {
            f["threat_id"]: f
            for f in state["validation_feedbacks"]
            if f.get("is_valid")
        }

    valid_threats = [
        t for t in state.get("draft_threats", []) 
        if t.get("threat_id") in valid_feedback_map
    ]
    
    logger.info(f"[Mediator] Found {len(valid_threats)} valid threats from final loop.")
    
    if not valid_threats:
        logger.warning("[Mediator] No valid threats to include in report")
        return {
            "final_report": [],
            "analysis_complete": True,
        }
    
    enrichment_prompt = f"""Given the following validated threats, provide detailed mitigation strategies and final recommendations.

VALIDATED THREATS:
{json.dumps(valid_threats, indent=2)}

ARCHITECTURE CONTEXT:
{state['hybrid_context'][:1000]}

For each threat, provide:
1. A clear, non-technical description of the risk.
2. Specific, actionable mitigation strategies.
3. Relevant compliance/standard references (NIST, OWASP, CWE).

Return a JSON object matching the requested schema.
"""
    
    try:
        response = litellm.completion(
            model="gemini/gemini-1.5-pro",
            messages=[
                {"role": "system", "content": MEDIATOR_SYSTEM_PROMPT},
                {"role": "user", "content": enrichment_prompt},
            ],
            temperature=0.5,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.debug(f"[Mediator] Enrichment response: {response_text[:200]}...")
        
        report_data = _extract_mediator_json(response_text)
        
        if not report_data or not report_data.get("recommendations"):
            logger.warning("[Mediator] Failed to parse enriched report, using basic structure")
            report_data = _create_basic_report(valid_threats, valid_feedback_map)
            final_threats = report_data
            markdown_report = _generate_markdown_report(final_threats, {}) # empty metadata
        else:
            final_threats: list[FinalThreatReport] = []
            for rec in report_data.get("recommendations", []):
                final_threat: FinalThreatReport = {
                    "threat_id": rec.get("threat_id", ""),
                    "category": rec.get("category", "Unknown"),
                    "target_component": rec.get("target_component", "Unknown"),
                    "attack_vector": rec.get("attack_vector", ""),
                    "severity": rec.get("severity", "High"),
                    "mitigation_strategy": rec.get("mitigation_strategy", ""),
                    "citations": rec.get("citations", []),
                }
                final_threats.append(final_threat)
                
            markdown_report = _generate_markdown_report(final_threats, report_data)

        logger.info(f"[Mediator] Generated final report with {len(final_threats)} threats")
        
        return {
            "final_report": final_threats,
            "analysis_complete": True,
        }
        
    except Exception as e:
        logger.error(f"[Mediator] Exception during report synthesis: {e}", exc_info=True)
        final_threats = _create_basic_report(valid_threats, valid_feedback_map)
        return {
            "final_report": final_threats,
            "analysis_complete": True,
        }


def _extract_mediator_json(response_text: str) -> dict | None:
    """
    Extract JSON report structure from Mediator response.
    
    Args:
        response_text: Raw LLM response
        
    Returns:
        Parsed JSON dict or None on failure
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        if "```json" in response_text:
            start = response_text.find("```json") + len("```json")
            end = response_text.find("```", start)
            if end > start:
                json_str = response_text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        logger.warning("[Mediator] Failed to extract JSON from response")
        return None


def _create_basic_report(
    valid_threats: list[ThreadThreat],
    feedback_map: dict[str, ValidationFeedback]
) -> list[FinalThreatReport]:
    """
    Create basic report structure when LLM enrichment fails.
    
    Args:
        valid_threats: List of validated ThreadThreat objects
        feedback_map: Map of threat_id to ValidationFeedback
        
    Returns:
        List of FinalThreatReport objects
    """
    final_threats: list[FinalThreatReport] = []
    
    severity_map = {
        8.0: "Critical",
        7.0: "High",
        5.0: "Medium",
        3.0: "Low",
    }
    
    for threat in valid_threats:
        feedback = feedback_map.get(threat["threat_id"])
        if not feedback:
            continue
        
        severity = feedback.get("adjusted_severity", "High")
        
        final_threat: FinalThreatReport = {
            "threat_id": threat["threat_id"],
            "category": threat["category"],
            "target_component": threat["target_component"],
            "attack_vector": threat["attack_vector"],
            "severity": severity,
            "mitigation_strategy": f"Implement controls aligned with {feedback.get('citation_reference', 'security best practices')}",
            "citations": [feedback.get("citation_reference", "Unknown")],
        }
        final_threats.append(final_threat)
    
    return final_threats


def _generate_markdown_report(
    final_threats: list[FinalThreatReport],
    metadata: dict
) -> str:
    """
    Generate a formatted Markdown report from final threats.
    
    Args:
        final_threats: List of FinalThreatReport objects
        metadata: Metadata including executive_summary and counts
        
    Returns:
        Formatted markdown string
    """
    markdown = "# Threat Model Report\n\n"
    
    if metadata.get("executive_summary"):
        markdown += f"## Executive Summary\n\n{metadata['executive_summary']}\n\n"
    
    markdown += "## Threat Summary\n\n"
    markdown += f"- **Total Threats:** {metadata.get('total_threats', len(final_threats))}\n"
    markdown += f"- **Critical:** {metadata.get('critical_count', 0)}\n"
    markdown += f"- **High:** {metadata.get('high_count', 0)}\n"
    markdown += f"- **Medium:** {metadata.get('medium_count', 0)}\n\n"
    
    # Detailed Threats
    markdown += "## Detailed Findings\n\n"
    
    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Informational": 4}
    sorted_threats = sorted(
        final_threats,
        key=lambda t: severity_order.get(t["severity"], 5)
    )
    
    for threat in sorted_threats:
        markdown += f"### [{threat['severity'].upper()}] {threat['threat_id']}: {threat['target_component']}\n\n"
        markdown += f"**Category:** {threat['category']}\n\n"
        markdown += f"**Attack Vector:** {threat['attack_vector']}\n\n"
        markdown += f"**Mitigation Strategy:**\n{threat['mitigation_strategy']}\n\n"
        
        if threat["citations"]:
            markdown += f"**References:**\n"
            for citation in threat["citations"]:
                markdown += f"- {citation}\n"
        markdown += "\n---\n\n"
    
    return markdown
