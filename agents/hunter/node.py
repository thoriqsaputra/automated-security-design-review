"""
Hunter Agent Node - Offensive threat proposal using STRIDE framework.

The Hunter Agent generates initial threat proposals based on the architecture.
It reads the hybrid_context (text + vision) and produces draft_threats.
"""

import json
import logging
from typing import Any

import litellm

from agents.state import AgentState, ThreadThreat
from core.prompts import HUNTER_SYSTEM_PROMPT
from rag_engine.settings import Settings

logger = logging.getLogger(__name__)


def hunter_node(state: AgentState) -> dict[str, Any]:
    """
    Hunter Agent Node: Propose threats based on architecture.
    """
    current_loop = state.get("recursion_count", 0)
    logger.info(f"[Hunter] Analyzing context for threats (recursion iteration {current_loop})")
    
    feedback_section = ""
    if current_loop > 0 and state.get("validation_feedbacks"):
        logger.info("[Hunter] Injecting Critic feedback into prompt for re-analysis...")
        feedback_section = "\nCRITIC FEEDBACK FROM PREVIOUS ROUND:\n"
        
        for f in state["validation_feedbacks"]:
            if not f.get("is_valid"):
                feedback_section += f"- Rejected Threat {f.get('threat_id')}: {f.get('validation_reason')}\n"
                
        feedback_section += "\nCRITICAL: DO NOT suggest these exact threats again. Pivot your analysis to find new vulnerabilities or fix the technical flaws in your previous attack vectors.\n"

    user_prompt = f"""Analyze the following architecture context and identify potential security threats using the STRIDE framework:

ARCHITECTURE CONTEXT:
{state['hybrid_context']}

ORIGINAL QUERY:
{state['query']}
{feedback_section}

Generate the most critical and realistic threats based on this architecture.
Focus on:
1. Trust boundary crossings and authentication bypasses
2. Data exposure risks in data flows
3. Component-level vulnerabilities
4. Integration points that could be exploited

Remember: Be specific to this architecture, not generic. Every threat must be tied to an actual component or data flow shown above.
"""
    
    try:
        response = litellm.completion(
            model="gemini/gemini-1.5-pro",
            messages=[
                {"role": "system", "content": HUNTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.debug(f"[Hunter] Raw LLM response: {response_text[:200]}...")
        
        threats = _extract_json_threats(response_text)
        
        if not threats:
            logger.warning("[Hunter] No threats extracted from LLM response")
            threats = []
            
        logger.info(f"[Hunter] Generated {len(threats)} draft threats")
        
        return {
            "draft_threats": threats
        }
        
    except Exception as e:
        logger.error(f"[Hunter] Exception during threat generation: {e}", exc_info=True)
        raise


def _extract_json_threats(response_text: str) -> list[ThreadThreat]:
    """
    Extract JSON array of threats from LLM response.
    Handles cases where LLM wraps JSON in markdown code blocks.
    
    Args:
        response_text: Raw text from LLM
        
    Returns:
        List of ThreadThreat dicts (or empty list on parse failure)
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        if "```json" in response_text:
            start = response_text.find("```json") + len("```json")
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                logger.error("[Hunter] Failed to extract JSON from markdown block")
                return []
        elif "```" in response_text:
            start = response_text.find("```") + len("```")
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                logger.error("[Hunter] Failed to extract JSON from code block")
                return []
        else:
            logger.error("[Hunter] No valid JSON found in response")
            return []
    
    if not isinstance(data, list):
        logger.error(f"[Hunter] Expected list of threats, got {type(data)}")
        return []
    
    threats: list[ThreadThreat] = []
    for item in data:
        try:
            threat: ThreadThreat = {
                "threat_id": item.get("threat_id", f"T-{len(threats)+1:03d}"),
                "category": item.get("category", "Elevation_of_Privilege"),
                "target_component": item.get("target_component", "Unknown"),
                "attack_vector": item.get("attack_vector", ""),
                "cvss_estimate": float(item.get("cvss_estimate", 5.0)),
                "confidence": float(item.get("confidence", 0.5)),
            }
            threats.append(threat)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"[Hunter] Skipping malformed threat item: {e}")
            continue
    
    return threats
