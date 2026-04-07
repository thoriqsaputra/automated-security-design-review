"""
Critic Agent Node - Defensive validation against NIST/OWASP knowledge base.

The Critic Agent evaluates Hunter's draft threats by:
1. Querying the HybridThreatRetriever for relevant NIST/OWASP guidelines
2. Cross-referencing architecture details
3. Marking threats as valid or hallucinations
"""

import json
import logging
from typing import Any

import litellm
from sqlalchemy.orm import Session

from agents.state import AgentState, ValidationFeedback
from core.prompts import CRITIC_SYSTEM_PROMPT, KB_QUERY_TEMPLATE
from rag_engine.retriever import HybridThreatRetriever
from db.session import SessionLocal

logger = logging.getLogger(__name__)


def critic_node(state: AgentState, db: Session = None) -> dict[str, Any]:
    """
    Critic Agent Node: Validate threats against knowledge base.
    """
    if db is None:
        db = SessionLocal()
        
    current_loop = state.get("recursion_count", 0)
    logger.info(f"[Critic] Validating {len(state['draft_threats'])} draft threats (Loop {current_loop})")
    
    if not state.get("draft_threats"):
        logger.info("[Critic] No threats to validate, returning empty feedbacks")
        return {
            "validation_feedbacks": [],
            "recursion_count": 1  # Always increment, even if skipping
        }
    
    retriever = HybridThreatRetriever(db=db, doc_id=state["document_id"])
    validation_feedbacks: list[ValidationFeedback] = []
    
    for threat in state["draft_threats"]:
        logger.debug(f"[Critic] Validating threat {threat['threat_id']}: {threat['target_component']}")
        
        kb_query = KB_QUERY_TEMPLATE.format(
            component=threat["target_component"],
            category=threat["category"],
            attack_vector=threat["attack_vector"][:50],
        )
        
        try:
            kb_results = retriever.retrieve(kb_query)
            kb_context = "\n\n".join([node.get_content() for node in kb_results])
        except Exception as e:
            logger.warning(f"[Critic] KB retrieval failed: {e}, using minimal context")
            kb_context = "(No knowledge base results available)"
            
        validation_prompt = f"""Validate the following threat against the knowledge base and architecture context.

CURRENT DEBATE ITERATION: {current_loop}

THREAT TO VALIDATE:
- ID: {threat['threat_id']}
- Category: {threat['category']}
- Target Component: {threat['target_component']}
- Attack Vector: {threat['attack_vector']}
- Proposed CVSS: {threat['cvss_estimate']}

ARCHITECTURE CONTEXT:
{state['hybrid_context'][:1000]}

RELEVANT KNOWLEDGE BASE GUIDELINES:
{kb_context[:1500]}

Determine if this threat is:
1. Valid and realistic based on the architecture and guidelines.
2. A hallucination or unsupported claim.
3. What severity level it should be (based on actual architectural controls).
"""
        
        try:
            response = litellm.completion(
                model="gemini/gemini-1.5-pro",
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": validation_prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"[Critic] Response for {threat['threat_id']}: {response_text[:150]}...")
            
            feedback = _extract_validation_feedback(response_text, threat["threat_id"], current_loop)
            
            if feedback:
                validation_feedbacks.append(feedback)
            else:
                logger.warning(f"[Critic] Failed to parse feedback for {threat['threat_id']}")
                
        except Exception as e:
            logger.error(f"[Critic] Exception validating threat {threat['threat_id']}: {e}", exc_info=True)
            validation_feedbacks.append({
                "threat_id": threat["threat_id"],
                "is_valid": True,
                "validation_reason": f"LLM evaluation failed, assuming valid: {str(e)[:100]}",
                "adjusted_severity": "High",
                "citation_reference": "Manual review required",
                "recursion_iteration": current_loop,
            })
            
    valid_count = sum(1 for f in validation_feedbacks if f["is_valid"])
    logger.info(f"[Critic] Validation complete. {valid_count}/{len(validation_feedbacks)} threats validated.")
    
    return {
        "validation_feedbacks": validation_feedbacks,
        "recursion_count": 1
    }


def _extract_validation_feedback(response_text: str, threat_id: str, iteration: int) -> ValidationFeedback | None:
    """Extracts JSON feedback, handling Markdown blocks gracefully."""
    try:
        json_str = response_text
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
            
        data = json.loads(json_str)
        
        item = data[0] if isinstance(data, list) and data else data
        if not item or not isinstance(item, dict):
            return None
            
        feedback: ValidationFeedback = {
            "threat_id": threat_id,
            "is_valid": bool(item.get("is_valid", True)),
            "validation_reason": str(item.get("validation_reason", "No reason provided")),
            "adjusted_severity": item.get("adjusted_severity", "High"),
            "citation_reference": str(item.get("citation_reference", "Unknown")),
            "recursion_iteration": iteration,
        }
        return feedback
        
    except Exception as e:
        logger.warning(f"[Critic] JSON parse failed: {e}. Falling back to text matching.")
        if "is_valid: true" in response_text.lower() or '"is_valid": true' in response_text.lower():
            return {
                "threat_id": threat_id,
                "is_valid": True,
                "validation_reason": response_text[:200],
                "adjusted_severity": "High",
                "citation_reference": "Extracted via regex fallback",
                "recursion_iteration": iteration,
            }
        return None