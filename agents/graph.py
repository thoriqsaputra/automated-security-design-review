"""
Multi-Agent Threat Modeling Graph (LangGraph).
Handles conditional routing based on validation feedback and recursion limits.
Natively integrates Langfuse Callbacks for LLM observability.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langfuse.langchain import CallbackHandler
from sqlalchemy.orm import Session

from agents.state import AgentState
from agents.hunter.node import hunter_node
from agents.critic.node import critic_node
from agents.mediator.node import mediator_node
from core.config import settings
from db.session import SessionLocal

logger = logging.getLogger(__name__)

_langfuse_enabled = settings.langfuse_enabled

def get_langfuse_handler(session_id: str):
    """Creates a fresh Langfuse callback handler for the current run."""
    if not _langfuse_enabled:
        return None
    return CallbackHandler(
        secret_key=settings.LANGFUSE_SECRET_KEY,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        host=settings.LANGFUSE_HOST,
        session_id=session_id,
        tags=["milestone_5", "threat_model_debate"]
    )

def should_continue_debate(state: AgentState) -> Literal["hunter", "mediator", "__end__"]:
    """
    Conditional edge routing logic. Prevents infinite loops by checking recursion_iteration.
    """
    current_loop = state.get("recursion_count", 0)
    
    logger.info(f"[Router] Evaluating debate state. Current Loop: {current_loop}/{state['recursion_limit']}")
    
    if current_loop >= state["recursion_limit"]:
        logger.warning("[Router] Maximum recursion reached. Forcing route to Mediator.")
        return "mediator"
    
    feedbacks = state.get("validation_feedbacks", [])

    if not feedbacks:
        logger.warning("[Router] No validation feedbacks available. Routing to Mediator.")
        return "mediator"
    
    invalid_count = sum(1 for f in feedbacks if not f.get("is_valid", False))

    logger.info(f"[Router] Feedback summary: {invalid_count} invalid threats found.")
    
    if invalid_count == 0:
        logger.info("[Router] All proposed threats validated successfully. Routing to Mediator.")
        return "mediator"
    else:
        logger.info("[Router] Critic rejected threats. Routing back to Hunter for refinement.")
        return "hunter"

def build_threat_modeling_graph(db: Session = None) -> StateGraph:
    """Compiles the multi-agent LangGraph."""
    if db is None:
        db = SessionLocal()
        
    graph = StateGraph(AgentState)
    
    graph.add_node("hunter", lambda state: hunter_node(state))
    graph.add_node("critic", lambda state: critic_node(state, db))
    graph.add_node("mediator", lambda state: mediator_node(state))
    
    graph.set_entry_point("hunter")
    
    graph.add_edge("hunter", "critic")
    
    graph.add_conditional_edges(
        "critic",
        should_continue_debate,
        {
            "hunter": "hunter",
            "mediator": "mediator",
            "__end__": END
        }
    )
    
    graph.add_edge("mediator", END)
    
    return graph.compile()


def run_threat_modeling(
    query: str,
    hybrid_context: str,
    document_id: str,
    db: Session = None,
    recursion_limit: int = 2,
    use_langfuse: bool = True,
) -> dict:
    """
    Executes the workflow. Injects Langfuse via LangChain Callbacks.
    """
    initial_state = {
        "query": query,
        "hybrid_context": hybrid_context,
        "document_id": document_id,
        "draft_threats": [],
        "validation_feedbacks": [],
        "recursion_count": 0, 
        "recursion_limit": recursion_limit,
        "final_report": [],
        "analysis_complete": False,
    }

    logger.info(f"[ThreatModeling] Starting AI Debate for document {document_id}")

    graph = build_threat_modeling_graph(db)
    
    callbacks = []
    langfuse_handler = get_langfuse_handler(session_id=document_id) if use_langfuse else None
    if langfuse_handler:
        callbacks.append(langfuse_handler)
        logger.info("[ThreatModeling] Langfuse tracing enabled.")
    
    try:
        final_state = graph.invoke(
            initial_state,
            config={"callbacks": callbacks} if callbacks else {}
        )
        
        logger.info("[ThreatModeling] Workflow completed successfully!")
        return final_state
        
    except Exception as e:
        logger.error(f"[ThreatModeling] Workflow crashed: {e}", exc_info=True)
        raise
    finally:
        if langfuse_handler:
            langfuse_handler.flush()