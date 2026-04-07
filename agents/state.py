"""
Agent State Definition for Multi-Agent Threat Modeling Workflow.
Uses LangGraph Reducers (Annotated + operator.add) to prevent state erasure.
"""

import operator
from typing import TypedDict, Literal, Annotated

class ThreadThreat(TypedDict):
    """Individual threat extracted by Hunter Agent."""
    threat_id: str
    category: Literal["Spoofing", "Tampering", "Repudiation", "Information_Disclosure", "Denial_of_Service", "Elevation_of_Privilege"]
    target_component: str
    attack_vector: str
    cvss_estimate: float
    confidence: float

class ValidationFeedback(TypedDict):
    """Feedback from Critic Agent on a specific threat."""
    threat_id: str
    is_valid: bool
    validation_reason: str
    adjusted_severity: Literal["Critical", "High", "Medium", "Low", "Informational"]
    citation_reference: str 
    recursion_iteration: int 

class FinalThreatReport(TypedDict):
    """Final threat item in the synthesized report."""
    threat_id: str
    category: str
    target_component: str
    attack_vector: str
    severity: Literal["Critical", "High", "Medium", "Low", "Informational"]
    mitigation_strategy: str
    citations: list[str]

class AgentState(TypedDict):
    """
    Shared state for the multi-agent threat modeling graph.
    """
    query: str
    hybrid_context: str 
    document_id: str
    
    draft_threats: list[ThreadThreat] 
    
    validation_feedbacks: Annotated[list[ValidationFeedback], operator.add] 
    
    recursion_count: Annotated[int, operator.add] 
    recursion_limit: int 
    
    final_report: list[FinalThreatReport] 
    analysis_complete: bool