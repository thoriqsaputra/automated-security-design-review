"""
Integration test for Milestone 5: Multi-Agent Threat Modeling Workflow.
Fully mocked for offline, zero-cost execution.
"""

import json
import logging
import os
import sys
import uuid
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.state import AgentState
from agents.graph import build_threat_modeling_graph, run_threat_modeling
from db.session import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

MOCK_HUNTER_RESPONSE = json.dumps([
    {
        "threat_id": "T-001",
        "category": "Spoofing",
        "target_component": "API Gateway",
        "attack_vector": "Attacker could bypass JWT validation if signature verification is missing",
        "cvss_estimate": 8.2,
        "confidence": 0.9,
    },
    {
        "threat_id": "T-002",
        "category": "Tampering",
        "target_component": "Message Queue",
        "attack_vector": "Unencrypted message transit allows queue poisoning",
        "cvss_estimate": 7.5,
        "confidence": 0.85,
    },
    {
        "threat_id": "T-003",
        "category": "Elevation_of_Privilege",
        "target_component": "Database",
        "attack_vector": "SQL injection in user input parameters",
        "cvss_estimate": 9.1,
        "confidence": 0.95,
    },
])

MOCK_CRITIC_RESPONSES = {
    "T-001": json.dumps([
        {
            "threat_id": "T-001",
            "is_valid": True,
            "validation_reason": "JWT bypass is realistic. OWASP API1:2023 explicitly warns about broken authentication.",
            "adjusted_severity": "Critical",
            "citation_reference": "OWASP API1:2023 - Broken Object Level Authorization",
        }
    ]),
    "T-002": json.dumps([
        {
            "threat_id": "T-002",
            "is_valid": True,
            "validation_reason": "Unencrypted message transit is a known vulnerability. NIST SP 800-52 requires TLS.",
            "adjusted_severity": "High",
            "citation_reference": "NIST SP 800-52 Rev 2 - Guidelines for TLS Implementations",
        }
    ]),
    "T-003": json.dumps([
        {
            "threat_id": "T-003",
            "is_valid": False,
            "validation_reason": "Architecture shows parameterized queries. SQL injection threat is a hallucination.",
            "adjusted_severity": "Low",
            "citation_reference": "NIST SP 800-92 - Guide to Computer Security Log Management",
        }
    ]),
}

MOCK_MEDIATOR_RESPONSE = json.dumps({
    "executive_summary": "The analyzed architecture has 2 critical security issues: JWT authentication bypass and unencrypted message transit. Immediate remediation required.",
    "total_threats": 2,
    "critical_count": 1,
    "high_count": 1,
    "medium_count": 0,
    "recommendations": [
        {
            "threat_id": "T-001",
            "category": "Spoofing",
            "target_component": "API Gateway",
            "severity": "Critical",
            "description": "The API Gateway lacks proper JWT signature verification.",
            "attack_vector": "Attacker could bypass JWT validation if signature verification is missing",
            "mitigation_strategy": "Implement strict JWT signature validation using RS256 or ES256. Rotate keys regularly. Validate expiration and issuer.",
            "citations": ["OWASP API1:2023", "NIST SP 800-52"],
        },
        {
            "threat_id": "T-002",
            "category": "Tampering",
            "target_component": "Message Queue",
            "severity": "High",
            "description": "Message queue communication lacks encryption.",
            "attack_vector": "Unencrypted message transit allows queue poisoning",
            "mitigation_strategy": "Enable TLS 1.3 encryption for all message queue connections. Implement message signing and authentication.",
            "citations": ["NIST SP 800-52 Rev 2", "OWASP A02:2021"],
        },
    ],
})

def mock_litellm_completion(model, messages, **kwargs):
    """Mock litellm.completion to return predefined responses."""
    system_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "")
    
    if "Offensive Security Analyst" in system_msg:
        response_content = MOCK_HUNTER_RESPONSE
    elif "Defensive Security Validator" in system_msg:
        threat_id = None
        if "T-001" in user_msg: threat_id = "T-001"
        elif "T-002" in user_msg: threat_id = "T-002"
        elif "T-003" in user_msg: threat_id = "T-003"
        response_content = MOCK_CRITIC_RESPONSES.get(threat_id, "{}")
    elif "Security Report Compiler" in system_msg:
        response_content = MOCK_MEDIATOR_RESPONSE
    else:
        response_content = '{"status": "unknown"}'
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_content
    return mock_response

@patch("agents.critic.node.HybridThreatRetriever")
@patch("litellm.completion", side_effect=mock_litellm_completion)
def test_threat_modeling_workflow(mock_completion, MockRetriever):
    logger.info("=== STARTING MILESTONE 5 MULTI-AGENT TEST ===")
    
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.retrieve.return_value = []
    MockRetriever.return_value = mock_retriever_instance

    db = MagicMock()
    doc_id = str(uuid.uuid4())
    
    sample_query = "Analyze the API Gateway architecture for security threats"
    sample_context = "API Gateway: FastAPI server listening on :8080 with JWT authentication..."
    
    try:
        final_state = run_threat_modeling(
            query=sample_query,
            hybrid_context=sample_context,
            document_id=doc_id,
            db=db,
            recursion_limit=2,
            use_langfuse=False,
        )
        
        logger.info("=== VALIDATING RESULTS ===")
        assert len(final_state["draft_threats"]) > 0
        assert len(final_state["validation_feedbacks"]) > 0
        assert len(final_state["final_report"]) > 0
        
        final_threat_ids = {t["threat_id"] for t in final_state["final_report"]}
        invalid_ids = {f["threat_id"] for f in final_state["validation_feedbacks"] if not f.get("is_valid")}
        
        assert not (final_threat_ids & invalid_ids), "Final report should not contain invalid threats"
        assert final_state["analysis_complete"]
        
        logger.info("\n=== ALL TESTS PASSED ===")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


@patch("agents.critic.node.HybridThreatRetriever")
@patch("litellm.completion", side_effect=mock_litellm_completion)
def test_recursion_limit_enforcement(mock_completion, MockRetriever):
    logger.info("=== TESTING RECURSION LIMIT ENFORCEMENT ===")
    
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.retrieve.return_value = []
    MockRetriever.return_value = mock_retriever_instance

    db = MagicMock()
    doc_id = str(uuid.uuid4())
    
    final_state = run_threat_modeling(
        query="Test recursion",
        hybrid_context="Test context",
        document_id=doc_id,
        db=db,
        recursion_limit=1,
        use_langfuse=False,
    )
    
    assert final_state.get("recursion_count", 0) <= 2, "Should respect recursion limit"
    logger.info(f"[Test] ✓ Recursion limit enforced: {final_state.get('recursion_count')} iterations")


def test_graph_compilation():
    logger.info("=== TESTING GRAPH COMPILATION ===")
    db = MagicMock()
    graph = build_threat_modeling_graph(db)
    assert graph is not None, "Graph should compile"
    logger.info("[Test] ✓ Graph compiled successfully")


if __name__ == "__main__":
    logger.info("Starting Milestone 5 Integration Tests...")
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "-s"],
        cwd=os.path.dirname(__file__) or "."
    )
    sys.exit(result.returncode)