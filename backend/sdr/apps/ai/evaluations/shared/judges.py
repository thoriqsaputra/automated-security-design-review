import json
import logging
from typing import Dict, List, Any

from sdr.apps.ai.client.manager import ai_service_manager
from sdr.apps.ai.retrieval.postprocessing.quote_grounding import is_quote_grounded

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """You are an expert evaluator for an AI Security Auditor.
Your task is to calculate the 'Faithfulness' score of an AI's response.
Faithfulness measures whether the AI's claims are entirely deducible from the provided retrieved context. It penalizes hallucination.

Given a Question, Retrieved Context, and the AI's Answer:
1. Extract a list of distinct factual claims made in the AI's Answer — i.e. positive
   assertions that something IS true or IS implemented.
2. For each claim, check if it can be directly inferred from the Retrieved Context.

IMPORTANT: Statements expressing uncertainty, refusal, or the ABSENCE of information
(e.g. "I cannot determine", "the context does not specify X", "there is no information
about Y", "this excerpt does not state Z") are NOT factual claims — they assert nothing
about the world that could be hallucinated. Do not extract these as claims. If the
Answer consists entirely of such non-claims, return an empty "claims" list and a
faithfulness_score of 1.0.

Return ONLY a JSON object:
{
    "claims": [
        {
            "claim": "The extracted claim",
            "is_faithful": true/false
        }
    ],
    "faithfulness_score": <float between 0.0 and 1.0> // (faithful claims / total claims)
}
"""

_REFUSAL_PHRASES = ("i cannot determine", "cannot be determined", "insufficient context", "i don't know")


def _is_pure_refusal(answer: str) -> bool:
    """True if the answer is, in substance, nothing but a refusal/abstention —
    no positive factual claim is made, so there's nothing to hallucinate."""
    normalized = (answer or "").strip().strip(".").strip().lower()
    return any(normalized == phrase or normalized.startswith(phrase) for phrase in _REFUSAL_PHRASES) and len(
        normalized
    ) < 40

CONTEXT_RECALL_PROMPT = """You are an expert evaluator for an AI Security Auditor.
Your task is to calculate the 'Context Recall' score of a Retrieval-Augmented Generation (RAG) system.
Context Recall measures whether the retrieved context contains ALL the information necessary to match the Ground Truth Context.

Given a Question, a Ground Truth Context, and the Retrieved Context from the system:
1. Break down the Ground Truth Context into distinct informational statements.
2. For each statement, check if it can be found in the Retrieved Context.

Return ONLY a JSON object:
{
    "statements": [
        {
            "statement": "The statement from Ground Truth",
            "is_retrieved": true/false
        }
    ],
    "context_recall_score": <float between 0.0 and 1.0> // (retrieved statements / total statements)
}
"""

def judge_faithfulness(question: str, retrieved_context: str, answer: str) -> Dict[str, Any]:
    if _is_pure_refusal(answer):
        return {"claims": [], "faithfulness_score": 1.0}

    user_prompt = f"Question: {question}\n\nRetrieved Context:\n{retrieved_context}\n\nAI Answer:\n{answer}"
    
    response = ai_service_manager.chat_completion_with_fallback(
        messages=[
            {"role": "system", "content": FAITHFULNESS_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        component="eval_judge",
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    if response.error:
        logger.error(f"Faithfulness judge failed: {response.error}")
        return {"faithfulness_score": 0.0, "error": response.error}
        
    try:
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"Failed to parse Faithfulness response: {e}")
        return {"faithfulness_score": 0.0, "error": "parse_failed"}

def judge_context_recall(question: str, ground_truth: str, retrieved_context: str) -> Dict[str, Any]:
    user_prompt = f"Question: {question}\n\nGround Truth Context:\n{ground_truth}\n\nRetrieved Context:\n{retrieved_context}"
    
    response = ai_service_manager.chat_completion_with_fallback(
        messages=[
            {"role": "system", "content": CONTEXT_RECALL_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        component="eval_judge",
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    if response.error:
        logger.error(f"Context Recall judge failed: {response.error}")
        return {"context_recall_score": 0.0, "error": response.error}
        
    try:
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"Failed to parse Context Recall response: {e}")
        return {"context_recall_score": 0.0, "error": "parse_failed"}


DIAGRAM_RELEVANCE_PROMPT = """You are an expert evaluator building ground truth \
for a diagram-retrieval evaluation in an AI Security Auditor system.

You will be shown a real architecture/security diagram and a single candidate \
security requirement. Your only \
job is to judge: is this requirement genuinely CHECKABLE from this diagram — i.e. \
could a well-annotated version of this diagram plausibly show evidence that \
would let a reviewer determine whether the requirement is met or not met?

This is a scope/relevance judgment, NOT a met/not_met verdict — do not judge \
whether the diagram shows the control correctly, only whether this diagram \
TYPE and CONTENT is capable of depicting this kind of control at all. A \
requirement about network segmentation is relevant to an architecture diagram \
even if segmentation isn't actually drawn (that would be a "not_met" \
judgment, made elsewhere) — it is NOT relevant if the requirement is about \
something a diagram fundamentally cannot show regardless of quality (e.g. \
process/documentation checks like "verify a security analysis was performed").

Return ONLY a JSON object:
{
    "relevant": true/false,
    "reasoning": "<one or two sentences explaining why, referencing what the diagram shows or what kind of diagram it is>"
}
"""


def judge_diagram_requirement_relevance(
    image_bytes: bytes,
    requirement_text: str,
    verification_hint: str,
    image_format: str = "png",
) -> Dict[str, Any]:
    """Vision LLM judge for diagram ground-truth relevance labeling — replaces
    manual human review of `build_diagram_ground_truth_template.py`'s output.
    Uses component="eval_judge" (a different model family from Hunter/Critic/
    Mediator) so this isn't the system grading its own homework."""
    user_prompt = (
        f"Candidate requirement: {requirement_text}\n\n"
        f"Verification hint: {verification_hint}\n\n"
        "Is this requirement checkable from the attached diagram?"
    )

    response = ai_service_manager.chat_completion_with_fallback(
        messages=[
            {"role": "system", "content": DIAGRAM_RELEVANCE_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        component="eval_judge",
        response_format={"type": "json_object"},
        temperature=0.0,
        image_bytes=image_bytes,
        image_format=image_format,
    )

    if response.error:
        logger.error(f"Diagram relevance judge failed: {response.error}")
        return {"relevant": None, "reasoning": "", "error": response.error}

    try:
        result = json.loads(response.content)
        result.setdefault("relevant", None)
        result.setdefault("reasoning", "")
        return result
    except Exception as e:
        logger.error(f"Failed to parse diagram relevance judge response: {e}")
        return {"relevant": None, "reasoning": "", "error": "parse_failed"}


def judge_faithfulness_deterministic(
    answer_quotes: List[str], retrieved_context_blocks: Dict[str, str]
) -> float:
    """
    Non-LLM faithfulness cross-check: what fraction of the quotes the answer
    claims as evidence are actually grounded (contiguous-match, same 85%
    coverage rule as the Critic's citation validator) in the retrieved
    blocks. Use this to sanity-check the LLM judge's faithfulness score —
    large, systematic disagreement between the two means the LLM judge
    itself needs recalibration, not just the system under test.
    """
    if not answer_quotes:
        return 1.0

    block_texts = list(retrieved_context_blocks.values())
    grounded_count = 0
    for quote in answer_quotes:
        if any(is_quote_grounded(quote, block_text) for block_text in block_texts):
            grounded_count += 1

    return grounded_count / len(answer_quotes)
