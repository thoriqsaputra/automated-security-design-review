import os
import re
import time
import logging
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from litellm import completion, embedding
from litellm.exceptions import RateLimitError, APIError, APIConnectionError

logger = logging.getLogger(__name__)

class RequirementNode(BaseModel):
    title: str = Field(..., description="Short exact title of the security requirement.")
    actionable_rule: str = Field(..., description="The comprehensive, pure security constraint or rule without administrative noise.")
    is_child_of: Optional[str] = Field(None, description="If this rule tightly depends on or is a sub-rule of another rule in this list, provide its exact title. Null otherwise.")
    db_id: Optional[str] = Field(None, exclude=True) # Internal tracker field for pgvector

class RequirementsGraphExtraction(BaseModel):
    requirements: List[RequirementNode] = Field(..., description="List of all hierarchical security requirements extracted from the text.")


def clean_llm_json(raw_string: str) -> str:
    """
    Defensively strips markdown code blocks (e.g., ```json ... ```) 
    that LLMs often wrap their outputs in, ensuring valid JSON parsing.
    """
    cleaned = raw_string.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:-3]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:-3]
    return cleaned.strip()


def extract_security_graph(text_content: str, max_retries: int = 4) -> RequirementsGraphExtraction:
    """
    Uses an LLM (via LiteLLM with smart fallbacks) to extract a strict parent-child 
    Knowledge Graph of security requirements from raw text.
    Designed to run safely within a Celery Worker.
    """
    # Safe chunking note: For production, implement a LangChain text splitter here.
    safe_text = text_content[:100000] 

    system_prompt = (
            "You are an expert Cybersecurity AI Data Modeler. Process the raw security documentation "
            "and extract a strict hierarchy of security requirements. "
            "You MUST output valid JSON matching the requested schema.\n\n"
            "Rules for Extraction:\n"
            "1. Extract ONLY pure, actionable security rules (e.g., 'Passwords must be at least 12 characters').\n"
            "2. Discard all administrative fluff, preambles, histories, and non-actionable text.\n"
            "3. Provide a short, precise 'title' for each rule.\n"
            "4. Provide the 'actionable_rule' containing the exact constraint.\n"
            "5. If a rule tightly depends on a broader rule, set 'is_child_of' to the EXACT 'title' of its parent. Otherwise, null.\n"
            "6. Ensure titles are perfectly consistent to map the Adjacency List securely.\n\n"
            "Example Output Format:\n"
            "```json\n"
            "{\n"
            "  \"requirements\": [\n"
            "    {\n"
            "      \"title\": \"Authentication Policy\",\n"
            "      \"actionable_rule\": \"The system must enforce strong authentication for all user access.\",\n"
            "      \"is_child_of\": null\n"
            "    },\n"
            "    {\n"
            "      \"title\": \"Password Minimum Length\",\n"
            "      \"actionable_rule\": \"All user passwords must be a minimum of 12 characters long.\",\n"
            "      \"is_child_of\": \"Authentication Policy\"\n"
            "    },\n"
            "    {\n"
            "      \"title\": \"Session Timeout\",\n"
            "      \"actionable_rule\": \"User sessions must automatically terminate after 15 minutes of inactivity.\",\n"
            "      \"is_child_of\": null\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )

    primary_model = "gemini/gemini-2.5-flash"
    # primary_model = "openrouter/google/gemma-4-26b-a4b-it:free"
    # primary_model = "openrouter/nvidia/nemotron-3-super-120b-a12b:free"
    fallbacks = [
        "openrouter/google/gemma-4-31b-it:free"
    ]

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt+1}/{max_retries}: Extracting rules using {primary_model} (with fallbacks)...")
            
            response = completion(
                model=primary_model,
                # fallbacks=fallbacks,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract rules from this text:\n\n{safe_text}"}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.1
            )

            raw_output = response.choices[0].message.content
            logger.info(f"Raw LLM output:\n{raw_output}")
            clean_json_str = clean_llm_json(raw_output)
            
            return RequirementsGraphExtraction.model_validate_json(clean_json_str)

        except RateLimitError as e:
            wait_time = 2 ** attempt
            logger.warning(f"RateLimitError. Backing off for {wait_time}s... Error: {e}")
            time.sleep(wait_time)
            
        except (APIConnectionError, APIError) as e:
            logger.error(f"Provider API Error during extraction: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
            
        except ValidationError as e:
            logger.error(f"LLM hallucinated invalid JSON structure: {e}")
            raise RuntimeError("Structured output validation failed") from e

    raise RuntimeError("Failed to extract rules from LLM after max retries")

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generates dense vector embeddings for a batch of texts in a SINGLE API call.
    Uses Google's free tier. 
    Crucial for Hybrid GraphRAG insertion in PostgreSQL (pgvector dimension: 768).
    """
    if not texts:
        return []

    try:
        response = embedding(
            model="gemini/gemini-embedding-2-preview", 
            input=texts,
            dimensions=768
        )
        
        return [item['embedding'] for item in response['data']]
        
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        raise RuntimeError("Batch embedding generation failed, aborting DB insertion") from e