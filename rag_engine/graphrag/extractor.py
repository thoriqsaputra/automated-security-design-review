import uuid
import logging
from typing import List
from pydantic import BaseModel, Field
from litellm import completion
from sqlalchemy.orm import Session
from db.models import ArchitectureEntity, ArchitectureEdge
from ingestion.ai_extractor import clean_llm_json

from llama_index.core import Settings 

logger = logging.getLogger(__name__)

class ExtractedEdge(BaseModel):
    source_entity: str
    target_entity: str
    relation_type: str = Field(..., description="Action verb e.g., 'READS_FROM', 'ENCRYPTS'")
    description: str

class ExtractedEntity(BaseModel):
    name: str = Field(..., description="Name of the component, e.g., 'PostgreSQL', 'Auth Service'")
    type: str = Field(..., description="E.g., 'Database', 'Service', 'User'")
    description: str

class ArchitectureGraph(BaseModel):
    entities: List[ExtractedEntity]
    edges: List[ExtractedEdge]

def process_graph_extraction(text_chunk: str, doc_id: str, db: Session):
    """LiteLLM structured extraction for Architectural GraphRAG."""
    system_prompt = (
        "You are an expert Security Architect. Extract system components and their exact "
        "data-flow relationships from the text.\n\n"
        "Example JSON Output:\n"
        "{\n"
        "  \"entities\": [\n"
        "    {\"name\": \"API Gateway\", \"type\": \"Service\", \"description\": \"Entry point\"}\n"
        "  ],\n"
        "  \"edges\": [\n"
        "    {\"source_entity\": \"API Gateway\", \"target_entity\": \"Auth Service\", \"relation_type\": \"AUTHENTICATES\", \"description\": \"Checks token\"}\n"
        "  ]\n"
        "}"
    )
    
    try:
        response = completion(
            model="openrouter/nvidia/nemotron-3-super-120b-a12b:free",
            # model="openrouter/qwen/qwen3.6-plus:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_chunk}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        raw_output = response.choices[0].message.content
        clean_json_str = clean_llm_json(raw_output)
        graph_data = ArchitectureGraph.model_validate_json(clean_json_str)
        
        existing_entities = db.query(ArchitectureEntity).filter(
            ArchitectureEntity.document_id == doc_id
        ).all()
        
        entity_map = {ent.name: ent.id for ent in existing_entities}
        
        new_entities = []
        for ent in graph_data.entities:
            clean_name = ent.name.strip().upper()
            if clean_name not in entity_map:
                new_entities.append(ent)
                
        if new_entities:
            descriptions = [ent.description for ent in new_entities]
            batch_vectors = Settings.embed_model.get_text_embedding_batch(descriptions)
            
            for ent, vector in zip(new_entities, batch_vectors):
                clean_name = ent.name.strip().upper()
                db_ent = ArchitectureEntity(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    name=clean_name,
                    entity_type=ent.type,
                    description=ent.description,
                    embedding=vector
                )
                db.add(db_ent)
                entity_map[clean_name] = db_ent.id
                
            db.flush()
        
        for edge in graph_data.edges:
            src_name = edge.source_entity.strip().upper()
            tgt_name = edge.target_entity.strip().upper()
            
            if src_name in entity_map and tgt_name in entity_map:
                db.add(ArchitectureEdge(
                    id=str(uuid.uuid4()),
                    source_id=entity_map[src_name],
                    target_id=entity_map[tgt_name],
                    relation_type=edge.relation_type.strip().upper(),
                    description=edge.description
                ))
                
    except Exception as e:
        logger.error(f"Graph extraction failed for chunk: {e}")
        raise