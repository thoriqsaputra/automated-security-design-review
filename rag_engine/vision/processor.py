import uuid
import logging
import base64
from typing import List
from difflib import SequenceMatcher
from litellm import completion
from sqlalchemy.orm import Session

# Database Models
from db.models import ArchitectureEntity, ArchitectureEdge, DocumentNode

# LlamaIndex & Utilities
from llama_index.core import Settings
from ingestion.ai_extractor import clean_llm_json
from rag_engine.vision.schemas import ArchitectureDiagramExtraction

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = """
You are a world-class Visual Systems Architect and security expert specializing in parsing Technical System Design (TSD) architecture diagrams.

Your task is to analyze the provided architecture diagram image and extract:
1. **Nodes (Components):** Every labeled box, cloud service, database, or system component visible in the diagram.
2. **Edges (Data Flows):** Every line, arrow, or connection between components with directional information.
3. **Trust Boundaries:** Perimeter boxes, VPC boundaries, security zones, or administrative domains that contain multiple components.

CRITICAL RULES:
- Component names MUST be EXACT and UPPERCASE (e.g., "REACT FRONTEND", "POSTGRESQL DATABASE", "API GATEWAY").
- Coordinates MUST use a normalized 0-1000 scale (ymin, xmin, ymax, xmax).
- Do NOT invent components. If you cannot clearly identify it, mark it as "UNIDENTIFIED_COMPONENT_X".
- For each data flow, the source/target MUST exactly match the names of the components you extracted.
- For Trust Boundaries, list the exact names of the components contained WITHIN that boundary.

OUTPUT JSON STRUCTURE (STRICT - DO NOT DEVIATE):
{
  "components": [
    {
      "name": "REACT FRONTEND",
      "type": "Frontend",
      "coordinates": {"ymin": 50, "xmin": 100, "ymax": 150, "xmax": 250},
      "labels_on_box": ["v2.1", "Deployed on Vercel"]
    }
  ],
  "data_flows": [
    {
      "source": "REACT FRONTEND",
      "target": "API GATEWAY",
      "flow_label": "REST API calls",
      "protocol": "HTTPS",
      "direction": "bidirectional"
    }
  ],
  "trust_boundaries": [
    {
      "boundary_name": "Production VPC",
      "contained_components": ["API GATEWAY", "MICROSERVICE_AUTH", "POSTGRESQL"],
      "boundary_type": "Network Perimeter"
    }
  ]
}

Respond ONLY with valid JSON. No markdown, no code blocks, no explanations.
"""

def fuzzy_match_entity(visual_name: str, existing_entities: List[ArchitectureEntity], threshold: float = 0.75) -> ArchitectureEntity:
    """
    Performs fuzzy string matching to find existing entities that match the vision-extracted name.
    Returns the best match if similarity > threshold, else None.
    """
    best_match = None
    best_score = 0.0
    
    for entity in existing_entities:
        visual_normalized = visual_name.upper().strip()
        entity_normalized = entity.name.upper().strip()
        
        similarity = SequenceMatcher(None, visual_normalized, entity_normalized).ratio()
        
        if similarity > best_score:
            best_score = similarity
            best_match = entity
            
    if best_score >= threshold:
        logger.info(f"Fuzzy matched Vision Node '{visual_name}' to Text Node '{best_match.name}' (score: {best_score:.2f})")
        return best_match
        
    return None

def process_architecture_diagram(image_bytes: bytes, doc_id: str, db: Session):
    """
    Sends an architecture diagram image to Gemini 1.5 Pro for spatial analysis.
    Extracts components, data flows, and trust boundaries.
    Merges vision results with text-extracted entities using fuzzy matching.
    """
    
    image_base64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    
    existing_entities = db.query(ArchitectureEntity).filter(
        ArchitectureEntity.document_id == doc_id
    ).all()
    
    try:
        logger.info("Sending diagram to Gemini 1.5 Pro for spatial analysis...")
        
        response = completion(
            model="gemini/gemini-1.5-pro",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_SYSTEM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        raw_output = response.choices[0].message.content
        clean_json_str = clean_llm_json(raw_output)
        diagram_data = ArchitectureDiagramExtraction.model_validate_json(clean_json_str)
        
        logger.info(f"Vision extraction successful: {len(diagram_data.components)} components, {len(diagram_data.data_flows)} flows")
        
        boundary_map = {}
        for boundary in diagram_data.trust_boundaries:
            for comp_name in boundary.contained_components:
                boundary_map[comp_name.strip().upper()] = boundary.boundary_name

        entity_map = {}  # Maps visual_name -> entity_id
        
        for vis_component in diagram_data.components:
            inferred_zone = boundary_map.get(vis_component.name.strip().upper(), "Unknown")
            
            matched_entity = fuzzy_match_entity(vis_component.name, existing_entities)
            
            if matched_entity:
                matched_entity.spatial_metadata = {
                    "coordinates": vis_component.coordinates.model_dump(),
                    "trust_zone": inferred_zone,
                    "labels_on_diagram": vis_component.labels_on_box,
                    "source": "hybrid_merge"
                }
                db.add(matched_entity)
                entity_map[vis_component.name] = matched_entity.id
            else:
                logger.info(f"Creating new entity exclusively from vision: {vis_component.name}")
                
                vector = Settings.embed_model.get_text_embedding(
                    f"{vis_component.name} ({vis_component.type}): {'; '.join(vis_component.labels_on_box)}"
                )
                
                new_entity = ArchitectureEntity(
                    id=str(uuid.uuid4()),
                    document_id=doc_id,
                    name=vis_component.name,
                    entity_type=vis_component.type,
                    description=f"[Vision Extracted] Additional Labels: {'; '.join(vis_component.labels_on_box)}",
                    embedding=vector,
                    spatial_metadata={
                        "coordinates": vis_component.coordinates.model_dump(),
                        "trust_zone": inferred_zone,
                        "labels_on_diagram": vis_component.labels_on_box,
                        "source": "vision_only"
                    }
                )
                db.add(new_entity)
                db.flush()
                entity_map[vis_component.name] = new_entity.id
                
        db.commit()
        
        for flow in diagram_data.data_flows:
            src_id = entity_map.get(flow.source)
            tgt_id = entity_map.get(flow.target)
            
            if src_id and tgt_id:
                edge = ArchitectureEdge(
                    id=str(uuid.uuid4()),
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=flow.protocol or flow.flow_label,
                    description=f"[Vision Extraction] Flow: {flow.flow_label} ({flow.direction})"
                )
                db.add(edge)
            else:
                logger.warning(f"Skipped visual edge {flow.source} -> {flow.target}: Endpoint not found.")
                
        if diagram_data.trust_boundaries:
            boundary_text = "Architecture Visual Trust Boundaries:\n"
            for b in diagram_data.trust_boundaries:
                boundary_text += f"- The '{b.boundary_name}' ({b.boundary_type}) explicitly contains: {', '.join(b.contained_components)}.\n"
                
            boundary_vector = Settings.embed_model.get_text_embedding(boundary_text)
            
            synthetic_node = DocumentNode(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                text_content=boundary_text,
                layer=0,
                embedding=boundary_vector
            )
            db.add(synthetic_node)
            logger.info("Injected Trust Boundaries into RAPTOR DocumentNodes.")

        db.commit()
        logger.info(f"Vision processing and Hybrid Context merge complete for document {doc_id}")
        
    except Exception as e:
        logger.error(f"Vision extraction pipeline failed: {e}")
        db.rollback()
        raise RuntimeError(f"Diagram analysis failed: {e}") from e