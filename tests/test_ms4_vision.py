import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uuid
import logging
from unittest.mock import patch, MagicMock
from db.session import SessionLocal
from db.models import TSDDocument, ArchitectureEntity, ArchitectureEdge, DocumentNode
from rag_engine.settings import configure_rag_settings
from rag_engine.vision.processor import process_architecture_diagram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOCK_GEMINI_JSON = """
{
  "components": [
    {
      "name": "API GATEWAY",
      "type": "Gateway",
      "coordinates": {"ymin": 100, "xmin": 100, "ymax": 200, "xmax": 200},
      "labels_on_box": ["Kong"]
    },
    {
      "name": "MAIN DATABASE",
      "type": "Database",
      "coordinates": {"ymin": 300, "xmin": 300, "ymax": 400, "xmax": 400},
      "labels_on_box": ["PostgreSQL v15"]
    }
  ],
  "data_flows": [
    {
      "source": "API GATEWAY",
      "target": "MAIN DATABASE",
      "flow_label": "Reads User Data",
      "protocol": "TCP",
      "direction": "unidirectional"
    }
  ],
  "trust_boundaries": [
    {
      "boundary_name": "Secure Subnet",
      "contained_components": ["API GATEWAY", "MAIN DATABASE"],
      "boundary_type": "VPC"
    }
  ]
}
"""

@patch('rag_engine.vision.processor.completion')
def test_milestone_4_vision_processing(mock_completion):
    logger.info("--- STARTING MILESTONE 4 VISION PROCESSING TEST (MOCKED) ---")
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = MOCK_GEMINI_JSON
    mock_completion.return_value = mock_response

    configure_rag_settings()
    db = SessionLocal()
    doc_id = str(uuid.uuid4())
    
    try:
        dummy_doc = TSDDocument(id=doc_id, title="Architecture Diagram Test")
        db.add(dummy_doc)
        
        text_entity = ArchitectureEntity(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            name="API GATEWAY",
            entity_type="Service",
            description="[Text Extracted] Main API entry point",
            embedding=[0.0] * 768
        )
        db.add(text_entity)
        db.commit()
        
        logger.info("Processing architecture diagram (Mocked API Call)...")
        fake_image_bytes = b"fake_image_data"
        process_architecture_diagram(fake_image_bytes, doc_id, db)
        
        logger.info("Validating Database State...")
        
        api_gateway = db.query(ArchitectureEntity).filter(
            ArchitectureEntity.name == "API GATEWAY",
            ArchitectureEntity.document_id == doc_id
        ).all()
        assert len(api_gateway) == 1, "Failed: API Gateway was duplicated!"
        assert api_gateway[0].spatial_metadata is not None, "Failed: API Gateway did not receive spatial metadata from vision!"
        assert api_gateway[0].spatial_metadata["trust_zone"] == "Secure Subnet", "Failed: Trust Zone not mapped to existing entity!"
        
        main_db = db.query(ArchitectureEntity).filter(
            ArchitectureEntity.name == "MAIN DATABASE",
            ArchitectureEntity.document_id == doc_id
        ).first()
        assert main_db is not None, "Failed: MAIN DATABASE was not created!"
        
        entity_ids = [api_gateway[0].id, main_db.id]
        edge_count = db.query(ArchitectureEdge).filter(
            ArchitectureEdge.source_id.in_(entity_ids)
        ).count()
        assert edge_count == 1, f"Failed: Expected 1 edge for doc, found {edge_count}"
        
        raptor_nodes = db.query(DocumentNode).filter(DocumentNode.document_id == doc_id).count()
        assert raptor_nodes == 1, "Failed: Synthetic Trust Boundary RAPTOR node was not created!"
        
        logger.info("SUCCESS: All deduplication, spatial mapping, and synthetic node logic works perfectly!")
        
    except AssertionError as e:
        logger.error(f"TEST FAILED ON ASSERTION: {e}")
    except Exception as e:
        logger.error(f"TEST CRASHED: {e}")
        
    finally:
        doc_to_delete = db.query(TSDDocument).filter(TSDDocument.id == doc_id).first()
        if doc_to_delete:
            db.delete(doc_to_delete)
        db.commit()
        db.close()

if __name__ == "__main__":
    test_milestone_4_vision_processing()