import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import uuid
import logging
from db.session import SessionLocal
from db.models import TSDDocument, DocumentNode, ArchitectureEntity, ArchitectureEdge
from rag_engine.settings import configure_rag_settings
from rag_engine.pipeline import build_raptor_and_graph
from rag_engine.retriever import HybridThreatRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_milestone_3_integration():
    logger.info("--- STARTING MILESTONE 3 INTEGRATION TEST ---")
    
    configure_rag_settings()
    
    db = SessionLocal()
    doc_id = str(uuid.uuid4())
    
    try:
        dummy_doc = TSDDocument(id=doc_id, title="Payment System Architecture v1")
        db.add(dummy_doc)
        db.commit()
        logger.info(f"Created Dummy Document ID: {doc_id}")
        
        text_chunks = [
            "The system uses a React Frontend deployed on Vercel.",
            "The React Frontend sends API requests to the Python FastAPI Gateway.",
            "The FastAPI Gateway authenticates users via the Auth0 Service.",
            "Once authenticated, the FastAPI Gateway reads and writes data to the PostgreSQL Database."
        ]
        
        logger.info("Running RAG Engine Pipeline (Calling Gemini APIs)...")
        build_raptor_and_graph(dummy_doc, text_chunks, db)
        
        logger.info("Validating PostgreSQL Database State...")
        
        node_count = db.query(DocumentNode).filter(DocumentNode.document_id == doc_id).count()
        entity_count = db.query(ArchitectureEntity).filter(ArchitectureEntity.document_id == doc_id).count()
        
        assert node_count > 0, "Failed: RAPTOR did not save any DocumentNodes!"
        assert entity_count > 0, "Failed: GraphRAG did not extract any ArchitectureEntities!"
        
        logger.info(f"SUCCESS: Found {node_count} text nodes and {entity_count} graph entities.")
        
        logger.info("\nTesting Hybrid Retriever Engine...")
        
        retriever = HybridThreatRetriever(db=db, doc_id=doc_id, top_k=2)
        
        query = "What databases are connected to the Gateway?"
        
        results = retriever.retrieve(query)
        
        assert len(results) > 0, "Failed: Retriever returned no context!"
        
        print(f"\n--- THREAT QUERY: '{query}' ---")
        print("--- RETRIEVED CONTEXT ---")
        
        for idx, node_with_score in enumerate(results):
            text = node_with_score.node.get_content()
            if "Architecture Topology Context:" in text:
                print("\n[Graph Topology (Node Edges)]:")
                print(text.replace("Architecture Topology Context:\n", ""))
            else:
                print(f"\n[RAPTOR Summary {idx+1}] (Score: {node_with_score.score:.2f}):")
                print(text)

        print("\n--- TEST PASSED SUCCESSFULLY! ---")

    except AssertionError as ae:
        logger.error(f"TEST FAILED on Assertion: {ae}")
    except Exception as e:
        logger.error(f"TEST CRASHED: {e}")
        
    finally:
        logger.info("\nCleaning up database...")
        doc_to_delete = db.query(TSDDocument).filter(TSDDocument.id == doc_id).first()
        if doc_to_delete:
            db.delete(doc_to_delete)
            db.commit()
            logger.info(f"Deleted Document {doc_id} and all related vectors/edges.")
        db.close()

if __name__ == "__main__":
    test_milestone_3_integration()