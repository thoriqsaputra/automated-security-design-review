import uuid
import logging
from typing import List
from sqlalchemy.orm import Session
from db.models import TSDDocument, DocumentNode

from llama_index.core import Settings
from rag_engine.graphrag.extractor import process_graph_extraction
from rag_engine.raptor.builder import summarize_cluster
import time

logger = logging.getLogger(__name__)

def build_raptor_and_graph(document: TSDDocument, text_chunks: List[str], db: Session):
    """
    Executes the highly-optimized ingestion pipeline for Milestone 3.
    Designed for atomicity (all-or-nothing) in a Celery Worker.
    """
    try:
        current_layer_nodes = []
        
        logger.info(f"Starting pipeline for Document {document.id} with {len(text_chunks)} chunks.")
        
        logger.info("Generating batch embeddings for Layer 0...")
        batch_vectors = Settings.embed_model.get_text_embedding_batch(text_chunks)
        
        for chunk, vector in zip(text_chunks, batch_vectors):
            node = DocumentNode(
                id=str(uuid.uuid4()), 
                document_id=document.id, 
                text_content=chunk, 
                layer=0, 
                embedding=vector
            )
            db.add(node)
            current_layer_nodes.append(node)
            
            process_graph_extraction(chunk, document.id, db)
            
            logger.info("Sleeping for 3 seconds to avoid rate limits...")
            time.sleep(3)
            
        db.flush()

        layer_num = 1
        cluster_size = 5 
        
        while len(current_layer_nodes) > 1:
            logger.info(f"Building RAPTOR Layer {layer_num} from {len(current_layer_nodes)} nodes...")
            next_layer_nodes = []
            
            for i in range(0, len(current_layer_nodes), cluster_size):
                cluster = current_layer_nodes[i:i+cluster_size]
                cluster_texts = [n.text_content for n in cluster]
                
                summary_text = summarize_cluster(cluster_texts)
                
                
                summary_vector = Settings.embed_model.get_text_embedding(summary_text)
                
                parent_node = DocumentNode(
                    id=str(uuid.uuid4()), 
                    document_id=document.id, 
                    text_content=summary_text,
                    layer=layer_num, 
                    embedding=summary_vector
                )
                db.add(parent_node)
                db.flush()
                
                for child in cluster:
                    child.parent_id = parent_node.id
                    
                next_layer_nodes.append(parent_node)
                
                logger.info("Sleeping for 3 seconds to avoid rate limits...")
                time.sleep(3)
                
            current_layer_nodes = next_layer_nodes
            layer_num += 1

        db.commit()
        logger.info(f"Pipeline complete! Document {document.id} ingested successfully.")

    except Exception as e:
        db.rollback()
        logger.error(f"Pipeline failed for document {document.id}. Rolled back DB. Error: {e}")
        raise