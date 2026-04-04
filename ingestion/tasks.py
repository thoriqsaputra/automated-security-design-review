from core.celery_app import celery_app
from ingestion.web_scraper.scraper import scrape_url
from ingestion.ai_extractor import extract_security_graph, generate_embeddings_batch
from db.session import SessionLocal
from db.models import SecurityRequirement
import logging

logger = logging.getLogger(__name__)

@celery_app.task(name="ingest_knowledge_base", bind=True, max_retries=3)
def ingest_knowledge_base(self, job_id: str, payload_type: str, payload_data: str, category: str):
    logger.info(f"Starting ingestion job: {job_id}")
    
    content = ""
    if payload_type == "url":
        content = scrape_url(payload_data)
    else:
        content = payload_data

    if not content:
        logger.warning(f"No content extracted for job {job_id}")
        return {"status": "failed", "error": "Empty content"}

    logger.info(f"Extracted {len(content)} characters. Engaging AI Extractor...")
    
    try:
        graph_nodes = extract_security_graph(content, 3)
        node_count = len(graph_nodes.requirements)
        logger.info(f"AI extracted {node_count} nodes cleanly.")
    except Exception as e:
        logger.error(f"LiteLLM Extraction failed: {e}. Re-queueing task...")
        raise self.retry(exc=e, countdown=120) 

    with SessionLocal() as db:
        try:
            title_to_node = {} 
            
            texts_to_embed = [f"{req.title}: {req.actionable_rule}" for req in graph_nodes.requirements]
            
            batch_vectors = generate_embeddings_batch(texts_to_embed)
            
            for req, vector in zip(graph_nodes.requirements, batch_vectors):
                kb_entry = SecurityRequirement(
                    title=req.title,
                    actionable_rule=req.actionable_rule,
                    category=category,
                    job_id=job_id,
                    embedding=vector
                )
                db.add(kb_entry)
                title_to_node[req.title.strip().casefold()] = kb_entry
                
            db.flush()
                
            for req in graph_nodes.requirements:
                if req.is_child_of:
                    parent_key = req.is_child_of.strip().casefold()
                    parent_node = title_to_node.get(parent_key)
                    
                    if parent_node:
                        child_node = title_to_node[req.title.strip().casefold()]
                        child_node.parent_id = parent_node.id
                    else:
                        logger.warning(
                            f"Graph Integrity Warning: LLM assigned child '{req.title}' "
                            f"to missing parent '{req.is_child_of}'"
                        )

            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"DB/Embedding error: {e}")
            raise self.retry(exc=e, countdown=60)

    logger.info(f"Job {job_id} constructed knowledge graph securely.")
    return {
        "job_id": job_id, 
        "status": "completed", 
        "graph_nodes_extracted": node_count,
        "raw_text_length": len(content)
    }