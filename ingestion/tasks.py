from core.celery_app import celery_app
from core.config import settings
from ingestion.web_scraper.scraper import scrape_url
from ingestion.ai_extractor import extract_security_graph, generate_embeddings_batch
from ingestion.dedup_logic import (
    cosine_similarity_from_distance,
    fuzzy_similarity,
    is_exact_duplicate,
    normalize_text,
    pick_parent,
)
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
        graph_nodes = extract_security_graph(content, 1)
        node_count = len(graph_nodes.requirements)
        logger.info(f"AI extracted {node_count} nodes cleanly.")
    except Exception as e:
        logger.error(f"LiteLLM Extraction failed: {e}. Re-queueing task...")
        raise self.retry(exc=e, countdown=120) 

    with SessionLocal() as db:
        try:
            title_to_node = {}
            title_to_confidence = {}
            dedup_decisions = []
            nodes_inserted = 0
            nodes_merged = 0
            nodes_skipped = 0
            child_parent_selection = {}
            
            texts_to_embed = [f"{req.title}: {req.actionable_rule}" for req in graph_nodes.requirements]
            
            batch_vectors = generate_embeddings_batch(texts_to_embed)

            existing_nodes = (
                db.query(SecurityRequirement)
                .filter(SecurityRequirement.category == category)
                .all()
            )
            exact_lookup = {}
            for existing in existing_nodes:
                exact_key = (
                    normalize_text(existing.title),
                    normalize_text(existing.actionable_rule),
                )
                exact_lookup.setdefault(exact_key, existing)
            
            for req, vector in zip(graph_nodes.requirements, batch_vectors):
                req_key = (normalize_text(req.title), normalize_text(req.actionable_rule))
                req_title_key = normalize_text(req.title)

                exact_node = exact_lookup.get(req_key)
                if exact_node and is_exact_duplicate(req.title, req.actionable_rule, exact_node.title, exact_node.actionable_rule):
                    resolved_node = exact_node
                    confidence = settings.KB_DEDUP_EXACT_THRESHOLD
                    nodes_skipped += 1
                    dedup_decisions.append(
                        {
                            "title": req.title,
                            "decision": "SKIP",
                            "node_id": str(exact_node.id),
                            "merged_into": None,
                            "similarity_score": 1.0,
                            "reason": "exact_duplicate",
                        }
                    )
                else:
                    semantic_candidate = None
                    best_similarity = 0.0
                    if vector is not None and existing_nodes:
                        semantic_matches = (
                            db.query(
                                SecurityRequirement,
                                SecurityRequirement.embedding.cosine_distance(vector).label("distance"),
                            )
                            .filter(SecurityRequirement.category == category)
                            .filter(SecurityRequirement.embedding.isnot(None))
                            .order_by(SecurityRequirement.embedding.cosine_distance(vector))
                            .limit(5)
                            .all()
                        )

                        for candidate, distance in semantic_matches:
                            similarity = cosine_similarity_from_distance(distance)
                            if similarity < settings.KB_DEDUP_SEMANTIC_THRESHOLD:
                                continue

                            title_similarity = fuzzy_similarity(req.title, candidate.title)
                            if title_similarity < settings.KB_DEDUP_FUZZY_TITLE_THRESHOLD:
                                continue

                            if similarity > best_similarity:
                                best_similarity = similarity
                                semantic_candidate = candidate

                    if semantic_candidate:
                        resolved_node = semantic_candidate
                        confidence = best_similarity
                        nodes_merged += 1
                        dedup_decisions.append(
                            {
                                "title": req.title,
                                "decision": "MERGE",
                                "node_id": str(semantic_candidate.id),
                                "merged_into": str(semantic_candidate.id),
                                "similarity_score": round(best_similarity, 4),
                                "reason": "semantic_match",
                            }
                        )
                    else:
                        kb_entry = SecurityRequirement(
                            title=req.title,
                            actionable_rule=req.actionable_rule,
                            category=category,
                            job_id=job_id,
                            embedding=vector,
                        )
                        db.add(kb_entry)
                        db.flush()
                        resolved_node = kb_entry
                        confidence = 1.0
                        nodes_inserted += 1
                        exact_lookup[req_key] = kb_entry
                        existing_nodes.append(kb_entry)
                        dedup_decisions.append(
                            {
                                "title": req.title,
                                "decision": "INSERT",
                                "node_id": str(kb_entry.id),
                                "merged_into": None,
                                "similarity_score": None,
                                "reason": "new_requirement",
                            }
                        )

                title_to_node[req_title_key] = resolved_node
                title_to_confidence[req_title_key] = confidence
                
            for req in graph_nodes.requirements:
                if req.is_child_of:
                    child_key = normalize_text(req.title)
                    parent_key = normalize_text(req.is_child_of)
                    parent_node = title_to_node.get(parent_key)
                    child_node = title_to_node.get(child_key)
                    
                    if parent_node and child_node:
                        selected_parent_id, selected_confidence = child_parent_selection.get(
                            child_key,
                            (None, -1.0),
                        )
                        candidate_parent_confidence = title_to_confidence.get(parent_key, settings.KB_DEDUP_SEMANTIC_THRESHOLD)
                        selected_parent_id, selected_confidence = pick_parent(
                            selected_parent_id,
                            selected_confidence,
                            parent_node.id,
                            candidate_parent_confidence,
                        )
                        child_parent_selection[child_key] = (selected_parent_id, selected_confidence)
                    else:
                        logger.warning(
                            f"Graph Integrity Warning: LLM assigned child '{req.title}' "
                            f"to missing parent '{req.is_child_of}'"
                        )

            for child_key, (parent_id, _) in child_parent_selection.items():
                child_node = title_to_node.get(child_key)
                if child_node and parent_id:
                    child_node.parent_id = parent_id

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
        "raw_text_length": len(content),
        "nodes_inserted": nodes_inserted,
        "nodes_merged": nodes_merged,
        "nodes_skipped": nodes_skipped,
        "dedup_decisions": dedup_decisions,
    }