import json
import logging
import redis
from itertools import chain

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from sdr.core.config import settings
from sdr.core.database import get_db
from sdr.apps.designs.models import Design
from sdr.apps.standards.models import StandardSourceDocument
from sdr.apps.designs.schemas import DesignSchema
from sdr.apps.standards.schemas import StandardSourceDocumentSchema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search")

# Initialize Redis client
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


@router.get("/")
def global_search(query: str = Query("", min_length=2), db: Session = Depends(get_db)):
    """
    Performs global search across designs and standards.
    Returns combined and sorted results, cached via Redis.
    """
    query_stripped = query.strip()
    if len(query_stripped) < 2:
        return []

    # Simple cache key
    cache_key = f"search_results_{hash(query_stripped)}"
    
    try:
        cached_results = redis_client.get(cache_key)
        if cached_results:
            logger.info(f"Search results served from cache for query: {query_stripped}")
            return json.loads(cached_results)
    except Exception as e:
        logger.warning(f"Redis cache read error: {e}")

    logger.info(f"Performing fresh search for query: {query_stripped}")
    
    # SQLAlchemy ILIKE query for case-insensitive search
    design_qs = select(Design).where(Design.name.ilike(f"%{query_stripped}%")).limit(5)
    standard_qs = select(StandardSourceDocument).where(StandardSourceDocument.name.ilike(f"%{query_stripped}%")).limit(5)
    
    design_results = db.execute(design_qs).scalars().all()
    standard_results = db.execute(standard_qs).scalars().all()

    # Convert to Pydantic dicts for response
    design_data = [DesignSchema.model_validate(d).model_dump() for d in design_results]
    standard_data = [StandardSourceDocumentSchema.model_validate(s).model_dump() for s in standard_results]
    
    # Add a type field for frontend distinction if needed
    for d in design_data:
        d["_search_type"] = "design"
    for s in standard_data:
        s["_search_type"] = "standard"
        
    combined_results = sorted(
        list(chain(design_data, standard_data)),
        key=lambda x: x.get('name', '')
    )
    
    try:
        redis_client.setex(cache_key, 300, json.dumps(combined_results, default=str))
        logger.info(f"Search results cached for query: {query_stripped}")
    except Exception as e:
        logger.warning(f"Redis cache write error: {e}")
        
    return combined_results