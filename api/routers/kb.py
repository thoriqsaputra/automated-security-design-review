from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from enum import Enum
import uuid
import logging
from ingestion.tasks import ingest_knowledge_base
from ingestion.document_parser.pdf_parser import parse_pdf

class KBCategory(str, Enum):
    OWASP = "OWASP"
    NIST = "NIST"
    INTERNAL = "INTERNAL"

router = APIRouter(prefix="/api/v1/kb", tags=["Knowledge Base"])
logger = logging.getLogger(__name__)

@router.post("/ingest", status_code=202)
async def ingest_kb(
    category: KBCategory = Form(...),
    standard_document: Optional[UploadFile] = File(None),
    source_url: Optional[str] = Form(None)
):
    if not standard_document and not source_url:
        raise HTTPException(status_code=400, detail="Must provide either standard_document or source_url")
    if standard_document and source_url:
        raise HTTPException(status_code=400, detail="Provide either standard_document OR source_url, not both")

    job_id = f"kb-ingest-{uuid.uuid4().hex[:8]}"

    if source_url:
        ingest_knowledge_base.delay(job_id, "url", source_url, category.value)  
    else:
        file_bytes = await standard_document.read()
        try:
            text_content, _ = parse_pdf(file_bytes)
        except Exception as e:
            logger.error(f"PDF parsing error: {e}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file")

        ingest_knowledge_base.delay(job_id, "text", text_content, category.value)

    return {
        "status": "success",
        "message": "Knowledge base ingestion queued",
        "job_id": job_id
    }
