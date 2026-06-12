from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from sdr.core.database import get_db
from sdr.apps.workspace.services.storage import storage_service
from ..models import Finding
from ..models.choices import FindingStatus
from ..schemas import FindingSchema

router = APIRouter(prefix="/findings")


def _get_diagram_image_metadata(finding: Finding) -> dict:
    metadata = finding.requirement_metadata or {}
    if not isinstance(metadata, dict):
        return {}
    image_metadata = metadata.get("diagram_image") or {}
    return image_metadata if isinstance(image_metadata, dict) else {}


@router.get("/{finding_id}/diagram-image")
def get_finding_diagram_image(finding_id: int, db: Session = Depends(get_db)):
    finding = db.get(Finding, finding_id)
    if not finding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Finding not found")

    image_metadata = _get_diagram_image_metadata(finding)
    object_name = image_metadata.get("object_name")
    if not object_name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagram image not found")

    try:
        content = storage_service.download_bytes(str(object_name))
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Diagram image not found")

    content_type = str(image_metadata.get("content_type") or "image/png")
    return Response(
        content=content,
        media_type=content_type,
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/{finding_id}/close", response_model=FindingSchema)
def close_finding(finding_id: int, db: Session = Depends(get_db)):
    finding = db.get(Finding, finding_id)
    if not finding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Finding not found")
        
    if finding.status == FindingStatus.CLOSED.value:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Finding is already closed")
        
    finding.status = FindingStatus.CLOSED.value
    db.commit()
    db.refresh(finding)
    return finding


@router.post("/{finding_id}/reopen", response_model=FindingSchema)
def reopen_finding(finding_id: int, db: Session = Depends(get_db)):
    finding = db.get(Finding, finding_id)
    if not finding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Finding not found")
        
    if finding.status == FindingStatus.OPEN.value:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Finding is already open")
        
    finding.status = FindingStatus.OPEN.value
    db.commit()
    db.refresh(finding)
    return finding
