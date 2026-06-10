from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from sdr.core.database import get_db
from ..models import Finding
from ..models.choices import FindingStatus
from ..schemas import FindingSchema

router = APIRouter(prefix="/findings")


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
