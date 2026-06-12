from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from sdr.core.database import get_db
from ..models import ASVSLevel
from ..schemas import ASVSLevelSchema

router = APIRouter(prefix="/asvs-levels")


@router.get("/", response_model=List[ASVSLevelSchema])
def list_asvs_levels(db: Session = Depends(get_db)):
    return db.execute(select(ASVSLevel).order_by(ASVSLevel.level)).scalars().all()
