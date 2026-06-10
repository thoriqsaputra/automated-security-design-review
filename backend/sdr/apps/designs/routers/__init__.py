from fastapi import APIRouter

from .design import router as design_router

router = APIRouter()
router.include_router(design_router)
