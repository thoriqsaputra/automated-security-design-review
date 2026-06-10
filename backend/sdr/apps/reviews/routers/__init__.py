from fastapi import APIRouter

from .approval import router as approval_router
from .findings import router as findings_router
from .reviews import router as reviews_router

router = APIRouter()
router.include_router(approval_router)
router.include_router(findings_router)
router.include_router(reviews_router)