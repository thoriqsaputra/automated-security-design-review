from fastapi import APIRouter

from .documents import router as documents_router
from .search import router as search_router

router = APIRouter()
router.include_router(documents_router)
router.include_router(search_router)
