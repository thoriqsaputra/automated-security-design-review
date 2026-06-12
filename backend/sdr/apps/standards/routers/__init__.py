from fastapi import APIRouter

from .asvs_levels import router as asvs_levels_router
from .categories import router as categories_router
from .ingestion import router as ingestion_router

router = APIRouter()
router.include_router(asvs_levels_router)
router.include_router(categories_router)
router.include_router(ingestion_router)
