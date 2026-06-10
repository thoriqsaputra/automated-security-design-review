import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from sdr.core.config import settings

# -----------------------------------------------------------------------------
# Router Imports
# -----------------------------------------------------------------------------
from sdr.apps.designs.routers import router as designs_router
from sdr.apps.reviews.routers import router as reviews_router
from sdr.apps.standards.routers import router as standards_router
from sdr.apps.workspace.routers import router as workspace_router

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    logger.info(f"Starting up {settings.PROJECT_NAME} in {settings.ENVIRONMENT} mode.")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}.")

# -----------------------------------------------------------------------------
# App Factory
# -----------------------------------------------------------------------------
def create_app() -> FastAPI:
    """
    Application factory for creating the FastAPI app instance.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=settings.PROJECT_DESCRIPTION,
        version=settings.PROJECT_VERSION,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs" if settings.ENVIRONMENT != "prod" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "prod" else None,
        lifespan=lifespan,
    )

    # Setup Middleware
    setup_middleware(app)

    # Setup Exception Handlers
    setup_exception_handlers(app)

    # Register Routers
    register_routers(app)

    # Register Base Endpoints
    register_base_endpoints(app)

    return app

def setup_middleware(app: FastAPI) -> None:
    """
    Configure application middleware.
    """
    if settings.CORS_ALLOWED_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.CORS_ALLOWED_ORIGINS],
            allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)

def setup_exception_handlers(app: FastAPI) -> None:
    """
    Configure global exception handlers.
    """
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "body": exc.body},
        )

def register_routers(app: FastAPI) -> None:
    """
    Register application routers from migrated Django apps.
    """
    app.include_router(designs_router, prefix=f"{settings.API_V1_STR}/designs", tags=["Designs"])
    app.include_router(reviews_router, prefix=f"{settings.API_V1_STR}/reviews", tags=["Reviews"])
    app.include_router(standards_router, prefix=f"{settings.API_V1_STR}/standards", tags=["Standards"])
    app.include_router(workspace_router, prefix=f"{settings.API_V1_STR}/workspace", tags=["Workspace"])

def register_base_endpoints(app: FastAPI) -> None:
    """
    Register base system endpoints.
    """
    @app.get("/health", tags=["System"])
    async def health_check():
        """
        Standard health check endpoint for container orchestration (Docker/K8s)
        and load balancers to verify the service is running.
        """
        return {
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
            "project": settings.PROJECT_NAME,
            "version": settings.PROJECT_VERSION
        }

# -----------------------------------------------------------------------------
# App Instance
# -----------------------------------------------------------------------------
app = create_app()