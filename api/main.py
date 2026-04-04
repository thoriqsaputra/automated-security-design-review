from fastapi import FastAPI
from core.config import settings
from api.routers import kb
from db.session import init_db

try:
    init_db()
except Exception as e:
    print(f"Warning: Could not initialize database (maybe starting without postgres?): {e}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG
)

app.include_router(kb.router)

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}
