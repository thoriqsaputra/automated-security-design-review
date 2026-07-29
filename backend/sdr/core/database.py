from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from sdr.core.config import settings

# -----------------------------------------------------------------------------
# 1. Engine Configuration
# -----------------------------------------------------------------------------
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_CONNECTION_POOL_SIZE,
    max_overflow=10,
    pool_recycle=settings.DATABASE_CONNECTION_MAX_AGE,
)

# -----------------------------------------------------------------------------
# 2. Session Factory
# -----------------------------------------------------------------------------
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# -----------------------------------------------------------------------------
# 3. Declarative Base
# -----------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass

# -----------------------------------------------------------------------------
# 4. Dependency Injection Generator
# -----------------------------------------------------------------------------
def get_db() -> Generator:
    """
    FastAPI Dependency that yields a database session for a single request,
    and safely closes it when the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()