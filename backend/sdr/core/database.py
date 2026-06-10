from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from sdr.core.config import settings

# -----------------------------------------------------------------------------
# 1. Engine Configuration
# -----------------------------------------------------------------------------
# The engine manages the actual connection pool to the database.
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_CONNECTION_POOL_SIZE,
    max_overflow=10,  # Allow up to 10 extra connections during traffic spikes
    pool_recycle=settings.DATABASE_CONNECTION_MAX_AGE,  # Recycle connections to prevent timeouts
)

# -----------------------------------------------------------------------------
# 2. Session Factory
# -----------------------------------------------------------------------------
# SessionLocal is a factory that generates new database sessions.
# autocommit=False and autoflush=False are standard best practices.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# -----------------------------------------------------------------------------
# 3. Declarative Base
# -----------------------------------------------------------------------------
# Every SQLAlchemy model you write will inherit from this Base.
# This is what Alembic looks at to figure out what tables to create.
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