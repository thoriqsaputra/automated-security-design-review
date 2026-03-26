# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN python -m pip install --no-cache-dir --upgrade pip uv

# Copy project metadata and source required for editable install
COPY pyproject.toml uv.lock* PROJECT_FOUNDATION.md ./
COPY api/ ./api/
COPY core/ ./core/
COPY agents/ ./agents/
COPY ingestion/ ./ingestion/
COPY rag_engine/ ./rag_engine/

# Create virtual environment and install dependencies using uv
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -e .

# Stage 2: Runtime stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy source code
COPY api/ ./api/
COPY core/ ./core/
COPY agents/ ./agents/
COPY ingestion/ ./ingestion/
COPY rag_engine/ ./rag_engine/
COPY pyproject.toml ./

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
