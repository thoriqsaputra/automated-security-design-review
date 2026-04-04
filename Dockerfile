FROM python:3.12-slim AS builder

# Set environment variables for optimal Python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install necessary build tools and copy uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies cleanly
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv venv $VIRTUAL_ENV && \
    uv sync --frozen --no-dev

# ==========================================
# Stage 2: Production Image
# ==========================================
FROM python:3.12-slim

# Security: Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Set environments to look at the isolated virtualenv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Copy the lightweight built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the actual application code
COPY --chown=appuser:appuser . /app/

# Switch to the non-root user
USER appuser

EXPOSE 8000
