# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never

# Install heavy build dependencies (these will NOT be in the final image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"



# Install strictly runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libpq5 \
    postgresql-client \
    curl \
    dos2unix \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN adduser --disabled-password --gecos '' appuser
WORKDIR /app

# Copy the compiled virtual environment from the builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application source code
COPY --chown=appuser:appuser sdr/ sdr/
COPY --chown=appuser:appuser alembic/ alembic/
COPY --chown=appuser:appuser alembic.ini ./

# Runtime dirs
RUN mkdir -p /app/static /app/staticfiles /app/media /app/logs

# Entrypoint setup
COPY --chown=appuser:appuser entrypoint.sh /app/entrypoint.sh
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Hand over ownership then switch user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f -L http://localhost:8000/health || exit 1

CMD ["/app/entrypoint.sh"]