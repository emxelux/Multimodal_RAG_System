# ============================================================
# Builder stage
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies required to build Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy uv dependency files
COPY pyproject.toml uv.lock ./

# Create the virtual environment using the EXACT versions
# recorded in uv.lock.
RUN uv sync --frozen --no-dev --no-install-project


# ============================================================
# Runtime stage
# ============================================================
FROM python:3.11-slim AS runtime

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd \
        --uid 1001 \
        --gid appgroup \
        --shell /bin/bash \
        --create-home \
        appuser

# Copy the exact virtual environment created by uv
COPY --from=builder /build/.venv /app/.venv

# Copy application
COPY --chown=appuser:appgroup . /app

# Create application/cache directories
RUN mkdir -p \
        /app/uploads \
        /app/qdrant_storage \
        /app/prompts \
        /app/.cache/huggingface \
        /app/.cache/fastembed \
    && \
    chown -R appuser:appgroup /app

# Run as non-root user
USER appuser

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    HF_HOME="/app/.cache/huggingface" \
    FASTEMBED_CACHE_PATH="/app/.cache/fastembed"

# FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]