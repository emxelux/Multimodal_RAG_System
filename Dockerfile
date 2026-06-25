# =============================================================
# ChatPDF Pro — Production Dockerfile
# Multi-stage build: keeps the final image lean (~400 MB vs ~1.5 GB)
# =============================================================

# ─────────────────────────────────────────────────────────────
# Stage 1 · builder
#   • Compiles every dependency.
#   • Pre-downloads the BM25 sparse model so cold starts are instant.
#   • Nothing from this stage leaks into the final image.
# ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# Build-time system deps (compilers, libpq headers for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install all Python packages into an isolated prefix.
# Using --prefix lets us COPY the exact tree into the runtime stage.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download fastembed's BM25 model (~15 MB).
# Without this, the first request after container start stalls
# while fastembed downloads the model at runtime.
RUN FASTEMBED_CACHE_PATH=/install/fastembed_cache \
    python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"


# ─────────────────────────────────────────────────────────────
# Stage 2 · runtime
#   • Starts from a clean slim base — no compilers, no headers.
#   • Receives only the installed packages from stage 1.
#   • Runs as a non-root user.
# ─────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Runtime-only system packages.
# libpq5: the runtime PostgreSQL client library (psycopg2 needs it).
# curl: used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Non-root user ─────────────────────────────────────────────
# Running as root inside a container is a security risk.
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup \
             --shell /bin/bash \
             --create-home appuser

# ── Copy installed packages from builder ─────────────────────
COPY --from=builder /install                       /usr/local
COPY --from=builder /install/fastembed_cache       /app/.cache/fastembed

# ── Copy application source ───────────────────────────────────
# .dockerignore excludes: .env, __pycache__, qdrant_storage/,
# uploads/, document_files/, .git/, .venv/
COPY --chown=appuser:appgroup . /app

# ── Runtime directories ───────────────────────────────────────
# These are mounted as Docker volumes in production so data
# persists across container restarts and deployments.
RUN mkdir -p \
        /app/uploads \
        /app/qdrant_storage \
        /app/prompts \
        /app/.cache/huggingface && \
    chown -R appuser:appgroup /app

USER appuser

# ── Environment variables ─────────────────────────────────────
ENV PYTHONPATH=/app \
    # Flush stdout/stderr immediately — essential for log visibility
    PYTHONUNBUFFERED=1 \
    # Don't write .pyc files inside the container
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}" \
    # Point fastembed at the pre-downloaded cache baked in stage 1
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    # Keep HuggingFace downloads inside the container volume
    HF_HOME=/app/.cache/huggingface

# ── Port ──────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────
# Docker (and Compose) marks the container healthy only once
# this passes. The API router has a GET / that returns 200.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Entrypoint ────────────────────────────────────────────────
# IMPORTANT — single worker only.
#
# Your vector store uses @lru_cache + a local Qdrant SQLite file.
# Multiple OS-level workers (--workers N) would each hold a separate
# cache instance and fight over the same SQLite file, causing data
# corruption. Keep --workers 1 until you migrate to Qdrant Cloud,
# at which point you can safely raise this to (2 × CPU cores + 1).
CMD ["uvicorn", "app.main:app", \
     "--host",               "0.0.0.0", \
     "--port",               "8000", \
     "--workers",            "1", \
     "--timeout-keep-alive", "120", \
     "--log-level",          "info"]
