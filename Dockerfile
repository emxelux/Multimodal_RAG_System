FROM python:3.13-slim AS builder

WORKDIR /build


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


RUN mkdir -p /install/fastembed_cache && chmod -R 777 /install/fastembed_cache


FROM python:3.13-slim AS runtime


RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup \
             --shell /bin/bash \
             --create-home appuser

COPY --from=builder /install                       /usr/local
COPY --from=builder /install/fastembed_cache       /app/.cache/fastembed


COPY --chown=appuser:appgroup . /app


RUN mkdir -p \
        /app/uploads \
        /app/qdrant_storage \
        /app/prompts \
        /app/.cache/huggingface && \
    chown -R appuser:appgroup /app

USER appuser

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

EXPOSE 8000


    --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1


CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]