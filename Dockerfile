FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --prefix=/install \
    -r requirements.txt


FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


RUN groupadd --gid 1001 appgroup && \
    useradd \
        --uid 1001 \
        --gid appgroup \
        --shell /bin/bash \
        --create-home \
        appuser


COPY --from=builder /install /usr/local

COPY --chown=appuser:appgroup . /app


RUN mkdir -p \
        /app/uploads \
        /app/qdrant_storage \
        /app/prompts \
        /app/.cache/huggingface \
        /app/.cache/fastembed && \
    chown -R appuser:appgroup /app


USER appuser


ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed


EXPOSE 8000


CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]