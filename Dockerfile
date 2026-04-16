FROM python:3.11-slim

WORKDIR /app

# Create non-root user (HuggingFace Spaces requirement)
RUN useradd -m -u 1000 user

# Install deps first (separate layer — only rebuilds when requirements.txt changes)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user:user . /app

# Ensure writable runtime directories
RUN mkdir -p /app/document_files /app/assets && \
    chown -R user:user /app/document_files /app/assets

USER user

ENV PYTHONPATH=/app
ENV PATH="/home/user/.local/bin:${PATH}"
# Tell HuggingFace where to cache models (writable location)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
