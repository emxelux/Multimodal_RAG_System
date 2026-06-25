FROM python:3.11-slim

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 user

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Switch to non-root for security
USER user


# Ensure the app code is owned by the user
COPY --chown=user:user . /app

EXPOSE 7860

# Production Uvicorn command
# Use --workers 1 if you are on a resource-constrained tier to avoid OOM
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "120"]