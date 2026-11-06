# ==============================================================================
# Enterprise Industrial Biofouling Classification API - Dockerfile (v4)
# ==============================================================================
# Base Image: Python 3.11 Slim (Debian Bookworm)
FROM python:3.11-slim

# Set working directory and runtime environment variables
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install system dependencies required for OpenCV, Pillow, ONNX Runtime, and health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and configuration
COPY app.py .
COPY src/ ./src/
COPY biofouling_model.onnx .
COPY checkpoints/ ./checkpoints/
COPY data/splits_cleaned/ ./data/splits_cleaned/
COPY metadata.csv .

# Create persistent storage directories for Human-in-the-Loop review queue
RUN mkdir -p /app/data/review_queue_images

# Expose REST API port
EXPOSE 8000

# Healthcheck probe querying /health endpoint for container orchestrators (K8s, ECS, Docker Compose)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production Uvicorn ASGI server with multi-worker concurrency
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
