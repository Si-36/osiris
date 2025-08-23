# AURA Intelligence Production Container - GPU Optimized
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONPATH=/app/core/src:/app
ENV REDIS_URL=redis://redis-service:6379
ENV NEO4J_URI=bolt://neo4j-service:7687

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    git \
    build-essential \
    redis-tools \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and user
WORKDIR /app
RUN groupadd -r aura && useradd -r -g aura aura

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .
COPY core/requirements_real.txt* ./core/

# Install Python dependencies with production optimizations
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt && \
    if [ -f core/requirements_real.txt ]; then pip3 install --no-cache-dir -r core/requirements_real.txt; fi && \
    pip3 install --no-cache-dir fastapi uvicorn psutil redis

# Copy application code
COPY . .

# Create directories and set permissions
RUN mkdir -p /app/logs /app/model_cache /app/data /tmp/prometheus_multiproc && \
    chown -R aura:aura /app && \
    chmod +x /app/real_time_dashboard.py

# Pre-download models during build (optional, adds build time but improves runtime)
RUN python3 -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('distilbert-base-uncased'); AutoTokenizer.from_pretrained('distilbert-base-uncased')" || true

# Switch to non-root user
USER aura

# Expose ports
EXPOSE 8080 8081 8082 9090

# Enhanced health check with model pre-loading validation
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Default command - production dashboard with GPU optimization
CMD ["python3", "-u", "real_time_dashboard.py", "--host", "0.0.0.0", "--port", "8080"]