FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .
COPY core/requirements_real.txt* ./core/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ -f core/requirements_real.txt ]; then pip install --no-cache-dir -r core/requirements_real.txt; fi

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/core/src
ENV PYTHONUNBUFFERED=1

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Expose ports for API and health checks
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Command to run the application
CMD ["python", "-m", "uvicorn", "working_aura_api:app", "--host", "0.0.0.0", "--port", "8000"]