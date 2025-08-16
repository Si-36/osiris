# Stage 1: Builder Environment
# This stage installs all dependencies, including the Modular SDK, and builds our Python environment.
FROM python:3.12-slim as builder

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIXI_HOME="/opt/pixi"
ENV PATH="/opt/pixi/bin:$PATH"

# Install essential build tools and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | sh

# Copy project files
WORKDIR /app
COPY ultimate_api_system/pixi.toml ultimate_api_system/pixi.lock ./

# Install all project dependencies (Mojo/MAX and Python libs) into a virtual environment
# This leverages the pixi.lock file for perfect reproducibility.
RUN pixi install

# Copy the rest of the application source code
COPY ultimate_api_system/ .

# ---

# Stage 2: Final Production Image
# This stage creates a minimal, secure image for running the application.
FROM python:3.12-slim as final

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MODULAR_HOME="/opt/venv/share/max"
ENV CONDA_PREFIX="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 -m -s /bin/bash appuser

# Copy the installed virtual environment from the builder stage
COPY --from=builder /app/.pixi/envs/default /opt/venv

# Copy the application source code
WORKDIR /app
COPY --from=builder /app .

# Set ownership to the non-root user
RUN chown -R appuser:appuser /app /opt/venv

# Switch to the non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Set the command to run the application in production using Gunicorn
# This provides a robust, multi-worker HTTP server.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "max_aura_api:app", "--bind", "0.0.0.0:8000"]
