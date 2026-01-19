# ZakOps Backend Production Dockerfile
# Phase 14: Deployment

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r zakops && useradd -r -g zakops zakops

# Set working directory
WORKDIR /app

# Install Python dependencies (as root for package installation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY db/ ./db/

# Change ownership to non-root user
RUN chown -R zakops:zakops /app

# Switch to non-root user
USER zakops

# Build metadata (set at build time)
ARG BUILD_TIME=unknown
ARG GIT_COMMIT=unknown
ARG APP_VERSION=dev

ENV BUILD_TIME=${BUILD_TIME} \
    GIT_COMMIT=${GIT_COMMIT} \
    APP_VERSION=${APP_VERSION}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run the API server
CMD ["uvicorn", "src.api.orchestration.main:app", "--host", "0.0.0.0", "--port", "8000"]
