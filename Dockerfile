# =============================================================================
# BluJay AI - Authoritative Backend Container
# =============================================================================
# Multi-stage/lean production container image for FastAPI Backend
# Follows security best practices: non-root user, slim base, integrated healthcheck
# =============================================================================

FROM python:3.11-slim AS runtime

# Set execution flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install minimal system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and group for runtime security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install Python server dependencies
COPY requirements-server.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-server.txt

# Copy application backend, frontend static assets, and runners
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY run_backend.py /app/

# Set proper ownership for non-root execution
RUN chown -R appuser:appgroup /app

# Switch to unprivileged user
USER appuser

# Expose standard port
EXPOSE 8000

# Integrated Container Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/health || exit 1

# Production server entrypoint
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
