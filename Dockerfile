# ==============================================================================
# AquaGuard: Industrial Multi-Stage Production Dockerfile (CPU & Edge Optimized)
# ==============================================================================
# Adheres to enterprise security standards:
# - Multi-stage build for minimal image surface area (<650MB)
# - Non-root execution (UID 10001: appuser)
# - Tini init system for PID 1 signal forwarding & zombie process reaping
# - Layered dependency caching
# - Strict healthcheck monitoring & unbuffered structured JSON logging
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1: Builder & Dependency Compiler
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

# Install compilation headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-compile wheels for rapid caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ------------------------------------------------------------------------------
# STAGE 2: Hardened Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=8000 \
    APP_ENV="production" \
    PATH="/home/appuser/.local/bin:$PATH"

# Install runtime shared libraries and tini init
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 appgroup \
    && useradd -u 10001 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Copy pre-installed Python packages from builder stage
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Copy application codebase, weights, and configurations
COPY --chown=appuser:appgroup . /app

# Create logs directory with correct permissions
RUN mkdir -p /app/logs && chown -R appuser:appgroup /app/logs /app/weights

# Switch to unprivileged non-root user
USER appuser

# Expose API and Studio Ports
EXPOSE 8000 7860

# Production Healthcheck Probe
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# PID 1 Tini Entrypoint with Uvicorn Production Workers
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--access-log"]
