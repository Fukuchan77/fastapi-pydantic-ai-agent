# ============================================================================
# FastAPI Pydantic AI Agent - Production Dockerfile
# ============================================================================
# Multi-stage build for minimal image size and security
# Requirements: 9.1-9.5

# ── builder stage ──────────────────────────────────────────────────────────────
# Install dependencies in isolated builder stage
FROM python:3.13-slim AS builder

# Install uv package manager
RUN pip install --no-cache-dir uv

# Set working directory for build
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen lockfile, no dev dependencies)
# Requirement 9.1: Multi-stage build with uv sync --frozen --no-dev
# Build venv at /app/.venv so shebang paths match the runtime stage
RUN uv sync --frozen --no-dev

# ── runtime stage ──────────────────────────────────────────────────────────────
# Minimal runtime image with application code
FROM python:3.13-slim AS runtime

# Requirement 9.3: Create non-root group and user for security (principle of least privilege)
RUN addgroup --system app && adduser --system --no-create-home --ingroup app app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv ./.venv

# Copy application code
COPY app/ ./app/

# Requirement 9.3: Switch to non-root user
USER app

# Add virtual environment to PATH; disable output buffering for reliable container logs
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port 8000 (documentation only)
EXPOSE 8000

# Health check: verify the app responds on /health
# Uses stdlib urllib to avoid any external dependency
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Requirement 9.4: Use exec form for proper SIGTERM handling
# Requirement 9.5: No secrets in image - all via environment variables at runtime
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
