FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY agents/ agents/
COPY shared/ shared/
COPY platform/ platform/
COPY config.yaml .

# Create data directories
RUN mkdir -p data logs artifacts

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "agents.orchestrator"]
CMD ["--config", "/app/config.yaml"]
