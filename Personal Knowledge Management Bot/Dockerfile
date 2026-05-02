FROM python:3.11-slim

WORKDIR /app

# Install system deps (git for GitPython)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir . && pip install --no-cache-dir httpx

COPY . .

RUN mkdir -p /app/data/knowledge /app/data

CMD ["python", "-m", "src.main"]
