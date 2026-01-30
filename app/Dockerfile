# DEPRECATED: Use Dockerfile.mcp and Dockerfile.sync at project root instead
# This file is kept for backward compatibility with existing docker-compose usage
#
# New usage:
#   docker build -f Dockerfile.mcp -t fpl-mcp:latest .
#   docker build -f Dockerfile.sync -t fpl-sync:latest .

FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
