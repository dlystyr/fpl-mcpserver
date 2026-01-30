# FPL MCP Server Deployment Guide

## Container Images

This project produces two separate container images:

| Image | Purpose | Dockerfile |
|-------|---------|------------|
| `fpl-mcp` | MCP server for FPL analytics | `Dockerfile.mcp` |
| `fpl-sync` | Data synchronization service | `Dockerfile.sync` |

---

## Building Images

### Build Both Images Locally

```bash
# Build MCP server
docker build -f Dockerfile.mcp -t fpl-mcp:latest .

# Build Sync service
docker build -f Dockerfile.sync -t fpl-sync:latest .
```

### Build with Version Tags

```bash
VERSION=1.0.0

docker build -f Dockerfile.mcp -t fpl-mcp:${VERSION} -t fpl-mcp:latest .
docker build -f Dockerfile.sync -t fpl-sync:${VERSION} -t fpl-sync:latest .
```

---

## Pushing to Container Registry

### Azure Container Registry (ACR)

```bash
# Login to ACR
az acr login --name <your-acr-name>

# Tag images
ACR_NAME=<your-acr-name>.azurecr.io
VERSION=1.0.0

docker tag fpl-mcp:latest ${ACR_NAME}/fpl-mcp:latest
docker tag fpl-mcp:latest ${ACR_NAME}/fpl-mcp:${VERSION}
docker tag fpl-sync:latest ${ACR_NAME}/fpl-sync:latest
docker tag fpl-sync:latest ${ACR_NAME}/fpl-sync:${VERSION}

# Push images
docker push ${ACR_NAME}/fpl-mcp:latest
docker push ${ACR_NAME}/fpl-mcp:${VERSION}
docker push ${ACR_NAME}/fpl-sync:latest
docker push ${ACR_NAME}/fpl-sync:${VERSION}
```

### Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag images
DOCKERHUB_USER=<your-username>
VERSION=1.0.0

docker tag fpl-mcp:latest ${DOCKERHUB_USER}/fpl-mcp:latest
docker tag fpl-mcp:latest ${DOCKERHUB_USER}/fpl-mcp:${VERSION}
docker tag fpl-sync:latest ${DOCKERHUB_USER}/fpl-sync:latest
docker tag fpl-sync:latest ${DOCKERHUB_USER}/fpl-sync:${VERSION}

# Push images
docker push ${DOCKERHUB_USER}/fpl-mcp:latest
docker push ${DOCKERHUB_USER}/fpl-mcp:${VERSION}
docker push ${DOCKERHUB_USER}/fpl-sync:latest
docker push ${DOCKERHUB_USER}/fpl-sync:${VERSION}
```

### GitHub Container Registry (GHCR)

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u <your-username> --password-stdin

# Tag images
GHCR_USER=<your-username>
VERSION=1.0.0

docker tag fpl-mcp:latest ghcr.io/${GHCR_USER}/fpl-mcp:latest
docker tag fpl-mcp:latest ghcr.io/${GHCR_USER}/fpl-mcp:${VERSION}
docker tag fpl-sync:latest ghcr.io/${GHCR_USER}/fpl-sync:latest
docker tag fpl-sync:latest ghcr.io/${GHCR_USER}/fpl-sync:${VERSION}

# Push images
docker push ghcr.io/${GHCR_USER}/fpl-mcp:latest
docker push ghcr.io/${GHCR_USER}/fpl-mcp:${VERSION}
docker push ghcr.io/${GHCR_USER}/fpl-sync:latest
docker push ghcr.io/${GHCR_USER}/fpl-sync:${VERSION}
```

### AWS Elastic Container Registry (ECR)

```bash
# Login to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

# Create repositories (first time only)
aws ecr create-repository --repository-name fpl-mcp
aws ecr create-repository --repository-name fpl-sync

# Tag and push
ECR_URI=<account-id>.dkr.ecr.<region>.amazonaws.com
VERSION=1.0.0

docker tag fpl-mcp:latest ${ECR_URI}/fpl-mcp:latest
docker tag fpl-mcp:latest ${ECR_URI}/fpl-mcp:${VERSION}
docker push ${ECR_URI}/fpl-mcp:latest
docker push ${ECR_URI}/fpl-mcp:${VERSION}

docker tag fpl-sync:latest ${ECR_URI}/fpl-sync:latest
docker tag fpl-sync:latest ${ECR_URI}/fpl-sync:${VERSION}
docker push ${ECR_URI}/fpl-sync:latest
docker push ${ECR_URI}/fpl-sync:${VERSION}
```

---

## Running with External Databases

### Environment Variables

Create a `.env` file with your external database credentials:

```bash
# PostgreSQL (Azure Database for PostgreSQL, AWS RDS, Supabase, etc.)
DATABASE_URL=postgresql://user:password@host:5432/database?sslmode=require

# Redis (Azure Cache for Redis, AWS ElastiCache, etc.)
REDIS_HOST=your-redis-host.redis.cache.windows.net
REDIS_PORT=6380
REDIS_PASSWORD=your-redis-key
REDIS_SSL=true

# Optional
API_KEY=your-api-key
LOG_LEVEL=INFO
```

### Run MCP Server (Standalone)

```bash
# Using docker run
docker run -d \
  --name fpl-mcp \
  -p 8000:8000 \
  --env-file .env \
  fpl-mcp:latest

# Or using docker-compose (external mode)
docker-compose up -d fpl-mcp
```

### Run Data Sync (One-time)

```bash
# Using docker run
docker run --rm \
  --env-file .env \
  fpl-sync:latest

# Or using docker-compose
docker-compose --profile sync up fpl-sync
```

---

## Local Development (All-in-One)

For local development with bundled PostgreSQL and Redis:

```bash
# Start everything (postgres, redis, mcp server)
docker-compose --profile local up -d

# Run data sync
docker-compose --profile local --profile sync up fpl-sync

# View logs
docker-compose logs -f fpl-mcp

# Stop everything
docker-compose --profile local down
```

---

## Azure Deployment Examples

### Azure Container Apps

```bash
# Create Container App for MCP Server
az containerapp create \
  --name fpl-mcp \
  --resource-group fpl-rg \
  --environment fpl-env \
  --image ${ACR_NAME}/fpl-mcp:latest \
  --target-port 8000 \
  --ingress external \
  --secrets \
    database-url="postgresql://..." \
    redis-password="..." \
  --env-vars \
    DATABASE_URL=secretref:database-url \
    REDIS_HOST=fpl-redis.redis.cache.windows.net \
    REDIS_PORT=6380 \
    REDIS_PASSWORD=secretref:redis-password \
    REDIS_SSL=true

# Create Container App Job for Sync (scheduled)
az containerapp job create \
  --name fpl-sync \
  --resource-group fpl-rg \
  --environment fpl-env \
  --image ${ACR_NAME}/fpl-sync:latest \
  --trigger-type Schedule \
  --cron-expression "0 6 * * 0,3" \
  --secrets \
    database-url="postgresql://..." \
    redis-password="..." \
  --env-vars \
    DATABASE_URL=secretref:database-url \
    REDIS_HOST=fpl-redis.redis.cache.windows.net \
    REDIS_PORT=6380 \
    REDIS_PASSWORD=secretref:redis-password \
    REDIS_SSL=true
```

### Azure Functions (Timer Trigger for Sync)

See `infra/` directory for Terraform/Bicep templates.

---

## Health Checks

The MCP server exposes a health endpoint:

```bash
curl http://localhost:8000/health
# Response: {"status": "ok"}
```

---

## Troubleshooting

### Connection Issues

```bash
# Test PostgreSQL connection
docker run --rm -it --env-file .env postgres:16-alpine \
  psql "$DATABASE_URL" -c "SELECT 1"

# Test Redis connection
docker run --rm -it --env-file .env redis:alpine \
  redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD --tls ping
```

### View Container Logs

```bash
# MCP Server logs
docker logs fpl-mcp -f

# Sync logs
docker logs fpl-sync
```
