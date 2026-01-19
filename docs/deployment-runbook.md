# ZakOps Backend Deployment Runbook

**Phase 14: Deployment**

This document provides step-by-step instructions for deploying the ZakOps backend.

## Prerequisites

### Required Infrastructure
- PostgreSQL 15+ database (managed service recommended: RDS, Cloud SQL)
- Docker and Docker Compose (for container deployment)
- Container registry (Docker Hub, ECR, GCR, etc.)

### Required Environment Variables
```bash
# Database (required)
DATABASE_URL=postgresql://user:password@host:5432/zakops

# Authentication (required in production)
SESSION_SECRET=your-secret-key-here
AUTH_REQUIRED=true

# Optional
CORS_ORIGINS=["https://app.zakops.com"]
LANGSMITH_API_KEY=your-langsmith-key
```

## Deployment Options

### Option 1: Docker Compose (Single Server)

Best for: Development, staging, small-scale production.

#### Step 1: Prepare Environment
```bash
# Clone repository
git clone https://github.com/your-org/zakops-backend.git
cd zakops-backend

# Set environment variables
export DATABASE_URL="postgresql://user:password@host:5432/zakops"
export SESSION_SECRET="$(openssl rand -base64 32)"
export VERSION="1.0.0"
export GIT_COMMIT="$(git rev-parse --short HEAD)"
export BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

#### Step 2: Build and Deploy
```bash
cd infra/docker

# Build the image
docker-compose -f docker-compose.production.yml build

# Start services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f
```

#### Step 3: Verify Deployment
```bash
# Health check
curl http://localhost:8000/health

# Version check
curl http://localhost:8000/api/version
```

### Option 2: Kubernetes (Cloud Native)

Best for: Large-scale production, auto-scaling, high availability.

See `infra/k8s/` for Kubernetes manifests (future phase).

## Database Migration

Before deploying a new version, run database migrations:

```bash
# Connect to database and run migrations
# (Migrations are idempotent and safe to re-run)

# Option 1: From docker container
docker exec -it zakops-backend python -m src.core.database.migrations

# Option 2: Directly with DATABASE_URL
DATABASE_URL=postgresql://... python -m src.core.database.migrations
```

## Health Checks

### Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /health` | Liveness probe | `{"status": "healthy"}` |
| `GET /api/health/ready` | Readiness probe | `{"status": "ready", "checks": {...}}` |
| `GET /api/health/startup` | Startup probe | `{"status": "started"}` |
| `GET /api/version` | Build information | `{"version": "...", "commit": "..."}` |

### Docker Health Check
The container includes a built-in health check that runs every 30 seconds:
```bash
curl -f http://localhost:8000/health || exit 1
```

## Service Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────┐    ┌─────────────────────┐     │
│  │   zakops-backend    │    │   zakops-outbox     │     │
│  │   (API Server)      │    │   (Outbox Worker)   │     │
│  │   Port: 8000        │    │   No port exposed   │     │
│  │   Healthcheck: /health    │   Polls outbox table│     │
│  └──────────┬──────────┘    └──────────┬──────────┘     │
│             │                          │                 │
│             └──────────────┬───────────┘                 │
│                            │                             │
│                    ┌───────▼───────┐                     │
│                    │   PostgreSQL  │                     │
│                    │   (External)  │                     │
│                    └───────────────┘                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Scaling

### Backend API
The backend is stateless and can be scaled horizontally:
```bash
# Docker Compose
docker-compose -f docker-compose.production.yml up -d --scale backend=3

# Note: When scaling, use a load balancer (nginx, HAProxy, cloud LB)
```

### Outbox Processor
Only one instance should run at a time (uses database locking for coordination).

## Monitoring

### Log Locations
- Container logs: `docker logs zakops-backend`
- JSON file logs: Configured in docker-compose with rotation

### Metrics Endpoints
- `GET /api/admin/outbox/stats` - Outbox processing statistics
- `GET /api/admin/sse/stats` - SSE connection statistics
- `GET /api/admin/dlq/stats` - Dead letter queue statistics

### Key Metrics to Monitor
1. HTTP response times (P50, P95, P99)
2. Outbox queue depth
3. DLQ entry count
4. Active SSE connections
5. Database connection pool utilization

## Common Operations

### Restart Services
```bash
docker-compose -f docker-compose.production.yml restart
```

### View Logs
```bash
# All services
docker-compose -f docker-compose.production.yml logs -f

# Specific service
docker-compose -f docker-compose.production.yml logs -f backend
docker-compose -f docker-compose.production.yml logs -f outbox-processor
```

### Update to New Version
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d
```

### Stop Services
```bash
docker-compose -f docker-compose.production.yml down
```

## Troubleshooting

### Container Won't Start
1. Check logs: `docker logs zakops-backend`
2. Verify DATABASE_URL is set and reachable
3. Verify SESSION_SECRET is set (required when AUTH_REQUIRED=true)

### Database Connection Errors
1. Verify DATABASE_URL format: `postgresql://user:pass@host:5432/dbname`
2. Check network connectivity to database
3. Verify database credentials
4. Check if database exists and user has permissions

### Outbox Not Processing
1. Check outbox-processor logs
2. Verify database connection
3. Query outbox table: `SELECT status, COUNT(*) FROM zakops.outbox GROUP BY status`

### High Memory Usage
1. Check SSE connection count
2. Review resource limits in docker-compose
3. Consider scaling horizontally

## Security Checklist

Before deploying to production, complete the security checklist:
- [ ] AUTH_REQUIRED=true is set
- [ ] SESSION_SECRET is a strong random value
- [ ] Database uses SSL connections
- [ ] CORS_ORIGINS is restricted to known domains
- [ ] Container runs as non-root user (default)
- [ ] No debug endpoints exposed

See `docs/security-checklist.md` for the full checklist.

## Emergency Contacts

| Role | Contact |
|------|---------|
| On-Call Engineer | [Your PagerDuty/OpsGenie integration] |
| Database Admin | [Contact info] |
| Security | [Contact info] |
