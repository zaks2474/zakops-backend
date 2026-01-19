# ZakOps Backend

Backend services for the ZakOps Deal Lifecycle OS — an AI-powered autonomous deal management platform.

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+ (or use Docker)

### Setup

```bash
# Clone the repository
git clone https://github.com/zaks2474/zakops-backend.git
cd zakops-backend

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Start development server
./scripts/dev.sh
```

The API will be available at:
- **API**: http://localhost:8091
- **Docs**: http://localhost:8091/docs
- **ReDoc**: http://localhost:8091/redoc

## Architecture

Implements the **Execution Plane** and **Data Plane** of the Four-Plane Architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT PLANE                               │
│                   (LangSmith Agent Builder)                      │
│                   See: Zaks-llm repository                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EXECUTION PLANE ← This Repo                 │
│                    (FastAPI + Workers)                           │
│      Orchestration API • Deal Lifecycle BFF • Workers            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PLANE                                │
│              (PostgreSQL + Filesystem + Vector)                  │
│         Deals • Artifacts • Events • Checkpoints                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY PLANE                           │
│                  (Events + Metrics + Audit)                      │
│              agent_events • Dashboards • Alerts                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
zakops-backend/
├── src/
│   ├── api/                 # API layer
│   │   ├── orchestration/   # Main API application (port 8091)
│   │   ├── deal_lifecycle/  # Deal BFF (port 8090)
│   │   └── shared/          # Shared utilities, middleware, routers
│   ├── core/                # Business logic
│   │   ├── auth/            # Authentication & authorization
│   │   ├── database/        # Database adapter (PostgreSQL)
│   │   ├── events/          # Event system
│   │   ├── hitl/            # Human-in-the-loop workflows
│   │   ├── outbox/          # Reliable event delivery
│   │   └── storage/         # Artifact storage
│   ├── workers/             # Background processors
│   └── agent/               # Agent integration
│       ├── bridge/          # MCP Server for LangSmith
│       └── tools/           # Tool implementations
├── db/
│   └── migrations/          # SQL migrations
├── infra/
│   └── docker/              # Docker configurations
├── scripts/                 # Development scripts
└── tests/                   # Test suites
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `AUTH_REQUIRED` | `false` | Enable authentication |
| `OUTBOX_ENABLED` | `true` | Enable outbox pattern |
| `SESSION_EXPIRY_HOURS` | `24` | Session duration |
| `ENVIRONMENT` | `development` | Environment name |

## API Documentation

### Authentication

```bash
# Login
curl -X POST http://localhost:8091/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secret"}'

# Use session cookie for subsequent requests
```

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness probe |
| `/api/deals` | GET | List deals |
| `/api/deals/{id}` | GET | Get deal details |
| `/api/actions` | GET | List actions |
| `/api/hitl/approval-queue` | GET | Pending approvals |
| `/api/agent/activity` | GET | Agent activity |
| `/api/events/recent` | GET | Recent events |

See `/docs` for complete API documentation.

## Development

### Running Tests

```bash
./scripts/test.sh
```

### Database Migrations

```bash
# Apply migrations
python db/migrate.py

# Create new migration
# Add SQL file to db/migrations/
```

### Docker

```bash
# Development (PostgreSQL only)
docker compose -f infra/docker/docker-compose.postgres.yml up -d

# Production (full stack)
docker compose -f infra/docker/docker-compose.yml up -d
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Orchestration API | 8091 | Main REST API |
| Deal Lifecycle API | 8090 | Deal-specific BFF |
| MCP Agent Bridge | 9100 | LangSmith MCP integration |

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [zakops-dashboard](https://github.com/zaks2474/zakops-dashboard) | Next.js frontend UI |
| [Zaks-llm](../Zaks-llm) | LangGraph agent development |

## Core Modules

### Event System (Phase 2)
Unified event taxonomy with 5 domains (deal, action, agent, worker, system).

### Outbox Pattern (Phase 3)
Reliable event delivery with exactly-once semantics.

### Artifact Storage (Phase 4)
Pluggable storage backends (local filesystem, S3).

### HITL Workflows (Phase 6)
Risk assessment, approval workflows, and durable checkpoints.

## License

Proprietary - All rights reserved.
