# ZakOps Backend

Backend services for ZakOps Deal Lifecycle OS.

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

## Structure

```
src/
├── api/
│   ├── orchestration/      # Unified API (port 8091)
│   ├── deal_lifecycle/     # Deal BFF (port 8090)
│   └── shared/             # Shared utilities
├── workers/                # Background processors
├── agent/
│   ├── bridge/             # MCP Server for LangSmith
│   └── tools/              # Tool implementations
└── core/
    ├── config/             # Configuration management
    ├── logging/            # Structured logging
    └── types/              # Shared type definitions

db/
└── migrations/             # Database migrations

infra/
└── docker/                 # Docker Compose files
```

## Related Repositories

| Repository | Purpose |
|------------|---------|
| [zakops-dashboard](../zakops-dashboard) | Next.js frontend UI |
| [Zaks-llm](../Zaks-llm) | LangGraph agent development |

## Getting Started

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment
cp .env.example .env

# Start services
docker compose -f infra/docker/docker-compose.yml up -d
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Deal Lifecycle API | 8090 | Main REST API for deals, actions, quarantine |
| MCP Agent Bridge | 9100 | LangSmith MCP integration |
| RAG REST API | 8052 | Document search with pgvector |

## API Documentation

- Deal Lifecycle API: http://localhost:8090/docs
- MCP Agent Bridge: http://localhost:9100/health
