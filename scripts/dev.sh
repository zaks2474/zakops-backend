#!/bin/bash
# Start development server with hot reload

set -e

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Set development environment
export ENVIRONMENT=development
export AUTH_REQUIRED=false
export OUTBOX_ENABLED=true

echo "🚀 Starting ZakOps Backend (Development)"
echo "========================================="
echo "API: http://localhost:8091"
echo "Docs: http://localhost:8091/docs"
echo ""

# Start with hot reload
python -m uvicorn src.api.orchestration.main:app \
    --reload \
    --host 0.0.0.0 \
    --port 8091 \
    --log-level info
