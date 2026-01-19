#!/bin/bash
# ZakOps Backend Setup Script

set -e

echo "🚀 ZakOps Backend Setup"
echo "======================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if PostgreSQL is running
if command -v docker &> /dev/null; then
    echo "Checking PostgreSQL..."
    if ! docker compose -f infra/docker/docker-compose.postgres.yml ps 2>/dev/null | grep -q "Up"; then
        echo "Starting PostgreSQL..."
        docker compose -f infra/docker/docker-compose.postgres.yml up -d
        echo "Waiting for PostgreSQL to be ready..."
        sleep 5
    fi
fi

# Run migrations
if [ -f "db/migrate.py" ]; then
    echo "Running database migrations..."
    python db/migrate.py
fi

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python -m uvicorn src.api.orchestration.main:app --reload --port 8091"
