#!/bin/bash
# Run tests

set -e

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "🧪 Running ZakOps Backend Tests"
echo "================================"

# Run pytest
python -m pytest tests/ -v --tb=short "$@"
