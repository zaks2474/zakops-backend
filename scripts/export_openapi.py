#!/usr/bin/env python3
"""
Export OpenAPI specification from FastAPI app.

Usage:
    python scripts/export_openapi.py
    python scripts/export_openapi.py --output shared/openapi/zakops-api.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def export_openapi(output_path: str = None):
    """Export OpenAPI spec from FastAPI app."""
    from src.api.orchestration.main import app

    # Get OpenAPI schema
    schema = app.openapi()

    # Add export metadata
    schema["info"]["x-exported-at"] = datetime.now(timezone.utc).isoformat()
    schema["info"]["x-export-version"] = "1.0.0"

    # Determine output path
    if output_path is None:
        output_path = "shared/openapi/zakops-api.json"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write spec
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"OpenAPI spec exported to: {output_path}")
    print(f"   Title: {schema['info']['title']}")
    print(f"   Version: {schema['info']['version']}")
    print(f"   Paths: {len(schema['paths'])} endpoints")

    return schema


def main():
    parser = argparse.ArgumentParser(description="Export OpenAPI specification")
    parser.add_argument("--output", "-o", help="Output file path")
    args = parser.parse_args()

    export_openapi(args.output)


if __name__ == "__main__":
    main()
