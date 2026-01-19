#!/usr/bin/env python3
"""
Generate TypeScript types from OpenAPI specification.

Usage:
    python scripts/generate_types.py
    python scripts/generate_types.py --output ../zakops-dashboard/src/types/api
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Optional

SPEC_PATH = "shared/openapi/zakops-api.json"


def openapi_type_to_ts(schema: Dict[str, Any], spec: Dict[str, Any], indent: int = 0) -> str:
    """Convert OpenAPI schema to TypeScript type."""
    prefix = "  " * indent

    # Handle non-dict schemas (e.g., boolean for additionalProperties: true)
    if not isinstance(schema, dict):
        return "unknown"

    # Handle $ref
    if "$ref" in schema:
        ref = schema["$ref"]
        type_name = ref.split("/")[-1]
        return type_name

    # Handle type
    schema_type = schema.get("type")

    if schema_type == "string":
        if "enum" in schema:
            return " | ".join(f'"{v}"' for v in schema["enum"])
        if schema.get("format") == "date-time":
            return "string"  # ISO date string
        if schema.get("format") == "uuid":
            return "string"
        return "string"

    elif schema_type == "integer" or schema_type == "number":
        return "number"

    elif schema_type == "boolean":
        return "boolean"

    elif schema_type == "array":
        items_type = openapi_type_to_ts(schema.get("items", {}), spec, indent)
        return f"{items_type}[]"

    elif schema_type == "object":
        if "properties" not in schema:
            if "additionalProperties" in schema:
                value_type = openapi_type_to_ts(schema["additionalProperties"], spec, indent)
                return f"Record<string, {value_type}>"
            return "Record<string, unknown>"

        lines = ["{"]
        for prop_name, prop_schema in schema.get("properties", {}).items():
            required = prop_name in schema.get("required", [])
            prop_type = openapi_type_to_ts(prop_schema, spec, indent + 1)
            optional = "" if required else "?"
            lines.append(f"{prefix}  {prop_name}{optional}: {prop_type};")
        lines.append(f"{prefix}}}")
        return "\n".join(lines)

    elif "allOf" in schema:
        types = [openapi_type_to_ts(s, spec, indent) for s in schema["allOf"]]
        return " & ".join(types)

    elif "anyOf" in schema or "oneOf" in schema:
        schemas = schema.get("anyOf") or schema.get("oneOf")
        types = [openapi_type_to_ts(s, spec, indent) for s in schemas]
        return " | ".join(types)

    # Handle null type
    if schema_type == "null":
        return "null"

    return "unknown"


def sanitize_type_name(name: str) -> str:
    """Sanitize a type name for TypeScript."""
    # Remove special characters that might appear in OpenAPI type names
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def generate_types(spec: Dict[str, Any]) -> str:
    """Generate TypeScript types from OpenAPI spec."""
    lines = [
        "/**",
        " * Auto-generated TypeScript types from OpenAPI specification.",
        f" * Generated from: {spec['info']['title']} v{spec['info']['version']}",
        " * DO NOT EDIT MANUALLY",
        " */",
        "",
        "/* eslint-disable @typescript-eslint/no-explicit-any */",
        "",
    ]

    # Generate schema types
    schemas = spec.get("components", {}).get("schemas", {})

    for name, schema in schemas.items():
        safe_name = sanitize_type_name(name)
        ts_type = openapi_type_to_ts(schema, spec)

        # Add JSDoc if description exists
        if "description" in schema:
            lines.append(f"/** {schema['description']} */")

        lines.append(f"export type {safe_name} = {ts_type};")
        lines.append("")

    # Generate endpoint types
    lines.append("// API Endpoints")
    lines.append("")

    endpoints: List[Dict[str, Any]] = []

    for path, path_item in spec.get("paths", {}).items():
        for method in ["get", "post", "put", "patch", "delete"]:
            if method not in path_item:
                continue

            operation = path_item[method]
            operation_id = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
            operation_id = sanitize_type_name(operation_id)

            # Response type
            response_200 = operation.get("responses", {}).get("200", {})
            response_schema = response_200.get("content", {}).get("application/json", {}).get("schema", {})
            response_type = openapi_type_to_ts(response_schema, spec) if response_schema else "void"

            # Request body type
            request_body = operation.get("requestBody", {})
            request_schema = request_body.get("content", {}).get("application/json", {}).get("schema", {})
            request_type = openapi_type_to_ts(request_schema, spec) if request_schema else "void"

            endpoints.append({
                "path": path,
                "method": method.upper(),
                "operationId": operation_id,
                "responseType": response_type,
                "requestType": request_type,
            })

    # Generate API interface
    lines.append("export interface ApiEndpoints {")
    for ep in endpoints:
        lines.append(f"  // {ep['method']} {ep['path']}")
        lines.append(f"  {ep['operationId']}: {{")
        lines.append(f"    request: {ep['requestType']};")
        lines.append(f"    response: {ep['responseType']};")
        lines.append("  };")
    lines.append("}")
    lines.append("")

    # Generate endpoint paths constant
    lines.append("export const API_PATHS = {")
    for ep in endpoints:
        clean_name = ep['operationId'].replace("-", "_").replace(".", "_")
        lines.append(f"  {clean_name}: '{ep['path']}',")
    lines.append("} as const;")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate TypeScript types from OpenAPI")
    parser.add_argument("--output", "-o", default="../zakops-dashboard/src/types/api",
                        help="Output directory")
    parser.add_argument("--spec", "-s", default=SPEC_PATH, help="OpenAPI spec path")
    args = parser.parse_args()

    # Load spec
    if not os.path.exists(args.spec):
        print(f"Error: OpenAPI spec not found at {args.spec}")
        print("Run 'python scripts/export_openapi.py' first")
        sys.exit(1)

    with open(args.spec) as f:
        spec = json.load(f)

    # Generate types
    types_content = generate_types(spec)

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Write types file
    output_file = os.path.join(args.output, "generated.ts")
    with open(output_file, "w") as f:
        f.write(types_content)

    print(f"TypeScript types generated: {output_file}")
    print(f"   Schemas: {len(spec.get('components', {}).get('schemas', {}))}")
    print(f"   Endpoints: {len(spec.get('paths', {}))}")


if __name__ == "__main__":
    main()
