"""
OpenAPI Schema Configuration

Phase 8: OpenAPI & Tooling (Enhanced from Phase 5)
Provides comprehensive API documentation for ZakOps Backend.
"""

from typing import Dict, Any

from fastapi import FastAPI


def customize_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Customize the OpenAPI schema with comprehensive documentation.

    Call this in your FastAPI app:
        app.openapi = lambda: customize_openapi(app)

    Returns:
        The customized OpenAPI schema dictionary.
    """
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title="ZakOps API",
        version="1.0.0",
        description="""
# ZakOps Deal Lifecycle OS API

Backend services for the ZakOps autonomous deal management platform.

## Overview

ZakOps provides AI-powered deal lifecycle management with:
- **Deal Management** — Create, track, and manage deals through their lifecycle
- **Agent Actions** — AI-generated actions with human-in-the-loop approval
- **Document Storage** — Secure artifact storage with cloud support
- **Real-time Events** — SSE streaming for live updates

## Authentication

Authentication is session-based using cookies.

```bash
# Login
curl -X POST /api/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"email": "user@example.com", "password": "secret"}' \\
  -c cookies.txt

# Use session
curl /api/deals -b cookies.txt
```

In development mode (`AUTH_REQUIRED=false`), authentication is optional.

## Response Format

### Success Response
```json
{
  "data": { ... },
  "meta": {
    "trace_id": "abc-123",
    "correlation_id": "deal-456",
    "timestamp": "2026-01-19T12:00:00Z"
  }
}
```

### Error Response
```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Deal not found",
    "details": null
  },
  "meta": {
    "trace_id": "abc-123",
    "timestamp": "2026-01-19T12:00:00Z"
  }
}
```

## Tracing

Include these headers for distributed tracing:
- `X-Trace-ID` — Unique request identifier (auto-generated if not provided)
- `X-Correlation-ID` — Business correlation ID (e.g., deal_id)

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `BAD_REQUEST` | 400 | Invalid request format |
| `VALIDATION_ERROR` | 422 | Request validation failed |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Permission denied |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource conflict |
| `INTERNAL_ERROR` | 500 | Server error |

## Rate Limiting

Rate limiting is not currently enforced but may be added in future versions.

## Webhooks

Webhook support is planned for future releases.
        """,
        routes=app.routes,
        tags=[
            {"name": "health", "description": "Health and readiness checks"},
            {"name": "auth", "description": "Authentication and session management"},
            {"name": "deals", "description": "Deal lifecycle management"},
            {"name": "actions", "description": "Agent actions and approvals"},
            {"name": "agent", "description": "Agent activity and events"},
            {"name": "hitl", "description": "Human-in-the-loop workflows"},
            {"name": "events", "description": "Event streaming and history"},
            {"name": "artifacts", "description": "Document and file storage"},
        ]
    )

    # Ensure components exists
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "sessionAuth": {
            "type": "apiKey",
            "in": "cookie",
            "name": "zakops_session",
            "description": "Session cookie obtained from /api/auth/login"
        }
    }

    # Ensure schemas exists
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}

    # Add common response schemas
    openapi_schema["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "error": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Machine-readable error code",
                        "example": "NOT_FOUND"
                    },
                    "message": {
                        "type": "string",
                        "description": "Human-readable error message",
                        "example": "Resource not found"
                    },
                    "details": {
                        "type": "array",
                        "nullable": True,
                        "description": "Additional error details for validation errors",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "description": "Field path that caused the error"
                                },
                                "message": {
                                    "type": "string",
                                    "description": "Error message for this field"
                                },
                                "code": {
                                    "type": "string",
                                    "description": "Error code for this field"
                                }
                            }
                        }
                    },
                    "trace_id": {
                        "type": "string",
                        "description": "Unique trace ID for debugging"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "When the error occurred"
                    }
                },
                "required": ["code", "message"]
            },
            "meta": {"$ref": "#/components/schemas/ResponseMeta"}
        }
    }

    openapi_schema["components"]["schemas"]["ResponseMeta"] = {
        "type": "object",
        "properties": {
            "trace_id": {
                "type": "string",
                "format": "uuid",
                "description": "Unique trace ID for this request"
            },
            "correlation_id": {
                "type": "string",
                "nullable": True,
                "description": "Correlation ID for related requests"
            },
            "timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "Response timestamp"
            }
        },
        "required": ["trace_id", "timestamp"]
    }

    openapi_schema["components"]["schemas"]["SuccessResponse"] = {
        "type": "object",
        "properties": {
            "data": {
                "description": "Response data (type varies by endpoint)"
            },
            "meta": {"$ref": "#/components/schemas/ResponseMeta"}
        }
    }

    openapi_schema["components"]["schemas"]["ListResponse"] = {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "description": "List of items"
            },
            "meta": {
                "type": "object",
                "properties": {
                    "total": {
                        "type": "integer",
                        "description": "Total number of items"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum items per page"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Current offset"
                    },
                    "has_more": {
                        "type": "boolean",
                        "description": "Whether more items exist"
                    },
                    "trace_id": {
                        "type": "string",
                        "description": "Unique trace ID"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time"
                    }
                },
                "required": ["total", "limit", "offset", "has_more", "trace_id"]
            }
        }
    }

    openapi_schema["components"]["schemas"]["PaginatedResponse"] = {
        "type": "object",
        "properties": {
            "data": {"type": "array", "items": {"type": "object"}},
            "meta": {"$ref": "#/components/schemas/ResponseMeta"},
            "pagination": {
                "type": "object",
                "properties": {
                    "total": {"type": "integer"},
                    "page": {"type": "integer"},
                    "per_page": {"type": "integer"},
                    "total_pages": {"type": "integer"}
                }
            }
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def setup_openapi(app: FastAPI) -> None:
    """
    Setup custom OpenAPI schema on the app.

    Args:
        app: The FastAPI application instance.
    """
    app.openapi = lambda: customize_openapi(app)
