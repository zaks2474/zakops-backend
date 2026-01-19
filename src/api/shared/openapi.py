"""
OpenAPI Schema Enhancements

Phase 5: API Stabilization

Customizes the OpenAPI schema for better documentation.
"""

from typing import Dict, Any

from fastapi import FastAPI


def customize_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Customize the OpenAPI schema.

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
## ZakOps Deal Lifecycle OS API

Backend services for the ZakOps autonomous deal management platform.

### Authentication

Authentication is handled via session cookies. Include credentials in requests.

### Response Format

All successful responses follow this structure:

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

### Error Format

All error responses follow this structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": [...],
    "trace_id": "abc-123"
  }
}
```

### Trace Headers

- `X-Trace-ID`: Include in requests for distributed tracing
- `X-Correlation-ID`: Include the deal_id for deal-related requests
        """,
        routes=app.routes,
    )

    # Ensure components exists
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "cookieAuth": {
            "type": "apiKey",
            "in": "cookie",
            "name": "session"
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
                        "description": "Machine-readable error code"
                    },
                    "message": {
                        "type": "string",
                        "description": "Human-readable error message"
                    },
                    "details": {
                        "type": "array",
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
                "required": ["code", "message", "trace_id"]
            }
        }
    }

    openapi_schema["components"]["schemas"]["SuccessResponse"] = {
        "type": "object",
        "properties": {
            "data": {
                "description": "Response data (type varies by endpoint)"
            },
            "meta": {
                "type": "object",
                "properties": {
                    "trace_id": {
                        "type": "string",
                        "description": "Unique trace ID for this request"
                    },
                    "correlation_id": {
                        "type": "string",
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

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def setup_openapi(app: FastAPI) -> None:
    """
    Setup custom OpenAPI schema on the app.

    Args:
        app: The FastAPI application instance.
    """
    app.openapi = lambda: customize_openapi(app)
