"""
Authentication Middleware

Phase 7: Authentication & Security

Validates session cookies and loads operator into request state.
"""

import os
from typing import Callable, Optional
from datetime import datetime, timezone
from uuid import UUID

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


def is_auth_required() -> bool:
    """Check if authentication is required."""
    return os.getenv("AUTH_REQUIRED", "false").lower() == "true"


# Paths that don't require authentication (even when AUTH_REQUIRED=true)
PUBLIC_PATHS = {
    "/api/auth/login",
    "/api/auth/register",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}

# Path prefixes that don't require authentication
PUBLIC_PREFIXES = [
    "/docs",
    "/redoc",
]


def is_public_path(path: str) -> bool:
    """Check if a path is public (doesn't require auth)."""
    if path in PUBLIC_PATHS:
        return True
    for prefix in PUBLIC_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Checks if auth is required (AUTH_REQUIRED env var)
    2. Validates session cookie
    3. Loads operator into request.state
    4. Allows public paths without auth
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Always set operator to None initially
        request.state.operator = None

        # Check if path is public
        if is_public_path(request.url.path):
            return await call_next(request)

        # Check if auth is required
        if not is_auth_required():
            # Auth disabled, create a mock operator for development
            request.state.operator = _create_dev_operator()
            return await call_next(request)

        # Import here to avoid circular imports
        from ....core.auth.session import validate_session, SESSION_COOKIE_NAME
        from ....core.auth.operator import get_operator_by_id

        # Get session cookie
        session_id = request.cookies.get(SESSION_COOKIE_NAME)

        if not session_id:
            # No session, but let the endpoint decide if auth is required
            return await call_next(request)

        # Validate session
        session = validate_session(session_id)

        if session is None:
            # Invalid or expired session
            return await call_next(request)

        # Load operator
        operator = await get_operator_by_id(session.operator_id)

        if operator and operator.is_active:
            request.state.operator = operator

        return await call_next(request)


def _create_dev_operator():
    """Create a mock operator for development mode."""
    from ....core.auth.operator import Operator

    return Operator(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        email="dev@zakops.local",
        name="Development User",
        role="admin",
        is_active=True,
        created_at=datetime.now(timezone.utc)
    )


def get_current_operator(request: Request) -> Optional["Operator"]:
    """
    Get the current operator from request state.

    Args:
        request: FastAPI request

    Returns:
        Operator if authenticated, None otherwise
    """
    return getattr(request.state, "operator", None)


def require_auth(request: Request) -> "Operator":
    """
    Require authentication and return the operator.

    Raises HTTPException if not authenticated.

    Args:
        request: FastAPI request

    Returns:
        Authenticated Operator

    Raises:
        HTTPException: If not authenticated
    """
    from fastapi import HTTPException

    operator = get_current_operator(request)

    if operator is None:
        if is_auth_required():
            raise HTTPException(status_code=401, detail="Authentication required")
        else:
            return _create_dev_operator()

    return operator
