"""
Permissions System

Phase 7: Authentication & Security

Role-based access control for API endpoints.
"""

import os
from enum import Enum
from typing import Set
from functools import wraps

from fastapi import HTTPException, Request


class Permission(str, Enum):
    """Available permissions."""

    # Deals
    DEALS_READ = "deals:read"
    DEALS_WRITE = "deals:write"
    DEALS_DELETE = "deals:delete"

    # Actions
    ACTIONS_READ = "actions:read"
    ACTIONS_APPROVE = "actions:approve"
    ACTIONS_EXECUTE = "actions:execute"

    # Artifacts
    ARTIFACTS_READ = "artifacts:read"
    ARTIFACTS_WRITE = "artifacts:write"

    # Quarantine
    QUARANTINE_READ = "quarantine:read"
    QUARANTINE_PROCESS = "quarantine:process"

    # Agent
    AGENT_READ = "agent:read"
    AGENT_EXECUTE = "agent:execute"

    # Admin
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"


# Role to permissions mapping
ROLE_PERMISSIONS: dict[str, Set[Permission]] = {
    "admin": {
        Permission.DEALS_READ, Permission.DEALS_WRITE, Permission.DEALS_DELETE,
        Permission.ACTIONS_READ, Permission.ACTIONS_APPROVE, Permission.ACTIONS_EXECUTE,
        Permission.ARTIFACTS_READ, Permission.ARTIFACTS_WRITE,
        Permission.QUARANTINE_READ, Permission.QUARANTINE_PROCESS,
        Permission.AGENT_READ, Permission.AGENT_EXECUTE,
        Permission.ADMIN_USERS, Permission.ADMIN_SYSTEM,
    },
    "analyst": {
        Permission.DEALS_READ, Permission.DEALS_WRITE,
        Permission.ACTIONS_READ, Permission.ACTIONS_APPROVE, Permission.ACTIONS_EXECUTE,
        Permission.ARTIFACTS_READ, Permission.ARTIFACTS_WRITE,
        Permission.QUARANTINE_READ, Permission.QUARANTINE_PROCESS,
        Permission.AGENT_READ, Permission.AGENT_EXECUTE,
    },
    "viewer": {
        Permission.DEALS_READ,
        Permission.ACTIONS_READ,
        Permission.ARTIFACTS_READ,
        Permission.QUARANTINE_READ,
        Permission.AGENT_READ,
    },
}


def get_permissions_for_role(role: str) -> Set[Permission]:
    """
    Get all permissions for a role.

    Args:
        role: Role name (admin, analyst, viewer)

    Returns:
        Set of permissions for the role
    """
    return ROLE_PERMISSIONS.get(role, set())


def has_permission(role: str, permission: Permission) -> bool:
    """
    Check if a role has a specific permission.

    Args:
        role: Role name
        permission: Permission to check

    Returns:
        True if role has the permission
    """
    return permission in get_permissions_for_role(role)


def require_permission(permission: Permission):
    """
    Decorator to require a permission for an endpoint.

    If AUTH_REQUIRED=false, the check is bypassed (dev mode).

    Usage:
        @router.post("/deals")
        @require_permission(Permission.DEALS_WRITE)
        async def create_deal(request: Request):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request in args or kwargs
            request = kwargs.get("request")
            if request is None:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request is None:
                raise HTTPException(status_code=500, detail="Request not found")

            # Check if auth is required
            if os.getenv("AUTH_REQUIRED", "false").lower() != "true":
                # Auth disabled, allow all
                return await func(*args, **kwargs)

            # Get operator from request state
            operator = getattr(request.state, "operator", None)
            if operator is None:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Check permission
            if not has_permission(operator.role, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {permission.value} required"
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


def check_permission(request: Request, permission: Permission) -> bool:
    """
    Check if the current request has a specific permission.

    Args:
        request: FastAPI request
        permission: Permission to check

    Returns:
        True if permission is granted
    """
    # Dev mode bypass
    if os.getenv("AUTH_REQUIRED", "false").lower() != "true":
        return True

    operator = getattr(request.state, "operator", None)
    if operator is None:
        return False

    return has_permission(operator.role, permission)
