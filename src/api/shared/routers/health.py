"""
Health Check Endpoints

Phase 8: OpenAPI & Tooling
Provides health, readiness, and liveness endpoints for container orchestration.
"""

from datetime import datetime
from typing import Dict, Any
import os

from fastapi import APIRouter, Response

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check.

    Returns 200 if the service is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.getenv("APP_VERSION", "1.0.0")
    }


@router.get("/health/live")
async def liveness_check() -> Dict[str, str]:
    """
    Kubernetes liveness probe.

    Returns 200 if the process is alive.
    Used by k8s to determine if the container should be restarted.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_check(response: Response) -> Dict[str, Any]:
    """
    Kubernetes readiness probe.

    Returns 200 if the service is ready to accept traffic.
    Checks database connectivity and other dependencies.
    """
    checks = {}
    all_healthy = True

    # Check database
    try:
        from ....core.database.adapter import get_database
        db = await get_database()
        await db.fetchrow("SELECT 1")
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)[:100]}"
        all_healthy = False

    # Check outbox processor (if enabled)
    if os.getenv("OUTBOX_ENABLED", "true").lower() == "true":
        try:
            from ....core.outbox.processor import _processor
            if _processor and _processor._running:
                checks["outbox_processor"] = "running"
            else:
                checks["outbox_processor"] = "not running"
        except Exception:
            checks["outbox_processor"] = "unknown"

    status = "ready" if all_healthy else "not_ready"

    if not all_healthy:
        response.status_code = 503

    return {
        "status": status,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/startup")
async def startup_check() -> Dict[str, Any]:
    """
    Kubernetes startup probe.

    Returns 200 if the service has completed initialization.
    Used by k8s to determine when the container is ready to receive probes.
    """
    # Check if database pool is initialized
    try:
        from ....core.database.adapter import get_database
        db = await get_database()
        await db.fetchrow("SELECT 1")
        return {
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "starting",
            "message": str(e)[:100],
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/api/version")
async def get_version() -> Dict[str, Any]:
    """
    Get build version information.

    Returns version, git commit, and build time for deployment verification.
    """
    return {
        "version": os.getenv("APP_VERSION", "dev"),
        "commit": os.getenv("GIT_COMMIT", "unknown"),
        "build_time": os.getenv("BUILD_TIME", "unknown"),
        "environment": os.getenv("APP_ENV", "development")
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with system information.

    Note: Should be protected in production.
    """
    import platform
    import sys

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "environment": os.getenv("APP_ENV", "development"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": {
            "auth_required": os.getenv("AUTH_REQUIRED", "false"),
            "outbox_enabled": os.getenv("OUTBOX_ENABLED", "true"),
            "database_backend": os.getenv("DATABASE_BACKEND", "postgresql"),
        }
    }
