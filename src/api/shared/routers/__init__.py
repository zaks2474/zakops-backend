"""Shared API routers."""

from .events import router as events_router
from .hitl import router as hitl_router

__all__ = ["events_router", "hitl_router"]
