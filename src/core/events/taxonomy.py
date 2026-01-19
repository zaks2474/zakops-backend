"""
ZakOps Event Taxonomy

Defines all event types following the Master Architecture Specification §5.1.

Event naming convention: {domain}.{action}
- domain: deal, action, agent, worker, system
- action: past tense verb (created, updated, completed, failed)
"""

from enum import Enum
from typing import Dict, Any


class EventDomain(str, Enum):
    """Top-level event domains."""
    DEAL = "deal"
    ACTION = "action"
    AGENT = "agent"
    WORKER = "worker"
    SYSTEM = "system"


class DealEventType(str, Enum):
    """Deal lifecycle events."""
    CREATED = "deal.created"
    UPDATED = "deal.updated"
    STAGE_CHANGED = "deal.stage_changed"
    PROFILE_ENRICHED = "deal.profile_enriched"
    ARCHIVED = "deal.archived"


class ActionEventType(str, Enum):
    """Action workflow events."""
    CREATED = "action.created"
    APPROVED = "action.approved"
    REJECTED = "action.rejected"
    EXECUTING = "action.executing"
    COMPLETED = "action.completed"
    FAILED = "action.failed"
    QUARANTINED = "action.quarantined"


class AgentEventType(str, Enum):
    """Agent execution events."""
    RUN_STARTED = "agent.run_started"
    RUN_COMPLETED = "agent.run_completed"
    RUN_FAILED = "agent.run_failed"
    TOOL_CALLED = "agent.tool_called"
    TOOL_COMPLETED = "agent.tool_completed"
    TOOL_FAILED = "agent.tool_failed"
    THINKING = "agent.thinking"
    WAITING_APPROVAL = "agent.waiting_approval"


class WorkerEventType(str, Enum):
    """Background worker events."""
    JOB_QUEUED = "worker.job_queued"
    JOB_STARTED = "worker.job_started"
    JOB_COMPLETED = "worker.job_completed"
    JOB_FAILED = "worker.job_failed"
    JOB_RETRYING = "worker.job_retrying"
    JOB_DLQ = "worker.job_dlq"


class SystemEventType(str, Enum):
    """System-level events."""
    STARTUP = "system.startup"
    SHUTDOWN = "system.shutdown"
    HEALTH_CHECK = "system.health_check"
    ERROR = "system.error"


# Combined lookup for all event types
ALL_EVENT_TYPES: Dict[str, str] = {
    **{e.value: e.name for e in DealEventType},
    **{e.value: e.name for e in ActionEventType},
    **{e.value: e.name for e in AgentEventType},
    **{e.value: e.name for e in WorkerEventType},
    **{e.value: e.name for e in SystemEventType},
}


def validate_event_type(event_type: str) -> bool:
    """Check if event type is valid."""
    return event_type in ALL_EVENT_TYPES


def get_domain(event_type: str) -> str:
    """Extract domain from event type."""
    return event_type.split(".")[0] if "." in event_type else "unknown"
