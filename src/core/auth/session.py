"""
Session Management

Phase 7: Authentication & Security

Handles session creation, validation, and invalidation.
Sessions are stored in memory for simplicity (can be upgraded to Redis).
"""

import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from uuid import UUID

# Session storage (in-memory for now, upgrade to Redis for production)
_sessions: Dict[str, "SessionData"] = {}

# Configuration
SESSION_EXPIRY_HOURS = int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
SESSION_COOKIE_NAME = "zakops_session"


@dataclass
class SessionData:
    """Session data stored server-side."""

    session_id: str
    operator_id: UUID
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict:
        """Convert session data to dictionary."""
        return {
            "session_id": self.session_id,
            "operator_id": str(self.operator_id),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
        }


def generate_session_id() -> str:
    """Generate a secure session ID."""
    return secrets.token_urlsafe(32)


def create_session(
    operator_id: UUID,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> SessionData:
    """
    Create a new session for an operator.

    Args:
        operator_id: The operator's UUID
        ip_address: Client IP address (for audit)
        user_agent: Client user agent (for audit)

    Returns:
        SessionData with session_id to be set as cookie
    """
    session_id = generate_session_id()
    now = datetime.now(timezone.utc)

    session = SessionData(
        session_id=session_id,
        operator_id=operator_id,
        created_at=now,
        expires_at=now + timedelta(hours=SESSION_EXPIRY_HOURS),
        ip_address=ip_address,
        user_agent=user_agent
    )

    _sessions[session_id] = session
    return session


def validate_session(session_id: str) -> Optional[SessionData]:
    """
    Validate a session ID and return session data.

    Args:
        session_id: The session ID from cookie

    Returns:
        SessionData if valid, None if invalid or expired
    """
    session = _sessions.get(session_id)

    if session is None:
        return None

    if session.is_expired():
        # Clean up expired session
        del _sessions[session_id]
        return None

    return session


def invalidate_session(session_id: str) -> bool:
    """
    Invalidate (logout) a session.

    Args:
        session_id: The session ID to invalidate

    Returns:
        True if session was found and invalidated
    """
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def get_active_sessions(operator_id: UUID) -> List[SessionData]:
    """Get all active sessions for an operator."""
    return [
        s for s in _sessions.values()
        if s.operator_id == operator_id and not s.is_expired()
    ]


def invalidate_all_sessions(operator_id: UUID) -> int:
    """
    Invalidate all sessions for an operator (logout everywhere).

    Args:
        operator_id: The operator's UUID

    Returns:
        Number of sessions invalidated
    """
    to_remove = [
        sid for sid, s in _sessions.items()
        if s.operator_id == operator_id
    ]
    for sid in to_remove:
        del _sessions[sid]
    return len(to_remove)


def cleanup_expired_sessions() -> int:
    """
    Remove all expired sessions.

    Returns:
        Number of sessions removed
    """
    expired = [
        sid for sid, s in _sessions.items()
        if s.is_expired()
    ]
    for sid in expired:
        del _sessions[sid]
    return len(expired)
