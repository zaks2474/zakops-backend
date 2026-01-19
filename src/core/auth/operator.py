"""
Operator Management

Phase 7: Authentication & Security

Handles operator authentication and management.
"""

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4
from dataclasses import dataclass

from ..database import get_database


@dataclass
class Operator:
    """Operator (user) model."""

    id: UUID
    email: str
    name: str
    role: str  # "admin", "analyst", "viewer"
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert operator to dictionary."""
        return {
            "id": str(self.id),
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a password with salt using PBKDF2.

    Args:
        password: Plain text password
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt.encode(),
        100000
    ).hex()
    return hashed, salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """
    Verify a password against hash.

    Args:
        password: Plain text password to verify
        hashed: Stored password hash
        salt: Stored salt

    Returns:
        True if password matches
    """
    check_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(check_hash, hashed)


async def authenticate_operator(email: str, password: str) -> Optional[Operator]:
    """
    Authenticate an operator by email and password.

    Args:
        email: Operator's email
        password: Plain text password

    Returns:
        Operator if authenticated, None if failed
    """
    db = await get_database()

    row = await db.fetchrow(
        """
        SELECT id, email, name, role, is_active, password_hash, password_salt,
               created_at, last_login_at
        FROM zakops.operators
        WHERE email = $1 AND is_active = true
        """,
        email.lower()
    )

    if not row:
        return None

    # Check if password fields exist
    if not row.get("password_hash") or not row.get("password_salt"):
        return None

    if not verify_password(password, row["password_hash"], row["password_salt"]):
        return None

    # Update last login
    now = datetime.now(timezone.utc)
    await db.execute(
        "UPDATE zakops.operators SET last_login_at = $1 WHERE id = $2",
        now,
        row["id"]
    )

    return Operator(
        id=row["id"],
        email=row["email"],
        name=row["name"],
        role=row["role"],
        is_active=row["is_active"],
        created_at=row["created_at"],
        last_login_at=now
    )


async def get_operator_by_id(operator_id: UUID) -> Optional[Operator]:
    """
    Get an operator by ID.

    Args:
        operator_id: The operator's UUID

    Returns:
        Operator if found, None otherwise
    """
    db = await get_database()

    row = await db.fetchrow(
        """
        SELECT id, email, name, role, is_active, created_at, last_login_at
        FROM zakops.operators
        WHERE id = $1
        """,
        operator_id
    )

    if not row:
        return None

    return Operator(
        id=row["id"],
        email=row["email"],
        name=row["name"],
        role=row["role"],
        is_active=row["is_active"],
        created_at=row["created_at"],
        last_login_at=row.get("last_login_at")
    )


async def create_operator(
    email: str,
    password: str,
    name: str,
    role: str = "analyst"
) -> Operator:
    """
    Create a new operator.

    Args:
        email: Operator's email (must be unique)
        password: Plain text password (will be hashed)
        name: Display name
        role: Role (admin, analyst, viewer)

    Returns:
        Created Operator
    """
    db = await get_database()

    operator_id = uuid4()
    password_hash, password_salt = hash_password(password)
    now = datetime.now(timezone.utc)

    await db.execute(
        """
        INSERT INTO zakops.operators (id, email, name, role, password_hash, password_salt,
                               is_active, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        operator_id,
        email.lower(),
        name,
        role,
        password_hash,
        password_salt,
        True,
        now
    )

    return Operator(
        id=operator_id,
        email=email.lower(),
        name=name,
        role=role,
        is_active=True,
        created_at=now
    )


async def update_operator_password(operator_id: UUID, new_password: str) -> bool:
    """
    Update an operator's password.

    Args:
        operator_id: The operator's UUID
        new_password: New plain text password

    Returns:
        True if updated successfully
    """
    db = await get_database()

    password_hash, password_salt = hash_password(new_password)

    result = await db.execute(
        """
        UPDATE zakops.operators
        SET password_hash = $1, password_salt = $2
        WHERE id = $3
        """,
        password_hash,
        password_salt,
        operator_id
    )

    return "UPDATE" in result


async def list_operators(
    limit: int = 50,
    offset: int = 0,
    include_inactive: bool = False
) -> list[Operator]:
    """
    List all operators.

    Args:
        limit: Maximum number of operators to return
        offset: Number of operators to skip
        include_inactive: Include inactive operators

    Returns:
        List of Operators
    """
    db = await get_database()

    if include_inactive:
        rows = await db.fetch(
            """
            SELECT id, email, name, role, is_active, created_at, last_login_at
            FROM zakops.operators
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset
        )
    else:
        rows = await db.fetch(
            """
            SELECT id, email, name, role, is_active, created_at, last_login_at
            FROM zakops.operators
            WHERE is_active = true
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset
        )

    return [
        Operator(
            id=row["id"],
            email=row["email"],
            name=row["name"],
            role=row["role"],
            is_active=row["is_active"],
            created_at=row["created_at"],
            last_login_at=row.get("last_login_at")
        )
        for row in rows
    ]
