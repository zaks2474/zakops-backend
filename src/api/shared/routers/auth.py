"""
Authentication API Endpoints

Phase 7: Authentication & Security

Provides login, logout, and session management.
"""

import os
from typing import Optional

from fastapi import APIRouter, Request, Response, HTTPException
from pydantic import BaseModel, EmailStr

from ....core.auth import (
    authenticate_operator,
    create_session,
    invalidate_session,
    create_operator,
    SESSION_COOKIE_NAME,
)
from ..middleware.auth import get_current_operator


router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Login request body."""
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    """Registration request body."""
    email: EmailStr
    password: str
    name: str


class OperatorResponse(BaseModel):
    """Operator response model."""
    id: str
    email: str
    name: str
    role: str


class LoginResponse(BaseModel):
    """Login response model."""
    message: str
    operator: OperatorResponse


class LogoutResponse(BaseModel):
    """Logout response model."""
    message: str


@router.post("/login", response_model=LoginResponse)
async def login(request: Request, response: Response, body: LoginRequest):
    """
    Login with email and password.

    Returns operator info and sets session cookie.
    """
    operator = await authenticate_operator(body.email, body.password)

    if operator is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Create session
    session = create_session(
        operator_id=operator.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )

    # Set cookie
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session.session_id,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
        samesite="lax",
        max_age=int((session.expires_at - session.created_at).total_seconds())
    )

    return LoginResponse(
        message="Login successful",
        operator=OperatorResponse(
            id=str(operator.id),
            email=operator.email,
            name=operator.name,
            role=operator.role
        )
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(request: Request, response: Response):
    """
    Logout and invalidate session.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)

    if session_id:
        invalidate_session(session_id)

    # Clear cookie
    response.delete_cookie(SESSION_COOKIE_NAME)

    return LogoutResponse(message="Logged out successfully")


@router.get("/me", response_model=OperatorResponse)
async def get_current_user(request: Request):
    """
    Get the current authenticated operator.
    """
    operator = get_current_operator(request)

    if operator is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return OperatorResponse(
        id=str(operator.id),
        email=operator.email,
        name=operator.name,
        role=operator.role
    )


@router.post("/register", response_model=OperatorResponse)
async def register(body: RegisterRequest):
    """
    Register a new operator.

    Note: In production, this should be admin-only or disabled.
    """
    # Check if registration is allowed
    if os.getenv("ALLOW_REGISTRATION", "true").lower() != "true":
        raise HTTPException(status_code=403, detail="Registration is disabled")

    try:
        operator = await create_operator(
            email=body.email,
            password=body.password,
            name=body.name,
            role="analyst"  # Default role
        )
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(status_code=409, detail="Email already registered")
        raise HTTPException(status_code=500, detail="Failed to create operator")

    return OperatorResponse(
        id=str(operator.id),
        email=operator.email,
        name=operator.name,
        role=operator.role
    )


@router.get("/check")
async def check_auth(request: Request):
    """
    Check if the current session is authenticated.

    Returns authentication status without requiring auth.
    """
    operator = get_current_operator(request)

    if operator is None:
        return {
            "authenticated": False,
            "operator": None
        }

    return {
        "authenticated": True,
        "operator": OperatorResponse(
            id=str(operator.id),
            email=operator.email,
            name=operator.name,
            role=operator.role
        )
    }
