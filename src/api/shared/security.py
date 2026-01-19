"""
Security Middleware and Utilities

Production security hardening.

Phase 13: Production Hardening
"""

import logging
import re
import time
from typing import Optional, Callable, Dict, List
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for production hardening.

    Features:
    - Security headers
    - Error sanitization
    - Request logging
    """

    def __init__(self, app, rate_limit: int = 100):
        super().__init__(app)
        self.rate_limit = rate_limit

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Add security headers
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Remove server header if present
        if "server" in response.headers:
            del response.headers["server"]

        return response


def sanitize_error_message(error: Exception) -> str:
    """
    Sanitize error message for client response.

    Removes sensitive information like:
    - File paths
    - Database connection strings
    - Stack traces
    - Internal IDs
    """
    message = str(error)

    # Remove file paths
    message = re.sub(r'/[\w/.-]+\.py', '[file]', message)
    message = re.sub(r'line \d+', 'line [N]', message)

    # Remove connection strings
    message = re.sub(r'postgresql://[^@]+@[^/]+/\w+', '[database]', message)
    message = re.sub(r'redis://[^@]+@[^/]+', '[redis]', message)

    # Remove potential secrets
    message = re.sub(r'password[=:][^\s,;]+', 'password=[REDACTED]', message, flags=re.IGNORECASE)
    message = re.sub(r'secret[=:][^\s,;]+', 'secret=[REDACTED]', message, flags=re.IGNORECASE)
    message = re.sub(r'key[=:][^\s,;]+', 'key=[REDACTED]', message, flags=re.IGNORECASE)

    # Truncate long messages
    if len(message) > 500:
        message = message[:500] + "..."

    return message


def validate_uuid(value: str) -> bool:
    """Validate UUID format."""
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(value))


def validate_correlation_id(value: str) -> bool:
    """Validate correlation_id format."""
    # Allow UUIDs and simple identifiers
    return validate_uuid(value) or bool(re.match(r'^[\w-]{1,100}$', value))


def validate_deal_id(value: str) -> bool:
    """Validate deal_id format (DEAL-XXXXXX)."""
    return bool(re.match(r'^DEAL-\d{6}$', value))


class RateLimiter:
    """
    Simple in-memory rate limiter.

    For production, use Redis-based rate limiting.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.limit = requests_per_minute
        self._timestamps: Dict[str, List[float]] = {}

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Clean old entries
        if key in self._timestamps:
            self._timestamps[key] = [t for t in self._timestamps[key] if t > window_start]
        else:
            self._timestamps[key] = []

        # Check limit
        count = len(self._timestamps[key])
        if count >= self.limit:
            logger.warning(f"Rate limit exceeded for {key}: {count}/{self.limit}")
            return False

        # Record request
        self._timestamps[key].append(now)

        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        window_start = now - 60

        if key not in self._timestamps:
            return self.limit

        current = len([t for t in self._timestamps[key] if t > window_start])
        return max(0, self.limit - current)


# Global rate limiters
general_rate_limiter = RateLimiter(requests_per_minute=60)
auth_rate_limiter = RateLimiter(requests_per_minute=10)  # Stricter for auth


def check_rate_limit(key: str, limiter: RateLimiter = None) -> None:
    """
    Check rate limit and raise HTTPException if exceeded.

    Args:
        key: Rate limit key (usually IP or user ID)
        limiter: RateLimiter instance (defaults to general)

    Raises:
        HTTPException: If rate limit exceeded
    """
    if limiter is None:
        limiter = general_rate_limiter

    if not limiter.is_allowed(key):
        remaining = limiter.get_remaining(key)
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
            headers={"Retry-After": "60", "X-RateLimit-Remaining": str(remaining)}
        )


class CORSConfig:
    """CORS configuration for production."""

    def __init__(
        self,
        allowed_origins: Optional[List[str]] = None,
        allow_credentials: bool = True,
        allow_methods: Optional[List[str]] = None,
        allow_headers: Optional[List[str]] = None,
    ):
        self.allowed_origins = allowed_origins or ["*"]
        self.allow_credentials = allow_credentials
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allow_headers = allow_headers or ["*"]

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if "*" in self.allowed_origins:
            return True
        return origin in self.allowed_origins


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    # Check X-Forwarded-For header (reverse proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Return first IP in the chain
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to direct client
    if request.client:
        return request.client.host

    return "unknown"


# Security headers for Content-Security-Policy
CSP_HEADERS = {
    "default-src": "'self'",
    "script-src": "'self'",
    "style-src": "'self' 'unsafe-inline'",
    "img-src": "'self' data:",
    "font-src": "'self'",
    "connect-src": "'self'",
    "frame-ancestors": "'none'",
}


def build_csp_header() -> str:
    """Build Content-Security-Policy header value."""
    return "; ".join(f"{k} {v}" for k, v in CSP_HEADERS.items())
