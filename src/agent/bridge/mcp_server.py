#!/usr/bin/env python3
"""
ZakOps Agent Bridge - MCP Server (FastMCP)
==========================================
True MCP server for LangSmith Agent Builder integration.

Architecture:
- LangSmith Agent Builder (cloud) -> Cloudflare Tunnel -> This MCP Server (local)
- Server proxies to: Deal Lifecycle API (:8090), RAG API (:8052), DataRoom filesystem

Transport: SSE (Server-Sent Events)
Endpoint: /sse
Port: 9100

Security:
- Bearer token authentication on all MCP requests
- Path traversal protection on all file operations
- Atomic writes with persistence verification
- No delete operations
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# =============================================================================
# Configuration
# =============================================================================

BRIDGE_PORT = int(os.getenv("ZAKOPS_BRIDGE_PORT", "9100"))
BRIDGE_HOST = os.getenv("ZAKOPS_BRIDGE_HOST", "127.0.0.1")
API_KEY = os.getenv("ZAKOPS_BRIDGE_API_KEY", "")

# Backend services
DEAL_API_URL = os.getenv("ZAKOPS_DEAL_API_URL", "http://localhost:8090")
RAG_API_URL = os.getenv("ZAKOPS_RAG_API_URL", "http://localhost:8052")

# Filesystem paths
DATAROOM_ROOT = Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom"))
PIPELINE_PATH = DATAROOM_ROOT / "00-PIPELINE" / "Inbound"
REGISTRY_PATH = DATAROOM_ROOT / ".deal-registry"
LOG_PATH = REGISTRY_PATH / "logs" / "agent_bridge.jsonl"

# Ensure log directory exists
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Logging Setup
# =============================================================================

class JSONLogHandler(logging.Handler):
    """Log handler that writes structured JSON to a file."""

    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath

    def emit(self, record: logging.LogRecord):
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
            }
            if hasattr(record, "correlation_id"):
                log_entry["correlation_id"] = record.correlation_id
            if hasattr(record, "extra_data"):
                log_entry["data"] = record.extra_data

            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass  # Don't fail on logging errors


# Setup logging
logger = logging.getLogger("agent_bridge")
logger.setLevel(logging.INFO)
logger.addHandler(JSONLogHandler(LOG_PATH))

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(console_handler)


def log_tool_call(tool_name: str, params: dict, correlation_id: str = None):
    """Log a tool call."""
    correlation_id = correlation_id or str(uuid.uuid4())[:8]
    record = logging.LogRecord(
        name="agent_bridge",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=f"Tool call: {tool_name}",
        args=(),
        exc_info=None,
    )
    record.correlation_id = correlation_id
    record.extra_data = params
    logger.handle(record)


def log_error(error: str, details: Optional[dict] = None, correlation_id: str = None):
    """Log an error."""
    correlation_id = correlation_id or str(uuid.uuid4())[:8]
    record = logging.LogRecord(
        name="agent_bridge",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg=error,
        args=(),
        exc_info=None,
    )
    record.correlation_id = correlation_id
    record.extra_data = details or {}
    logger.handle(record)


# =============================================================================
# Authentication Middleware
# =============================================================================

class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to verify Bearer token on all requests except health."""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request, call_next):
        # Allow health check without auth
        if request.url.path == "/health":
            return await call_next(request)

        # If no API key configured, allow all (development mode)
        if not self.api_key:
            logger.warning("No API key configured - running in development mode")
            return await call_next(request)

        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")

        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing Authorization header"}
            )

        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid Authorization header format"}
            )

        token = auth_header[7:]
        if token != self.api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"}
            )

        return await call_next(request)


# =============================================================================
# HTTP Access Logging Middleware
# =============================================================================

ACCESS_LOG_PATH = Path("/var/log/zakops/access.log")


class HTTPAccessLogMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests to access.log."""

    async def dispatch(self, request, call_next):
        # Capture request details
        timestamp = datetime.now(timezone.utc).isoformat()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = str(request.url.path)
        query = str(request.url.query) if request.url.query else ""

        # Capture headers (redact sensitive values)
        headers = dict(request.headers)
        if "authorization" in headers:
            # Redact the token, show only first 8 chars
            auth_val = headers["authorization"]
            if len(auth_val) > 15:
                headers["authorization"] = auth_val[:15] + "..."

        # Log the request
        log_entry = {
            "timestamp": timestamp,
            "ip": client_ip,
            "method": method,
            "path": path,
            "query": query,
            "headers": headers,
        }

        try:
            with open(ACCESS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            # Also print to console if file write fails
            print(f"[ACCESS] {timestamp} {client_ip} {method} {path} (log write failed: {e})")

        # Also print a summary line to console/journal
        print(f"[ACCESS] {timestamp} {client_ip} {method} {path}")

        # Continue processing the request
        response = await call_next(request)
        return response


# =============================================================================
# Path Safety Utilities
# =============================================================================

def validate_path_safe(base_path: Path, relative_path: str) -> Path:
    """
    Validate that a relative path is safe (no traversal) and returns the resolved path.
    Raises ValueError if path is unsafe.
    """
    # Block obvious traversal patterns
    if ".." in relative_path:
        raise ValueError("Path traversal blocked: '..' not allowed")

    if relative_path.startswith("/"):
        raise ValueError("Absolute paths not allowed")

    # Resolve and verify containment
    resolved = (base_path / relative_path).resolve()

    try:
        resolved.relative_to(base_path.resolve())
    except ValueError:
        raise ValueError("Path traversal blocked: path escapes base directory")

    return resolved


def get_deal_folder(deal_id: str) -> Path:
    """Get the folder path for a deal, validating it exists.

    Folder names follow the pattern: {CanonicalName}--{YearNumber}
    e.g., 'Textile-Art-Education-Business--2026-003' for deal_id 'DEAL-2026-003'
    """
    # Extract the numeric suffix from deal_id (e.g., "2026-003" from "DEAL-2026-003")
    suffix = deal_id.replace("DEAL-", "")

    for folder in PIPELINE_PATH.iterdir():
        if folder.is_dir() and folder.name.endswith(f"--{suffix}"):
            return folder

    raise ValueError(f"Deal folder not found for {deal_id}")


# =============================================================================
# Atomic Write Utilities
# =============================================================================

def atomic_json_write(filepath: Path, data: dict[str, Any], verify_field: str = "updated_at") -> bool:
    """
    Atomically write JSON to a file with verification.
    Returns True if successful and verified.
    """
    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp if not present
    if verify_field not in data:
        data[verify_field] = datetime.now(timezone.utc).isoformat()

    # Write to temp file
    temp_path = filepath.parent / f".{filepath.name}.{uuid.uuid4().hex[:8]}.tmp"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

        # Atomic rename
        temp_path.rename(filepath)

        # Verify
        with open(filepath, "r", encoding="utf-8") as f:
            verified = json.load(f)

        if verified.get(verify_field) != data.get(verify_field):
            raise RuntimeError("Verification failed: written data does not match")

        return True

    except Exception as e:
        # Cleanup temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise e


# =============================================================================
# Create MCP Server
# =============================================================================

mcp = FastMCP(
    name="ZakOps Bridge",
    instructions="""
    ZakOps Agent Bridge - Local Infrastructure Access for LangSmith Agents.

    This server provides tools to interact with the local ZakOps deal lifecycle system:
    - List, read, and update deals in the pipeline
    - Create and manage actions (proposals that may require human approval)
    - Query the local RAG database for document search
    - Write artifacts to deal folders

    All state changes that require human approval will be queued for review.
    """,
)


# =============================================================================
# DEAL TOOLS (6)
# =============================================================================

@mcp.tool()
def zakops_list_deals(
    status: Optional[str] = None,
    stage: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """
    List all deals in the local ZakOps pipeline with metadata.

    Args:
        status: Optional filter by status (active, inactive)
        stage: Optional filter by stage (INBOUND, SCREENING, QUALIFIED, LOI, DILIGENCE, etc.)
        limit: Maximum number of deals to return (default 50)

    Returns:
        Dictionary containing count and list of deals with their metadata
    """
    log_tool_call("zakops_list_deals", {"status": status, "stage": stage, "limit": limit})

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{DEAL_API_URL}/api/deals")
            if resp.status_code != 200:
                return {"error": f"Failed to fetch deals: HTTP {resp.status_code}"}
            data = resp.json()

        deals = data.get("deals", [])

        # Apply filters
        if status:
            deals = [d for d in deals if d.get("status") == status]
        if stage:
            deals = [d for d in deals if d.get("stage") == stage]

        # Apply limit
        deals = deals[:limit]

        return {
            "count": len(deals),
            "deals": deals,
        }

    except httpx.HTTPError as e:
        log_error(f"Failed to list deals: {e}")
        return {"error": f"Deal API error: {str(e)}"}


@mcp.tool()
def zakops_get_deal(deal_id: str) -> dict:
    """
    Get complete deal state including deal_profile.json, classified_links.json, and triage_summary.json.

    Args:
        deal_id: Deal ID in format DEAL-YYYY-XXX (e.g., DEAL-2026-001)

    Returns:
        Complete deal state with all enrichments from filesystem
    """
    log_tool_call("zakops_get_deal", {"deal_id": deal_id})

    try:
        # Get deal from API
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{DEAL_API_URL}/api/deals/{deal_id}")
            if resp.status_code != 200:
                return {"error": f"Deal {deal_id} not found"}
            deal = resp.json()

        # Get deal folder
        try:
            deal_folder = get_deal_folder(deal_id)
            deal["_folder_exists"] = True
            deal["_folder_path"] = str(deal_folder)
        except ValueError:
            deal["_folder_exists"] = False
            return deal

        # Load deal_profile.json if exists
        profile_path = deal_folder / "deal_profile.json"
        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    deal["deal_profile"] = json.load(f)
            except Exception as e:
                deal["deal_profile"] = {"_error": str(e)}

        # Load classified_links.json if exists
        classified_path = deal_folder / "07-Correspondence" / "classified_links.json"
        if classified_path.exists():
            try:
                with open(classified_path, "r", encoding="utf-8") as f:
                    deal["classified_links"] = json.load(f)
            except Exception as e:
                deal["classified_links"] = {"_error": str(e)}

        # Load triage_summary.json from first bundle if exists
        corr_dir = deal_folder / "07-Correspondence"
        if corr_dir.exists():
            for bundle_dir in sorted(corr_dir.iterdir()):
                if bundle_dir.is_dir():
                    manifest_path = bundle_dir / "manifest.json"
                    if manifest_path.exists():
                        try:
                            with open(manifest_path, "r", encoding="utf-8") as f:
                                manifest = json.load(f)
                            quarantine_dir = manifest.get("quarantine_dir")
                            if quarantine_dir:
                                triage_path = Path(quarantine_dir) / "triage_summary.json"
                                if triage_path.exists():
                                    with open(triage_path, "r", encoding="utf-8") as f:
                                        deal["triage_summary_source"] = json.load(f)
                                    break
                        except Exception:
                            pass

        return deal

    except httpx.HTTPError as e:
        log_error(f"Failed to get deal {deal_id}: {e}")
        return {"error": f"Deal API error: {str(e)}"}


@mcp.tool()
def zakops_update_deal_profile(deal_id: str, profile_patch: dict) -> dict:
    """
    Update deal_profile.json with atomic write and persistence verification.

    Args:
        deal_id: Deal ID to update (e.g., DEAL-2026-001)
        profile_patch: Fields to update/add (merged with existing profile)

    Returns:
        Confirmation with updated fields and verification status
    """
    log_tool_call("zakops_update_deal_profile", {"deal_id": deal_id, "patch_keys": list(profile_patch.keys())})

    try:
        deal_folder = get_deal_folder(deal_id)
    except ValueError as e:
        return {"error": str(e)}

    profile_path = deal_folder / "deal_profile.json"

    # Load existing profile or create new
    if profile_path.exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception:
            profile = {}
    else:
        profile = {"deal_id": deal_id, "created_at": datetime.now(timezone.utc).isoformat()}

    # Apply patch
    profile.update(profile_patch)
    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    profile["updated_by"] = "langsmith_bridge"

    # Atomic write with verification
    try:
        verified = atomic_json_write(profile_path, profile, verify_field="updated_at")

        return {
            "success": True,
            "verified": verified,
            "deal_id": deal_id,
            "updated_fields": list(profile_patch.keys()),
        }

    except Exception as e:
        log_error(f"Failed to update deal profile: {e}")
        return {"error": f"Failed to persist deal profile: {str(e)}"}


@mcp.tool()
def zakops_write_deal_artifact(
    deal_id: str,
    relative_path: str,
    content: str,
    content_type: str = "text/plain",
) -> dict:
    """
    Write a file artifact to a deal's folder with safety checks.

    Args:
        deal_id: Deal ID (e.g., DEAL-2026-001)
        relative_path: Path relative to deal folder (cannot contain '..' or start with '/')
        content: File content to write
        content_type: Content type (default text/plain)

    Returns:
        Confirmation with path and size
    """
    log_tool_call("zakops_write_deal_artifact", {"deal_id": deal_id, "relative_path": relative_path})

    try:
        deal_folder = get_deal_folder(deal_id)
    except ValueError as e:
        return {"error": str(e)}

    # Validate path safety
    try:
        target_path = validate_path_safe(deal_folder, relative_path)
    except ValueError as e:
        return {"error": str(e)}

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file atomically
    temp_path = target_path.parent / f".{target_path.name}.{uuid.uuid4().hex[:8]}.tmp"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)

        temp_path.rename(target_path)

        # Verify
        with open(target_path, "r", encoding="utf-8") as f:
            written = f.read()

        if written != content:
            raise RuntimeError("Verification failed: written content does not match")

        return {
            "success": True,
            "verified": True,
            "deal_id": deal_id,
            "path": relative_path,
            "size_bytes": len(content),
        }

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        log_error(f"Failed to write artifact: {e}")
        return {"error": f"Failed to write artifact: {str(e)}"}


@mcp.tool()
def zakops_list_deal_artifacts(deal_id: str) -> dict:
    """
    List all files in a deal's folder with sizes and modification dates.

    Args:
        deal_id: Deal ID (e.g., DEAL-2026-001)

    Returns:
        List of artifacts with metadata (path, size, modified_at)
    """
    log_tool_call("zakops_list_deal_artifacts", {"deal_id": deal_id})

    try:
        deal_folder = get_deal_folder(deal_id)
    except ValueError as e:
        return {"error": str(e)}

    artifacts = []

    def scan_dir(path: Path, prefix: str = ""):
        for item in sorted(path.iterdir()):
            rel_path = f"{prefix}/{item.name}" if prefix else item.name
            if item.is_file():
                artifacts.append({
                    "path": rel_path,
                    "size_bytes": item.stat().st_size,
                    "modified_at": datetime.fromtimestamp(
                        item.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                })
            elif item.is_dir() and not item.name.startswith("."):
                scan_dir(item, rel_path)

    scan_dir(deal_folder)

    return {
        "deal_id": deal_id,
        "folder_path": str(deal_folder),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


@mcp.tool()
def zakops_list_quarantine(limit: int = 20) -> dict:
    """
    List emails in quarantine awaiting human review.

    Args:
        limit: Maximum number of items to return (default 20)

    Returns:
        List of quarantine items pending review
    """
    log_tool_call("zakops_list_quarantine", {"limit": limit})

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"{DEAL_API_URL}/api/actions/quarantine",
                params={"limit": limit},
            )
            if resp.status_code != 200:
                return {"error": f"Failed to list quarantine: HTTP {resp.status_code}"}
            return resp.json()

    except httpx.HTTPError as e:
        log_error(f"Failed to list quarantine: {e}")
        return {"error": f"Deal API error: {str(e)}"}


# =============================================================================
# ACTION TOOLS (4)
# =============================================================================

@mcp.tool()
def zakops_create_action(
    action_type: str,
    title: str,
    inputs: Optional[dict] = None,
    deal_id: Optional[str] = None,
    requires_approval: bool = True,
) -> dict:
    """
    Create an action (proposal) in the local ZakOps system.

    Actions with requires_approval=True wait for human approval before executing.
    This is the primary way to propose state changes.

    Args:
        action_type: Action type (e.g., DEAL.CREATE_FROM_EMAIL, DEAL.UPDATE_STAGE, RAG.REINDEX_DEAL)
        title: Human-readable title describing the action
        inputs: Action-specific input parameters (optional)
        deal_id: Associated deal ID if applicable (optional)
        requires_approval: Whether action needs human approval (default True)

    Returns:
        Created action with ID and status
    """
    log_tool_call("zakops_create_action", {
        "action_type": action_type,
        "title": title,
        "deal_id": deal_id,
        "requires_approval": requires_approval,
    })

    try:
        payload = {
            "action_type": action_type,
            "title": title,
            "inputs": inputs or {},
            "deal_id": deal_id,
            "source": "langsmith_bridge",
            "created_by": "langsmith_agent",
            "requires_human_review": requires_approval,
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{DEAL_API_URL}/api/actions", json=payload)
            if resp.status_code not in (200, 201):
                return {"error": f"Failed to create action: HTTP {resp.status_code}"}
            data = resp.json()

        return {
            "success": True,
            "action_id": data.get("action", {}).get("action_id"),
            "status": data.get("action", {}).get("status"),
            "requires_approval": requires_approval,
            "message": "Action created successfully",
        }

    except httpx.HTTPError as e:
        log_error(f"Failed to create action: {e}")
        return {"error": f"Deal API error: {str(e)}"}


@mcp.tool()
def zakops_get_action(action_id: str) -> dict:
    """
    Get action status, outputs, and timestamps.

    Args:
        action_id: Action ID (e.g., ACTION-123456)

    Returns:
        Action details including status and outputs
    """
    log_tool_call("zakops_get_action", {"action_id": action_id})

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{DEAL_API_URL}/api/actions/{action_id}")
            if resp.status_code != 200:
                return {"error": f"Action {action_id} not found"}
            return resp.json()

    except httpx.HTTPError as e:
        log_error(f"Failed to get action {action_id}: {e}")
        return {"error": f"Deal API error: {str(e)}"}


@mcp.tool()
def zakops_list_actions(
    status: Optional[str] = None,
    action_type: Optional[str] = None,
    deal_id: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """
    List actions with optional filters.

    Args:
        status: Filter by status (PENDING_APPROVAL, QUEUED, RUNNING, COMPLETED, FAILED)
        action_type: Filter by action type
        deal_id: Filter by associated deal
        limit: Maximum number of actions to return (default 20)

    Returns:
        List of actions matching filters
    """
    log_tool_call("zakops_list_actions", {
        "status": status,
        "action_type": action_type,
        "deal_id": deal_id,
        "limit": limit,
    })

    try:
        params = {"limit": limit}
        if status:
            params["status"] = status
        if action_type:
            params["action_type"] = action_type
        if deal_id:
            params["deal_id"] = deal_id

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{DEAL_API_URL}/api/actions", params=params)
            if resp.status_code != 200:
                return {"error": f"Failed to list actions: HTTP {resp.status_code}"}
            return resp.json()

    except httpx.HTTPError as e:
        log_error(f"Failed to list actions: {e}")
        return {"error": f"Deal API error: {str(e)}"}


@mcp.tool()
def zakops_approve_quarantine(action_id: str) -> dict:
    """
    Approve a quarantine item, triggering deal creation from email.

    Args:
        action_id: Quarantine action ID to approve

    Returns:
        Approval result
    """
    log_tool_call("zakops_approve_quarantine", {"action_id": action_id})

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{DEAL_API_URL}/api/actions/quarantine/{action_id}/approve")
            if resp.status_code != 200:
                return {"error": f"Failed to approve: HTTP {resp.status_code}"}
            return resp.json()

    except httpx.HTTPError as e:
        log_error(f"Failed to approve quarantine {action_id}: {e}")
        return {"error": f"Deal API error: {str(e)}"}


# =============================================================================
# RAG TOOLS (2)
# =============================================================================

@mcp.tool()
def rag_query_local(
    query: str,
    deal_id: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Search the local RAG database for relevant document chunks.

    Args:
        query: Search query (e.g., 'customer concentration', 'EBITDA adjustments')
        deal_id: Optional filter to specific deal
        top_k: Number of results to return (1-20, default 5)

    Returns:
        Relevant document chunks with similarity scores
    """
    log_tool_call("rag_query_local", {"query": query[:50], "deal_id": deal_id, "top_k": top_k})

    # Clamp top_k
    top_k = max(1, min(20, top_k))

    try:
        payload = {
            "query": query,
            "top_k": top_k,
        }

        # Filter by deal if specified
        if deal_id:
            payload["filter"] = {"deal_id": deal_id}

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{RAG_API_URL}/rag/query", json=payload)
            if resp.status_code != 200:
                return {"error": f"RAG query failed: HTTP {resp.status_code}"}
            return resp.json()

    except httpx.ConnectError:
        return {"error": "RAG API unreachable"}
    except httpx.HTTPError as e:
        log_error(f"Failed to query RAG: {e}")
        return {"error": f"RAG API error: {str(e)}"}


@mcp.tool()
def rag_reindex_deal(deal_id: str, artifact_paths: Optional[list] = None) -> dict:
    """
    Trigger reindexing of deal documents in RAG database.

    Args:
        deal_id: Deal ID to reindex (e.g., DEAL-2026-001)
        artifact_paths: Optional specific paths to reindex (if empty, reindexes all)

    Returns:
        Reindex action created
    """
    log_tool_call("rag_reindex_deal", {"deal_id": deal_id, "artifact_paths": artifact_paths})

    try:
        # Get deal folder
        deal_folder = get_deal_folder(deal_id)

        # Call RAG index endpoint
        payload = {
            "paths": [str(deal_folder)],
            "force": True,
        }

        if artifact_paths:
            # Validate and use specific paths
            validated_paths = []
            for rel_path in artifact_paths:
                try:
                    full_path = validate_path_safe(deal_folder, rel_path)
                    validated_paths.append(str(full_path))
                except ValueError:
                    pass  # Skip invalid paths
            if validated_paths:
                payload["paths"] = validated_paths

        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{RAG_API_URL}/rag/index", json=payload)
            if resp.status_code != 200:
                return {"error": f"RAG index failed: HTTP {resp.status_code}"}
            return resp.json()

    except ValueError as e:
        return {"error": str(e)}
    except httpx.ConnectError:
        return {"error": "RAG API unreachable"}
    except httpx.HTTPError as e:
        log_error(f"Failed to reindex deal {deal_id}: {e}")
        return {"error": f"RAG API error: {str(e)}"}


# =============================================================================
# Health Check (Custom Route)
# =============================================================================

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint (no authentication required)."""
    from starlette.responses import JSONResponse

    checks = {
        "bridge": "healthy",
        "deal_api": "unknown",
        "rag_api": "unknown",
        "dataroom": "unknown",
    }

    # Check Deal API
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{DEAL_API_URL}/api/deals")
            checks["deal_api"] = "healthy" if resp.status_code == 200 else "degraded"
    except Exception:
        checks["deal_api"] = "unhealthy"

    # Check RAG API
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{RAG_API_URL}/rag/stats")
            checks["rag_api"] = "healthy" if resp.status_code == 200 else "degraded"
    except Exception:
        checks["rag_api"] = "unhealthy"

    # Check DataRoom
    checks["dataroom"] = "healthy" if DATAROOM_ROOT.exists() else "unhealthy"

    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"

    return JSONResponse({
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "mcp_version": "fastmcp",
        "transport": "sse",
    })


# =============================================================================
# Main Entry Point
# =============================================================================

def create_app():
    """Create the HTTP app with authentication and logging middleware."""
    app = mcp.http_app(
        transport="sse",
    )

    # Add authentication middleware (runs second - after logging)
    app.add_middleware(BearerAuthMiddleware, api_key=API_KEY)

    # Add access logging middleware (runs first - logs all requests)
    app.add_middleware(HTTPAccessLogMiddleware)

    return app


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("ZakOps Agent Bridge - MCP Server (FastMCP)")
    print("=" * 60)
    print(f"Transport: SSE")
    print(f"Endpoint: http://{BRIDGE_HOST}:{BRIDGE_PORT}/sse")
    print(f"Health: http://{BRIDGE_HOST}:{BRIDGE_PORT}/health")
    print(f"Deal API: {DEAL_API_URL}")
    print(f"RAG API: {RAG_API_URL}")
    print(f"DataRoom: {DATAROOM_ROOT}")
    print(f"Auth: {'Enabled' if API_KEY else 'DISABLED (dev mode)'}")
    print("=" * 60)

    # Create app with auth middleware
    app = create_app()

    # Run with uvicorn
    uvicorn.run(app, host=BRIDGE_HOST, port=BRIDGE_PORT)
