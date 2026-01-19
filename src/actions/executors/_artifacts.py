"""
Artifact Utilities for Action Executors

Phase 4: Updated to use ArtifactStore abstraction while maintaining
backward compatibility with existing code.

This module provides helper functions for executors to work with artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union

from actions.engine.models import ActionError
from actions.executors.base import ActionExecutionError, ExecutionContext

# Import storage abstraction
from core.storage import (
    ArtifactStore,
    ArtifactMetadata,
    LocalFilesystemArtifactStore,
    get_artifact_store,
)


def _dataroom_root() -> Path:
    """
    Get the DataRoom root path.

    Backward compatible function - prefer using get_artifact_store() for new code.
    """
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom"))


def resolve_action_artifact_dir(ctx: ExecutionContext) -> Path:
    """
    Return the directory where this action should write artifacts:
    {deal.folder_path}/99-ACTIONS/{action_id}/

    Backward compatible function - prefer using get_action_artifact_store() for new code.
    """
    deal = ctx.deal or {}
    folder_path = (deal.get("folder_path") or "").strip()
    if not folder_path:
        raise ActionExecutionError(
            ActionError(
                code="deal_folder_path_missing",
                message="Deal folder_path missing; cannot determine artifact destination",
                category="validation",
                retryable=False,
            )
        )

    action_id = ctx.action.action_id
    base = (_dataroom_root() / folder_path / "99-ACTIONS" / action_id).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


# ============================================================================
# New ArtifactStore-based helpers (Phase 4)
# ============================================================================


def get_action_artifact_store() -> ArtifactStore:
    """
    Get the ArtifactStore instance for action artifacts.

    Returns the configured storage backend (local by default).
    """
    return get_artifact_store()


def get_action_storage_key(ctx: ExecutionContext, filename: str) -> str:
    """
    Build a storage key for an action artifact.

    Args:
        ctx: Execution context
        filename: Artifact filename

    Returns:
        Storage key following the spec convention:
        {deal_folder_path}/99-ACTIONS/{action_id}/{filename}
    """
    deal = ctx.deal or {}
    folder_path = (deal.get("folder_path") or "").strip()
    if not folder_path:
        raise ActionExecutionError(
            ActionError(
                code="deal_folder_path_missing",
                message="Deal folder_path missing; cannot determine artifact destination",
                category="validation",
                retryable=False,
            )
        )

    action_id = ctx.action.action_id
    return f"{folder_path}/99-ACTIONS/{action_id}/{filename}"


def store_action_artifact(
    ctx: ExecutionContext,
    filename: str,
    data: Union[bytes, BinaryIO],
    *,
    mime_type: str = "application/octet-stream",
    metadata: Optional[Dict[str, str]] = None,
) -> ArtifactMetadata:
    """
    Store an artifact for the current action using the ArtifactStore.

    This is the preferred method for new code. It uses the configured
    storage backend and returns proper metadata.

    Args:
        ctx: Execution context
        filename: Artifact filename
        data: Binary content or file-like object
        mime_type: MIME type of the content
        metadata: Additional metadata

    Returns:
        ArtifactMetadata with storage location and file info
    """
    store = get_action_artifact_store()
    key = get_action_storage_key(ctx, filename)

    return store.put(
        key,
        data,
        filename=filename,
        mime_type=mime_type,
        metadata=metadata,
    )


def get_deal_storage_key(deal_id: str, category: str, filename: str, subcategory: Optional[str] = None) -> str:
    """
    Build a storage key for a deal artifact following the spec convention.

    Args:
        deal_id: Deal identifier
        category: Category (emails, documents, generated, extracted)
        filename: Artifact filename
        subcategory: Optional subcategory (e.g., "cim", "teasers")

    Returns:
        Storage key: {deal_id}/{category}/[{subcategory}/]{filename}
    """
    store = get_action_artifact_store()
    return store.build_key(deal_id, category, filename, subcategory)


def store_deal_artifact(
    deal_id: str,
    category: str,
    filename: str,
    data: Union[bytes, BinaryIO],
    *,
    subcategory: Optional[str] = None,
    mime_type: str = "application/octet-stream",
    metadata: Optional[Dict[str, str]] = None,
) -> ArtifactMetadata:
    """
    Store an artifact for a deal using the standard key convention.

    Args:
        deal_id: Deal identifier
        category: Category (emails, documents, generated, extracted)
        filename: Artifact filename
        data: Binary content or file-like object
        subcategory: Optional subcategory
        mime_type: MIME type
        metadata: Additional metadata

    Returns:
        ArtifactMetadata
    """
    store = get_action_artifact_store()
    key = get_deal_storage_key(deal_id, category, filename, subcategory)

    return store.put(
        key,
        data,
        filename=filename,
        mime_type=mime_type,
        metadata=metadata,
    )


def get_artifact(key: str) -> bytes:
    """
    Retrieve artifact content by storage key.

    Args:
        key: Storage key or absolute path (for backward compatibility)

    Returns:
        Binary content
    """
    store = get_action_artifact_store()
    return store.get(key)


def artifact_exists(key: str) -> bool:
    """
    Check if an artifact exists.

    Args:
        key: Storage key or absolute path

    Returns:
        True if exists
    """
    store = get_action_artifact_store()
    return store.exists(key)


def get_artifact_url(key: str, *, expires_in: int = 3600) -> str:
    """
    Get a URL for an artifact.

    For local storage, returns a file:// URI.
    For S3, returns a presigned URL.

    Args:
        key: Storage key
        expires_in: URL expiration in seconds

    Returns:
        URL string
    """
    store = get_action_artifact_store()
    return store.get_url(key, expires_in=expires_in)
