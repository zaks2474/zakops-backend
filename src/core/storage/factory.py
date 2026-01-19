"""
ArtifactStore Factory

Phase 4: Artifact Storage Abstraction
Spec Reference: Storage Abstraction section

Factory function for creating ArtifactStore instances based on configuration.

CRITICAL RULES:
1. Local filesystem is the DEFAULT backend
2. S3/cloud storage is OPT-IN only
3. Existing DataRoom paths must continue to work
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from .base import ArtifactStore, StorageBackend
from .local import LocalFilesystemArtifactStore

logger = logging.getLogger(__name__)

# Global singleton instance
_default_store: Optional[ArtifactStore] = None


def get_artifact_store(
    backend: Optional[str] = None,
    *,
    force_new: bool = False,
    **kwargs,
) -> ArtifactStore:
    """
    Get an ArtifactStore instance.

    This is the primary entry point for obtaining storage access.
    By default, returns a singleton LocalFilesystemArtifactStore.

    Args:
        backend: Storage backend to use. Options:
                 - "local" (default): Local filesystem
                 - "s3": S3-compatible storage (requires boto3)
                 If not provided, uses ARTIFACT_STORAGE_BACKEND env var,
                 defaulting to "local".
        force_new: If True, create a new instance instead of returning singleton.
        **kwargs: Backend-specific configuration options.

    Returns:
        ArtifactStore instance

    Environment Variables:
        ARTIFACT_STORAGE_BACKEND: Default backend ("local" or "s3")
        ALLOW_CLOUD_DEFAULT: Must be "true" to allow cloud storage as default

    Examples:
        # Get default store (local filesystem)
        store = get_artifact_store()

        # Explicitly request local storage
        store = get_artifact_store("local")

        # Request S3 storage (must be opt-in)
        store = get_artifact_store("s3", bucket="my-bucket")

        # Get a fresh instance
        store = get_artifact_store(force_new=True)
    """
    global _default_store

    # Determine backend
    if backend is None:
        backend = os.getenv("ARTIFACT_STORAGE_BACKEND", "local").lower()

    # Validate backend
    backend = backend.lower()
    if backend not in ("local", "s3"):
        logger.warning(
            f"Unknown storage backend '{backend}', falling back to 'local'"
        )
        backend = "local"

    # Safety check: cloud storage must be opt-in
    if backend != "local":
        allow_cloud = os.getenv("ALLOW_CLOUD_DEFAULT", "false").lower() == "true"
        if not allow_cloud and not kwargs:
            logger.warning(
                f"Cloud storage backend '{backend}' requested but ALLOW_CLOUD_DEFAULT "
                "is not 'true'. Set ALLOW_CLOUD_DEFAULT=true to enable cloud storage "
                "as default, or pass explicit configuration. Falling back to 'local'."
            )
            backend = "local"

    # Return singleton for default local store
    if backend == "local" and not force_new and not kwargs:
        if _default_store is None:
            _default_store = _create_local_store()
        return _default_store

    # Create new instance
    if backend == "local":
        return _create_local_store(**kwargs)
    elif backend == "s3":
        return _create_s3_store(**kwargs)
    else:
        # Fallback (shouldn't reach here due to validation above)
        return _create_local_store(**kwargs)


def _create_local_store(**kwargs) -> LocalFilesystemArtifactStore:
    """Create a LocalFilesystemArtifactStore instance."""
    base_path = kwargs.get("base_path")
    create_dirs = kwargs.get("create_dirs", True)

    store = LocalFilesystemArtifactStore(
        base_path=base_path,
        create_dirs=create_dirs,
    )

    logger.info(f"Created LocalFilesystemArtifactStore at {store.base_path}")
    return store


def _create_s3_store(**kwargs) -> ArtifactStore:
    """Create an S3ArtifactStore instance."""
    # Import here to avoid requiring boto3 when not using S3
    from .s3 import S3ArtifactStore

    bucket = kwargs.get("bucket")
    prefix = kwargs.get("prefix", "")
    endpoint_url = kwargs.get("endpoint_url")
    region = kwargs.get("region")

    store = S3ArtifactStore(
        bucket=bucket,
        prefix=prefix,
        endpoint_url=endpoint_url,
        region=region,
    )

    logger.info(f"Created S3ArtifactStore for bucket {store.bucket}")
    return store


def reset_default_store() -> None:
    """
    Reset the default store singleton.

    Useful for testing or when configuration changes.
    """
    global _default_store
    _default_store = None


# ============================================================================
# Convenience functions
# ============================================================================


def get_dataroom_root() -> str:
    """
    Get the DataRoom root path.

    This is a convenience function for backward compatibility with code
    that needs the raw filesystem path.

    Returns:
        Absolute path to DataRoom root
    """
    return os.getenv(
        "DATAROOM_ROOT",
        os.getenv("ARTIFACT_STORAGE_PATH", "/home/zaks/DataRoom"),
    )


def storage_backend_from_uri(uri: str) -> StorageBackend:
    """
    Determine storage backend from a URI.

    Args:
        uri: Storage URI (e.g., "file:///path" or "s3://bucket/key")

    Returns:
        StorageBackend enum value
    """
    if uri.startswith("file://"):
        return StorageBackend.LOCAL
    elif uri.startswith("s3://"):
        return StorageBackend.S3
    elif uri.startswith("/"):
        # Assume local filesystem path
        return StorageBackend.LOCAL
    else:
        # Default to local
        return StorageBackend.LOCAL


def get_store_for_uri(uri: str) -> ArtifactStore:
    """
    Get an appropriate ArtifactStore for a given URI.

    Args:
        uri: Storage URI

    Returns:
        ArtifactStore instance capable of handling the URI
    """
    backend = storage_backend_from_uri(uri)
    return get_artifact_store(backend.value)
