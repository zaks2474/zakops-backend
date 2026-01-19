"""
ArtifactStore Abstract Base Class

Phase 4: Artifact Storage Abstraction
Spec Reference: Storage Abstraction section

Defines the interface for artifact storage backends.
All storage implementations must inherit from ArtifactStore.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Optional, Union


class StorageBackend(str, Enum):
    """Supported storage backend types."""

    LOCAL = "local"
    S3 = "s3"
    # Future backends can be added here
    # GCS = "gcs"
    # AZURE = "azure"


@dataclass
class ArtifactMetadata:
    """
    Metadata for a stored artifact.

    This is the unified metadata structure returned by all storage backends.
    """

    # Storage location
    storage_key: str  # Relative key within storage (e.g., "DEAL-001/documents/cim.pdf")
    storage_uri: str  # Full URI (e.g., "file:///path/to/file" or "s3://bucket/key")
    storage_backend: StorageBackend

    # File info
    filename: str
    mime_type: str
    size_bytes: int
    sha256: Optional[str] = None

    # Timestamps
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None

    # Additional metadata
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "storage_key": self.storage_key,
            "storage_uri": self.storage_uri,
            "storage_backend": self.storage_backend.value,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "metadata": self.metadata,
        }


class ArtifactStore(ABC):
    """
    Abstract base class for artifact storage backends.

    All storage implementations must provide these methods:
    - put: Store an artifact
    - get: Retrieve artifact content
    - delete: Remove an artifact
    - exists: Check if artifact exists
    - get_url: Get a (signed) URL for the artifact
    - list: List artifacts by prefix

    Storage Key Convention:
        {deal_id}/
        ├── emails/
        │   └── {message_id}.eml
        ├── documents/
        │   ├── cim/
        │   ├── teasers/
        │   └── financials/
        ├── generated/
        │   ├── loi/
        │   └── responses/
        └── extracted/

    Storage URI Format:
        - Local: file://{absolute_path}
        - S3: s3://{bucket}/{key}
    """

    @property
    @abstractmethod
    def backend_type(self) -> StorageBackend:
        """Return the storage backend type."""
        ...

    @abstractmethod
    def put(
        self,
        key: str,
        data: Union[bytes, BinaryIO],
        *,
        filename: Optional[str] = None,
        mime_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> ArtifactMetadata:
        """
        Store an artifact.

        Args:
            key: Storage key (e.g., "DEAL-001/documents/cim.pdf")
            data: Binary content or file-like object
            filename: Original filename (defaults to key basename)
            mime_type: MIME type of the content
            metadata: Additional metadata to store

        Returns:
            ArtifactMetadata with storage location and file info
        """
        ...

    @abstractmethod
    def get(self, key: str) -> bytes:
        """
        Retrieve artifact content.

        Args:
            key: Storage key

        Returns:
            Binary content of the artifact

        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        ...

    @abstractmethod
    def get_stream(self, key: str) -> BinaryIO:
        """
        Get a streaming handle to artifact content.

        Args:
            key: Storage key

        Returns:
            File-like object for streaming reads

        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete an artifact.

        Args:
            key: Storage key

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if an artifact exists.

        Args:
            key: Storage key

        Returns:
            True if exists, False otherwise
        """
        ...

    @abstractmethod
    def get_url(
        self,
        key: str,
        *,
        expires_in: int = 3600,
        for_download: bool = False,
    ) -> str:
        """
        Get a URL for the artifact.

        For local storage, returns a file:// URI.
        For S3, returns a presigned URL.

        Args:
            key: Storage key
            expires_in: URL expiration in seconds (for presigned URLs)
            for_download: If True, set Content-Disposition for download

        Returns:
            URL string

        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        ...

    @abstractmethod
    def list(
        self,
        prefix: str = "",
        *,
        recursive: bool = True,
        max_results: int = 1000,
    ) -> Iterator[ArtifactMetadata]:
        """
        List artifacts by prefix.

        Args:
            prefix: Key prefix to filter by (e.g., "DEAL-001/documents/")
            recursive: Include items in subdirectories
            max_results: Maximum number of results to return

        Yields:
            ArtifactMetadata for each matching artifact
        """
        ...

    @abstractmethod
    def get_metadata(self, key: str) -> ArtifactMetadata:
        """
        Get metadata for an artifact without downloading content.

        Args:
            key: Storage key

        Returns:
            ArtifactMetadata

        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        ...

    # =========================================================================
    # Utility Methods (non-abstract, common to all backends)
    # =========================================================================

    def compute_sha256(self, data: Union[bytes, BinaryIO]) -> str:
        """Compute SHA256 hash of data."""
        hasher = hashlib.sha256()
        if isinstance(data, bytes):
            hasher.update(data)
        else:
            # Reset to beginning if possible
            if hasattr(data, "seek"):
                data.seek(0)
            for chunk in iter(lambda: data.read(8192), b""):
                hasher.update(chunk)
            if hasattr(data, "seek"):
                data.seek(0)
        return hasher.hexdigest()

    def normalize_key(self, key: str) -> str:
        """
        Normalize a storage key.

        - Removes leading/trailing slashes
        - Collapses multiple slashes
        - Converts backslashes to forward slashes
        """
        # Convert backslashes
        key = key.replace("\\", "/")
        # Remove leading/trailing slashes
        key = key.strip("/")
        # Collapse multiple slashes
        while "//" in key:
            key = key.replace("//", "/")
        return key

    def key_to_deal_id(self, key: str) -> Optional[str]:
        """
        Extract deal_id from a storage key.

        Keys are expected to start with {deal_id}/ prefix.
        Returns None if key doesn't follow convention.
        """
        normalized = self.normalize_key(key)
        if "/" in normalized:
            return normalized.split("/")[0]
        return None

    def build_key(
        self,
        deal_id: str,
        category: str,
        filename: str,
        subcategory: Optional[str] = None,
    ) -> str:
        """
        Build a storage key following the spec convention.

        Args:
            deal_id: Deal identifier (e.g., "DEAL-2026-001")
            category: Top-level category (emails, documents, generated, extracted)
            filename: File name
            subcategory: Optional subcategory (e.g., "cim", "teasers")

        Returns:
            Normalized storage key
        """
        parts = [deal_id, category]
        if subcategory:
            parts.append(subcategory)
        parts.append(filename)
        return self.normalize_key("/".join(parts))
