"""
Local Filesystem Artifact Store

Phase 4: Artifact Storage Abstraction
Spec Reference: Storage Abstraction section

Default storage backend using the local filesystem.
Maintains backward compatibility with existing DataRoom paths.
"""

from __future__ import annotations

import mimetypes
import os
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Union

from .base import ArtifactMetadata, ArtifactStore, StorageBackend


class LocalFilesystemArtifactStore(ArtifactStore):
    """
    Local filesystem implementation of ArtifactStore.

    This is the DEFAULT backend for ZakOps. It stores artifacts on the
    local filesystem with the following structure:

        {base_path}/
        └── {deal_id}/
            ├── emails/
            ├── documents/
            ├── generated/
            └── extracted/

    Environment Variables:
        DATAROOM_ROOT: Base path for storage (default: /home/zaks/DataRoom)
        ARTIFACT_STORAGE_PATH: Alternative base path

    Backward Compatibility:
        - Existing DataRoom paths continue to work
        - Legacy paths can be read via get() using absolute paths
        - New artifacts use the structured key convention
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        *,
        create_dirs: bool = True,
    ):
        """
        Initialize local filesystem store.

        Args:
            base_path: Base directory for storage. If not provided, uses
                       DATAROOM_ROOT or ARTIFACT_STORAGE_PATH env vars.
            create_dirs: If True, create base directory if it doesn't exist.
        """
        if base_path is None:
            base_path = os.getenv(
                "DATAROOM_ROOT",
                os.getenv("ARTIFACT_STORAGE_PATH", "/home/zaks/DataRoom"),
            )

        self._base_path = Path(base_path).resolve()

        if create_dirs:
            self._base_path.mkdir(parents=True, exist_ok=True)

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.LOCAL

    @property
    def base_path(self) -> Path:
        """Get the base storage path."""
        return self._base_path

    def _resolve_path(self, key: str) -> Path:
        """
        Resolve a storage key to an absolute path.

        Handles both:
        - Relative keys: "DEAL-001/documents/cim.pdf" -> {base_path}/DEAL-001/documents/cim.pdf
        - Absolute paths (legacy): "/home/zaks/DataRoom/..." -> as-is
        """
        # Handle absolute paths for backward compatibility
        if key.startswith("/"):
            return Path(key)

        normalized = self.normalize_key(key)
        return self._base_path / normalized

    def _path_to_key(self, path: Path) -> str:
        """Convert an absolute path back to a storage key."""
        try:
            return str(path.relative_to(self._base_path))
        except ValueError:
            # Path is outside base_path, return as-is
            return str(path)

    def _path_to_uri(self, path: Path) -> str:
        """Convert a path to a file:// URI."""
        return f"file://{path.resolve()}"

    def put(
        self,
        key: str,
        data: Union[bytes, BinaryIO],
        *,
        filename: Optional[str] = None,
        mime_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> ArtifactMetadata:
        """Store an artifact on the local filesystem."""
        path = self._resolve_path(key)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get the data as bytes
        if isinstance(data, bytes):
            content = data
        else:
            # Read from file-like object
            if hasattr(data, "seek"):
                data.seek(0)
            content = data.read()

        # Compute hash
        sha256 = self.compute_sha256(content)

        # Write file
        with open(path, "wb") as f:
            f.write(content)

        # Get file stats
        stat = path.stat()

        # Determine filename
        if filename is None:
            filename = path.name

        # Try to guess mime type if not provided
        if mime_type == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(filename)
            if guessed:
                mime_type = guessed

        return ArtifactMetadata(
            storage_key=self.normalize_key(key),
            storage_uri=self._path_to_uri(path),
            storage_backend=StorageBackend.LOCAL,
            filename=filename,
            mime_type=mime_type,
            size_bytes=stat.st_size,
            sha256=sha256,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            metadata=metadata or {},
        )

    def get(self, key: str) -> bytes:
        """Retrieve artifact content."""
        path = self._resolve_path(key)

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")

        with open(path, "rb") as f:
            return f.read()

    def get_stream(self, key: str) -> BinaryIO:
        """Get a streaming handle to artifact content."""
        path = self._resolve_path(key)

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")

        return open(path, "rb")

    def delete(self, key: str) -> bool:
        """Delete an artifact."""
        path = self._resolve_path(key)

        if not path.exists():
            return False

        path.unlink()
        return True

    def exists(self, key: str) -> bool:
        """Check if an artifact exists."""
        path = self._resolve_path(key)
        return path.exists() and path.is_file()

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
        Note: expires_in and for_download are ignored for local storage.
        """
        path = self._resolve_path(key)

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")

        return self._path_to_uri(path)

    def list(
        self,
        prefix: str = "",
        *,
        recursive: bool = True,
        max_results: int = 1000,
    ) -> Iterator[ArtifactMetadata]:
        """List artifacts by prefix."""
        if prefix.startswith("/"):
            # Absolute path prefix
            search_path = Path(prefix)
        else:
            normalized_prefix = self.normalize_key(prefix) if prefix else ""
            search_path = self._base_path / normalized_prefix if normalized_prefix else self._base_path

        if not search_path.exists():
            return

        count = 0

        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        # Handle case where search_path is a file
        if search_path.is_file():
            if count < max_results:
                yield self.get_metadata(self._path_to_key(search_path))
            return

        for item in search_path.glob(pattern):
            if count >= max_results:
                break

            if item.is_file():
                try:
                    yield self.get_metadata(self._path_to_key(item))
                    count += 1
                except Exception:
                    # Skip files we can't read
                    continue

    def get_metadata(self, key: str) -> ArtifactMetadata:
        """Get metadata for an artifact without downloading content."""
        path = self._resolve_path(key)

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {key}")

        stat = path.stat()

        # Guess mime type
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type is None:
            mime_type = "application/octet-stream"

        # For metadata, we don't compute sha256 by default (expensive)
        return ArtifactMetadata(
            storage_key=self.normalize_key(key),
            storage_uri=self._path_to_uri(path),
            storage_backend=StorageBackend.LOCAL,
            filename=path.name,
            mime_type=mime_type,
            size_bytes=stat.st_size,
            sha256=None,  # Not computed for metadata-only calls
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            metadata={},
        )

    # =========================================================================
    # Local-specific helper methods
    # =========================================================================

    def get_absolute_path(self, key: str) -> Path:
        """
        Get the absolute filesystem path for a key.

        This is useful for backward compatibility with code that
        expects direct filesystem paths.
        """
        return self._resolve_path(key)

    def copy_from_path(
        self,
        source_path: Union[str, Path],
        key: str,
        *,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ArtifactMetadata:
        """
        Copy a file from the filesystem into storage.

        This is more efficient than reading the file and calling put()
        for large files.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        dest = self._resolve_path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(source, dest)

        # Compute hash
        with open(dest, "rb") as f:
            sha256 = self.compute_sha256(f)

        stat = dest.stat()

        # Determine mime type
        if mime_type is None:
            guessed, _ = mimetypes.guess_type(source.name)
            mime_type = guessed or "application/octet-stream"

        return ArtifactMetadata(
            storage_key=self.normalize_key(key),
            storage_uri=self._path_to_uri(dest),
            storage_backend=StorageBackend.LOCAL,
            filename=source.name,
            mime_type=mime_type,
            size_bytes=stat.st_size,
            sha256=sha256,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            metadata=metadata or {},
        )

    def move_to_key(
        self,
        source_path: Union[str, Path],
        key: str,
        *,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ArtifactMetadata:
        """
        Move a file from the filesystem into storage.

        Similar to copy_from_path but removes the source file.
        """
        result = self.copy_from_path(
            source_path, key, mime_type=mime_type, metadata=metadata
        )
        Path(source_path).unlink()
        return result
