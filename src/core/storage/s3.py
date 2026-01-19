"""
S3 Artifact Store

Phase 4: Artifact Storage Abstraction
Spec Reference: Storage Abstraction section

Optional S3-compatible storage backend for cloud deployments.
Supports AWS S3, MinIO, and other S3-compatible services.

IMPORTANT: This backend is OPT-IN only. Local filesystem is the default.
"""

from __future__ import annotations

import mimetypes
import os
from datetime import datetime, timezone
from io import BytesIO
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterator, Optional, Union

from .base import ArtifactMetadata, ArtifactStore, StorageBackend

# Type hints for boto3 without requiring it at import time
if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class S3ArtifactStore(ArtifactStore):
    """
    S3-compatible storage implementation of ArtifactStore.

    This backend is OPTIONAL and OPT-IN only. It requires:
    1. boto3 package installed: pip install boto3
    2. ARTIFACT_STORAGE_BACKEND=s3 environment variable
    3. AWS credentials or S3-compatible endpoint configured

    Environment Variables:
        ARTIFACT_STORAGE_BACKEND: Must be "s3" to use this backend
        AWS_S3_BUCKET: S3 bucket name (required)
        AWS_S3_PREFIX: Optional key prefix for all artifacts
        AWS_S3_ENDPOINT_URL: Custom endpoint for MinIO/localstack
        AWS_ACCESS_KEY_ID: AWS access key
        AWS_SECRET_ACCESS_KEY: AWS secret key
        AWS_REGION: AWS region (default: us-east-1)

    Storage URI Format:
        s3://{bucket}/{key}
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        *,
        prefix: str = "",
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
        client: Optional["S3Client"] = None,
    ):
        """
        Initialize S3 storage.

        Args:
            bucket: S3 bucket name. If not provided, uses AWS_S3_BUCKET env var.
            prefix: Key prefix for all artifacts (e.g., "artifacts/").
            endpoint_url: Custom endpoint for S3-compatible services.
            region: AWS region.
            client: Pre-configured boto3 S3 client (for testing).
        """
        self._bucket = bucket or os.getenv("AWS_S3_BUCKET")
        if not self._bucket:
            raise ValueError(
                "S3 bucket name required. Set AWS_S3_BUCKET environment variable "
                "or pass bucket parameter."
            )

        self._prefix = prefix.strip("/")
        if self._prefix:
            self._prefix += "/"

        self._endpoint_url = endpoint_url or os.getenv("AWS_S3_ENDPOINT_URL")
        self._region = region or os.getenv("AWS_REGION", "us-east-1")

        if client:
            self._client = client
        else:
            self._client = self._create_client()

    def _create_client(self) -> "S3Client":
        """Create a boto3 S3 client."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage backend. "
                "Install with: pip install boto3"
            )

        client_kwargs = {"region_name": self._region}

        if self._endpoint_url:
            client_kwargs["endpoint_url"] = self._endpoint_url

        return boto3.client("s3", **client_kwargs)

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.S3

    @property
    def bucket(self) -> str:
        """Get the S3 bucket name."""
        return self._bucket

    def _resolve_key(self, key: str) -> str:
        """Resolve a storage key to a full S3 key with prefix."""
        normalized = self.normalize_key(key)
        return f"{self._prefix}{normalized}"

    def _strip_prefix(self, s3_key: str) -> str:
        """Strip the prefix from an S3 key to get storage key."""
        if self._prefix and s3_key.startswith(self._prefix):
            return s3_key[len(self._prefix):]
        return s3_key

    def _key_to_uri(self, key: str) -> str:
        """Convert a storage key to an S3 URI."""
        s3_key = self._resolve_key(key)
        return f"s3://{self._bucket}/{s3_key}"

    def put(
        self,
        key: str,
        data: Union[bytes, BinaryIO],
        *,
        filename: Optional[str] = None,
        mime_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> ArtifactMetadata:
        """Store an artifact in S3."""
        s3_key = self._resolve_key(key)

        # Get data as bytes for hashing
        if isinstance(data, bytes):
            content = data
            body = content
        else:
            if hasattr(data, "seek"):
                data.seek(0)
            content = data.read()
            body = content

        # Compute hash
        sha256 = self.compute_sha256(content)

        # Determine filename
        if filename is None:
            filename = key.split("/")[-1]

        # Try to guess mime type if not provided
        if mime_type == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(filename)
            if guessed:
                mime_type = guessed

        # Prepare S3 metadata
        s3_metadata = metadata or {}
        s3_metadata["sha256"] = sha256
        s3_metadata["original-filename"] = filename

        # Upload to S3
        self._client.put_object(
            Bucket=self._bucket,
            Key=s3_key,
            Body=body,
            ContentType=mime_type,
            Metadata=s3_metadata,
        )

        now = datetime.now(tz=timezone.utc)

        return ArtifactMetadata(
            storage_key=self.normalize_key(key),
            storage_uri=self._key_to_uri(key),
            storage_backend=StorageBackend.S3,
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(content),
            sha256=sha256,
            created_at=now,
            modified_at=now,
            metadata=metadata or {},
        )

    def get(self, key: str) -> bytes:
        """Retrieve artifact content from S3."""
        s3_key = self._resolve_key(key)

        try:
            response = self._client.get_object(Bucket=self._bucket, Key=s3_key)
            return response["Body"].read()
        except self._client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Artifact not found: {key}")
        except Exception as e:
            if "NoSuchKey" in str(e) or "Not Found" in str(e):
                raise FileNotFoundError(f"Artifact not found: {key}")
            raise

    def get_stream(self, key: str) -> BinaryIO:
        """Get a streaming handle to artifact content from S3."""
        s3_key = self._resolve_key(key)

        try:
            response = self._client.get_object(Bucket=self._bucket, Key=s3_key)
            return response["Body"]
        except self._client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Artifact not found: {key}")
        except Exception as e:
            if "NoSuchKey" in str(e) or "Not Found" in str(e):
                raise FileNotFoundError(f"Artifact not found: {key}")
            raise

    def delete(self, key: str) -> bool:
        """Delete an artifact from S3."""
        s3_key = self._resolve_key(key)

        # Check if exists first
        if not self.exists(key):
            return False

        self._client.delete_object(Bucket=self._bucket, Key=s3_key)
        return True

    def exists(self, key: str) -> bool:
        """Check if an artifact exists in S3."""
        s3_key = self._resolve_key(key)

        try:
            self._client.head_object(Bucket=self._bucket, Key=s3_key)
            return True
        except Exception:
            return False

    def get_url(
        self,
        key: str,
        *,
        expires_in: int = 3600,
        for_download: bool = False,
    ) -> str:
        """
        Get a presigned URL for the artifact.

        Args:
            key: Storage key
            expires_in: URL expiration in seconds
            for_download: If True, set Content-Disposition for download

        Returns:
            Presigned URL
        """
        s3_key = self._resolve_key(key)

        if not self.exists(key):
            raise FileNotFoundError(f"Artifact not found: {key}")

        params = {
            "Bucket": self._bucket,
            "Key": s3_key,
        }

        if for_download:
            filename = key.split("/")[-1]
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'

        return self._client.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=expires_in,
        )

    def list(
        self,
        prefix: str = "",
        *,
        recursive: bool = True,
        max_results: int = 1000,
    ) -> Iterator[ArtifactMetadata]:
        """List artifacts by prefix in S3."""
        search_prefix = self._resolve_key(prefix) if prefix else self._prefix

        paginator = self._client.get_paginator("list_objects_v2")

        list_kwargs = {
            "Bucket": self._bucket,
            "Prefix": search_prefix,
            "MaxKeys": min(max_results, 1000),
        }

        if not recursive:
            list_kwargs["Delimiter"] = "/"

        count = 0
        for page in paginator.paginate(**list_kwargs):
            if "Contents" not in page:
                break

            for obj in page["Contents"]:
                if count >= max_results:
                    return

                s3_key = obj["Key"]
                storage_key = self._strip_prefix(s3_key)

                # Skip "directory" markers
                if s3_key.endswith("/"):
                    continue

                # Get mime type
                mime_type, _ = mimetypes.guess_type(s3_key)
                if mime_type is None:
                    mime_type = "application/octet-stream"

                yield ArtifactMetadata(
                    storage_key=storage_key,
                    storage_uri=f"s3://{self._bucket}/{s3_key}",
                    storage_backend=StorageBackend.S3,
                    filename=s3_key.split("/")[-1],
                    mime_type=mime_type,
                    size_bytes=obj["Size"],
                    sha256=None,  # Not available in list response
                    created_at=None,  # Not available in list response
                    modified_at=obj["LastModified"],
                    metadata={},
                )
                count += 1

    def get_metadata(self, key: str) -> ArtifactMetadata:
        """Get metadata for an artifact without downloading content."""
        s3_key = self._resolve_key(key)

        try:
            response = self._client.head_object(Bucket=self._bucket, Key=s3_key)
        except Exception as e:
            if "NoSuchKey" in str(e) or "Not Found" in str(e) or "404" in str(e):
                raise FileNotFoundError(f"Artifact not found: {key}")
            raise

        # Get custom metadata
        s3_metadata = response.get("Metadata", {})

        # Get mime type
        mime_type = response.get("ContentType", "application/octet-stream")

        # Get original filename from metadata or key
        filename = s3_metadata.get("original-filename", key.split("/")[-1])

        return ArtifactMetadata(
            storage_key=self.normalize_key(key),
            storage_uri=self._key_to_uri(key),
            storage_backend=StorageBackend.S3,
            filename=filename,
            mime_type=mime_type,
            size_bytes=response["ContentLength"],
            sha256=s3_metadata.get("sha256"),
            created_at=None,  # S3 doesn't track creation time
            modified_at=response.get("LastModified"),
            metadata={k: v for k, v in s3_metadata.items() if k not in ("sha256", "original-filename")},
        )
