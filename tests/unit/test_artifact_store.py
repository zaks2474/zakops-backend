"""
Tests for ArtifactStore abstraction layer.

Phase 4: Artifact Storage Tests
"""

import os
import tempfile
from io import BytesIO
from pathlib import Path

import pytest

from core.storage import (
    ArtifactStore,
    ArtifactMetadata,
    StorageBackend,
    LocalFilesystemArtifactStore,
    get_artifact_store,
)
from core.storage.factory import reset_default_store, get_dataroom_root


class TestLocalFilesystemArtifactStore:
    """Tests for LocalFilesystemArtifactStore."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, temp_dir):
        """Create a store with temp directory."""
        return LocalFilesystemArtifactStore(base_path=temp_dir)

    def test_backend_type(self, store):
        """Test that backend type is LOCAL."""
        assert store.backend_type == StorageBackend.LOCAL

    def test_put_bytes(self, store, temp_dir):
        """Test storing bytes content."""
        content = b"Hello, World!"
        key = "test/hello.txt"

        result = store.put(key, content, mime_type="text/plain")

        assert result.storage_key == "test/hello.txt"
        assert result.storage_backend == StorageBackend.LOCAL
        assert result.filename == "hello.txt"
        assert result.mime_type == "text/plain"
        assert result.size_bytes == len(content)
        assert result.sha256 is not None
        assert result.storage_uri.startswith("file://")

        # Verify file exists
        assert (temp_dir / "test" / "hello.txt").exists()

    def test_put_file_like(self, store):
        """Test storing from file-like object."""
        content = b"File-like content"
        key = "docs/readme.md"

        result = store.put(key, BytesIO(content), mime_type="text/markdown")

        assert result.storage_key == "docs/readme.md"
        assert result.size_bytes == len(content)

    def test_get(self, store):
        """Test retrieving content."""
        content = b"Test content for get"
        key = "retrieve/test.bin"

        store.put(key, content)
        retrieved = store.get(key)

        assert retrieved == content

    def test_get_not_found(self, store):
        """Test getting non-existent artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            store.get("nonexistent/file.txt")

    def test_get_stream(self, store):
        """Test streaming retrieval."""
        content = b"Streaming content"
        key = "stream/data.bin"

        store.put(key, content)

        with store.get_stream(key) as stream:
            retrieved = stream.read()

        assert retrieved == content

    def test_delete(self, store, temp_dir):
        """Test deleting an artifact."""
        content = b"To be deleted"
        key = "delete/me.txt"

        store.put(key, content)
        assert (temp_dir / "delete" / "me.txt").exists()

        result = store.delete(key)

        assert result is True
        assert not (temp_dir / "delete" / "me.txt").exists()

    def test_delete_not_found(self, store):
        """Test deleting non-existent artifact returns False."""
        result = store.delete("nonexistent/file.txt")
        assert result is False

    def test_exists(self, store):
        """Test existence check."""
        key = "exists/check.txt"

        assert store.exists(key) is False

        store.put(key, b"content")

        assert store.exists(key) is True

    def test_get_url(self, store):
        """Test URL generation."""
        content = b"URL test"
        key = "url/test.txt"

        store.put(key, content)
        url = store.get_url(key)

        assert url.startswith("file://")
        assert "url/test.txt" in url or "url" in url

    def test_get_url_not_found(self, store):
        """Test URL for non-existent artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            store.get_url("nonexistent/file.txt")

    def test_list(self, store):
        """Test listing artifacts."""
        # Create some test files
        store.put("deals/DEAL-001/doc1.pdf", b"doc1")
        store.put("deals/DEAL-001/doc2.pdf", b"doc2")
        store.put("deals/DEAL-002/doc3.pdf", b"doc3")

        # List all
        all_items = list(store.list("deals"))
        assert len(all_items) == 3

        # List specific deal
        deal_items = list(store.list("deals/DEAL-001"))
        assert len(deal_items) == 2

    def test_get_metadata(self, store):
        """Test metadata retrieval."""
        content = b"Metadata test content"
        key = "meta/test.txt"

        store.put(key, content, mime_type="text/plain")
        meta = store.get_metadata(key)

        assert meta.storage_key == "meta/test.txt"
        assert meta.filename == "test.txt"
        assert meta.size_bytes == len(content)
        assert meta.mime_type == "text/plain"

    def test_normalize_key(self, store):
        """Test key normalization."""
        assert store.normalize_key("/foo/bar/") == "foo/bar"
        assert store.normalize_key("foo//bar") == "foo/bar"
        assert store.normalize_key("foo\\bar") == "foo/bar"

    def test_build_key(self, store):
        """Test key building."""
        key = store.build_key("DEAL-001", "documents", "cim.pdf", "cim")
        assert key == "DEAL-001/documents/cim/cim.pdf"

        key = store.build_key("DEAL-001", "emails", "msg.eml")
        assert key == "DEAL-001/emails/msg.eml"

    def test_key_to_deal_id(self, store):
        """Test deal ID extraction from key."""
        assert store.key_to_deal_id("DEAL-001/documents/cim.pdf") == "DEAL-001"
        assert store.key_to_deal_id("single_file.txt") is None

    def test_copy_from_path(self, store, temp_dir):
        """Test copying from filesystem path."""
        # Create source file
        source = temp_dir / "source.txt"
        source.write_bytes(b"Source content")

        result = store.copy_from_path(source, "copied/file.txt")

        assert result.storage_key == "copied/file.txt"
        assert result.filename == "source.txt"
        assert store.exists("copied/file.txt")
        # Source should still exist
        assert source.exists()

    def test_move_to_key(self, store, temp_dir):
        """Test moving from filesystem path."""
        # Create source file
        source = temp_dir / "to_move.txt"
        source.write_bytes(b"Move content")

        result = store.move_to_key(source, "moved/file.txt")

        assert result.storage_key == "moved/file.txt"
        assert store.exists("moved/file.txt")
        # Source should be deleted
        assert not source.exists()

    def test_absolute_path_backward_compat(self, store, temp_dir):
        """Test backward compatibility with absolute paths."""
        # Create a file using absolute path
        abs_path = temp_dir / "absolute" / "test.txt"
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(b"Absolute path content")

        # Should be able to read using absolute path
        content = store.get(str(abs_path))
        assert content == b"Absolute path content"


class TestGetArtifactStore:
    """Tests for get_artifact_store factory function."""

    @pytest.fixture(autouse=True)
    def reset_store(self):
        """Reset default store before and after each test."""
        reset_default_store()
        yield
        reset_default_store()

    def test_default_local_store(self, monkeypatch, tmp_path):
        """Test that default store is local filesystem."""
        monkeypatch.setenv("DATAROOM_ROOT", str(tmp_path))

        store = get_artifact_store()

        assert store.backend_type == StorageBackend.LOCAL
        assert isinstance(store, LocalFilesystemArtifactStore)

    def test_explicit_local_store(self, monkeypatch, tmp_path):
        """Test explicitly requesting local store."""
        monkeypatch.setenv("DATAROOM_ROOT", str(tmp_path))

        store = get_artifact_store("local")

        assert store.backend_type == StorageBackend.LOCAL

    def test_singleton_behavior(self, monkeypatch, tmp_path):
        """Test that default store is a singleton."""
        monkeypatch.setenv("DATAROOM_ROOT", str(tmp_path))

        store1 = get_artifact_store()
        store2 = get_artifact_store()

        assert store1 is store2

    def test_force_new(self, monkeypatch, tmp_path):
        """Test force_new creates new instance."""
        monkeypatch.setenv("DATAROOM_ROOT", str(tmp_path))

        store1 = get_artifact_store()
        store2 = get_artifact_store(force_new=True)

        assert store1 is not store2

    def test_s3_requires_opt_in(self, monkeypatch, tmp_path):
        """Test that S3 storage requires explicit opt-in."""
        monkeypatch.setenv("DATAROOM_ROOT", str(tmp_path))
        monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "s3")
        # Don't set ALLOW_CLOUD_DEFAULT

        # Should fall back to local
        store = get_artifact_store()
        assert store.backend_type == StorageBackend.LOCAL

    def test_s3_with_explicit_config(self, monkeypatch, tmp_path):
        """Test S3 store with explicit configuration."""
        monkeypatch.setenv("DATAROOM_ROOT", str(tmp_path))

        # This should attempt to create S3 store (will fail without boto3/credentials)
        # But demonstrates the configuration path
        try:
            store = get_artifact_store("s3", bucket="test-bucket")
            assert store.backend_type == StorageBackend.S3
        except (ImportError, ValueError):
            # Expected if boto3 not installed or no credentials
            pass

    def test_unknown_backend_fallback(self, monkeypatch, tmp_path):
        """Test unknown backend falls back to local."""
        monkeypatch.setenv("DATAROOM_ROOT", str(tmp_path))

        store = get_artifact_store("unknown")

        assert store.backend_type == StorageBackend.LOCAL


class TestArtifactMetadata:
    """Tests for ArtifactMetadata dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from datetime import datetime, timezone

        meta = ArtifactMetadata(
            storage_key="test/file.txt",
            storage_uri="file:///path/to/file.txt",
            storage_backend=StorageBackend.LOCAL,
            filename="file.txt",
            mime_type="text/plain",
            size_bytes=100,
            sha256="abc123",
            created_at=datetime(2026, 1, 19, 12, 0, 0, tzinfo=timezone.utc),
            metadata={"custom": "value"},
        )

        d = meta.to_dict()

        assert d["storage_key"] == "test/file.txt"
        assert d["storage_backend"] == "local"
        assert d["filename"] == "file.txt"
        assert d["sha256"] == "abc123"
        assert d["metadata"] == {"custom": "value"}


class TestStorageBackend:
    """Tests for StorageBackend enum."""

    def test_values(self):
        """Test enum values."""
        assert StorageBackend.LOCAL.value == "local"
        assert StorageBackend.S3.value == "s3"

    def test_string_conversion(self):
        """Test string conversion."""
        assert str(StorageBackend.LOCAL) == "StorageBackend.LOCAL"
        assert StorageBackend.LOCAL == "local"  # Due to str inheritance
