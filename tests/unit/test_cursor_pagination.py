"""
Tests for cursor-based pagination.
"""

import pytest
from src.api.orchestration.routers.search import PaginationCursor


class TestPaginationCursor:
    """Test cursor encoding/decoding."""

    def test_encode_decode_roundtrip(self):
        """Cursor should survive encode/decode roundtrip."""
        original = PaginationCursor(
            last_id="550e8400-e29b-41d4-a716-446655440000",
            last_timestamp="2026-01-19T12:00:00"
        )

        encoded = original.encode()
        decoded = PaginationCursor.decode(encoded)

        assert decoded.last_id == original.last_id
        assert decoded.last_timestamp == original.last_timestamp

    def test_decode_invalid_cursor(self):
        """Invalid cursor should raise ValueError."""
        with pytest.raises(ValueError):
            PaginationCursor.decode("invalid-cursor")

    def test_decode_malformed_base64(self):
        """Malformed base64 should raise ValueError."""
        with pytest.raises(ValueError):
            PaginationCursor.decode("!!!not-base64!!!")

    def test_decode_empty_cursor(self):
        """Empty cursor should raise ValueError."""
        with pytest.raises(ValueError):
            PaginationCursor.decode("")

    def test_cursor_is_url_safe(self):
        """Encoded cursor should be URL-safe."""
        cursor = PaginationCursor(
            last_id="550e8400-e29b-41d4-a716-446655440000",
            last_timestamp="2026-01-19T12:00:00+00:00"
        )

        encoded = cursor.encode()

        # Should not contain URL-unsafe characters
        assert "+" not in encoded
        assert " " not in encoded
        # / and = are acceptable in URL-safe base64 (padding)

    def test_cursor_with_special_characters(self):
        """Cursor should handle special characters in timestamp."""
        cursor = PaginationCursor(
            last_id="deal-with-dashes-123",
            last_timestamp="2026-01-19T12:00:00.123456+00:00"
        )

        encoded = cursor.encode()
        decoded = PaginationCursor.decode(encoded)

        assert decoded.last_id == cursor.last_id
        assert decoded.last_timestamp == cursor.last_timestamp

    def test_cursor_deterministic(self):
        """Same cursor data should produce same encoding."""
        cursor1 = PaginationCursor(
            last_id="test-id",
            last_timestamp="2026-01-19T00:00:00"
        )
        cursor2 = PaginationCursor(
            last_id="test-id",
            last_timestamp="2026-01-19T00:00:00"
        )

        assert cursor1.encode() == cursor2.encode()

    def test_different_cursors_different_encoding(self):
        """Different cursor data should produce different encoding."""
        cursor1 = PaginationCursor(
            last_id="id-1",
            last_timestamp="2026-01-19T00:00:00"
        )
        cursor2 = PaginationCursor(
            last_id="id-2",
            last_timestamp="2026-01-19T00:00:00"
        )

        assert cursor1.encode() != cursor2.encode()


class TestPaginationCursorStructure:
    """Test cursor structure and validation."""

    def test_cursor_contains_id_and_timestamp(self):
        """Cursor must contain both id and timestamp."""
        cursor = PaginationCursor(
            last_id="abc-123",
            last_timestamp="2026-01-19T12:00:00"
        )

        assert cursor.last_id == "abc-123"
        assert cursor.last_timestamp == "2026-01-19T12:00:00"

    def test_cursor_with_uuid_id(self):
        """Cursor should work with UUID-style IDs."""
        uuid_id = "550e8400-e29b-41d4-a716-446655440000"
        cursor = PaginationCursor(
            last_id=uuid_id,
            last_timestamp="2026-01-19T12:00:00"
        )

        encoded = cursor.encode()
        decoded = PaginationCursor.decode(encoded)

        assert decoded.last_id == uuid_id

    def test_cursor_with_long_id(self):
        """Cursor should handle long IDs."""
        long_id = "a" * 100
        cursor = PaginationCursor(
            last_id=long_id,
            last_timestamp="2026-01-19T12:00:00"
        )

        encoded = cursor.encode()
        decoded = PaginationCursor.decode(encoded)

        assert decoded.last_id == long_id


class TestCursorEdgeCases:
    """Test edge cases for cursor pagination."""

    def test_cursor_with_null_equivalent(self):
        """Cursor with empty strings should work."""
        cursor = PaginationCursor(
            last_id="",
            last_timestamp=""
        )

        encoded = cursor.encode()
        decoded = PaginationCursor.decode(encoded)

        assert decoded.last_id == ""
        assert decoded.last_timestamp == ""

    def test_cursor_with_unicode(self):
        """Cursor should handle unicode in ID."""
        cursor = PaginationCursor(
            last_id="deal-日本語-123",
            last_timestamp="2026-01-19T12:00:00"
        )

        encoded = cursor.encode()
        decoded = PaginationCursor.decode(encoded)

        assert decoded.last_id == "deal-日本語-123"
