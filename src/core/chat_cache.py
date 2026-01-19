#!/usr/bin/env python3
"""
Chat Evidence Cache

TTL-based cache for evidence bundles to speed up repeated queries.
Implements separate caches for global, deal, and document scopes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from chat_evidence_builder import EvidenceBundle

# Configuration from environment
CACHE_ENABLED = os.getenv("CHAT_CACHE_ENABLED", "true").lower() == "true"
GLOBAL_TTL_SECONDS = int(os.getenv("CHAT_CACHE_GLOBAL_TTL", "45"))
DEAL_TTL_SECONDS = int(os.getenv("CHAT_CACHE_DEAL_TTL", "180"))
DOC_TTL_SECONDS = int(os.getenv("CHAT_CACHE_DOC_TTL", "180"))
MAX_CACHE_ENTRIES = int(os.getenv("CHAT_CACHE_MAX_ENTRIES", "1000"))


@dataclass
class CacheEntry:
    """A cached evidence bundle with metadata."""
    bundle: Any  # EvidenceBundle
    created_at: float
    ttl_seconds: int
    query_hash: str
    scope_type: str
    deal_id: Optional[str] = None
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    def touch(self):
        """Record a cache hit."""
        self.hit_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    entries_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "entries_count": self.entries_count,
            "hit_rate": round(self.hit_rate, 3),
        }


class EvidenceCache:
    """
    TTL-based cache for evidence bundles.

    Separate caches for each scope type with different TTLs:
    - Global: 45 seconds (queries change frequently)
    - Deal: 180 seconds (3 min, deal data more stable)
    - Document: 180 seconds (docs rarely change)
    """

    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
        self._enabled = CACHE_ENABLED

    def _get_ttl(self, scope_type: str) -> int:
        """Get TTL based on scope type."""
        if scope_type == "global":
            return GLOBAL_TTL_SECONDS
        elif scope_type == "deal":
            return DEAL_TTL_SECONDS
        elif scope_type == "document":
            return DOC_TTL_SECONDS
        return GLOBAL_TTL_SECONDS  # Default

    def cache_key(self, query: str, scope: Dict[str, Any]) -> str:
        """
        Generate cache key from query + scope.

        Key includes:
        - Query text (normalized)
        - Scope type
        - Deal ID (if deal scope)
        - Doc ID (if document scope)
        """
        normalized_query = query.lower().strip()
        scope_type = scope.get("type", "global")
        deal_id = scope.get("deal_id", "")
        doc_id = scope.get("doc", "")

        key_parts = [normalized_query, scope_type, deal_id, doc_id]
        key_string = "|".join(str(p) for p in key_parts)

        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def get(self, key: str, scope_type: str) -> Tuple[Optional[Any], bool]:
        """
        Get cached bundle if not expired.

        Args:
            key: Cache key from cache_key()
            scope_type: "global" | "deal" | "document"

        Returns:
            (bundle, cache_hit) tuple
        """
        if not self._enabled:
            return None, False

        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None, False

            if entry.is_expired():
                # Expired, remove and miss
                del self._cache[key]
                self._stats.misses += 1
                self._stats.entries_count = len(self._cache)
                return None, False

            # Hit
            entry.touch()
            self._stats.hits += 1
            return entry.bundle, True

    async def set(
        self,
        key: str,
        bundle: Any,
        scope_type: str,
        deal_id: Optional[str] = None
    ):
        """
        Cache a bundle with appropriate TTL.

        Args:
            key: Cache key
            bundle: EvidenceBundle to cache
            scope_type: "global" | "deal" | "document"
            deal_id: Deal ID for deal-scoped entries (for invalidation)
        """
        if not self._enabled:
            return

        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= MAX_CACHE_ENTRIES:
                self._evict_oldest()

            ttl = self._get_ttl(scope_type)
            entry = CacheEntry(
                bundle=bundle,
                created_at=time.time(),
                ttl_seconds=ttl,
                query_hash=key,
                scope_type=scope_type,
                deal_id=deal_id,
            )
            self._cache[key] = entry
            self._stats.entries_count = len(self._cache)

    async def invalidate_deal(self, deal_id: str):
        """
        Invalidate all cache entries for a deal.

        Call this when:
        - New event added to deal
        - Deal stage changed
        - New document uploaded
        """
        if not self._enabled:
            return

        async with self._lock:
            keys_to_delete = [
                k for k, v in self._cache.items()
                if v.deal_id == deal_id
            ]
            for key in keys_to_delete:
                del self._cache[key]
                self._stats.invalidations += 1

            self._stats.entries_count = len(self._cache)

    async def invalidate_global(self):
        """Invalidate all global-scope entries."""
        if not self._enabled:
            return

        async with self._lock:
            keys_to_delete = [
                k for k, v in self._cache.items()
                if v.scope_type == "global"
            ]
            for key in keys_to_delete:
                del self._cache[key]
                self._stats.invalidations += 1

            self._stats.entries_count = len(self._cache)

    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats.entries_count = 0

    def _evict_oldest(self):
        """Evict oldest entry (LRU-like)."""
        if not self._cache:
            return

        # Find oldest by created_at
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        del self._cache[oldest_key]
        self._stats.evictions += 1

    def _cleanup_expired(self):
        """Remove all expired entries (called periodically)."""
        expired_keys = [
            k for k, v in self._cache.items()
            if v.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]
        self._stats.entries_count = len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Clean up expired entries first
        self._cleanup_expired()
        return {
            **self._stats.to_dict(),
            "enabled": self._enabled,
            "config": {
                "global_ttl_seconds": GLOBAL_TTL_SECONDS,
                "deal_ttl_seconds": DEAL_TTL_SECONDS,
                "doc_ttl_seconds": DOC_TTL_SECONDS,
                "max_entries": MAX_CACHE_ENTRIES,
            },
        }

    def enable(self):
        """Enable caching."""
        self._enabled = True

    def disable(self):
        """Disable caching."""
        self._enabled = False


# Singleton instance
_cache_instance: Optional[EvidenceCache] = None


def get_cache() -> EvidenceCache:
    """Get or create the singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = EvidenceCache()
    return _cache_instance


if __name__ == "__main__":
    # Quick test
    import asyncio

    async def test_cache():
        cache = get_cache()

        # Generate key
        key = cache.cache_key("How many deals?", {"type": "global"})
        print(f"Cache key: {key}")

        # Set a value
        await cache.set(key, {"test": "bundle"}, "global")
        print(f"Set bundle")

        # Get it back
        bundle, hit = await cache.get(key, "global")
        print(f"Get result: hit={hit}, bundle={bundle}")

        # Check stats
        print(f"Stats: {cache.stats()}")

    asyncio.run(test_cache())
