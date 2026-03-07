"""
CacheManager — thin wrapper around diskcache with per-category TTLs.

Usage:
    from cache.cache_manager import cache

    # Per-ticker caching
    data = cache.get("fundamentals", "NVDA")
    if data is None:
        data = expensive_fetch("NVDA")
        cache.set("fundamentals", "NVDA", data)

    # Batch caching (e.g. earnings calendar)
    data = cache.get_batch("earnings_calendar", ["NVDA", "TSLA"])
    if data is None:
        data = expensive_batch_fetch(["NVDA", "TSLA"])
        cache.set_batch("earnings_calendar", ["NVDA", "TSLA"], data)

    # Filter stale tickers from a list
    fresh, stale = cache.partition("fundamentals", ["NVDA", "TSLA", "VOO"])
    # fresh = {"NVDA": cached_data}  (still valid)
    # stale = ["TSLA", "VOO"]        (need re-fetch)
"""
from __future__ import annotations

import os
from pathlib import Path

import diskcache

from cache.ttl_config import TTL_MAP, cache_key, batch_cache_key

_CACHE_DIR = os.getenv("CACHE_DIR", ".cache/data")
_MAX_SIZE = int(float(os.getenv("CACHE_MAX_SIZE_GB", "2")) * 1e9)


class CacheManager:
    """Singleton cache backed by diskcache.Cache with per-category TTLs."""

    def __init__(self, directory: str = _CACHE_DIR, max_size: int = _MAX_SIZE):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(
            directory,
            size_limit=max_size,
            eviction_policy="least-recently-used",
        )

    def get(self, category: str, ticker: str):
        """
        Get cached data for a category+ticker.
        Returns None if not cached or expired.
        """
        key = cache_key(category, ticker)
        return self._cache.get(key)

    def set(self, category: str, ticker: str, data) -> None:
        """Cache data for a category+ticker with the appropriate TTL."""
        key = cache_key(category, ticker)
        ttl = TTL_MAP.get(category, 86400)  # default 24hr
        self._cache.set(key, data, expire=ttl)

    def get_batch(self, category: str, tickers: list[str]):
        """Get cached data for a batch request. Returns None if expired."""
        key = batch_cache_key(category, tickers)
        return self._cache.get(key)

    def set_batch(self, category: str, tickers: list[str], data) -> None:
        """Cache data for a batch request."""
        key = batch_cache_key(category, tickers)
        ttl = TTL_MAP.get(category, 86400)
        self._cache.set(key, data, expire=ttl)

    def partition(
        self, category: str, tickers: list[str]
    ) -> tuple[dict, list[str]]:
        """
        Split tickers into fresh (cached) and stale (need re-fetch).

        Returns:
            (fresh_dict, stale_list)
            fresh_dict: {ticker: cached_data} for tickers with valid cache
            stale_list: [ticker, ...] for tickers needing re-fetch
        """
        fresh: dict = {}
        stale: list[str] = []

        for ticker in tickers:
            data = self.get(category, ticker)
            if data is not None:
                fresh[ticker] = data
            else:
                stale.append(ticker)

        return fresh, stale

    def invalidate(self, category: str, ticker: str) -> None:
        """Remove a specific cache entry."""
        key = cache_key(category, ticker)
        self._cache.delete(key)

    def clear_category(self, category: str) -> int:
        """Remove all entries for a category. Returns count removed."""
        prefix = f"{category}:"
        removed = 0
        for key in list(self._cache):
            if isinstance(key, str) and key.startswith(prefix):
                self._cache.delete(key)
                removed += 1
        return removed

    def clear_all(self) -> None:
        """Clear entire cache."""
        self._cache.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        volume = self._cache.volume()
        return {
            "entries": len(self._cache),
            "size_mb": round(volume / 1e6, 1),
            "directory": str(self._cache.directory),
        }

    def close(self) -> None:
        """Close the cache (call on shutdown)."""
        self._cache.close()


# Module-level singleton
cache = CacheManager()
