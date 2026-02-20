from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger

T = TypeVar("T")


class CacheEntry:
    def __init__(self, value: Any, metadata: dict[str, Any]):
        self.value = value
        self.metadata = metadata


class DiskCache:
    def __init__(self, cache_dir: Path, max_size_mb: int = 100):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = cache_dir / "metadata.json"
        self._metadata: dict[str, dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self):
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    self._metadata = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._metadata = {}

    def _save_metadata(self):
        with open(self._metadata_file, "w") as f:
            json.dump(self._metadata, f)

    def _get_cache_path(self, key: str) -> Path:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def _compute_size(self) -> float:
        total_size = 0.0
        for file in self.cache_dir.glob("*.cache"):
            total_size += file.stat().st_size
        return total_size / (1024 * 1024)

    def get(self, key: str) -> Any | None:
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        if key in self._metadata:
            import time
            entry = self._metadata[key]
            if entry.get("expired", 0) > 0 and time.time() > entry["expired"]:
                self.delete(key)
                return None

        try:
            import pickle
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl_seconds: int = 0):
        import pickle
        import time

        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)

            expired = time.time() + ttl_seconds if ttl_seconds > 0 else 0
            self._metadata[key] = {
                "path": str(cache_path),
                "created": time.time(),
                "expired": expired,
            }

            current_size = self._compute_size()
            if current_size > self.max_size_mb:
                self._evict_oldest(int(current_size - self.max_size_mb) + 10)

            self._save_metadata()
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")

    def delete(self, key: str):
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
        if key in self._metadata:
            del self._metadata[key]
            self._save_metadata()

    def clear(self):
        for file in self.cache_dir.glob("*.cache"):
            file.unlink()
        self._metadata.clear()
        self._save_metadata()

    def _evict_oldest(self, size_mb: float):
        sorted_entries = sorted(
            self._metadata.items(),
            key=lambda x: x[1].get("created", 0)
        )

        for key, entry in sorted_entries:
            if size_mb <= 0:
                break
            cache_path = Path(entry["path"])
            if cache_path.exists():
                size_mb -= cache_path.stat().st_size / (1024 * 1024)
                self.delete(key)


def memoize(func: Callable[..., T]) -> Callable[..., T]:
    cache: dict[str, T] = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key = f"{func.__name__}:{':'.join(key_parts)}"

        if key not in cache:
            cache[key] = func(*args, **kwargs)

        return cache[key]

    wrapper.cache_clear = lambda: cache.clear()
    wrapper.cache_info = lambda: {"size": len(cache), "keys": list(cache.keys())}
    return wrapper


def lru_cache(maxsize: int = 128):
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict[str, T] = {}
        access_order: list[str] = []

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = f"{func.__name__}:{':'.join(key_parts)}"

            if key in cache:
                access_order.remove(key)
                access_order.append(key)
                return cache[key]

            result = func(*args, **kwargs)

            if len(cache) >= maxsize:
                oldest = access_order.pop(0)
                del cache[oldest]

            cache[key] = result
            access_order.append(key)

            return result

        wrapper.cache_clear = lambda: (cache.clear(), access_order.clear())
        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize, "keys": list(cache.keys())}
        return wrapper

    return decorator


class ComputationCache:
    def __init__(self):
        self._cache: dict[str, Any] = {}

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def invalidate(self, key: str):
        if key in self._cache:
            del self._cache[key]

    def clear(self):
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        return {"entries": len(self._cache), "keys": list(self._cache.keys())}


_global_cache = ComputationCache()


def get_global_cache() -> ComputationCache:
    return _global_cache
