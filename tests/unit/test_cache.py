import pytest
import time
from pathlib import Path
import tempfile
import shutil

from attention_studio.utils.cache import (
    DiskCache,
    ComputationCache,
    memoize,
    lru_cache,
    get_global_cache,
)


class TestDiskCache:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache = DiskCache(self.temp_dir, max_size_mb=1)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_cache_set_and_get(self):
        self.cache.set("key1", {"data": "value1"})
        result = self.cache.get("key1")
        assert result == {"data": "value1"}

    def test_cache_miss(self):
        result = self.cache.get("nonexistent")
        assert result is None

    def test_cache_delete(self):
        self.cache.set("key1", "value1")
        self.cache.delete("key1")
        result = self.cache.get("key1")
        assert result is None

    def test_cache_clear(self):
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.clear()
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None


class TestComputationCache:
    def test_get_or_compute(self):
        cache = ComputationCache()
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "computed"

        result1 = cache.get_or_compute("key", compute)
        assert result1 == "computed"
        assert call_count == 1

        result2 = cache.get_or_compute("key", compute)
        assert result2 == "computed"
        assert call_count == 1

    def test_invalidate(self):
        cache = ComputationCache()
        cache.get_or_compute("key", lambda: "value")
        cache.invalidate("key")
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "new_value"

        result = cache.get_or_compute("key", compute)
        assert result == "new_value"
        assert call_count == 1


class TestMemoize:
    def test_memoize(self):
        call_count = 0

        @memoize
        def expensive_func(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        assert expensive_func(1, 2) == 3
        assert call_count == 1

        assert expensive_func(1, 2) == 3
        assert call_count == 1

        assert expensive_func(3, 4) == 7
        assert call_count == 2

    def test_memoize_clear(self):
        @memoize
        def func(x):
            return x * 2

        func(5)
        func(5)
        assert len(func.cache_info()["keys"]) == 1
        func.cache_clear()
        assert len(func.cache_info()["keys"]) == 0


class TestLRUCache:
    def test_lru_cache(self):
        call_count = 0

        @lru_cache(maxsize=2)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive_func(1) == 2
        assert call_count == 1

        assert expensive_func(1) == 2
        assert call_count == 1

        assert expensive_func(2) == 4
        assert call_count == 2

        assert expensive_func(3) == 6
        assert call_count == 3

    def test_lru_cache_eviction(self):
        @lru_cache(maxsize=2)
        def func(x):
            return x * 2

        func(1)
        func(2)
        func(3)

        assert func.cache_info()["size"] == 2


class TestGlobalCache:
    def test_global_cache(self):
        cache = get_global_cache()
        cache.clear()

        result = cache.get_or_compute("test_key", lambda: "test_value")
        assert result == "test_value"

        stats = cache.get_stats()
        assert stats["entries"] == 1
        assert "test_key" in stats["keys"]
