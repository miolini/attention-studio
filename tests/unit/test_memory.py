import pytest
import torch
import torch.nn as nn

from attention_studio.utils.memory import (
    MemoryManager,
    ModelMemoryOptimizer,
    ActivationCache,
    empty_cache_context,
    get_tensor_size,
    estimate_model_size,
)


class TestMemoryManager:
    def test_get_memory_stats(self):
        manager = MemoryManager(device="cpu")
        stats = manager.get_memory_stats()
        assert hasattr(stats, "allocated")
        assert hasattr(stats, "reserved")
        assert hasattr(stats, "free")
        assert hasattr(stats, "peak_allocated")

    def test_clear_cache(self):
        manager = MemoryManager(device="cpu")
        manager.clear_cache()

    def test_memory_tracker(self):
        manager = MemoryManager(device="cpu")
        with manager.memory_tracker("test"):
            x = torch.randn(100, 100)
            _ = x * 2


class TestModelMemoryOptimizer:
    def test_get_model_memory_footprint(self):
        model = nn.Linear(10, 10)
        footprint = ModelMemoryOptimizer.get_model_memory_footprint(model)
        assert footprint > 0

    def test_estimate_model_size(self):
        model = nn.Linear(10, 10)
        sizes = estimate_model_size(model)
        assert "total_mb" in sizes
        assert "parameters_mb" in sizes
        assert sizes["parameters_mb"] > 0


class TestActivationCache:
    def test_cache_put_and_get(self):
        cache = ActivationCache(max_size=2)
        tensor = torch.randn(10, 10)
        cache.put("key1", tensor)
        result = cache.get("key1")
        assert result is not None

    def test_cache_miss(self):
        cache = ActivationCache(max_size=2)
        result = cache.get("nonexistent")
        assert result is None

    def test_cache_eviction(self):
        cache = ActivationCache(max_size=2)
        cache.put("key1", torch.randn(10))
        cache.put("key2", torch.randn(10))
        cache.put("key3", torch.randn(10))
        assert cache.size() == 2
        assert cache.get("key1") is None

    def test_cache_clear(self):
        cache = ActivationCache(max_size=2)
        cache.put("key1", torch.randn(10))
        cache.clear()
        assert cache.size() == 0


class TestHelpers:
    def test_get_tensor_size(self):
        tensor = torch.randn(10, 10)
        size = get_tensor_size(tensor)
        assert size > 0

    def test_empty_cache_context(self):
        with empty_cache_context():
            x = torch.randn(100, 100)
            _ = x * 2
