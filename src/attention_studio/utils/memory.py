from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from loguru import logger


@dataclass
class MemoryStats:
    allocated: float
    reserved: float
    free: float
    peak_allocated: float


class MemoryManager:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._peak_allocated = 0

    def get_memory_stats(self) -> MemoryStats:
        if torch.cuda.is_available() and self.device == "cuda":
            return MemoryStats(
                allocated=torch.cuda.memory_allocated(self.device) / 1024**3,
                reserved=torch.cuda.memory_reserved(self.device) / 1024**3,
                free=(torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)) / 1024**3,
                peak_allocated=torch.cuda.max_memory_allocated(self.device) / 1024**3,
            )
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return MemoryStats(
                allocated=0,
                reserved=0,
                free=0,
                peak_allocated=0,
            )
        return MemoryStats(allocated=0, reserved=0, free=0, peak_allocated=0)

    def clear_cache(self):
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        gc.collect()

    def optimize_memory(self):
        gc.collect()
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()

    @contextmanager
    def memory_tracker(self, name: str = ""):
        stats_before = self.get_memory_stats()
        logger.debug(f"[{name}] Memory before: {stats_before.allocated:.2f} GB")

        try:
            yield
        finally:
            stats_after = self.get_memory_stats()
            diff = stats_after.allocated - stats_before.allocated
            logger.debug(f"[{name}] Memory after: {stats_after.allocated:.2f} GB (delta: {diff:+.2f} GB)")


class ModelMemoryOptimizer:
    @staticmethod
    def convert_to_half(model: torch.nn.Module) -> torch.nn.Module:
        return model.half()

    @staticmethod
    def convert_to_bf16(model: torch.nn.Module) -> torch.nn.Module:
        return model.to(dtype=torch.bfloat16)

    @staticmethod
    def enable_gradient_checkpointing(model: torch.nn.Module):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

    @staticmethod
    def freeze_layers(model: torch.nn.Module, num_layers: int = 0):
        if hasattr(model, 'transformer'):
            transformer = model.transformer
            if hasattr(transformer, 'h'):
                for i, layer in enumerate(transformer.h):
                    if i < num_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

    @staticmethod
    def get_model_memory_footprint(model: torch.nn.Module) -> float:
        total = 0
        for param in model.parameters():
            total += param.nelement() * param.element_size()
        for buffer in model.buffers():
            total += buffer.nelement() * buffer.element_size()
        return total / 1024**3


class ActivationCache:
    def __init__(self, max_size: int = 10):
        self._cache: dict[str, torch.Tensor] = {}
        self._max_size = max_size
        self._access_order: list[str] = []

    def get(self, key: str) -> torch.Tensor | None:
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: torch.Tensor):
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = value.detach()
        self._access_order.append(key)

    def clear(self):
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        return len(self._cache)


@contextmanager
def empty_cache_context():
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_tensor_size(tensor: torch.Tensor) -> int:
    return tensor.nelement() * tensor.element_size()


def estimate_model_size(model: torch.nn.Module) -> dict[str, float]:
    param_sizes = {}
    buffer_sizes = {}

    for name, param in model.named_parameters():
        size_mb = param.nelement() * param.element_size() / 1024**2
        param_sizes[name] = size_mb

    for name, buffer in model.named_buffers():
        size_mb = buffer.nelement() * buffer.element_size() / 1024**2
        buffer_sizes[name] = size_mb

    total_params = sum(param_sizes.values())
    total_buffers = sum(buffer_sizes.values())

    return {
        "total_mb": total_params + total_buffers,
        "parameters_mb": total_params,
        "buffers_mb": total_buffers,
    }


def print_memory_summary():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
