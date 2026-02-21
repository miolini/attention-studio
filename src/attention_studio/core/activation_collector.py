from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ActivationSnapshot:
    prompt: str
    layer_activations: dict[int, torch.Tensor]
    timestamp: float = 0.0


@dataclass
class ActivationCollection:
    snapshots: list[ActivationSnapshot] = field(default_factory=list)


class ActivationCollector:
    def __init__(self, model_manager: Any, layer_indices: list[int] | None = None):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer
        self.layer_indices = layer_indices
        self.hooks: list[Any] = []
        self.collection = ActivationCollection()

    def register_hooks(self) -> None:
        if self.layer_indices is None:
            self.layer_indices = list(range(len(self.model.transformer.h)))

        for layer_idx in self.layer_indices:
            layer = self.model.transformer.h[layer_idx]

            def create_hook(idx):
                def hook(module, inp, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._store_activation(idx, hidden.detach())
                return hook

            handle = layer.register_forward_hook(create_hook(layer_idx))
            self.hooks.append(handle)

    def remove_hooks(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def _store_activation(self, layer_idx: int, activation: torch.Tensor) -> None:
        if not hasattr(self, "_current_activations"):
            self._current_activations = {}
        self._current_activations[layer_idx] = activation

    def collect(self, prompt: str) -> ActivationSnapshot:
        self._current_activations = {}

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        with torch.no_grad():
            _ = self.model(input_ids)

        snapshot = ActivationSnapshot(
            prompt=prompt,
            layer_activations=self._current_activations.copy(),
        )

        self.collection.snapshots.append(snapshot)

        return snapshot

    def get_snapshot(self, index: int) -> ActivationSnapshot | None:
        if 0 <= index < len(self.collection.snapshots):
            return self.collection.snapshots[index]
        return None

    def get_all_snapshots(self) -> list[ActivationSnapshot]:
        return list(self.collection.snapshots)

    def get_layer_activations(
        self,
        layer_idx: int,
        aggregate: str = "mean",
    ) -> torch.Tensor | None:
        if not self.collection.snapshots:
            return None

        activations = []
        for snapshot in self.collection.snapshots:
            if layer_idx in snapshot.layer_activations:
                activations.append(snapshot.layer_activations[layer_idx])

        if not activations:
            return None

        stacked = torch.stack(activations)

        if aggregate == "mean":
            return stacked.mean(dim=0)
        elif aggregate == "max":
            return stacked.max(dim=0).values
        elif aggregate == "sum":
            return stacked.sum(dim=0)
        else:
            return stacked

    def clear(self) -> None:
        self.collection.snapshots = []
        self._current_activations = {}

    def __len__(self) -> int:
        return len(self.collection.snapshots)


class ActivationComparator:
    def __init__(self, collector: ActivationCollector):
        self.collector = collector

    def compute_layer_similarity(
        self,
        snapshot_a_idx: int,
        snapshot_b_idx: int,
        layer_idx: int,
    ) -> float:
        snapshot_a = self.collector.get_snapshot(snapshot_a_idx)
        snapshot_b = self.collector.get_snapshot(snapshot_b_idx)

        if not snapshot_a or not snapshot_b:
            return 0.0

        if layer_idx not in snapshot_a.layer_activations:
            return 0.0

        if layer_idx not in snapshot_b.layer_activations:
            return 0.0

        act_a = snapshot_a.layer_activations[layer_idx].flatten()
        act_b = snapshot_b.layer_activations[layer_idx].flatten()

        cos_sim = torch.cosine_similarity(act_a.unsqueeze(0), act_b.unsqueeze(0)).item()

        return cos_sim

    def compute_prompt_similarity_matrix(
        self,
        layer_idx: int,
    ) -> list[list[float]]:
        n = len(self.collector)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.compute_layer_similarity(i, j, layer_idx)

        return matrix

    def find_most_similar_pairs(
        self,
        layer_idx: int,
        top_k: int = 5,
    ) -> list[tuple[int, int, float]]:
        n = len(self.collector)
        pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_layer_similarity(i, j, layer_idx)
                pairs.append((i, j, sim))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]
