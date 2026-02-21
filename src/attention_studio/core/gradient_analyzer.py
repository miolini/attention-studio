from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass
class GradientStats:
    layer_idx: int
    mean_grad_norm: float
    max_grad_norm: float
    grad_std: float
    zero_fraction: float


class GradientAnalyzer:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

    def compute_gradients(
        self,
        prompt: str,
        target_token: str | None = None,
    ) -> dict[int, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        self.model.zero_grad()

        outputs = self.model(input_ids=input_ids, output_hidden_states=True)

        if target_token is not None:
            target_id = self.tokenizer.encode(target_token)[0]
            target_logits = outputs.logits[0, -1, target_id]
            target_logits.backward()
        else:
            loss = outputs.logits[0, -1].sum()
            loss.backward()

        gradients = {}
        for i, layer in enumerate(self.model.transformer.h):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                gradients[i] = layer.weight.grad.detach().clone()

        self.model.zero_grad()
        return gradients

    def compute_layer_gradient_stats(
        self,
        prompt: str,
        layer_indices: list[int] | None = None,
    ) -> list[GradientStats]:
        gradients = self.compute_gradients(prompt)

        if layer_indices is None:
            layer_indices = list(gradients.keys())

        stats_list = []
        for layer_idx in layer_indices:
            if layer_idx not in gradients:
                continue

            grad = gradients[layer_idx]
            grad_flat = grad.flatten().abs().cpu().numpy()

            mean_norm = float(np.mean(grad_flat))
            max_norm = float(np.max(grad_flat))
            std_norm = float(np.std(grad_flat))
            zero_frac = float(np.mean(grad_flat == 0))

            stats = GradientStats(
                layer_idx=layer_idx,
                mean_grad_norm=mean_norm,
                max_grad_norm=max_norm,
                grad_std=std_norm,
                zero_fraction=zero_frac,
            )
            stats_list.append(stats)

        return stats_list

    def compute_gradient_similarity(
        self,
        prompt_a: str,
        prompt_b: str,
    ) -> np.ndarray:
        grads_a = self.compute_gradients(prompt_a)
        grads_b = self.compute_gradients(prompt_b)

        common_layers = set(grads_a.keys()) & set(grads_b.keys())
        num_layers = len(common_layers)

        similarity = np.zeros((num_layers, num_layers))

        layers_a = sorted(common_layers)
        layers_b = sorted(common_layers)

        for i, ly_a in enumerate(layers_a):
            for j, ly_b in enumerate(layers_b):
                g_a = grads_a[ly_a].flatten()
                g_b = grads_b[ly_b].flatten()

                if g_a.shape != g_b.shape:
                    similarity[i, j] = 0.0
                else:
                    cos_sim = functional.cosine_similarity(
                        g_a.unsqueeze(0), g_b.unsqueeze(0)
                    ).item()
                    similarity[i, j] = cos_sim

        return similarity

    def find_gradient_hero_layers(
        self,
        prompts: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        layer_grads: dict[int, list[float]] = {}

        for prompt in prompts:
            grads = self.compute_gradients(prompt)
            for layer_idx, grad in grads.items():
                if layer_idx not in layer_grads:
                    layer_grads[layer_idx] = []
                grad_norm = grad.norm().item()
                layer_grads[layer_idx].append(grad_norm)

        layer_avg_grads = {
            layer_idx: np.mean(grads) for layer_idx, grads in layer_grads.items()
        }

        sorted_layers = sorted(
            layer_avg_grads.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_layers[:top_k]


class GradientFlowTracker:
    def __init__(self, model_manager: Any):
        self.analyzer = GradientAnalyzer(model_manager)
        self.history: list[dict[int, float]] = []

    def record(self, prompt: str) -> None:
        grads = self.analyzer.compute_gradients(prompt)
        grad_norms = {layer: grad.norm().item() for layer, grad in grads.items()}
        self.history.append(grad_norms)

    def get_layer_history(self, layer_idx: int) -> list[float]:
        return [h.get(layer_idx, 0.0) for h in self.history]

    def get_total_history(self) -> list[dict[int, float]]:
        return list(self.history)

    def compute_stability(self, layer_idx: int) -> float:
        history = self.get_layer_history(layer_idx)
        if len(history) < 2:
            return 0.0

        history_arr = np.array(history)
        variance = float(np.var(history_arr))
        return variance

    def clear(self) -> None:
        self.history = []
