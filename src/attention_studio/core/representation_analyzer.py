from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class RepresentationMetrics:
    layer_idx: int
    mean_norm: float
    std_norm: float
    variance_explained: float
    rank: int


class RepresentationAnalyzer:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

    def get_hidden_states(
        self,
        prompt: str,
        layer_indices: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        all_hidden_states = outputs.hidden_states

        if layer_indices is None:
            layer_indices = list(range(len(all_hidden_states)))

        states = {}
        for idx in layer_indices:
            if 0 <= idx < len(all_hidden_states):
                states[idx] = all_hidden_states[idx].squeeze(0)

        return states

    def compute_representation_norms(
        self,
        prompt: str,
        layer_indices: list[int] | None = None,
    ) -> dict[int, float]:
        states = self.get_hidden_states(prompt, layer_indices)

        norms = {}
        for layer_idx, hidden in states.items():
            flat = hidden.reshape(-1, hidden.shape[-1])
            norm = torch.norm(flat, dim=1).mean().item()
            norms[layer_idx] = norm

        return norms

    def compute_pca(
        self,
        prompts: list[str],
        layer_idx: int,
        n_components: int = 10,
    ) -> dict[str, np.ndarray]:
        all_representations = []

        for prompt in prompts:
            states = self.get_hidden_states(prompt, [layer_idx])
            hidden = states[layer_idx]
            all_representations.append(hidden.cpu().numpy())

        combined = np.concatenate(all_representations, axis=0)
        combined_flat = combined.reshape(-1, combined.shape[-1])

        centered = combined_flat - combined_flat.mean(axis=0)

        u, s, vt = np.linalg.svd(centered, full_matrices=False)

        components = vt[:n_components]
        explained_variance = (s[:n_components] ** 2).sum() / (s ** 2).sum()

        projected = centered @ components.T

        return {
            "components": components,
            "explained_variance": explained_variance,
            "singular_values": s[:n_components],
            "projected": projected,
        }

    def compute_svd(
        self,
        prompts: list[str],
        layer_idx: int,
        n_components: int = 10,
    ) -> dict[str, np.ndarray]:
        all_representations = []

        for prompt in prompts:
            states = self.get_hidden_states(prompt, [layer_idx])
            hidden = states[layer_idx]
            all_representations.append(hidden.cpu().numpy())

        combined = np.concatenate(all_representations, axis=0)
        combined_flat = combined.reshape(-1, combined.shape[-1])

        u, s, vt = np.linalg.svd(combined_flat, full_matrices=False)

        return {
            "U": u[:, :n_components],
            "S": s[:n_components],
            "Vt": vt[:n_components],
        }

    def compute_representation_similarity(
        self,
        prompt_a: str,
        prompt_b: str,
        layer_idx: int,
    ) -> float:
        states_a = self.get_hidden_states(prompt_a, [layer_idx])
        states_b = self.get_hidden_states(prompt_b, [layer_idx])

        hidden_a = states_a[layer_idx].flatten()
        hidden_b = states_b[layer_idx].flatten()

        cos_sim = torch.cosine_similarity(
            hidden_a.unsqueeze(0), hidden_b.unsqueeze(0)
        ).item()

        return cos_sim

    def compute_intrinsic_dimension(
        self,
        prompts: list[str],
        layer_idx: int,
        threshold: float = 0.95,
    ) -> int:
        all_representations = []

        for prompt in prompts:
            states = self.get_hidden_states(prompt, [layer_idx])
            hidden = states[layer_idx]
            all_representations.append(hidden.cpu().numpy())

        combined = np.concatenate(all_representations, axis=0)
        combined_flat = combined.reshape(-1, combined.shape[-1])

        u, s, _ = np.linalg.svd(combined_flat, full_matrices=False)

        total_variance = (s ** 2).sum()
        cumsum = np.cumsum(s ** 2)
        intrinsic_dim = np.searchsorted(cumsum / total_variance, threshold) + 1

        return int(intrinsic_dim)

    def compute_metrics(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> RepresentationMetrics:
        all_representations = []

        for prompt in prompts:
            states = self.get_hidden_states(prompt, [layer_idx])
            hidden = states[layer_idx]
            all_representations.append(hidden.cpu().numpy())

        combined = np.concatenate(all_representations, axis=0)
        combined_flat = combined.reshape(-1, combined.shape[-1])

        norms = np.linalg.norm(combined_flat, axis=1)
        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))

        u, s, _ = np.linalg.svd(combined_flat, full_matrices=False)
        total_var = (s ** 2).sum()
        variance_explained = float((s[0] ** 2) / total_var)

        rank = int(np.sum(s > 1e-10))

        return RepresentationMetrics(
            layer_idx=layer_idx,
            mean_norm=mean_norm,
            std_norm=std_norm,
            variance_explained=variance_explained,
            rank=rank,
        )


class RepresentationComparator:
    def __init__(self, model_manager: Any):
        self.analyzer = RepresentationAnalyzer(model_manager)

    def compare_prompts(
        self,
        prompt_a: str,
        prompt_b: str,
        layer_indices: list[int] | None = None,
    ) -> dict[int, float]:
        if layer_indices is None:
            states = self.analyzer.get_hidden_states(prompt_a)
            layer_indices = list(states.keys())

        similarities = {}
        for layer_idx in layer_indices:
            sim = self.analyzer.compute_representation_similarity(
                prompt_a, prompt_b, layer_idx
            )
            similarities[layer_idx] = sim

        return similarities

    def find_similar_prompts(
        self,
        query_prompt: str,
        candidate_prompts: list[str],
        layer_idx: int,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        similarities = []

        for candidate in candidate_prompts:
            sim = self.analyzer.compute_representation_similarity(
                query_prompt, candidate, layer_idx
            )
            similarities.append((candidate, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
