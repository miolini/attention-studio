from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass
class LayerAnalysis:
    layer_idx: int
    representation_norm: float
    representation_variance: float
    cosine_similarity: float


class LayerRepresentationAnalyzer:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager

    def compute_layer_norms(
        self,
        prompts: list[str],
        layer_indices: list[int] | None = None,
    ) -> dict[int, float]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        if layer_indices is None:
            num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else 12
            layer_indices = list(range(num_layers))

        norms = {}

        for layer_idx in layer_indices:
            layer_norms = []

            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(model.device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                hidden_states = outputs.hidden_states[layer_idx]
                norm = hidden_states.norm(dim=-1).mean().item()
                layer_norms.append(norm)

            norms[layer_idx] = np.mean(layer_norms)

        return norms

    def compute_layer_variance(
        self,
        prompts: list[str],
        layer_indices: list[int] | None = None,
    ) -> dict[int, float]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        if layer_indices is None:
            num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else 12
            layer_indices = list(range(num_layers))

        variances = {}

        for layer_idx in layer_indices:
            layer_vars = []

            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(model.device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                hidden_states = outputs.hidden_states[layer_idx]
                var = hidden_states.var(dim=1).mean().item()
                layer_vars.append(var)

            variances[layer_idx] = np.mean(layer_vars)

        return variances

    def compute_layer_similarities(
        self,
        prompts: list[str],
        layer_indices: list[int] | None = None,
    ) -> dict[tuple[int, int], float]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        if layer_indices is None:
            num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else 12
            layer_indices = list(range(num_layers))

        representations = {idx: [] for idx in layer_indices}

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            for layer_idx in layer_indices:
                hidden_states = outputs.hidden_states[layer_idx]
                rep = hidden_states[0, -1, :].cpu().numpy()
                representations[layer_idx].append(rep)

        similarities = {}
        for i, layer1 in enumerate(layer_indices):
            for layer2 in layer_indices[i + 1:]:
                reps1 = np.mean(representations[layer1], axis=0)
                reps2 = np.mean(representations[layer2], axis=0)

                cos_sim = np.dot(reps1, reps2) / (np.linalg.norm(reps1) * np.linalg.norm(reps2) + 1e-8)
                similarities[(layer1, layer2)] = cos_sim

        return similarities

    def compute_representational_similarity(
        self,
        prompts: list[str],
        layer_indices: list[int] | None = None,
    ) -> np.ndarray:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        if layer_indices is None:
            num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else 12
            layer_indices = list(range(num_layers))

        representations = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            layer_reps = []
            for layer_idx in layer_indices:
                hidden_states = outputs.hidden_states[layer_idx]
                rep = hidden_states[0, -1, :].cpu().numpy()
                layer_reps.append(rep)

            representations.append(layer_reps)

        n_layers = len(layer_indices)
        rdm = np.zeros((n_layers, n_layers))

        for i in range(n_layers):
            for j in range(n_layers):
                if i == j:
                    rdm[i, j] = 0.0
                else:
                    reps_i = [r[i] for r in representations]
                    reps_j = [r[j] for r in reps_i]

                    diffs = [np.linalg.norm(a - b) for a, b in zip(reps_i, reps_j, strict=True)]
                    rdm[i, j] = np.mean(diffs)

        return rdm


class LayerTransitionAnalyzer:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager

    def compute_information_flow(
        self,
        prompt: str,
        layer_indices: list[int],
    ) -> dict[int, dict[str, float]]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        results = {}

        for i, layer_idx in enumerate(layer_indices[:-1]):
            hidden_curr = outputs.hidden_states[layer_idx]
            hidden_next = outputs.hidden_states[layer_indices[i + 1]]

            diff = (hidden_next - hidden_curr).abs()
            change_magnitude = diff.mean().item()

            corr = functional.cosine_similarity(
                hidden_curr.flatten(),
                hidden_next.flatten(),
                dim=0,
            ).item()

            results[layer_idx] = {
                "change_magnitude": change_magnitude,
                "correlation": corr,
            }

        return results

    def compute_layerwise_entropy(
        self,
        prompts: list[str],
        layer_indices: list[int],
    ) -> dict[int, float]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        entropies = {idx: [] for idx in layer_indices}

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            for layer_idx in layer_indices:
                hidden_states = outputs.hidden_states[layer_idx]
                probs = functional.softmax(hidden_states, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
                entropies[layer_idx].append(entropy)

        return {idx: np.mean(vals) for idx, vals in entropies.items()}

    def find_transfer_layers(
        self,
        prompts: list[str],
        layer_indices: list[int],
        threshold: float = 0.8,
    ) -> list[tuple[int, int]]:
        similarities = self._compute_pairwise_similarities(prompts, layer_indices)

        transfer_layers = []
        for (layer1, layer2), sim in similarities.items():
            if sim > threshold:
                transfer_layers.append((layer1, layer2))

        return transfer_layers

    def _compute_pairwise_similarities(
        self,
        prompts: list[str],
        layer_indices: list[int],
    ) -> dict[tuple[int, int], float]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        representations = {idx: [] for idx in layer_indices}

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            for layer_idx in layer_indices:
                hidden_states = outputs.hidden_states[layer_idx]
                rep = hidden_states[0, -1, :].cpu().numpy()
                representations[layer_idx].append(rep)

        similarities = {}
        for i, layer1 in enumerate(layer_indices):
            for layer2 in layer_indices[i + 1:]:
                reps1 = np.mean(representations[layer1], axis=0)
                reps2 = np.mean(representations[layer2], axis=0)

                cos_sim = np.dot(reps1, reps2) / (np.linalg.norm(reps1) * np.linalg.norm(reps2) + 1e-8)
                similarities[(layer1, layer2)] = cos_sim

        return similarities


class LayerComparison:
    @staticmethod
    def compare_layers(
        analysis1: dict[int, float],
        analysis2: dict[int, float],
    ) -> dict[int, float]:
        common_layers = set(analysis1.keys()) & set(analysis2.keys())
        return {layer: analysis1[layer] - analysis2[layer] for layer in common_layers}

    @staticmethod
    def compute_diversity_score(
        layer_metrics: dict[int, dict[str, float]],
    ) -> float:
        if not layer_metrics:
            return 0.0

        all_values = []
        for metrics in layer_metrics.values():
            all_values.extend(metrics.values())

        if not all_values:
            return 0.0

        return np.std(all_values)

    @staticmethod
    def find_optimal_layer(
        layer_metrics: dict[int, float],
        criterion: str = "max",
    ) -> int | None:
        if not layer_metrics:
            return None

        if criterion == "max":
            return max(layer_metrics.items(), key=lambda x: x[1])[0]
        elif criterion == "min":
            return min(layer_metrics.items(), key=lambda x: x[1])[0]
        else:
            return None
