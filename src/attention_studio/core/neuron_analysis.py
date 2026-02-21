from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class NeuronStats:
    layer_idx: int
    neuron_idx: int
    mean_activation: float
    std_activation: float
    max_activation: float
    zero_fraction: float
    dead_neuron: bool


@dataclass
class NeuronProfile:
    neuron_id: tuple[int, int]
    stats: NeuronStats
    top_activating_prompts: list[tuple[str, float]] = field(default_factory=list)
    activation_distribution: list[float] = field(default_factory=list)


class NeuronAnalyzer:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

    def get_neuron_activations(
        self,
        prompt: str,
        layer_idx: int,
    ) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        hidden_states = None

        def hook(module, inp, output):
            nonlocal hidden_states
            hidden_states = output[0] if isinstance(output, tuple) else output

        layer = self.model.transformer.h[layer_idx]
        handle = layer.register_forward_hook(hook)

        with torch.no_grad():
            _ = self.model(input_ids)

        handle.remove()
        return hidden_states

    def compute_neuron_statistics(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> list[NeuronStats]:
        all_activations = []

        for prompt in prompts:
            activations = self.get_neuron_activations(prompt, layer_idx)
            all_activations.append(activations.squeeze(0))

        stacked = torch.cat(all_activations, dim=0)
        num_neurons = stacked.shape[-1]

        stats_list = []
        for neuron_idx in range(num_neurons):
            neuron_acts = stacked[:, neuron_idx].numpy()

            mean_act = float(np.mean(neuron_acts))
            std_act = float(np.std(neuron_acts))
            max_act = float(np.max(neuron_acts))
            zero_frac = float(np.mean(neuron_acts == 0))
            dead = zero_frac > 0.99

            stats = NeuronStats(
                layer_idx=layer_idx,
                neuron_idx=neuron_idx,
                mean_activation=mean_act,
                std_activation=std_act,
                max_activation=max_act,
                zero_fraction=zero_frac,
                dead_neuron=dead,
            )
            stats_list.append(stats)

        return stats_list

    def find_dead_neurons(
        self,
        prompts: list[str],
        layer_idx: int,
        threshold: float = 0.99,
    ) -> list[int]:
        stats = self.compute_neuron_statistics(prompts, layer_idx)
        return [s.neuron_idx for s in stats if s.zero_fraction > threshold]

    def find_important_neurons(
        self,
        prompts: list[str],
        layer_idx: int,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        stats = self.compute_neuron_statistics(prompts, layer_idx)
        by_variance = sorted(stats, key=lambda s: s.std_activation, reverse=True)
        return [(s.neuron_idx, s.std_activation) for s in by_variance[:top_k]]

    def compute_neuron_correlation(
        self,
        prompts: list[str],
        layer_idx: int,
        neuron_indices: list[int],
    ) -> np.ndarray:
        all_activations = []

        for prompt in prompts:
            activations = self.get_neuron_activations(prompt, layer_idx)
            all_activations.append(activations.squeeze(0))

        stacked = torch.cat(all_activations, dim=0)
        selected = stacked[:, neuron_indices].numpy()

        corr = np.corrcoef(selected.T)
        return corr


class NeuronProfiler:
    def __init__(self, model_manager: Any):
        self.analyzer = NeuronAnalyzer(model_manager)

    def profile_neurons(
        self,
        prompts: list[str],
        layer_indices: list[int] | None = None,
    ) -> dict[int, list[NeuronProfile]]:
        model = self.analyzer.model
        num_layers = len(model.transformer.h)

        if layer_indices is None:
            layer_indices = list(range(num_layers))

        profiles = {}
        for layer_idx in layer_indices:
            stats_list = self.analyzer.compute_neuron_statistics(prompts, layer_idx)
            layer_profiles = []

            for stats in stats_list:
                profile = NeuronProfile(
                    neuron_id=(stats.layer_idx, stats.neuron_idx),
                    stats=stats,
                )
                layer_profiles.append(profile)

            profiles[layer_idx] = layer_profiles

        return profiles

    def find_maximally_activating_prompts(
        self,
        prompts: list[str],
        layer_idx: int,
        neuron_idx: int,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        prompt_activations = []

        for prompt in prompts:
            activations = self.analyzer.get_neuron_activations(prompt, layer_idx)
            max_act = activations[:, :, neuron_idx].max().item()
            prompt_activations.append((prompt, max_act))

        prompt_activations.sort(key=lambda x: x[1], reverse=True)
        return prompt_activations[:top_k]
