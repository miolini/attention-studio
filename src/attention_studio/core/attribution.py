from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional


class GradientAttribution:
    @staticmethod
    def compute_gradients(
        model: nn.Module,
        input_ids: torch.Tensor,
        target_idx: int,
        layer_idx: int,
    ) -> torch.Tensor:
        model.eval()
        input_embeds = model.get_input_embeddings()(input_ids)
        input_embeds.requires_grad_(True)

        outputs = model(inputs_embeds=input_embeds, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[layer_idx]

        target = hidden_states[0, target_idx].sum()
        target.backward()

        gradients = input_embeds.grad
        return gradients

    @staticmethod
    def integrated_gradients(
        model: nn.Module,
        input_ids: torch.Tensor,
        target_idx: int,
        layer_idx: int,
        baseline_ids: Optional[torch.Tensor] = None,
        steps: int = 50,
    ) -> torch.Tensor:
        model.eval()

        if baseline_ids is None:
            baseline_ids = torch.zeros_like(input_ids)

        input_embeds = model.get_input_embeddings()(input_ids)
        baseline_embeds = model.get_input_embeddings()(baseline_ids)

        integrated_grad = torch.zeros_like(input_embeds)

        for step in range(steps):
            alpha = step / steps
            current_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
            current_embeds.requires_grad_(True)

            outputs = model(inputs_embeds=current_embeds, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states[layer_idx]

            target = hidden_states[0, target_idx].sum()
            model.zero_grad()
            target.backward()

            integrated_grad += current_embeds.grad

        integrated_grad = (input_embeds - baseline_embeds) * integrated_grad / steps
        return integrated_grad


class AttentionAttribution:
    @staticmethod
    def attention_rollout(
        model: nn.Module,
        input_ids: torch.Tensor,
        num_layers: int,
    ) -> torch.Tensor:
        model.eval()

        outputs = model(input_ids=input_ids, output_attentions=True, return_dict=True)

        if not hasattr(outputs, "attentions") or outputs.attentions is None:
            return torch.zeros(1, 1, 1, 1)

        attentions = outputs.attentions
        num_heads = attentions[0].shape[1]

        rollout = attentions[0]
        for i in range(1, len(attentions)):
            attention = attentions[i]
            batch_size, heads, seq_len, _ = attention.shape

            attention = attention + torch.eye(seq_len, device=attention.device).unsqueeze(0).unsqueeze(0)
            attention = attention / attention.sum(dim=-1, keepdim=True)

            rollout = torch.matmul(rollout, attention)

        return rollout

    @staticmethod
    def attention_flow(
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        model.eval()

        outputs = model(input_ids=input_ids, output_attentions=True, return_dict=True)

        if not hasattr(outputs, "attentions") or outputs.attentions is None:
            return {}

        attentions = outputs.attentions

        flow_by_layer = {}
        for layer_idx, attention in enumerate(attentions):
            attn = attention[0].mean(dim=0)
            flow_by_layer[layer_idx] = attn

        return flow_by_layer


class LRPAttribution:
    @staticmethod
    def compute_lrp(
        model: nn.Module,
        input_ids: torch.Tensor,
        target_idx: int,
    ) -> torch.Tensor:
        model.eval()

        embeddings = model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)

        outputs = model(inputs_embeds=embeddings, output_hidden_states=True, return_dict=True)
        logits = outputs.logits[0, target_idx]

        relevances = embeddings.grad
        if relevances is None:
            relevances = torch.zeros_like(embeddings)

        return relevances.abs()


class ShapAttribution:
    @staticmethod
    def compute_shapley(
        model: nn.Module,
        input_ids: torch.Tensor,
        target_idx: int,
        num_samples: int = 100,
    ) -> torch.Tensor:
        model.eval()

        seq_len = input_ids.shape[1]
        embeddings = model.get_input_embeddings()(input_ids)

        baseline = torch.zeros_like(embeddings)

        attributions = torch.zeros_like(embeddings)

        for _ in range(num_samples):
            mask = torch.rand(seq_len) > 0.5

            perturbed = embeddings.clone()
            perturbed[:, mask[None, :, None], :] = baseline[:, mask[None, :, None], :]

            outputs = model(inputs_embeds=perturbed, output_hidden_states=True, return_dict=True)
            score_perturbed = outputs.hidden_states[-1][0, target_idx].sum()

            outputs = model(inputs_embeds=embeddings, output_hidden_states=True, return_dict=True)
            score_original = outputs.hidden_states[-1][0, target_idx].sum()

            contribution = (score_perturbed - score_original).item()
            attributions[:, mask[None, :, None], :] += contribution

        attributions /= num_samples
        return attributions


class FeatureAttribution:
    def __init__(self, model: nn.Module, transcoder: Optional[Any] = None):
        self.model = model
        self.transcoder = transcoder

    def attribute_by_gradient(
        self,
        input_ids: torch.Tensor,
        target_idx: int = -1,
        layer_idx: int = -1,
    ) -> dict[str, torch.Tensor]:
        gradients = GradientAttribution.compute_gradients(
            self.model, input_ids, target_idx, layer_idx
        )
        return {
            "gradient": gradients,
            "gradient_norm": gradients.norm(dim=-1),
        }

    def attribute_by_integrated_gradients(
        self,
        input_ids: torch.Tensor,
        target_idx: int = -1,
        layer_idx: int = -1,
        steps: int = 50,
    ) -> dict[str, torch.Tensor]:
        ig = GradientAttribution.integrated_gradients(
            self.model, input_ids, target_idx, layer_idx, steps=steps
        )
        return {
            "integrated_gradient": ig,
            "ig_norm": ig.norm(dim=-1),
        }

    def attribute_by_attention(
        self,
        input_ids: torch.Tensor,
    ) -> dict[str, Any]:
        rollout = AttentionAttribution.attention_rollout(
            self.model, input_ids, len(list(self.model.parameters()))
        )
        return {
            "rollout": rollout,
            "head_importance": rollout.mean(dim=(1, 2)),
        }

    def attribute_by_lrp(
        self,
        input_ids: torch.Tensor,
        target_idx: int = -1,
    ) -> dict[str, torch.Tensor]:
        relevances = LRPAttribution.compute_lrp(
            self.model, input_ids, target_idx
        )
        return {
            "relevance": relevances,
            "relevance_norm": relevances.abs().sum(dim=-1),
        }

    def attribute_comprehensive(
        self,
        input_ids: torch.Tensor,
        target_idx: int = -1,
        layer_idx: int = -1,
    ) -> dict[str, Any]:
        results = {}

        try:
            results["gradient"] = self.attribute_by_gradient(input_ids, target_idx, layer_idx)
        except Exception:
            results["gradient"] = None

        try:
            results["integrated_gradients"] = self.attribute_by_integrated_gradients(
                input_ids, target_idx, layer_idx
            )
        except Exception:
            results["integrated_gradients"] = None

        try:
            results["attention"] = self.attribute_by_attention(input_ids)
        except Exception:
            results["attention"] = None

        try:
            results["lrp"] = self.attribute_by_lrp(input_ids, target_idx)
        except Exception:
            results["lrp"] = None

        return results


class NeuronLevelAttribution:
    @staticmethod
    def attribute_to_neurons(
        model: nn.Module,
        input_ids: torch.Tensor,
        layer_idx: int,
    ) -> dict[str, torch.Tensor]:
        model.eval()

        embeddings = model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)

        outputs = model(inputs_embeds=embeddings, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[layer_idx]

        neuron_activations = hidden_states[0].mean(dim=0)

        neuron_activations.sum().backward()

        neuron_gradients = embeddings.grad[0].mean(dim=1)

        return {
            "activations": neuron_activations,
            "gradients": neuron_gradients,
            "gradient_times_activation": neuron_activations * neuron_gradients,
        }


class FeatureInteractionAttribution:
    @staticmethod
    def compute_feature_interactions(
        model: nn.Module,
        input_ids: torch.Tensor,
        feature_indices: list[int],
    ) -> torch.Tensor:
        model.eval()

        embeddings = model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)

        outputs = model(inputs_embeds=embeddings, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[-1]

        interactions = torch.zeros(len(feature_indices), len(feature_indices))

        for i, idx1 in enumerate(feature_indices):
            for j, idx2 in enumerate(feature_indices):
                if i != j:
                    hidden_states[0, :, idx1].sum().backward(retain_graph=True)
                    grad1 = embeddings.grad[0, :, idx2].sum().item()
                    interactions[i, j] = grad1
                    model.zero_grad()

        return interactions
