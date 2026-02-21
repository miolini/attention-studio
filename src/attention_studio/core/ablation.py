from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class AblationResult:
    original_score: float
    ablated_score: float
    change: float
    change_percent: float
    feature_indices: list[int]
    layer_idx: int


@dataclass
class AblationStudy:
    name: str
    results: list[AblationResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class FeatureAblator:
    def __init__(self, model_manager: Any, transcoder: Any):
        self.model_manager = model_manager
        self.transcoder = transcoder

    def ablate_features(
        self,
        prompt: str,
        feature_indices: list[int],
        layer_idx: int,
        ablation_value: float = 0.0,
    ) -> AblationResult:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs_original = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            original_logits = outputs_original.logits[0, -1, :]
            original_score = torch.softmax(original_logits, dim=-1).max().item()

        def create_hook(feat_indices):
            def hook(module, inp, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                _, features = self.transcoder(hidden)
                for idx in feat_indices:
                    features[:, :, idx] = ablation_value
                reconstructed = self.transcoder.decoder(features.view(-1, features.shape[0] * features.shape[1], features.shape[2]))
                reconstructed = reconstructed.view(hidden.shape)
                if isinstance(output, tuple):
                    return (reconstructed,) + output[1:]
                return reconstructed
            return hook

        hook_handle = None
        try:
            for name, module in model.named_modules():
                if f".h.{layer_idx}" in name or f".layers.{layer_idx}" in name:
                    hook_handle = module.register_forward_pre_hook(create_hook(feature_indices))
                    break

            with torch.no_grad():
                outputs_ablated = model(
                    input_ids=input_ids,
                    return_dict=True,
                )
                ablated_logits = outputs_ablated.logits[0, -1, :]
                ablated_score = torch.softmax(ablated_logits, dim=-1).max().item()
        finally:
            if hook_handle:
                hook_handle.remove()

        change = original_score - ablated_score
        change_percent = (change / original_score * 100) if original_score > 0 else 0

        return AblationResult(
            original_score=original_score,
            ablated_score=ablated_score,
            change=change,
            change_percent=change_percent,
            feature_indices=feature_indices,
            layer_idx=layer_idx,
        )

    def ablate_random_features(
        self,
        prompt: str,
        layer_idx: int,
        n_features: int,
        n_trials: int = 5,
    ) -> list[AblationResult]:
        results = []

        for _ in range(n_trials):
            feature_indices = np.random.choice(100, n_features, replace=False).tolist()
            result = self.ablate_features(prompt, feature_indices, layer_idx)
            results.append(result)

        return results

    def ablate_by_importance(
        self,
        prompt: str,
        layer_idx: int,
        top_k: int,
    ) -> AblationResult:
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
            hidden_states = outputs.hidden_states[layer_idx]

        _, features = self.transcoder(hidden_states)
        mean_activations = features.abs().mean(dim=(0, 1))
        top_features = torch.topk(mean_activations, top_k).indices.tolist()

        return self.ablate_features(prompt, top_features, layer_idx)


class LayerAblator:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager

    def ablate_layer(
        self,
        prompt: str,
        layer_idx: int,
        ablation_type: str = "zero",
    ) -> dict[str, float]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs_original = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            original_logits = outputs_original.logits[0, -1, :]
            original_score = torch.softmax(original_logits, dim=-1).max().item()

        def create_hook(layer_idx, ablation_type):
            def hook(module, inp, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                if ablation_type == "zero":
                    hidden = torch.zeros_like(hidden)
                elif ablation_type == "mean":
                    hidden = hidden.mean(dim=0, keepdim=True).expand_as(hidden)
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook

        hook_handle = None
        try:
            for name, module in model.named_modules():
                if f".h.{layer_idx}" in name or f".layers.{layer_idx}" in name:
                    hook_handle = module.register_forward_pre_hook(create_hook(layer_idx, ablation_type))
                    break

            with torch.no_grad():
                outputs_ablated = model(
                    input_ids=input_ids,
                    return_dict=True,
                )
                ablated_logits = outputs_ablated.logits[0, -1, :]
                ablated_score = torch.softmax(ablated_logits, dim=-1).max().item()
        finally:
            if hook_handle:
                hook_handle.remove()

        return {
            "original_score": original_score,
            "ablated_score": ablated_score,
            "change": original_score - ablated_score,
            "layer_idx": layer_idx,
            "ablation_type": ablation_type,
        }


class AblationStudyManager:
    def __init__(self):
        self.studies: dict[str, AblationStudy] = {}

    def create_study(self, name: str, metadata: Optional[dict] = None) -> AblationStudy:
        study = AblationStudy(name=name, metadata=metadata or {})
        self.studies[name] = study
        return study

    def add_result(self, study_name: str, result: AblationResult) -> None:
        if study_name not in self.studies:
            raise ValueError(f"Study {study_name} not found")
        self.studies[study_name].results.append(result)

    def get_study(self, name: str) -> Optional[AblationStudy]:
        return self.studies.get(name)

    def compare_ablations(
        self,
        study_name: str,
    ) -> dict[str, Any]:
        study = self.get_study(study_name)
        if not study:
            return {}

        results = study.results
        if not results:
            return {}

        sorted_results = sorted(results, key=lambda x: abs(x.change), reverse=True)

        return {
            "most_impactful": {
                "features": sorted_results[0].feature_indices,
                "change": sorted_results[0].change,
                "layer": sorted_results[0].layer_idx,
            },
            "least_impactful": {
                "features": sorted_results[-1].feature_indices,
                "change": sorted_results[-1].change,
                "layer": sorted_results[-1].layer_idx,
            },
            "average_change": np.mean([r.change for r in results]),
            "total_ablations": len(results),
        }

    def rank_features(
        self,
        study_name: str,
    ) -> list[tuple[int, float]]:
        study = self.get_study(study_name)
        if not study:
            return []

        feature_impact: dict[int, list[float]] = {}

        for result in study.results:
            for feature_idx in result.feature_indices:
                if feature_idx not in feature_impact:
                    feature_impact[feature_idx] = []
                feature_impact[feature_idx].append(abs(result.change))

        avg_impact = {k: np.mean(v) for k, v in feature_impact.items()}
        return sorted(avg_impact.items(), key=lambda x: x[1], reverse=True)

    def save_study(self, path: str) -> None:
        import json
        data = {}
        for name, study in self.studies.items():
            data[name] = {
                "metadata": study.metadata,
                "results": [
                    {
                        "original_score": r.original_score,
                        "ablated_score": r.ablated_score,
                        "change": r.change,
                        "change_percent": r.change_percent,
                        "feature_indices": r.feature_indices,
                        "layer_idx": r.layer_idx,
                    }
                    for r in study.results
                ],
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_study(self, path: str) -> None:
        import json
        with open(path, "r") as f:
            data = json.load(f)

        for name, study_data in data.items():
            study = AblationStudy(
                name=name,
                metadata=study_data.get("metadata", {}),
            )
            for r_data in study_data.get("results", []):
                result = AblationResult(
                    original_score=r_data["original_score"],
                    ablated_score=r_data["ablated_score"],
                    change=r_data["change"],
                    change_percent=r_data["change_percent"],
                    feature_indices=r_data["feature_indices"],
                    layer_idx=r_data["layer_idx"],
                )
                study.results.append(result)
            self.studies[name] = study
