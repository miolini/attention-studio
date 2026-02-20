from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelComparisonResult:
    model_a: str
    model_b: str
    prompt: str
    features_a: list[dict[str, Any]] = field(default_factory=list)
    features_b: list[dict[str, Any]] = field(default_factory=list)
    circuits_a: dict[str, list] = field(default_factory=dict)
    circuits_b: dict[str, list] = field(default_factory=dict)
    differences: dict[str, Any] = field(default_factory=dict)


class ModelComparator:
    def __init__(self, model_managers: list[Any], trainers: list[Any] | None = None):
        self.model_managers = model_managers
        self.trainers = trainers or [None] * len(model_managers)
        self._comparison_history: list[ModelComparisonResult] = []

    def compare_features(
        self,
        prompt: str,
        layer_indices: list[int] | None = None,
    ) -> ModelComparisonResult:
        if len(self.model_managers) < 2:
            raise ValueError("Need at least 2 models to compare")

        model_a = self.model_managers[0]
        model_b = self.model_managers[1]
        trainer_a = self.trainers[0] if self.trainers else None
        trainer_b = self.trainers[1] if len(self.trainers) > 1 else None

        features_a = self._extract_features(model_a, trainer_a, prompt, layer_indices or [0])
        features_b = self._extract_features(model_b, trainer_b, prompt, layer_indices or [0])

        circuits_a = self._find_circuits(model_a, trainer_a, prompt, layer_indices or [0])
        circuits_b = self._find_circuits(model_b, trainer_b, prompt, layer_indices or [0])

        differences = self._compute_differences(features_a, features_b, circuits_a, circuits_b)

        result = ModelComparisonResult(
            model_a=getattr(model_a, 'model_name', 'model_a'),
            model_b=getattr(model_b, 'model_name', 'model_b'),
            prompt=prompt,
            features_a=features_a,
            features_b=features_b,
            circuits_a=circuits_a,
            circuits_b=circuits_b,
            differences=differences,
        )

        self._comparison_history.append(result)
        return result

    def _extract_features(
        self,
        model_manager: Any,
        trainer: Any,
        prompt: str,
        layer_indices: list[int],
    ) -> list[dict[str, Any]]:
        if not trainer or not trainer.transcoders:
            return []

        features = []
        tokenizer = model_manager.tokenizer

        for layer_idx in layer_indices:
            transcoder = trainer.get_transcoder(layer_idx)
            if transcoder is None:
                continue

            import torch

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model_manager.model.device)

            with torch.no_grad():
                outputs = model_manager.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[layer_idx]

            _, layer_features = transcoder(hidden_states)
            layer_features = layer_features.squeeze(0)

            mean_activations = layer_features.mean(dim=0)
            top_k = min(20, len(mean_activations))
            top_features = torch.topk(mean_activations.abs(), top_k)

            for idx, act in zip(top_features.indices, top_features.values, strict=True):
                features.append({
                    "layer": layer_idx,
                    "idx": idx.item(),
                    "activation": act.item(),
                    "norm": torch.norm(transcoder.decoder.weight[:, idx]).item() if transcoder.decoder.weight is not None else 0,
                })

        return features

    def _find_circuits(
        self,
        model_manager: Any,
        trainer: Any,
        prompt: str,
        layer_indices: list[int],
    ) -> dict[str, list]:
        if not trainer or not trainer.transcoders:
            return {}

        from attention_studio.core.feature_extractor import GlobalCircuitAnalyzer

        analyzer = GlobalCircuitAnalyzer(
            model_manager,
            trainer.transcoders,
            getattr(trainer, 'lorsas', None),
            layer_indices,
        )

        return analyzer.analyze_all_circuits(prompt)

    def _compute_differences(
        self,
        features_a: list[dict],
        features_b: list[dict],
        circuits_a: dict,
        circuits_b: dict,
    ) -> dict[str, Any]:
        feat_dict_a = {f["idx"]: f for f in features_a}
        feat_dict_b = {f["idx"]: f for f in features_b}

        all_indices = set(feat_dict_a.keys()) | set(feat_dict_b.keys())

        feature_diffs = []
        for idx in all_indices:
            act_a = feat_dict_a.get(idx, {}).get("activation", 0)
            act_b = feat_dict_b.get(idx, {}).get("activation", 0)
            diff = abs(act_a - act_b)
            if diff > 0.01:
                feature_diffs.append({
                    "idx": idx,
                    "model_a": act_a,
                    "model_b": act_b,
                    "diff": diff,
                })

        feature_diffs.sort(key=lambda x: x["diff"], reverse=True)

        circuit_types = set(circuits_a.keys()) | set(circuits_b.keys())
        circuit_diffs = {}

        for ct in circuit_types:
            has_a = ct in circuits_a and circuits_a[ct]
            has_b = ct in circuits_b and circuits_b[ct]
            if has_a != has_b:
                circuit_diffs[ct] = {"in_a": has_a, "in_b": has_b}

        return {
            "top_feature_differences": feature_diffs[:20],
            "circuit_differences": circuit_diffs,
            "unique_to_a": len(set(feat_dict_a.keys()) - set(feat_dict_b.keys())),
            "unique_to_b": len(set(feat_dict_b.keys()) - set(feat_dict_a.keys())),
        }

    def get_comparison_history(self) -> list[ModelComparisonResult]:
        return self._comparison_history

    def clear_history(self):
        self._comparison_history.clear()
