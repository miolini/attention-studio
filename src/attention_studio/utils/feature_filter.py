from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class FilterCriteria:
    min_activation: float = 0.0
    max_activation: float = float("inf")
    layers: list[int] | None = None
    feature_indices: list[int] | None = None
    search_text: str = ""
    regex: bool = False


class FeatureFilter:
    def __init__(self, features: list[dict[str, Any]]):
        self._features = features
        self._original = list(features)

    def filter_by_activation(
        self,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> list[dict[str, Any]]:
        min_val = min_val if min_val is not None else 0.0
        max_val = max_val if max_val is not None else float("inf")

        return [
            f for f in self._features
            if min_val <= f.get("activation", 0) <= max_val
        ]

    def filter_by_layer(self, layers: list[int]) -> list[dict[str, Any]]:
        return [f for f in self._features if f.get("layer") in layers]

    def filter_by_feature_index(self, indices: list[int]) -> list[dict[str, Any]]:
        return [f for f in self._features if f.get("idx") in indices]

    def search(self, text: str, case_sensitive: bool = False) -> list[dict[str, Any]]:
        if not text:
            return self._features

        if case_sensitive:
            return [f for f in self._features if text in str(f.get("token", ""))]
        else:
            text_lower = text.lower()
            return [f for f in self._features if text_lower in str(f.get("token", "")).lower()]

    def regex_search(self, pattern: str) -> list[dict[str, Any]]:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            return [
                f for f in self._features
                if regex.search(str(f.get("token", "")))
            ]
        except re.error:
            return self._features

    def apply_criteria(self, criteria: FilterCriteria) -> list[dict[str, Any]]:
        result = self._features

        result = self.filter_by_activation(criteria.min_activation, criteria.max_activation)

        if criteria.layers:
            result = [f for f in result if f.get("layer") in criteria.layers]

        if criteria.feature_indices:
            result = [f for f in result if f.get("idx") in criteria.feature_indices]

        if criteria.search_text:
            if criteria.regex:
                result = self.regex_search(criteria.search_text)
            else:
                result = self.search(criteria.search_text)

        return result

    def sort_by(
        self,
        key: str = "activation",
        reverse: bool = True,
    ) -> list[dict[str, Any]]:
        return sorted(self._features, key=lambda f: f.get(key, 0), reverse=reverse)

    def reset(self):
        self._features = list(self._original)

    def get_statistics(self) -> dict[str, Any]:
        if not self._features:
            return {}

        activations = [f.get("activation", 0) for f in self._features]
        layers = {f.get("layer") for f in self._features if f.get("layer") is not None}

        return {
            "total_features": len(self._features),
            "min_activation": min(activations) if activations else 0,
            "max_activation": max(activations) if activations else 0,
            "mean_activation": sum(activations) / len(activations) if activations else 0,
            "layers": sorted(layers),
            "unique_features": len({f.get("idx") for f in self._features}),
        }


class FeatureSearchEngine:
    def __init__(self, features: list[dict[str, Any]]):
        self._features = features
        self._index: dict[int, list[dict[str, Any]]] = {}

        for f in features:
            layer = f.get("layer", -1)
            if layer not in self._index:
                self._index[layer] = []
            self._index[layer].append(f)

    def search_by_layer(self, layer: int) -> list[dict[str, Any]]:
        return self._index.get(layer, [])

    def search_by_activation_range(
        self,
        min_val: float,
        max_val: float,
    ) -> list[dict[str, Any]]:
        return [
            f for f in self._features
            if min_val <= f.get("activation", 0) <= max_val
        ]

    def find_similar(
        self,
        feature_idx: int,
        layer: int | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        target = None
        for f in self._features:
            if f.get("idx") == feature_idx and (layer is None or f.get("layer") == layer):
                target = f
                break

        if not target:
            return []

        target_act = target.get("activation", 0)

        similarities = []
        for f in self._features:
            if f.get("idx") == feature_idx:
                continue
            if layer is not None and f.get("layer") != layer:
                continue

            act = f.get("activation", 0)
            similarity = 1 - abs(act - target_act)
            similarities.append((f, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in similarities[:top_k]]

    def find_by_pattern(self, pattern: str) -> list[dict[str, Any]]:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            return [
                f for f in self._features
                if regex.search(str(f.get("token", "")))
            ]
        except re.error:
            return []
