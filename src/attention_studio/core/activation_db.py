from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class ActivationRecord:
    prompt: str
    layer: int
    position: int
    features: np.ndarray
    tokens: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "layer": self.layer,
            "position": self.position,
            "features": self.features.tolist(),
            "tokens": self.tokens,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActivationRecord":
        return cls(
            prompt=data["prompt"],
            layer=data["layer"],
            position=data["position"],
            features=np.array(data["features"]),
            tokens=data["tokens"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


class ActivationDatabase:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.records: list[ActivationRecord] = []
        self._index: dict[int, dict[int, list[int]]] = {}
        if storage_path and storage_path.exists():
            self.load()

    def add(
        self,
        prompt: str,
        layer: int,
        position: int,
        features: torch.Tensor | np.ndarray,
        tokens: list[str],
    ) -> int:
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        record = ActivationRecord(
            prompt=prompt,
            layer=layer,
            position=position,
            features=features,
            tokens=tokens,
        )

        record_id = len(self.records)
        self.records.append(record)

        if layer not in self._index:
            self._index[layer] = {}
        if position not in self._index[layer]:
            self._index[layer][position] = []
        self._index[layer][position].append(record_id)

        return record_id

    def get(self, record_id: int) -> Optional[ActivationRecord]:
        if 0 <= record_id < len(self.records):
            return self.records[record_id]
        return None

    def query_by_layer(self, layer: int) -> list[ActivationRecord]:
        if layer in self._index:
            record_ids = set()
            for pos_ids in self._index[layer].values():
                record_ids.update(pos_ids)
            return [self.records[i] for i in sorted(record_ids)]
        return []

    def query_by_position(self, position: int) -> list[ActivationRecord]:
        result = []
        for record_ids in self._index.values():
            if position in record_ids:
                result.extend(self.records[i] for i in record_ids[position])
        return result

    def query_by_layer_and_position(
        self, layer: int, position: int
    ) -> list[ActivationRecord]:
        if layer in self._index and position in self._index[layer]:
            return [self.records[i] for i in self._index[layer][position]]
        return []

    def query_by_prompt(self, prompt_substring: str) -> list[ActivationRecord]:
        return [r for r in self.records if prompt_substring.lower() in r.prompt.lower()]

    def query_by_feature_threshold(
        self, feature_idx: int, threshold: float, comparison: str = "above"
    ) -> list[ActivationRecord]:
        results = []
        for record in self.records:
            if feature_idx < len(record.features):
                value = record.features[feature_idx]
                if comparison == "above" and value > threshold:
                    results.append(record)
                elif comparison == "below" and value < threshold:
                    results.append(record)
                elif comparison == "equals" and abs(value - threshold) < 1e-6:
                    results.append(record)
        return results

    def get_top_features(
        self, layer: int, position: int, top_k: int = 10
    ) -> list[tuple[int, float]]:
        records = self.query_by_layer_and_position(layer, position)
        if not records:
            return []

        all_features = np.mean([r.features for r in records], axis=0)
        top_indices = np.argsort(all_features)[-top_k:][::-1]

        return [(int(i), float(all_features[i])) for i in top_indices]

    def compute_feature_statistics(self, feature_idx: int) -> dict[str, float]:
        values = []
        for record in self.records:
            if feature_idx < len(record.features):
                values.append(record.features[feature_idx])

        if not values:
            return {}

        values = np.array(values)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
        }

    def get_feature_correlation(
        self, feature_idx1: int, feature_idx2: int
    ) -> float:
        values1 = []
        values2 = []
        for record in self.records:
            if feature_idx1 < len(record.features) and feature_idx2 < len(
                record.features
            ):
                values1.append(record.features[feature_idx1])
                values2.append(record.features[feature_idx2])

        if len(values1) < 2:
            return 0.0

        return float(np.corrcoef(values1, values2)[0, 1])

    def get_all_prompts(self) -> list[str]:
        return list(set(r.prompt for r in self.records))

    def get_all_layers(self) -> list[int]:
        return sorted(self._index.keys())

    def get_all_positions(self, layer: int) -> list[int]:
        if layer in self._index:
            return sorted(self._index[layer].keys())
        return []

    def count_records(self) -> int:
        return len(self.records)

    def clear(self) -> None:
        self.records.clear()
        self._index.clear()

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self.storage_path
        if path is None:
            raise ValueError("No storage path specified")

        data = [r.to_dict() for r in self.records]
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Optional[Path] = None) -> None:
        path = path or self.storage_path
        if path is None or not path.exists():
            return

        data = json.loads(path.read_text())
        self.records = [ActivationRecord.from_dict(d) for d in data]

        self._index.clear()
        for record_id, record in enumerate(self.records):
            if record.layer not in self._index:
                self._index[record.layer] = {}
            if record.position not in self._index[record.layer]:
                self._index[record.layer][record.position] = []
            self._index[record.layer][record.position].append(record_id)


class ActivationBatcher:
    def __init__(self, database: ActivationDatabase, batch_size: int = 32):
        self.database = database
        self.batch_size = batch_size

    def get_batches_by_layer(self, layer: int) -> list[list[ActivationRecord]]:
        records = self.database.query_by_layer(layer)
        batches = []
        for i in range(0, len(records), self.batch_size):
            batches.append(records[i : i + self.batch_size])
        return batches

    def get_batches_by_prompt(self, prompt_substring: str) -> list[list[ActivationRecord]]:
        records = self.database.query_by_prompt(prompt_substring)
        batches = []
        for i in range(0, len(records), self.batch_size):
            batches.append(records[i : i + self.batch_size])
        return batches


class ActivationAggregator:
    @staticmethod
    def aggregate_mean(
        records: list[ActivationRecord], feature_idx: Optional[int] = None
    ) -> np.ndarray:
        if not records:
            return np.array([])

        if feature_idx is not None:
            return np.array([r.features[feature_idx] for r in records if feature_idx < len(r.features)])

        all_features = np.array([r.features for r in records])
        return np.mean(all_features, axis=0)

    @staticmethod
    def aggregate_max(
        records: list[ActivationRecord], feature_idx: Optional[int] = None
    ) -> np.ndarray:
        if not records:
            return np.array([])

        if feature_idx is not None:
            return np.array([r.features[feature_idx] for r in records if feature_idx < len(r.features)])

        all_features = np.array([r.features for r in records])
        return np.max(all_features, axis=0)

    @staticmethod
    def aggregate_sum(
        records: list[ActivationRecord], feature_idx: Optional[int] = None
    ) -> np.ndarray:
        if not records:
            return np.array([])

        if feature_idx is not None:
            return np.array([r.features[feature_idx] for r in records if feature_idx < len(r.features)])

        all_features = np.array([r.features for r in records])
        return np.sum(all_features, axis=0)

    @staticmethod
    def aggregate_by_position(
        records: list[ActivationRecord],
    ) -> dict[int, np.ndarray]:
        result = {}
        by_position: dict[int, list[np.ndarray]] = {}

        for record in records:
            if record.position not in by_position:
                by_position[record.position] = []
            by_position[record.position].append(record.features)

        for pos, features_list in by_position.items():
            result[pos] = np.mean(features_list, axis=0)

        return result
