from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AnalysisResult:
    id: str
    timestamp: str
    analysis_type: str
    model_name: str
    prompt: str
    metrics: dict[str, float] = field(default_factory=dict)
    features: list[dict[str, Any]] = field(default_factory=list)
    graph_data: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "analysis_type": self.analysis_type,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "metrics": self.metrics,
            "features": self.features,
            "graph_data": self.graph_data,
            "notes": self.notes,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisResult:
        return cls(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", ""),
            analysis_type=data.get("analysis_type", ""),
            model_name=data.get("model_name", ""),
            prompt=data.get("prompt", ""),
            metrics=data.get("metrics", {}),
            features=data.get("features", []),
            graph_data=data.get("graph_data", {}),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
        )


class AnalysisHistory:
    def __init__(self, storage_dir: Path | None = None):
        if storage_dir is None:
            storage_dir = Path.home() / ".attention_studio" / "history"
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._storage_dir / "analysis_history.json"
        self._results: list[AnalysisResult] = []
        self._load()

    def _load(self) -> None:
        if self._history_file.exists():
            try:
                data = json.loads(self._history_file.read_text())
                self._results = [AnalysisResult.from_dict(r) for r in data]
            except (json.JSONDecodeError, KeyError):
                self._results = []
        else:
            self._results = []

    def _save(self) -> None:
        data = [r.to_dict() for r in self._results]
        self._history_file.write_text(json.dumps(data, indent=2))

    def add(
        self,
        analysis_type: str,
        model_name: str,
        prompt: str,
        metrics: dict[str, float] | None = None,
        features: list[dict[str, Any]] | None = None,
        graph_data: dict[str, Any] | None = None,
        notes: str = "",
        tags: list[str] | None = None,
    ) -> AnalysisResult:
        result = AnalysisResult(
            id=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._results)}",
            timestamp=datetime.now().isoformat(),
            analysis_type=analysis_type,
            model_name=model_name,
            prompt=prompt,
            metrics=metrics or {},
            features=features or [],
            graph_data=graph_data or {},
            notes=notes,
            tags=tags or [],
        )
        self._results.append(result)
        self._save()
        return result

    def get(self, result_id: str) -> AnalysisResult | None:
        for result in self._results:
            if result.id == result_id:
                return result
        return None

    def list_all(self) -> list[AnalysisResult]:
        return list(self._results)

    def list_by_type(self, analysis_type: str) -> list[AnalysisResult]:
        return [r for r in self._results if r.analysis_type == analysis_type]

    def list_by_model(self, model_name: str) -> list[AnalysisResult]:
        return [r for r in self._results if r.model_name == model_name]

    def list_by_tag(self, tag: str) -> list[AnalysisResult]:
        return [r for r in self._results if tag in r.tags]

    def search(self, query: str) -> list[AnalysisResult]:
        query_lower = query.lower()
        return [
            r for r in self._results
            if query_lower in r.prompt.lower()
            or query_lower in r.notes.lower()
            or query_lower in r.analysis_type.lower()
        ]

    def delete(self, result_id: str) -> bool:
        for i, result in enumerate(self._results):
            if result.id == result_id:
                self._results.pop(i)
                self._save()
                return True
        return False

    def clear(self) -> None:
        self._results.clear()
        self._save()

    def export_json(self, path: Path | str, result_ids: list[str] | None = None) -> None:
        path = Path(path)
        results = [r for r in self._results if r.id in result_ids] if result_ids else self._results

        data = {
            "exported_at": datetime.now().isoformat(),
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }
        path.write_text(json.dumps(data, indent=2))

    def export_csv(self, path: Path | str) -> None:
        path = Path(path)
        if not self._results:
            return

        rows = []
        for r in self._results:
            row = {
                "id": r.id,
                "timestamp": r.timestamp,
                "analysis_type": r.analysis_type,
                "model_name": r.model_name,
                "prompt": r.prompt[:100],
                "notes": r.notes,
                "tags": ",".join(r.tags),
            }
            for key, value in r.metrics.items():
                row[f"metric_{key}"] = value
            rows.append(row)

        if rows:
            fieldnames = list(rows[0].keys())
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def get_summary(self) -> dict[str, Any]:
        if not self._results:
            return {"total": 0, "by_type": {}, "by_model": {}}

        by_type: dict[str, int] = {}
        by_model: dict[str, int] = {}

        for r in self._results:
            by_type[r.analysis_type] = by_type.get(r.analysis_type, 0) + 1
            by_model[r.model_name] = by_model.get(r.model_name, 0) + 1

        return {
            "total": len(self._results),
            "by_type": by_type,
            "by_model": by_model,
            "oldest": self._results[0].timestamp if self._results else None,
            "newest": self._results[-1].timestamp if self._results else None,
        }

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self) -> Iterator[AnalysisResult]:
        return iter(self._results)

    def __getitem__(self, index: int) -> AnalysisResult:
        return self._results[index]


_history: AnalysisHistory | None = None


def get_analysis_history() -> AnalysisHistory:
    global _history
    if _history is None:
        _history = AnalysisHistory()
    return _history
