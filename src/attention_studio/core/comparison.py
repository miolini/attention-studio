from __future__ import annotations

from dataclasses import dataclass, field
from statistics import StatisticsError, mean, stdev
from typing import Any


@dataclass
class ComparisonItem:
    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    metric_name: str
    comparison_type: str
    items: list[str]
    values: list[float]
    statistic: float | None = None
    p_value: float | None = None
    interpretation: str = ""


class ComparisonEngine:
    def __init__(self):
        self._items: list[ComparisonItem] = []

    def add_item(self, name: str, metrics: dict[str, float], metadata: dict[str, Any] | None = None) -> None:
        item = ComparisonItem(name=name, metrics=metrics, metadata=metadata or {})
        self._items.append(item)

    def add_items(self, items: list[ComparisonItem]) -> None:
        self._items.extend(items)

    def clear(self) -> None:
        self._items.clear()

    def get_item(self, name: str) -> ComparisonItem | None:
        for item in self._items:
            if item.name == name:
                return item
        return None

    def list_items(self) -> list[ComparisonItem]:
        return list(self._items)

    def get_metric_names(self) -> list[str]:
        names = set()
        for item in self._items:
            names.update(item.metrics.keys())
        return sorted(names)

    def compare_metric(self, metric_name: str) -> ComparisonResult | None:
        items_with_metric = [item for item in self._items if metric_name in item.metrics]
        if not items_with_metric:
            return None

        names = [item.name for item in items_with_metric]
        values = [item.metrics[metric_name] for item in items_with_metric]

        result = ComparisonResult(
            metric_name=metric_name,
            comparison_type="values",
            items=names,
            values=values,
        )

        if len(values) >= 2:
            result.statistic = max(values) - min(values)
            result.interpretation = f"Range: {result.statistic:.4f}"

        return result

    def compare_all_metrics(self) -> list[ComparisonResult]:
        results = []
        for metric_name in self.get_metric_names():
            result = self.compare_metric(metric_name)
            if result:
                results.append(result)
        return results

    def rank_items(self, metric_name: str, ascending: bool = False) -> list[tuple[str, float]]:
        items_with_metric = [item for item in self._items if metric_name in item.metrics]
        ranked = [(item.name, item.metrics[metric_name]) for item in items_with_metric]
        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        return ranked

    def get_best_item(self, metric_name: str, higher_is_better: bool = True) -> tuple[str, float] | None:
        ranked = self.rank_items(metric_name, ascending=not higher_is_better)
        return ranked[0] if ranked else None

    def get_summary(self) -> dict[str, Any]:
        if not self._items:
            return {"count": 0, "metrics": []}

        metric_names = self.get_metric_names()
        summary = {
            "count": len(self._items),
            "metrics": metric_names,
            "items": {},
        }

        for item in self._items:
            summary["items"][item.name] = {
                "metrics": item.metrics,
                "metadata": item.metadata,
            }

        for metric_name in metric_names:
            values = [item.metrics[metric_name] for item in self._items if metric_name in item.metrics]
            if values:
                summary[f"{metric_name}_stats"] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": mean(values),
                    "range": max(values) - min(values),
                }
                if len(values) > 1:
                    summary[f"{metric_name}_stats"]["stdev"] = stdev(values)

        return summary

    def compute_correlation(self, metric1: str, metric2: str) -> float | None:
        pairs = [
            (item.metrics[metric1], item.metrics[metric2])
            for item in self._items
            if metric1 in item.metrics and metric2 in item.metrics
        ]
        if len(pairs) < 2:
            return None

        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]

        return self._pearson_correlation(x_vals, y_vals)

    def _pearson_correlation(self, x: list[float], y: list[float]) -> float:
        n = len(x)
        if n == 0:
            return 0.0

        mean_x = mean(x)
        mean_y = mean(y)

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
        denominator = (sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def find_outliers(self, metric_name: str, threshold: float = 2.0) -> list[tuple[str, float, str]]:
        items_with_metric = [item for item in self._items if metric_name in item.metrics]
        if len(items_with_metric) < 3:
            return []

        values = [item.metrics[metric_name] for item in items_with_metric]
        mean_val = mean(values)

        if len(values) < 2:
            return []

        try:
            std_val = stdev(values)
        except StatisticsError:
            return []

        if std_val == 0:
            return []

        outliers = []
        for item in items_with_metric:
            z_score = abs((item.metrics[metric_name] - mean_val) / std_val)
            if z_score > threshold:
                direction = "above" if item.metrics[metric_name] > mean_val else "below"
                outliers.append((item.name, item.metrics[metric_name], direction))

        return outliers

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [
                {
                    "name": item.name,
                    "metrics": item.metrics,
                    "metadata": item.metadata,
                }
                for item in self._items
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComparisonEngine:
        engine = cls()
        for item_data in data.get("items", []):
            engine.add_item(
                name=item_data["name"],
                metrics=item_data.get("metrics", {}),
                metadata=item_data.get("metadata", {}),
            )
        return engine

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> ComparisonItem:
        return self._items[index]


def compare_models(
    model_results: dict[str, dict[str, float]],
    metrics_to_compare: list[str] | None = None,
) -> ComparisonEngine:
    engine = ComparisonEngine()
    for model_name, metrics in model_results.items():
        if metrics_to_compare:
            filtered_metrics = {k: v for k, v in metrics.items() if k in metrics_to_compare}
            engine.add_item(model_name, filtered_metrics, {"type": "model"})
        else:
            engine.add_item(model_name, metrics, {"type": "model"})
    return engine


def compare_prompts(
    prompt_results: dict[str, dict[str, float]],
) -> ComparisonEngine:
    engine = ComparisonEngine()
    for prompt_name, metrics in prompt_results.items():
        engine.add_item(prompt_name, metrics, {"type": "prompt"})
    return engine


def compare_datasets(
    dataset_results: dict[str, dict[str, float]],
) -> ComparisonEngine:
    engine = ComparisonEngine()
    for dataset_name, metrics in dataset_results.items():
        engine.add_item(dataset_name, metrics, {"type": "dataset"})
    return engine
