from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    name: str
    model_name: str
    dictionary_size: int
    top_k: int
    learning_rate: float
    batch_size: int
    epochs: int
    layer_indices: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class MetricRecord:
    name: str
    value: float
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    phase: str = "train"


@dataclass
class Experiment:
    id: str
    config: ExperimentConfig
    metrics: list[MetricRecord] = field(default_factory=list)
    status: str = "running"
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None


class ExperimentTracker:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: dict[str, Experiment] = {}
        self._load_experiments()

    def _load_experiments(self) -> None:
        for exp_file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(exp_file.read_text())
                config = ExperimentConfig(**data["config"])
                metrics = [MetricRecord(**m) for m in data.get("metrics", [])]
                exp = Experiment(
                    id=data["id"],
                    config=config,
                    metrics=metrics,
                    status=data.get("status", "unknown"),
                    start_time=data.get("start_time", ""),
                    end_time=data.get("end_time"),
                )
                self.experiments[exp.id] = exp
            except Exception:
                pass

    def create_experiment(
        self,
        name: str,
        model_name: str,
        dictionary_size: int,
        top_k: int,
        learning_rate: float,
        batch_size: int,
        epochs: int,
        layer_indices: Optional[list[int]] = None,
        tags: Optional[list[str]] = None,
        notes: str = "",
    ) -> Experiment:
        exp_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config = ExperimentConfig(
            name=name,
            model_name=model_name,
            dictionary_size=dictionary_size,
            top_k=top_k,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            layer_indices=layer_indices or [],
            tags=tags or [],
            notes=notes,
        )
        exp = Experiment(id=exp_id, config=config)
        self.experiments[exp_id] = exp
        self._save_experiment(exp)
        return exp

    def log_metric(
        self,
        exp_id: str,
        name: str,
        value: float,
        step: int,
        phase: str = "train",
    ) -> None:
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")

        metric = MetricRecord(name=name, value=value, step=step, phase=phase)
        self.experiments[exp_id].metrics.append(metric)

    def log_metrics(
        self,
        exp_id: str,
        metrics: dict[str, float],
        step: int,
        phase: str = "train",
    ) -> None:
        for name, value in metrics.items():
            self.log_metric(exp_id, name, value, step, phase)

    def complete_experiment(self, exp_id: str, status: str = "completed") -> None:
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")

        exp = self.experiments[exp_id]
        exp.status = status
        exp.end_time = datetime.now().isoformat()
        self._save_experiment(exp)

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        return self.experiments.get(exp_id)

    def list_experiments(
        self,
        status: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[Experiment]:
        results = []
        for exp in self.experiments.values():
            if status and exp.status != status:
                continue
            if tags:
                if not any(tag in exp.config.tags for tag in tags):
                    continue
            results.append(exp)
        return results

    def compare_experiments(
        self,
        exp_ids: list[str],
        metric_name: str,
    ) -> dict[str, list[tuple[int, float]]]:
        results = {}
        for exp_id in exp_ids:
            if exp_id not in self.experiments:
                continue
            exp = self.experiments[exp_id]
            values = [(m.step, m.value) for m in exp.metrics if m.name == metric_name]
            results[exp_id] = values
        return results

    def get_best_epoch(
        self,
        exp_id: str,
        metric_name: str,
        higher_is_better: bool = True,
    ) -> Optional[tuple[int, float]]:
        if exp_id not in self.experiments:
            return None

        exp = self.experiments[exp_id]
        values = [(m.step, m.value) for m in exp.metrics if m.name == metric_name]

        if not values:
            return None

        if higher_is_better:
            return max(values, key=lambda x: x[1])
        else:
            return min(values, key=lambda x: x[1])

    def get_metrics_summary(self, exp_id: str) -> dict[str, Any]:
        if exp_id not in self.experiments:
            return {}

        exp = self.experiments[exp_id]
        metrics_by_name: dict[str, list[float]] = {}

        for m in exp.metrics:
            if m.name not in metrics_by_name:
                metrics_by_name[m.name] = []
            metrics_by_name[m.name].append(m.value)

        summary = {}
        for name, values in metrics_by_name.items():
            summary[name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "last": values[-1] if values else None,
            }

        return summary

    def _save_experiment(self, exp: Experiment) -> None:
        data = {
            "id": exp.id,
            "config": {
                "name": exp.config.name,
                "model_name": exp.config.model_name,
                "dictionary_size": exp.config.dictionary_size,
                "top_k": exp.config.top_k,
                "learning_rate": exp.config.learning_rate,
                "batch_size": exp.config.batch_size,
                "epochs": exp.config.epochs,
                "layer_indices": exp.config.layer_indices,
                "tags": exp.config.tags,
                "notes": exp.config.notes,
            },
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "step": m.step,
                    "timestamp": m.timestamp,
                    "phase": m.phase,
                }
                for m in exp.metrics
            ],
            "status": exp.status,
            "start_time": exp.start_time,
            "end_time": exp.end_time,
        }

        exp_file = self.storage_dir / f"{exp.id}.json"
        exp_file.write_text(json.dumps(data, indent=2))

    def export_to_csv(self, exp_id: str, output_path: Path) -> None:
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")

        exp = self.experiments[exp_id]
        lines = ["step,phase,name,value,timestamp"]

        for m in exp.metrics:
            lines.append(f"{m.step},{m.phase},{m.name},{m.value},{m.timestamp}")

        output_path.write_text("\n".join(lines))

    def delete_experiment(self, exp_id: str) -> None:
        if exp_id in self.experiments:
            del self.experiments[exp_id]
            exp_file = self.storage_dir / f"{exp_id}.json"
            if exp_file.exists():
                exp_file.unlink()


class HyperparameterSearch:
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker

    def random_search(
        self,
        base_config: dict[str, Any],
        param_ranges: dict[str, tuple[Any, Any]],
        num_trials: int,
    ) -> list[Experiment]:
        import random

        experiments = []
        for i in range(num_trials):
            config = base_config.copy()
            for param, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    config[param] = random.randint(min_val, max_val)
                elif isinstance(min_val, float) and isinstance(max_val, float):
                    config[param] = random.uniform(min_val, max_val)
                else:
                    config[param] = random.choice([min_val, max_val])

            exp = self.tracker.create_experiment(
                name=f"trial_{i}",
                model_name=config.get("model_name", "gpt2"),
                dictionary_size=config.get("dictionary_size", 32768),
                top_k=config.get("top_k", 128),
                learning_rate=config.get("learning_rate", 1e-4),
                batch_size=config.get("batch_size", 4),
                epochs=config.get("epochs", 10),
            )
            experiments.append(exp)

        return experiments

    def grid_search(
        self,
        base_config: dict[str, Any],
        param_grid: dict[str, list[Any]],
    ) -> list[Experiment]:
        import itertools

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        experiments = []
        for values in itertools.product(*param_values):
            config = base_config.copy()
            for name, value in zip(param_names, values):
                config[name] = value

            exp = self.tracker.create_experiment(
                name=f"grid_{'_'.join(str(v) for v in values)}",
                model_name=config.get("model_name", "gpt2"),
                dictionary_size=config.get("dictionary_size", 32768),
                top_k=config.get("top_k", 128),
                learning_rate=config.get("learning_rate", 1e-4),
                batch_size=config.get("batch_size", 4),
                epochs=config.get("epochs", 10),
            )
            experiments.append(exp)

        return experiments


class TrainingProgressTracker:
    def __init__(self, experiment_id: str, tracker: ExperimentTracker):
        self.experiment_id = experiment_id
        self.tracker = tracker
        self.current_epoch = 0
        self.current_step = 0

    def update_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def update_step(self, step: int) -> None:
        self.current_step = step

    def log_epoch_metrics(self, metrics: dict[str, float], phase: str = "train") -> None:
        for name, value in metrics.items():
            self.tracker.log_metric(
                self.experiment_id,
                name,
                value,
                self.current_epoch,
                phase,
            )

    def log_step_metrics(self, metrics: dict[str, float], phase: str = "train") -> None:
        for name, value in metrics.items():
            self.tracker.log_metric(
                self.experiment_id,
                name,
                value,
                self.current_step,
                phase,
            )
