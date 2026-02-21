from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HyperparameterConfig:
    name: str
    values: list[Any]
    type: str = "categorical"


@dataclass
class TrialResult:
    trial_id: int
    params: dict[str, Any]
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "completed"


class HyperparameterOptimizer:
    def __init__(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        maximize: bool = True,
    ):
        self.objective_fn = objective_fn
        self.maximize = maximize
        self.trials: list[TrialResult] = []
        self.best_trial: TrialResult | None = None

    def suggest_random(self, configs: list[HyperparameterConfig]) -> dict[str, Any]:
        params = {}
        for config in configs:
            params[config.name] = random.choice(config.values)
        return params

    def suggest_grid(self, configs: list[HyperparameterConfig]) -> dict[str, Any]:
        params = {}
        for config in configs:
            index = len(self.trials) % len(config.values)
            params[config.name] = config.values[index]
        return params

    def run_trial(self, params: dict[str, Any]) -> TrialResult:
        trial_id = len(self.trials)
        try:
            score = self.objective_fn(params)
            status = "completed"
        except Exception:
            score = float("-inf") if self.maximize else float("inf")
            status = "failed"

        result = TrialResult(
            trial_id=trial_id,
            params=params,
            score=score,
            status=status,
        )
        self.trials.append(result)

        if self.best_trial is None or (
            self.maximize and score > self.best_trial.score
        ) or (not self.maximize and score < self.best_trial.score):
            self.best_trial = result

        return result

    def get_best_params(self) -> dict[str, Any] | None:
        if self.best_trial:
            return self.best_trial.params
        return None

    def get_best_score(self) -> float | None:
        if self.best_trial:
            return self.best_trial.score
        return None

    def get_trials(self) -> list[TrialResult]:
        return list(self.trials)

    def get_top_k_trials(self, k: int = 10) -> list[TrialResult]:
        sorted_trials = sorted(
            self.trials,
            key=lambda t: t.score,
            reverse=self.maximize,
        )
        return sorted_trials[:k]


class GridSearchOptimizer(HyperparameterOptimizer):
    def __init__(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        maximize: bool = True,
    ):
        super().__init__(objective_fn, maximize)
        self.configs: list[HyperparameterConfig] = []
        self.grid_indices: list[int] = []

    def set_search_space(self, configs: list[HyperparameterConfig]) -> None:
        self.configs = configs
        total_combinations = 1
        for config in configs:
            total_combinations *= len(config.values)
        self.grid_indices = list(range(total_combinations))
        random.shuffle(self.grid_indices)

    def suggest(self) -> dict[str, Any] | None:
        if not self.grid_indices:
            return None

        index = self.grid_indices.pop(0)

        params = {}
        for config in self.configs:
            param_index = index % len(config.values)
            params[config.name] = config.values[param_index]
            index //= len(config.values)

        return params


class RandomSearchOptimizer(HyperparameterOptimizer):
    def __init__(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        maximize: bool = True,
        n_iter: int = 100,
    ):
        super().__init__(objective_fn, maximize)
        self.n_iter = n_iter
        self.current_iter = 0
        self.configs: list[HyperparameterConfig] = []

    def set_search_space(self, configs: list[HyperparameterConfig]) -> None:
        self.configs = configs

    def suggest(self) -> dict[str, Any] | None:
        if self.current_iter >= self.n_iter:
            return None

        self.current_iter += 1
        return self.suggest_random(self.configs)


class BayesianOptimizer(HyperparameterOptimizer):
    def __init__(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        maximize: bool = True,
        n_iter: int = 50,
    ):
        super().__init__(objective_fn, maximize)
        self.n_iter = n_iter
        self.current_iter = 0
        self.configs: list[HyperparameterConfig] = []
        self._initialize_bounds()

    def _initialize_bounds(self) -> None:
        self.lower_bounds: dict[str, float] = {}
        self.upper_bounds: dict[str, float] = {}

    def set_search_space(self, configs: list[HyperparameterConfig]) -> None:
        self.configs = configs
        for config in configs:
            if config.type == "continuous":
                self.lower_bounds[config.name] = float(min(config.values))
                self.upper_bounds[config.name] = float(max(config.values))

    def suggest(self) -> dict[str, Any] | None:
        if self.current_iter >= self.n_iter:
            return None

        if len(self.trials) < 3:
            return self.suggest_random(self.configs)

        self.current_iter += 1

        params = {}
        for config in self.configs:
            if config.type == "continuous":
                best_trial = max(
                    self.trials[-3:],
                    key=lambda t: t.score,
                )
                mu = best_trial.params.get(config.name, 0.0)
                sigma = (self.upper_bounds[config.name] - self.lower_bounds[config.name]) / 10
                value = np.random.normal(mu, sigma)
                value = float(np.clip(value, self.lower_bounds[config.name], self.upper_bounds[config.name]))
                params[config.name] = value
            else:
                params[config.name] = random.choice(config.values)

        return params
