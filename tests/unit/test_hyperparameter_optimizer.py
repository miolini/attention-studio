from __future__ import annotations

import pytest

from attention_studio.core.hyperparameter_optimizer import (
    HyperparameterConfig,
    TrialResult,
    HyperparameterOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
)


class TestHyperparameterConfig:
    def test_config_creation(self):
        config = HyperparameterConfig(
            name="learning_rate",
            values=[0.001, 0.01, 0.1],
            type="categorical",
        )
        assert config.name == "learning_rate"
        assert len(config.values) == 3


class TestTrialResult:
    def test_result_creation(self):
        result = TrialResult(
            trial_id=0,
            params={"lr": 0.01},
            score=0.95,
        )
        assert result.trial_id == 0
        assert result.score == 0.95


class TestHyperparameterOptimizer:
    def test_optimizer_creation(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = HyperparameterOptimizer(objective, maximize=True)
        assert optimizer.maximize is True

    def test_suggest_random(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = HyperparameterOptimizer(objective, maximize=True)
        config = HyperparameterConfig("lr", [0.001, 0.01, 0.1])
        params = optimizer.suggest_random([config])
        assert params["lr"] in [0.001, 0.01, 0.1]

    def test_run_trial(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = HyperparameterOptimizer(objective, maximize=True)
        result = optimizer.run_trial({"lr": 0.01})
        assert result.score == 0.1
        assert result.trial_id == 0

    def test_get_best_params(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = HyperparameterOptimizer(objective, maximize=True)
        optimizer.run_trial({"lr": 0.01})
        optimizer.run_trial({"lr": 0.1})
        best = optimizer.get_best_params()
        assert best == {"lr": 0.1}

    def test_get_top_k_trials(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = HyperparameterOptimizer(objective, maximize=True)
        optimizer.run_trial({"lr": 0.01})
        optimizer.run_trial({"lr": 0.05})
        optimizer.run_trial({"lr": 0.1})
        top = optimizer.get_top_k_trials(2)
        assert len(top) == 2


class TestGridSearchOptimizer:
    def test_grid_search_creation(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = GridSearchOptimizer(objective, maximize=True)
        assert isinstance(optimizer, HyperparameterOptimizer)

    def test_grid_search_suggest(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = GridSearchOptimizer(objective, maximize=True)
        config = HyperparameterConfig("lr", [0.001, 0.01, 0.1])
        optimizer.set_search_space([config])
        params = optimizer.suggest()
        assert params["lr"] in [0.001, 0.01, 0.1]


class TestRandomSearchOptimizer:
    def test_random_search_creation(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = RandomSearchOptimizer(objective, maximize=True, n_iter=5)
        assert optimizer.n_iter == 5

    def test_random_search_suggest(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = RandomSearchOptimizer(objective, maximize=True, n_iter=5)
        config = HyperparameterConfig("lr", [0.001, 0.01, 0.1])
        optimizer.set_search_space([config])
        params = optimizer.suggest()
        assert params["lr"] in [0.001, 0.01, 0.1]


class TestBayesianOptimizer:
    def test_bayesian_creation(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = BayesianOptimizer(objective, maximize=True, n_iter=10)
        assert optimizer.n_iter == 10

    def test_bayesian_suggest(self):
        def objective(params):
            return params["lr"] * 10

        optimizer = BayesianOptimizer(objective, maximize=True, n_iter=10)
        config = HyperparameterConfig("lr", [0.001, 0.01, 0.1], type="continuous")
        optimizer.set_search_space([config])
        optimizer.run_trial({"lr": 0.01})
        optimizer.run_trial({"lr": 0.05})
        optimizer.run_trial({"lr": 0.1})
        params = optimizer.suggest()
        assert "lr" in params
