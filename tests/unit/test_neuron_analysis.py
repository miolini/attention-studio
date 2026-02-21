from __future__ import annotations

import torch
from unittest.mock import MagicMock, patch

import pytest

from attention_studio.core.neuron_analysis import (
    NeuronStats,
    NeuronProfile,
    NeuronAnalyzer,
    NeuronProfiler,
)


class TestNeuronStats:
    def test_stats_creation(self):
        stats = NeuronStats(
            layer_idx=5,
            neuron_idx=10,
            mean_activation=0.5,
            std_activation=0.3,
            max_activation=1.2,
            zero_fraction=0.1,
            dead_neuron=False,
        )
        assert stats.layer_idx == 5
        assert stats.neuron_idx == 10
        assert stats.mean_activation == 0.5
        assert stats.dead_neuron is False


class TestNeuronProfile:
    def test_profile_creation(self):
        stats = NeuronStats(
            layer_idx=3,
            neuron_idx=7,
            mean_activation=0.2,
            std_activation=0.1,
            max_activation=0.8,
            zero_fraction=0.05,
            dead_neuron=False,
        )
        profile = NeuronProfile(
            neuron_id=(3, 7),
            stats=stats,
        )
        assert profile.neuron_id == (3, 7)
        assert profile.stats.neuron_idx == 7


class TestNeuronAnalyzer:
    def test_analyzer_initialization(self):
        mock_manager = MagicMock()
        analyzer = NeuronAnalyzer(mock_manager)
        assert analyzer.model_manager is mock_manager

    def test_find_dead_neurons(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"
        mock_model.transformer.h = [MagicMock() for _ in range(12)]

        analyzer = NeuronAnalyzer(mock_manager)

        with patch.object(analyzer, "compute_neuron_statistics") as mock_stats:
            mock_stats.return_value = [
                NeuronStats(0, 0, 0.0, 0.0, 0.0, 1.0, True),
                NeuronStats(0, 1, 0.1, 0.1, 0.5, 0.1, False),
            ]

            dead = analyzer.find_dead_neurons(["prompt"], 0)
            assert 0 in dead
            assert 1 not in dead

    def test_find_important_neurons(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"
        mock_model.transformer.h = [MagicMock() for _ in range(12)]

        analyzer = NeuronAnalyzer(mock_manager)

        with patch.object(analyzer, "compute_neuron_statistics") as mock_stats:
            mock_stats.return_value = [
                NeuronStats(0, 0, 0.5, 0.3, 1.0, 0.1, False),
                NeuronStats(0, 1, 0.2, 0.8, 1.5, 0.05, False),
                NeuronStats(0, 2, 0.1, 0.1, 0.3, 0.2, False),
            ]

            important = analyzer.find_important_neurons(["prompt"], 0, top_k=2)
            assert len(important) == 2
            assert important[0][0] == 1
            assert important[0][1] > important[1][1]


class TestNeuronProfiler:
    def test_profiler_initialization(self):
        mock_manager = MagicMock()
        profiler = NeuronProfiler(mock_manager)
        assert isinstance(profiler.analyzer, NeuronAnalyzer)

    def test_profile_neurons(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"
        mock_model.transformer.h = [MagicMock() for _ in range(12)]

        profiler = NeuronProfiler(mock_manager)

        with patch.object(profiler.analyzer, "compute_neuron_statistics") as mock_stats:
            mock_stats.return_value = [
                NeuronStats(0, i, 0.1, 0.1, 0.5, 0.1, False)
                for i in range(5)
            ]

            profiles = profiler.profile_neurons(["test"], layer_indices=[0])

            assert 0 in profiles
            assert len(profiles[0]) == 5

    def test_find_maximally_activating_prompts(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_manager.model = mock_model
        mock_manager.tokenizer = mock_tokenizer

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))
        }
        mock_model.device = "cpu"
        mock_model.transformer.h = [MagicMock() for _ in range(12)]

        profiler = NeuronProfiler(mock_manager)

        with patch.object(profiler.analyzer, "get_neuron_activations") as mock_get:
            mock_tensor = torch.randn(1, 10, 768)
            mock_get.return_value = mock_tensor

            results = profiler.find_maximally_activating_prompts(
                ["prompt1", "prompt2", "prompt3"], 0, 0
            )

            assert len(results) <= 3
