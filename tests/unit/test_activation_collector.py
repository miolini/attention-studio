from __future__ import annotations

import torch
from unittest.mock import MagicMock

import pytest

from attention_studio.core.activation_collector import (
    ActivationSnapshot,
    ActivationCollection,
    ActivationCollector,
    ActivationComparator,
)


class TestActivationSnapshot:
    def test_snapshot_creation(self):
        snapshot = ActivationSnapshot(
            prompt="test prompt",
            layer_activations={0: torch.randn(1, 10, 768)},
        )
        assert snapshot.prompt == "test prompt"
        assert 0 in snapshot.layer_activations


class TestActivationCollection:
    def test_collection_creation(self):
        collection = ActivationCollection()
        assert len(collection.snapshots) == 0


class TestActivationCollector:
    def test_collector_initialization(self):
        mock_manager = MagicMock()
        collector = ActivationCollector(mock_manager)
        assert collector.model_manager is mock_manager

    def test_collector_with_layer_indices(self):
        mock_manager = MagicMock()
        collector = ActivationCollector(mock_manager, layer_indices=[0, 1, 2])
        assert collector.layer_indices == [0, 1, 2]

    def test_collect(self):
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

        mock_output = MagicMock()
        mock_output.hidden_states = None
        mock_model.return_value = mock_output

        collector = ActivationCollector(mock_manager, layer_indices=[0])

        collector._current_activations = {0: torch.randn(1, 10, 768)}

        snapshot = collector.collect("test prompt")
        assert isinstance(snapshot, ActivationSnapshot)

    def test_get_snapshot(self):
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

        collector = ActivationCollector(mock_manager)
        collector._current_activations = {0: torch.randn(1, 10, 768)}

        collector.collect("prompt1")
        collector.collect("prompt2")

        snapshot = collector.get_snapshot(1)
        assert snapshot is not None

    def test_clear(self):
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

        collector = ActivationCollector(mock_manager)
        collector._current_activations = {0: torch.randn(1, 10, 768)}

        collector.collect("prompt1")
        assert len(collector) == 1

        collector.clear()
        assert len(collector) == 0


class TestActivationComparator:
    def test_comparator_initialization(self):
        mock_manager = MagicMock()
        collector = ActivationCollector(mock_manager)
        comparator = ActivationComparator(collector)
        assert comparator.collector is collector

    def test_compute_layer_similarity(self):
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

        collector = ActivationCollector(mock_manager)

        collector._current_activations = {0: torch.randn(1, 10, 768)}
        collector.collect("prompt1")
        collector._current_activations = {0: torch.randn(1, 10, 768)}
        collector.collect("prompt2")

        comparator = ActivationComparator(collector)

        sim = comparator.compute_layer_similarity(0, 1, 0)
        assert isinstance(sim, float)

    def test_find_most_similar_pairs(self):
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

        collector = ActivationCollector(mock_manager)

        for i in range(3):
            collector._current_activations = {0: torch.randn(1, 10, 768)}
            collector.collect(f"prompt{i}")

        comparator = ActivationComparator(collector)

        pairs = comparator.find_most_similar_pairs(0, top_k=2)
        assert len(pairs) <= 2
