from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch

import pytest

from attention_studio.core.attention_pattern import (
    AttentionHeadStats,
    AttentionPattern,
    AttentionPatternAnalyzer,
    AttentionPatternVisualizer,
)


class TestAttentionHeadStats:
    def test_stats_creation(self):
        stats = AttentionHeadStats(
            layer_idx=5,
            head_idx=3,
            mean_attention=0.15,
            attention_std=0.1,
            max_attention=0.8,
            entropy=2.5,
            sparsity=0.3,
        )
        assert stats.layer_idx == 5
        assert stats.head_idx == 3
        assert stats.mean_attention == 0.15


class TestAttentionPattern:
    def test_pattern_creation(self):
        pattern = AttentionPattern(
            layer_idx=2,
            head_idx=1,
            pattern_matrix=np.random.rand(10, 10),
            tokens=["tok1", "tok2"],
        )
        assert pattern.layer_idx == 2
        assert pattern.head_idx == 1
        assert pattern.pattern_matrix.shape == (10, 10)


class TestAttentionPatternAnalyzer:
    def test_analyzer_initialization(self):
        mock_manager = MagicMock()
        analyzer = AttentionPatternAnalyzer(mock_manager)
        assert analyzer.model_manager is mock_manager

    def test_find_attention_heroes(self):
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

        analyzer = AttentionPatternAnalyzer(mock_manager)

        with patch.object(analyzer, "compute_head_statistics") as mock_stats:
            mock_stats.return_value = [
                AttentionHeadStats(0, 0, 0.1, 0.1, 0.5, 3.0, 0.2),
                AttentionHeadStats(0, 1, 0.2, 0.15, 0.6, 2.5, 0.3),
                AttentionHeadStats(0, 2, 0.15, 0.1, 0.4, 2.0, 0.4),
            ]

            heroes = analyzer.find_attention_heroes(["prompt"], 0, top_k=2)
            assert len(heroes) == 2
            assert heroes[0][0] == 0
            assert heroes[0][1] > heroes[1][1]

    def test_find_diverse_attention(self):
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

        analyzer = AttentionPatternAnalyzer(mock_manager)

        with patch.object(analyzer, "compute_head_statistics") as mock_stats:
            mock_stats.return_value = [
                AttentionHeadStats(0, 0, 0.1, 0.1, 0.5, 3.0, 0.8),
                AttentionHeadStats(0, 1, 0.2, 0.15, 0.6, 2.5, 0.3),
                AttentionHeadStats(0, 2, 0.15, 0.1, 0.4, 2.0, 0.1),
            ]

            diverse = analyzer.find_diverse_attention(["prompt"], 0, top_k=2)
            assert len(diverse) == 2


class TestAttentionPatternVisualizer:
    def test_visualizer_initialization(self):
        mock_manager = MagicMock()
        analyzer = AttentionPatternAnalyzer(mock_manager)
        visualizer = AttentionPatternVisualizer(analyzer)
        assert visualizer.analyzer is analyzer

    def test_get_attention_heatmap_data(self):
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

        analyzer = AttentionPatternAnalyzer(mock_manager)
        visualizer = AttentionPatternVisualizer(analyzer)

        with patch.object(analyzer, "get_attention_patterns") as mock_patterns:
            mock_patterns.return_value = {
                (0, 0): np.random.rand(5, 5)
            }

            data = visualizer.get_attention_heatmap_data("test prompt", 0, 0)
            assert "pattern" in data

    def test_get_layer_attention_summary(self):
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

        analyzer = AttentionPatternAnalyzer(mock_manager)
        visualizer = AttentionPatternVisualizer(analyzer)

        with patch.object(analyzer, "compute_head_statistics") as mock_stats:
            mock_stats.return_value = [
                AttentionHeadStats(0, 0, 0.1, 0.1, 0.5, 3.0, 0.2),
                AttentionHeadStats(0, 1, 0.2, 0.15, 0.6, 2.5, 0.3),
            ]

            summary = visualizer.get_layer_attention_summary(["prompt"], 0)
            assert summary["layer_idx"] == 0
            assert summary["num_heads"] == 2
