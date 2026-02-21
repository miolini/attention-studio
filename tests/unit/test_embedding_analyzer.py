from __future__ import annotations

import torch
from unittest.mock import MagicMock

import pytest

from attention_studio.core.embedding_analyzer import (
    EmbeddingStats,
    EmbeddingAnalyzer,
    EmbeddingVisualizer,
)


class TestEmbeddingStats:
    def test_stats_creation(self):
        stats = EmbeddingStats(
            mean_norm=1.0,
            std_norm=0.2,
            min_norm=0.5,
            max_norm=1.5,
        )
        assert stats.mean_norm == 1.0
        assert stats.min_norm == 0.5


class TestEmbeddingAnalyzer:
    def test_analyzer_initialization(self):
        mock_manager = MagicMock()
        analyzer = EmbeddingAnalyzer(mock_manager)
        assert analyzer.model_manager is mock_manager

    def test_get_embedding(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()

        mock_manager.model = mock_model

        mock_embeddings = torch.randn(50257, 768)
        mock_model.transformer.wte.weight.data = mock_embeddings

        analyzer = EmbeddingAnalyzer(mock_manager)

        emb = analyzer.get_embedding(0)
        assert emb.shape == (768,)

    def test_compute_stats(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()

        mock_manager.model = mock_model

        mock_embeddings = torch.randn(50257, 768)
        mock_model.transformer.wte.weight.data = mock_embeddings

        analyzer = EmbeddingAnalyzer(mock_manager)

        stats = analyzer.compute_stats()
        assert isinstance(stats, EmbeddingStats)

    def test_compute_similarity(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()

        mock_manager.model = mock_model

        mock_embeddings = torch.randn(50257, 768)
        mock_model.transformer.wte.weight.data = mock_embeddings

        analyzer = EmbeddingAnalyzer(mock_manager)

        sim = analyzer.compute_similarity(0, 1)
        assert -1.0 <= sim <= 1.0

    def test_find_most_similar(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()

        mock_manager.model = mock_model

        mock_embeddings = torch.randn(50257, 768)
        mock_model.transformer.wte.weight.data = mock_embeddings

        analyzer = EmbeddingAnalyzer(mock_manager)

        similar = analyzer.find_most_similar(0, top_k=5)
        assert len(similar) == 5


class TestEmbeddingVisualizer:
    def test_visualizer_initialization(self):
        mock_manager = MagicMock()
        analyzer = EmbeddingAnalyzer(mock_manager)
        visualizer = EmbeddingVisualizer(analyzer)
        assert visualizer.analyzer is analyzer

    def test_get_similarity_matrix_data(self):
        mock_manager = MagicMock()
        mock_model = MagicMock()

        mock_manager.model = mock_model

        mock_embeddings = torch.randn(50257, 768)
        mock_model.transformer.wte.weight.data = mock_embeddings

        analyzer = EmbeddingAnalyzer(mock_manager)
        visualizer = EmbeddingVisualizer(analyzer)

        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_ids_to_tokens.return_value = ["tok1", "tok2", "tok3"]
        analyzer.tokenizer = mock_tokenizer

        data = visualizer.get_similarity_matrix_data([0, 1, 2])
        assert "matrix" in data
        assert "tokens" in data
