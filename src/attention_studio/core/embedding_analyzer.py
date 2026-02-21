from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EmbeddingStats:
    mean_norm: float
    std_norm: float
    min_norm: float
    max_norm: float


class EmbeddingAnalyzer:
    def __init__(self, model_manager: any):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

    def get_embeddings(self) -> torch.Tensor:
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            return self.model.transformer.wte.weight.data
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens.weight.data
        raise ValueError("Could not find embedding layer")

    def get_embedding(self, token_id: int) -> np.ndarray:
        embeddings = self.get_embeddings()
        return embeddings[token_id].cpu().numpy()

    def compute_stats(self) -> EmbeddingStats:
        embeddings = self.get_embeddings()

        norms = torch.norm(embeddings, dim=1)

        return EmbeddingStats(
            mean_norm=float(norms.mean()),
            std_norm=float(norms.std()),
            min_norm=float(norms.min()),
            max_norm=float(norms.max()),
        )

    def compute_similarity(self, token_id_a: int, token_id_b: int) -> float:
        emb_a = self.get_embedding(token_id_a)
        emb_b = self.get_embedding(token_id_b)

        cos_sim = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        )

        return float(cos_sim)

    def find_most_similar(
        self,
        token_id: int,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        embeddings = self.get_embeddings()
        target = embeddings[token_id]

        similarities = torch.cosine_similarity(
            target.unsqueeze(0), embeddings, dim=1
        )

        top_k = min(top_k, len(similarities))
        top_indices = torch.topk(similarities, top_k).indices

        return [(idx.item(), similarities[idx].item()) for idx in top_indices]

    def compute_variance(self) -> float:
        embeddings = self.get_embeddings()
        return float(embeddings.var().item())

    def compute_principal_components(
        self,
        n_components: int = 10,
    ) -> dict[str, np.ndarray]:
        embeddings = self.get_embeddings().cpu().numpy()

        centered = embeddings - embeddings.mean(axis=0)

        u, s, vt = np.linalg.svd(centered, full_matrices=False)

        components = vt[:n_components]
        explained_variance = (s[:n_components] ** 2).sum() / (s ** 2).sum()

        return {
            "components": components,
            "singular_values": s[:n_components],
            "explained_variance": explained_variance,
        }

    def compute_token_distance_matrix(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        n = len(token_ids)
        matrix = np.zeros((n, n))

        embeddings = [self.get_embedding(tid) for tid in token_ids]

        for i in range(n):
            for j in range(n):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                matrix[i, j] = dist

        return matrix

    def find_outliers(
        self,
        threshold: float = 2.0,
    ) -> list[int]:
        embeddings = self.get_embeddings()

        norms = torch.norm(embeddings, dim=1)
        mean_norm = norms.mean()
        std_norm = norms.std()

        outlier_indices = torch.where(
            torch.abs(norms - mean_norm) > threshold * std_norm
        )[0]

        return outlier_indices.tolist()


class EmbeddingVisualizer:
    def __init__(self, analyzer: EmbeddingAnalyzer):
        self.analyzer = analyzer

    def get_similarity_matrix_data(
        self,
        token_ids: list[int],
    ) -> dict[str, any]:
        matrix = self.analyzer.compute_token_distance_matrix(token_ids)

        tokens = self.analyzer.tokenizer.convert_ids_to_tokens(token_ids)

        return {
            "matrix": matrix.tolist(),
            "tokens": tokens,
        }

    def get_pca_projection_data(
        self,
        token_ids: list[int],
        n_components: int = 2,
    ) -> dict[str, any]:
        embeddings = self.analyzer.get_embeddings()
        selected = embeddings[token_ids].cpu().numpy()

        centered = selected - selected.mean(axis=0)

        u, s, vt = np.linalg.svd(centered, full_matrices=False)

        projection = centered @ vt[:n_components].T

        tokens = self.analyzer.tokenizer.convert_ids_to_tokens(token_ids)

        return {
            "projection": projection.tolist(),
            "tokens": tokens,
            "explained_variance": float((s[:n_components] ** 2).sum() / (s ** 2).sum()),
        }
