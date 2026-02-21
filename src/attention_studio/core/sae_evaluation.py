from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SAEEvaluationResult:
    reconstruction_error: float
    sparsity: float
    l0_norm: float
    l1_norm: float
    feature_utilization: float
    dead_features: float
    explained_variance: float
    downstream_accuracy: Optional[float] = None
    metadata: Optional[dict] = None


class SAEEvaluationSuite:
    def __init__(self, transcoder: Any, model: Optional[torch.nn.Module] = None):
        self.transcoder = transcoder
        self.model = model

    def compute_reconstruction_error(
        self,
        hidden_states: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            reconstructed, features = self.transcoder(hidden_states)
            mse = F.mse_loss(reconstructed, hidden_states).item()
        return mse

    def compute_feature_sparsity(
        self,
        hidden_states: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        active_features = (features > 0).float()
        sparsity_per_token = 1.0 - active_features.mean(dim=-1).mean(dim=0)
        overall_sparsity = sparsity_per_token.mean().item()

        l0 = (features != 0).sum(dim=-1).float().mean().item()
        l1 = features.abs().sum(dim=-1).mean().item()

        return {
            "sparsity_ratio": overall_sparsity,
            "l0_norm": l0,
            "l1_norm": l1,
        }

    def compute_feature_utilization(
        self,
        hidden_states: torch.Tensor,
        threshold: float = 1e-6,
    ) -> dict[str, float]:
        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        mean_activations = features.abs().mean(dim=(0, 1))
        active_features = (mean_activations > threshold).sum().item()
        total_features = len(mean_activations)

        utilization = active_features / total_features if total_features > 0 else 0.0
        dead_features = 1.0 - utilization

        return {
            "utilization": utilization,
            "dead_features": dead_features,
            "active_count": active_features,
            "total_features": total_features,
        }

    def compute_explained_variance(
        self,
        hidden_states: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            reconstructed, _ = self.transcoder(hidden_states)

        original_var = hidden_states.var(dim=0).sum()
        residual_var = (hidden_states - reconstructed).var(dim=0).sum()

        if original_var == 0:
            return 0.0

        explained = 1.0 - (residual_var / original_var)
        return explained.item()

    def compute_ceiling_analysis(
        self,
        hidden_states: torch.Tensor,
        feature_counts: list[int],
    ) -> dict[int, float]:
        results = {}
        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

            for k in feature_counts:
                top_k_features = features.clone()
                values, indices = torch.topk(top_k_features, k, dim=-1)
                mask = torch.zeros_like(top_k_features)
                mask.scatter_(-1, indices, values)
                top_k_features = mask

                reconstructed_k = self.transcoder.decoder(top_k_features.view(-1, features.shape[-1]))
                reconstructed_k = reconstructed_k.view(hidden_states.shape)

                mse_k = F.mse_loss(reconstructed_k, hidden_states).item()
                results[k] = mse_k

        return results

    def compute_feature_clustering(
        self,
        hidden_states: torch.Tensor,
        n_clusters: int = 10,
    ) -> dict[str, Any]:
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except ImportError:
            return {"error": "sklearn required"}

        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        features_flat = features.view(-1, features.shape[-1]).cpu().numpy()

        if features_flat.shape[0] < n_clusters:
            return {"error": "Not enough samples for clustering"}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_flat)

        pca = PCA(n_components=min(2, features_flat.shape[0], features_flat.shape[1]))
        coords = pca.fit_transform(features_flat)

        return {
            "labels": labels.tolist(),
            "centers": kmeans.cluster_centers_.tolist(),
            "pca_coords": coords.tolist() if len(coords) < 1000 else coords[:1000].tolist(),
            "inertia": kmeans.inertia_,
        }

    def compute_cosine_similarity_matrix(
        self,
        hidden_states: torch.Tensor,
    ) -> np.ndarray:
        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        features_flat = features.view(-1, features.shape[-1])
        features_flat = F.normalize(features_flat, dim=-1)

        similarity = torch.mm(features_flat, features_flat.t())
        return similarity.cpu().numpy()

    def compute_neuron_statistics(
        self,
        hidden_states: torch.Tensor,
    ) -> dict[str, Any]:
        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        stats = {
            "mean": features.mean(dim=(0, 1)).cpu().tolist(),
            "std": features.std(dim=(0, 1)).cpu().tolist(),
            "max": features.max(dim=1)[0].max(dim=0)[0].cpu().tolist(),
            "min": features.min(dim=1)[0].min(dim=0)[0].cpu().tolist(),
            "median": features.median(dim=1)[0].median(dim=0)[0].cpu().tolist(),
        }

        return stats

    def evaluate(
        self,
        hidden_states: torch.Tensor,
    ) -> SAEEvaluationResult:
        reconstruction_error = self.compute_reconstruction_error(hidden_states)

        sparsity_dict = self.compute_feature_sparsity(hidden_states)
        sparsity = sparsity_dict["sparsity_ratio"]
        l0_norm = sparsity_dict["l0_norm"]
        l1_norm = sparsity_dict["l1_norm"]

        utilization_dict = self.compute_feature_utilization(hidden_states)
        feature_utilization = utilization_dict["utilization"]
        dead_features = utilization_dict["dead_features"]

        explained_variance = self.compute_explained_variance(hidden_states)

        return SAEEvaluationResult(
            reconstruction_error=reconstruction_error,
            sparsity=sparsity,
            l0_norm=l0_norm,
            l1_norm=l1_norm,
            feature_utilization=feature_utilization,
            dead_features=dead_features,
            explained_variance=explained_variance,
            metadata={
                "active_count": utilization_dict["active_count"],
                "total_features": utilization_dict["total_features"],
            },
        )


class SAEComparison:
    @staticmethod
    def compare_reconstruction(
        results1: SAEEvaluationResult,
        results2: SAEEvaluationResult,
    ) -> dict[str, float]:
        return {
            "reconstruction_improvement": results1.reconstruction_error - results2.reconstruction_error,
            "sparsity_improvement": results2.sparsity - results1.sparsity,
            "utilization_improvement": results2.feature_utilization - results1.feature_utilization,
            "dead_features_improvement": results1.dead_features - results2.dead_features,
        }

    @staticmethod
    def rank_saes(
        results: list[SAEEvaluationResult],
        weights: Optional[dict[str, float]] = None,
    ) -> list[tuple[int, float]]:
        if weights is None:
            weights = {
                "reconstruction": -1.0,
                "sparsity": 0.5,
                "utilization": 0.3,
                "dead_features": -0.2,
            }

        scores = []
        for i, result in enumerate(results):
            score = 0.0
            score += weights.get("reconstruction", 0) * result.reconstruction_error
            score += weights.get("sparsity", 0) * result.sparsity
            score += weights.get("utilization", 0) * result.feature_utilization
            score += weights.get("dead_features", 0) * result.dead_features
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class SAEAnalyzer:
    def __init__(self, transcoder: Any):
        self.transcoder = transcoder
        self.evaluation_suite = SAEEvaluationSuite(transcoder)

    def analyze_feature_neurons(
        self,
        hidden_states: torch.Tensor,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        mean_activations = features.abs().mean(dim=(0, 1))
        top_indices = torch.topk(mean_activations, min(top_k, len(mean_activations)))

        decoder_weights = self.transcoder.decoder.weight

        results = []
        for idx, val in zip(top_indices.indices, top_indices.values, strict=True):
            idx_val = idx.item()
            neuron_weights = decoder_weights[:, idx_val].cpu().detach()

            results.append({
                "feature_idx": idx_val,
                "mean_activation": val.item(),
                "neuron_weight_norm": torch.norm(neuron_weights).item(),
                "weight_std": neuron_weights.std().item(),
                "top_weight": neuron_weights.max().item(),
                "bottom_weight": neuron_weights.min().item(),
            })

        return results

    def detect_anomalies(
        self,
        hidden_states: torch.Tensor,
        z_threshold: float = 3.0,
    ) -> dict[str, Any]:
        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        mean_activations = features.abs().mean(dim=(0, 1))
        overall_mean = mean_activations.mean().item()
        overall_std = mean_activations.std().item()

        z_scores = (mean_activations - overall_mean) / (overall_std + 1e-8)
        anomalies = torch.where(z_scores > z_threshold)[0]

        return {
            "anomaly_indices": anomalies.cpu().tolist(),
            "anomaly_count": len(anomalies),
            "z_scores": z_scores.cpu().tolist(),
            "threshold": z_threshold,
        }

    def compute_feature_diversity(
        self,
        hidden_states: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            _, features = self.transcoder(hidden_states)

        features_binary = (features > 0).float()
        unique_patterns = features_binary.unique(dim=0)

        diversity = len(unique_patterns) / features.shape[0]

        return {
            "unique_activation_patterns": len(unique_patterns),
            "total_activations": features.shape[0],
            "diversity_ratio": diversity,
        }
