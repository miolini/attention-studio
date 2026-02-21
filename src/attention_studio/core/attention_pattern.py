from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class AttentionHeadStats:
    layer_idx: int
    head_idx: int
    mean_attention: float
    attention_std: float
    max_attention: float
    entropy: float
    sparsity: float


@dataclass
class AttentionPattern:
    layer_idx: int
    head_idx: int
    pattern_matrix: np.ndarray
    tokens: list[str]


class AttentionPatternAnalyzer:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

    def get_attention_patterns(
        self,
        prompt: str,
        layer_idx: int | None = None,
        head_idx: int | None = None,
    ) -> dict[tuple[int, int], np.ndarray]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs.get("attention_mask", None)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        _ = input_ids.shape[1]

        attention_patterns = {}

        def create_hook(lyr_idx, h_idx):
            def hook(module, inp, output):
                attn_weights = output[0]
                attn = attn_weights[0, h_idx].detach().cpu().numpy()
                attention_patterns[(lyr_idx, h_idx)] = attn
            return hook

        handles = []
        num_layers = len(self.model.transformer.h)

        for lyr in range(num_layers):
            if layer_idx is not None and lyr != layer_idx:
                continue

            attn_layer = self.model.transformer.h[lyr].attn

            if head_idx is not None:
                handles.append(attn_layer.register_forward_hook(create_hook(lyr, head_idx)))
            else:
                num_heads = attn_layer.head_dim
                for h in range(num_heads):
                    handles.append(attn_layer.register_forward_hook(create_hook(lyr, h)))

        with torch.no_grad():
            _ = self.model(input_ids, attention_mask=attention_mask)

        for handle in handles:
            handle.remove()

        return attention_patterns

    def compute_head_statistics(
        self,
        prompts: list[str],
        layer_idx: int | None = None,
    ) -> list[AttentionHeadStats]:
        all_patterns = {}

        for prompt in prompts:
            patterns = self.get_attention_patterns(prompt, layer_idx=layer_idx)
            for key, val in patterns.items():
                if key not in all_patterns:
                    all_patterns[key] = []
                all_patterns[key].append(val)

        stats_list = []
        for (lyr, h), patterns in all_patterns.items():
            stacked = np.stack(patterns)
            mean_attn = float(stacked.mean())
            attn_std = float(stacked.std())
            max_attn = float(stacked.max())

            entropy = 0.0
            for p in patterns:
                p = np.clip(p, 1e-10, 1.0)
                entropy -= (p * np.log(p)).sum() / p.shape[0]
            entropy /= len(patterns)

            sparsity = float((np.abs(stacked) < 0.01).mean())

            stats = AttentionHeadStats(
                layer_idx=lyr,
                head_idx=h,
                mean_attention=mean_attn,
                attention_std=attn_std,
                max_attention=max_attn,
                entropy=entropy,
                sparsity=sparsity,
            )
            stats_list.append(stats)

        return stats_list

    def find_attention_heroes(
        self,
        prompts: list[str],
        layer_idx: int,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        stats = self.compute_head_statistics(prompts, layer_idx=layer_idx)
        sorted_stats = sorted(stats, key=lambda s: s.entropy, reverse=True)
        return [(s.head_idx, s.entropy) for s in sorted_stats[:top_k]]

    def find_diverse_attention(
        self,
        prompts: list[str],
        layer_idx: int,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        stats = self.compute_head_statistics(prompts, layer_idx=layer_idx)
        sorted_stats = sorted(stats, key=lambda s: s.sparsity, reverse=False)
        return [(s.head_idx, 1.0 - s.sparsity) for s in sorted_stats[:top_k]]

    def compute_head_similarity(
        self,
        prompt: str,
        layer_idx: int,
    ) -> np.ndarray:
        patterns = self.get_attention_patterns(prompt, layer_idx=layer_idx)
        num_heads = len(patterns)
        similarity = np.zeros((num_heads, num_heads))

        pattern_list = [patterns[(layer_idx, h)] for h in range(num_heads)]

        for i in range(num_heads):
            for j in range(num_heads):
                p1 = pattern_list[i].flatten()
                p2 = pattern_list[j].flatten()
                similarity[i, j] = float(np.corrcoef(p1, p2)[0, 1])

        return similarity


class AttentionPatternVisualizer:
    def __init__(self, analyzer: AttentionPatternAnalyzer):
        self.analyzer = analyzer

    def get_attention_heatmap_data(
        self,
        prompt: str,
        layer_idx: int,
        head_idx: int,
    ) -> dict[str, Any]:
        patterns = self.analyzer.get_attention_patterns(
            prompt, layer_idx=layer_idx, head_idx=head_idx
        )

        pattern = patterns.get((layer_idx, head_idx))
        if pattern is None:
            return {"error": "No pattern found"}

        tokens = self.analyzer.tokenizer.tokenize(prompt)
        if len(tokens) != pattern.shape[0]:
            tokens = [f"tok_{i}" for i in range(pattern.shape[0])]

        return {
            "pattern": pattern.tolist(),
            "tokens": tokens,
            "shape": pattern.shape,
        }

    def get_layer_attention_summary(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> dict[str, Any]:
        stats = self.analyzer.compute_head_statistics(prompts, layer_idx=layer_idx)

        return {
            "layer_idx": layer_idx,
            "num_heads": len(stats),
            "heads": [
                {
                    "head_idx": s.head_idx,
                    "mean_attention": s.mean_attention,
                    "entropy": s.entropy,
                    "sparsity": s.sparsity,
                }
                for s in stats
            ],
        }
