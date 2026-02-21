from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TokenInfluence:
    token_id: int
    token_text: str
    position: int
    influence_score: float
    layer_contributions: dict[int, float] = field(default_factory=dict)


@dataclass
class InfluenceReport:
    prompt: str
    target_position: int
    tokens: list[TokenInfluence]
    total_influence: float


class TokenInfluenceTracker:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

    def compute_influence_by_ablation(
        self,
        prompt: str,
        target_position: int | None = None,
    ) -> InfluenceReport:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            original_output = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            original_logits = original_output.logits[0, -1, :]

        if target_position is None:
            target_position = len(tokens) - 1

        original_prob = torch.softmax(original_logits, dim=-1).max().item()

        token_influences = []

        for pos in range(len(tokens)):
            if pos == target_position:
                continue

            masked_input = input_ids.clone()
            masked_input[0, pos] = self.tokenizer.mask_token_id

            with torch.no_grad():
                masked_output = self.model(
                    input_ids=masked_input,
                    output_hidden_states=True,
                    return_dict=True,
                )
                masked_logits = masked_output.logits[0, -1, :]

            masked_prob = torch.softmax(masked_logits, dim=-1).max().item()

            influence = original_prob - masked_prob

            token_influences.append(
                TokenInfluence(
                    token_id=input_ids[0, pos].item(),
                    token_text=tokens[pos],
                    position=pos,
                    influence_score=influence,
                )
            )

        token_influences.sort(key=lambda x: x.influence_score, reverse=True)

        return InfluenceReport(
            prompt=prompt,
            target_position=target_position,
            tokens=token_influences,
            total_influence=sum(t.influence_score for t in token_influences),
        )

    def compute_influence_by_gradient(
        self,
        prompt: str,
        target_token_id: int | None = None,
    ) -> dict[int, float]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        self.model.zero_grad()

        outputs = self.model(input_ids=input_ids, output_hidden_states=True)

        if target_token_id is None:
            target_token_id = outputs.logits[0, -1, :].argmax().item()

        loss = outputs.logits[0, -1, target_token_id]
        loss.backward()

        layer_contributions = {}

        for idx, layer in enumerate(self.model.transformer.h):
            if layer.weight.grad is not None:
                grad_norm = layer.weight.grad.norm().item()
                layer_contributions[idx] = grad_norm

        self.model.zero_grad()

        return layer_contributions

    def compute_attention_contribution(
        self,
        prompt: str,
    ) -> dict[tuple[int, int], float]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        attention_weights = {}

        def create_hook(layer_idx, head_idx):
            def hook(module, inp, output):
                attn = output[0]
                weight = attn[0, head_idx].detach().cpu().numpy()
                attention_weights[(layer_idx, head_idx)] = weight
            return hook

        handles = []
        num_layers = len(self.model.transformer.h)

        for layer_idx in range(num_layers):
            layer = self.model.transformer.h[layer_idx]
            if hasattr(layer, "attn"):
                num_heads = getattr(layer.attn, "head_dim", 12)
                for head_idx in range(num_heads):
                    handles.append(
                        layer.attn.register_forward_hook(
                            create_hook(layer_idx, head_idx)
                        )
                    )

        with torch.no_grad():
            _ = self.model(input_ids)

        for handle in handles:
            handle.remove()

        return attention_weights


class InfluenceVisualizer:
    def __init__(self, tracker: TokenInfluenceTracker):
        self.tracker = tracker

    def get_influence_heatmap_data(
        self,
        prompt: str,
        target_position: int | None = None,
    ) -> dict[str, Any]:
        report = self.tracker.compute_influence_by_ablation(
            prompt, target_position
        )

        tokens = self.tracker.tokenizer.tokenize(prompt)
        influence_scores = [t.influence_score for t in report.tokens]

        return {
            "tokens": tokens,
            "influence_scores": influence_scores,
            "total_influence": report.total_influence,
        }

    def get_top_influential_tokens(
        self,
        prompt: str,
        target_position: int | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        report = self.tracker.compute_influence_by_ablation(
            prompt, target_position
        )

        top_tokens = report.tokens[:top_k]

        return [
            {
                "token": t.token_text,
                "position": t.position,
                "influence": t.influence_score,
            }
            for t in top_tokens
        ]
