from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass
class PredictionResult:
    token_id: int
    token_text: str
    logit: float
    probability: float
    rank: int


@dataclass
class LogitStats:
    mean_logit: float
    std_logit: float
    max_logit: float
    min_logit: float
    entropy: float


class LogitAnalyzer:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

    def get_logits(
        self,
        prompt: str,
        position: int | None = None,
    ) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits

        if position is None:
            position = logits.shape[1] - 1

        return logits[0, position, :]

    def get_top_predictions(
        self,
        prompt: str,
        top_k: int = 10,
        position: int | None = None,
    ) -> list[PredictionResult]:
        logits = self.get_logits(prompt, position)

        probs = functional.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(top_k)

        results = []
        for idx, (prob, token_id) in enumerate(zip(top_probs, top_indices, strict=True)):
            token_text = self.tokenizer.decode(token_id.item())
            results.append(
                PredictionResult(
                    token_id=token_id.item(),
                    token_text=token_text,
                    logit=logits[token_id].item(),
                    probability=prob.item(),
                    rank=idx + 1,
                )
            )

        return results

    def compute_logit_stats(
        self,
        prompt: str,
        position: int | None = None,
    ) -> LogitStats:
        logits = self.get_logits(prompt, position)

        logits_np = logits.cpu().numpy()

        mean_logit = float(np.mean(logits_np))
        std_logit = float(np.std(logits_np))
        max_logit = float(np.max(logits_np))
        min_logit = float(np.min(logits_np))

        probs = functional.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy()
        probs_np = np.clip(probs_np, 1e-10, 1.0)
        entropy = float(-np.sum(probs_np * np.log(probs_np)))

        return LogitStats(
            mean_logit=mean_logit,
            std_logit=std_logit,
            max_logit=max_logit,
            min_logit=min_logit,
            entropy=entropy,
        )

    def compute_prediction_confidence(
        self,
        prompt: str,
        position: int | None = None,
    ) -> float:
        logits = self.get_logits(prompt, position)
        probs = functional.softmax(logits, dim=-1)

        confidence = probs.max().item()
        return confidence

    def compare_predictions(
        self,
        prompt_a: str,
        prompt_b: str,
    ) -> dict[str, Any]:
        preds_a = self.get_top_predictions(prompt_a, top_k=5)
        preds_b = self.get_top_predictions(prompt_b, top_k=5)

        top_token_a = preds_a[0].token_id
        top_token_b = preds_b[0].token_id

        overlap = {p.token_id for p in preds_a} & {p.token_id for p in preds_b}

        return {
            "prompt_a_top": preds_a[0].token_text,
            "prompt_b_top": preds_b[0].token_text,
            "same_top": top_token_a == top_token_b,
            "overlap_count": len(overlap),
            "overlap_tokens": list(overlap),
        }

    def get_token_probability(
        self,
        prompt: str,
        target_token: str,
        position: int | None = None,
    ) -> float | None:
        target_ids = self.tokenizer.encode(target_token)

        if not target_ids:
            return None

        target_id = target_ids[0]
        logits = self.get_logits(prompt, position)

        probs = functional.softmax(logits, dim=-1)

        return probs[target_id].item()

    def compute_perplexity(
        self,
        prompt: str,
    ) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits
        labels = input_ids

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )

        perplexity = torch.exp(loss).item()
        return perplexity


class PredictionStabilityAnalyzer:
    def __init__(self, model_manager: Any):
        self.analyzer = LogitAnalyzer(model_manager)

    def measure_stability(
        self,
        prompts: list[str],
        num_samples: int = 5,
    ) -> dict[str, float]:
        results = {}

        for prompt in prompts:
            top_tokens = []
            for _ in range(num_samples):
                preds = self.analyzer.get_top_predictions(prompt, top_k=1)
                top_tokens.append(preds[0].token_id)

            most_common = max(set(top_tokens), key=top_tokens.count)
            stability = top_tokens.count(most_common) / num_samples

            results[prompt] = stability

        return results

    def find_consistent_predictions(
        self,
        prompts: list[str],
        stability_threshold: float = 0.8,
    ) -> list[tuple[str, float]]:
        stabilities = self.measure_stability(prompts)

        consistent = [
            (prompt, stability)
            for prompt, stability in stabilities.items()
            if stability >= stability_threshold
        ]

        return consistent
