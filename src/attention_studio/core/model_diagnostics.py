from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class DiagnosticResult:
    name: str
    passed: bool
    value: float
    threshold: float
    message: str


@dataclass
class DiagnosticReport:
    model_name: str
    results: list[DiagnosticResult]
    overall_passed: bool


class ModelDiagnostics:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

    def check_nan_values(self, prompts: list[str]) -> DiagnosticResult:
        has_nan = False

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, return_dict=True)

            logits = outputs.logits
            if torch.isnan(logits).any():
                has_nan = True
                break

            if outputs.hidden_states:
                for hidden in outputs.hidden_states:
                    if torch.isnan(hidden).any():
                        has_nan = True
                        break

        return DiagnosticResult(
            name="NaN Check",
            passed=not has_nan,
            value=float(has_nan),
            threshold=0.0,
            message="Model outputs contain NaN values" if has_nan else "No NaN values found",
        )

    def check_inf_values(self, prompts: list[str]) -> DiagnosticResult:
        has_inf = False

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, return_dict=True)

            logits = outputs.logits
            if torch.isinf(logits).any():
                has_inf = True
                break

        return DiagnosticResult(
            name="Inf Check",
            passed=not has_inf,
            value=float(has_inf),
            threshold=0.0,
            message="Model outputs contain Inf values" if has_inf else "No Inf values found",
        )

    def check_output_variance(self, prompts: list[str]) -> DiagnosticResult:
        variances = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, return_dict=True)

            logits = outputs.logits
            variance = logits.var().item()
            variances.append(variance)

        avg_variance = np.mean(variances)

        passed = avg_variance > 0.01

        return DiagnosticResult(
            name="Output Variance",
            passed=passed,
            value=avg_variance,
            threshold=0.01,
            message=f"Average variance: {avg_variance:.4f}",
        )

    def check_token_probabilities(self, prompt: str) -> DiagnosticResult:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        max_prob = probs.max().item()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        passed = max_prob < 0.99 and entropy > 0.5

        return DiagnosticResult(
            name="Token Probability Distribution",
            passed=passed,
            value=max_prob,
            threshold=0.99,
            message=f"Max prob: {max_prob:.4f}, Entropy: {entropy:.4f}",
        )

    def check_layer_consistency(self, prompts: list[str]) -> DiagnosticResult:
        layer_means = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            if not outputs.hidden_states:
                return DiagnosticResult(
                    name="Layer Consistency",
                    passed=False,
                    value=0.0,
                    threshold=0.0,
                    message="No hidden states available",
                )

            means = []
            for hidden in outputs.hidden_states:
                mean_val = hidden.abs().mean().item()
                means.append(mean_val)

            layer_means.append(means)

        layer_means_arr = np.array(layer_means)
        layer_variance = layer_means_arr.var(axis=0).mean()

        passed = layer_variance > 0.001

        return DiagnosticResult(
            name="Layer Consistency",
            passed=passed,
            value=layer_variance,
            threshold=0.001,
            message=f"Layer variance: {layer_variance:.6f}",
        )

    def check_embedding_norms(self) -> DiagnosticResult:
        if not hasattr(self.model, "transformer"):
            return DiagnosticResult(
                name="Embedding Norms",
                passed=False,
                value=0.0,
                threshold=0.0,
                message="Model does not have transformer attribute",
            )

        if not hasattr(self.model.transformer, "wte"):
            return DiagnosticResult(
                name="Embedding Norms",
                passed=False,
                value=0.0,
                threshold=0.0,
                message="Model does not have wte attribute",
            )

        embeddings = self.model.transformer.wte.weight.data
        norms = torch.norm(embeddings, dim=1)
        mean_norm = norms.mean().item()

        passed = 0.1 < mean_norm < 20.0

        return DiagnosticResult(
            name="Embedding Norms",
            passed=passed,
            value=mean_norm,
            threshold=0.1,
            message=f"Mean embedding norm: {mean_norm:.4f}",
        )

    def run_full_diagnostics(
        self,
        prompts: list[str] | None = None,
    ) -> DiagnosticReport:
        if prompts is None:
            prompts = ["Hello, how are you?", "The quick brown fox jumps over the lazy dog."]

        results = []

        results.append(self.check_nan_values(prompts))
        results.append(self.check_inf_values(prompts))
        results.append(self.check_output_variance(prompts))
        results.append(self.check_token_probabilities(prompts[0]))
        results.append(self.check_layer_consistency(prompts))
        results.append(self.check_embedding_norms())

        overall_passed = all(r.passed for r in results)

        model_name = getattr(self.model, "name_or_path", "unknown")

        return DiagnosticReport(
            model_name=model_name,
            results=results,
            overall_passed=overall_passed,
        )
