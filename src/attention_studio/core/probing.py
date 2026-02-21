from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    accuracy: float
    f1_score: float
    confusion_matrix: Optional[list[list[int]]] = None
    cross_val_scores: Optional[list[float]] = None
    coefficients: Optional[list[float]] = None
    intercept: Optional[float] = None


@dataclass
class ProbeConfig:
    input_dim: int
    num_classes: int
    probe_type: str = "linear"
    regularization: float = 1.0
    max_iter: int = 1000
    test_size: float = 0.2
    random_state: int = 42


class ProbeClassifier:
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_cross_validation: bool = False,
    ) -> ProbeResult:
        X = self.scaler.fit_transform(X)

        if self.config.test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y

        if self.config.probe_type == "linear":
            if self.config.num_classes == 2:
                self.model = LogisticRegression(
                    C=self.config.regularization,
                    max_iter=self.config.max_iter,
                    random_state=self.config.random_state,
                )
            else:
                self.model = LogisticRegression(
                    C=self.config.regularization,
                    max_iter=self.config.max_iter,
                    random_state=self.config.random_state,
                    multi_class="multinomial",
                )
        elif self.config.probe_type == "ridge":
            self.model = Ridge(
                alpha=self.config.regularization,
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(f"Unknown probe type: {self.config.probe_type}")

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test)

        if self.config.probe_type == "ridge":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            accuracy = -mse
            f1 = r2
            confusion_matrix = None
        else:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            confusion_matrix = self._compute_confusion_matrix(y_test, y_pred)

        cross_val_scores = None
        if use_cross_validation and self.config.probe_type == "linear":
            cv_scores = cross_val_score(
                self.model,
                X,
                y,
                cv=min(5, min(np.bincount(y)) if len(np.bincount(y)) > 1 else 3),
                scoring="accuracy",
            )
            cross_val_scores = cv_scores.tolist()

        coefficients = None
        intercept = None
        if hasattr(self.model, "coef_"):
            coefficients = self.model.coef_.tolist()
            if self.model.intercept_ is not None:
                intercept = float(self.model.intercept_.item()) if self.model.intercept_.ndim == 0 else self.model.intercept_.tolist()

        return ProbeResult(
            accuracy=accuracy,
            f1_score=f1,
            confusion_matrix=confusion_matrix,
            cross_val_scores=cross_val_scores,
            coefficients=coefficients,
            intercept=intercept,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        X = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            preds = self.model.predict(X)
            proba = F.one_hot(torch.tensor(preds), num_classes=self.config.num_classes).float().numpy()
            return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> list[list[int]]:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        matrix = [[0] * n_classes for _ in range(n_classes)]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for true, pred in zip(y_true, y_pred):
            i = class_to_idx[true]
            j = class_to_idx[pred]
            matrix[i][j] += 1
        return matrix

    def save(self, path: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted probe")
        state = {
            "config": {
                "input_dim": self.config.input_dim,
                "num_classes": self.config.num_classes,
                "probe_type": self.config.probe_type,
                "regularization": self.config.regularization,
                "max_iter": self.config.max_iter,
                "random_state": self.config.random_state,
            },
            "model": self.model,
            "scaler": self.scaler,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Path) -> "ProbeClassifier":
        state = torch.load(path)
        config = ProbeConfig(**state["config"])
        probe = cls(config)
        probe.model = state["model"]
        probe.scaler = state["scaler"]
        probe.is_fitted = True
        return probe


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_classes)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class ProbingAnalyzer:
    def __init__(self, model_manager: Any):
        self.model_manager = model_manager

    def extract_representations(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        representations = []
        labels = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states[layer_idx]
            rep = hidden_states[0, -1, :].cpu().numpy()
            representations.append(rep)

        return np.array(representations), np.array(labels)

    def probe_syntax(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> ProbeResult:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        representations = []
        labels = []

        pos_tags = {"NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
        neg_tags = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS"}

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states[layer_idx]
            rep = hidden_states[0, -1, :].cpu().numpy()
            representations.append(rep)
            labels.append(1 if any(tag in prompt for tag in pos_tags) else 0)

        X = np.array(representations)
        y = np.array(labels)

        config = ProbeConfig(input_dim=X.shape[1], num_classes=2)
        probe = ProbeClassifier(config)
        return probe.fit(X, y)

    def probe_sentiment(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> ProbeResult:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        positive_words = {"good", "great", "excellent", "amazing", "wonderful", "love", "best", "fantastic"}
        negative_words = {"bad", "terrible", "awful", "horrible", "worst", "hate", "poor", "disappointing"}

        representations = []
        labels = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states[layer_idx]
            rep = hidden_states[0, -1, :].cpu().numpy()
            representations.append(rep)

            prompt_lower = prompt.lower()
            if any(w in prompt_lower for w in positive_words):
                labels.append(2)
            elif any(w in prompt_lower for w in negative_words):
                labels.append(0)
            else:
                labels.append(1)

        X = np.array(representations)
        y = np.array(labels)

        config = ProbeConfig(input_dim=X.shape[1], num_classes=3)
        probe = ProbeClassifier(config)
        return probe.fit(X, y)

    def probe_task(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> ProbeResult:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        representations = []
        labels = []

        task_keywords = {
            "classification": {"classify", "categorize", "identify", "determine"},
            "extraction": {"extract", "find", "locate", "search"},
            "generation": {"write", "create", "generate", "compose", "produce"},
            "reasoning": {"explain", "why", "how", "because", "therefore"},
        }

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states[layer_idx]
            rep = hidden_states[0, -1, :].cpu().numpy()
            representations.append(rep)

            prompt_lower = prompt.lower()
            label = -1
            for idx, (task, keywords) in enumerate(task_keywords.items()):
                if any(kw in prompt_lower for kw in keywords):
                    label = idx
                    break
            if label == -1:
                label = len(task_keywords)
            labels.append(label)

        X = np.array(representations)
        y = np.array(labels)

        config = ProbeConfig(input_dim=X.shape[1], num_classes=len(task_keywords) + 1)
        probe = ProbeClassifier(config)
        return probe.fit(X, y)

    def compare_layers(
        self,
        prompts: list[str],
        layer_indices: list[int],
    ) -> dict[int, ProbeResult]:
        results = {}
        for layer_idx in layer_indices:
            try:
                result = self.probe_sentiment(prompts, layer_idx)
                results[layer_idx] = result
            except Exception as e:
                results[layer_idx] = None
        return results
