import pytest
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from attention_studio.core.probing import (
    ProbeClassifier,
    ProbeConfig,
    ProbeResult,
    LinearProbe,
)


class TestProbeClassifier:
    def test_probe_config_creation(self):
        config = ProbeConfig(input_dim=512, num_classes=2)
        assert config.input_dim == 512
        assert config.num_classes == 2
        assert config.probe_type == "linear"

    def test_linear_probe_binary(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        config = ProbeConfig(input_dim=10, num_classes=2)
        probe = ProbeClassifier(config)
        result = probe.fit(X, y)

        assert isinstance(result, ProbeResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert result.confusion_matrix is not None

    def test_linear_probe_multiclass(self):
        np.random.seed(42)
        X = np.random.randn(150, 20)
        y = np.random.randint(0, 3, 150)

        config = ProbeConfig(input_dim=20, num_classes=3, test_size=0.0)
        probe = ProbeClassifier(config)
        probe.scaler.fit(X)
        X_scaled = probe.scaler.transform(X)
        probe.model = LogisticRegression(max_iter=1000, solver="lbfgs")
        probe.model.fit(X_scaled, y)
        probe.is_fitted = True

        predictions = probe.predict(X)
        assert len(predictions) == 150

    def test_ridge_probe(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

        config = ProbeConfig(input_dim=10, num_classes=1, probe_type="ridge", test_size=0.0)
        probe = ProbeClassifier(config)
        result = probe.fit(X, y)

        assert isinstance(result, ProbeResult)

    def test_predict(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)

        config = ProbeConfig(input_dim=10, num_classes=2)
        probe = ProbeClassifier(config)
        probe.fit(X_train, y_train)

        predictions = probe.predict(X_test)
        assert len(predictions) == 20
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)

        config = ProbeConfig(input_dim=10, num_classes=2)
        probe = ProbeClassifier(config)
        probe.fit(X_train, y_train)

        proba = probe.predict_proba(X_test)
        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_cross_validation(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        config = ProbeConfig(input_dim=10, num_classes=2)
        probe = ProbeClassifier(config)
        result = probe.fit(X, y, use_cross_validation=True)

        assert result.cross_val_scores is not None
        assert len(result.cross_val_scores) > 0

    def test_score(self):
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randint(0, 2, 20)

        config = ProbeConfig(input_dim=10, num_classes=2)
        probe = ProbeClassifier(config)
        probe.fit(X_train, y_train)

        score = probe.score(X_test, y_test)
        assert 0.0 <= score <= 1.0


class TestLinearProbe:
    def test_linear_probe_creation(self):
        probe = LinearProbe(input_dim=512, num_classes=10)
        assert probe.weight.shape == (10, 512)

    def test_linear_probe_forward(self):
        probe = LinearProbe(input_dim=512, num_classes=10)
        x = torch.randn(2, 512)
        output = probe(x)
        assert output.shape == (2, 10)

    def test_linear_probe_predict(self):
        probe = LinearProbe(input_dim=512, num_classes=10)
        x = torch.randn(2, 512)
        predictions = probe.predict(x)
        assert predictions.shape == (2,)

    def test_linear_probe_predict_proba(self):
        probe = LinearProbe(input_dim=512, num_classes=10)
        x = torch.randn(2, 512)
        proba = probe.predict_proba(x)
        assert proba.shape == (2, 10)
        assert np.allclose(proba.detach().numpy().sum(axis=1), 1.0)

    def test_linear_probe_with_bias(self):
        probe = LinearProbe(input_dim=512, num_classes=10, bias=True)
        assert probe.bias is not None
        assert probe.bias.shape == (10,)

    def test_linear_probe_without_bias(self):
        probe = LinearProbe(input_dim=512, num_classes=10, bias=False)
        assert probe.bias is None
