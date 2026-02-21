import pytest
import numpy as np
import torch
from attention_studio.core.activation_db import (
    ActivationDatabase,
    ActivationRecord,
    ActivationBatcher,
    ActivationAggregator,
)


class TestActivationRecord:
    def test_record_creation(self):
        record = ActivationRecord(
            prompt="Hello world",
            layer=5,
            position=0,
            features=np.array([1.0, 2.0, 3.0]),
            tokens=["Hello", "world"],
        )
        assert record.prompt == "Hello world"
        assert record.layer == 5
        assert record.position == 0

    def test_to_dict(self):
        record = ActivationRecord(
            prompt="Test",
            layer=1,
            position=0,
            features=np.array([1.0, 2.0]),
            tokens=["Test"],
        )
        d = record.to_dict()
        assert d["prompt"] == "Test"
        assert d["layer"] == 1

    def test_from_dict(self):
        data = {
            "prompt": "Test",
            "layer": 1,
            "position": 0,
            "features": [1.0, 2.0],
            "tokens": ["Test"],
            "timestamp": "2024-01-01T00:00:00",
        }
        record = ActivationRecord.from_dict(data)
        assert record.prompt == "Test"
        assert record.features[0] == 1.0


class TestActivationDatabase:
    def test_database_creation(self):
        db = ActivationDatabase()
        assert db.count_records() == 0

    def test_add_record(self):
        db = ActivationDatabase()
        record_id = db.add(
            prompt="Hello world",
            layer=5,
            position=0,
            features=np.array([1.0, 2.0, 3.0]),
            tokens=["Hello", "world"],
        )
        assert record_id == 0
        assert db.count_records() == 1

    def test_add_record_with_tensor(self):
        db = ActivationDatabase()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        record_id = db.add(
            prompt="Test",
            layer=1,
            position=0,
            features=tensor,
            tokens=["Test"],
        )
        assert record_id == 0

    def test_get_record(self):
        db = ActivationDatabase()
        db.add(
            prompt="Hello",
            layer=1,
            position=0,
            features=np.array([1.0, 2.0]),
            tokens=["Hello"],
        )
        record = db.get(0)
        assert record is not None
        assert record.prompt == "Hello"

    def test_get_record_out_of_bounds(self):
        db = ActivationDatabase()
        record = db.get(999)
        assert record is None

    def test_query_by_layer(self):
        db = ActivationDatabase()
        db.add("prompt1", 1, 0, np.array([1.0]), ["p1"])
        db.add("prompt2", 1, 1, np.array([2.0]), ["p2"])
        db.add("prompt3", 2, 0, np.array([3.0]), ["p3"])

        results = db.query_by_layer(1)
        assert len(results) == 2

    def test_query_by_position(self):
        db = ActivationDatabase()
        db.add("prompt1", 1, 0, np.array([1.0]), ["p1"])
        db.add("prompt2", 1, 1, np.array([2.0]), ["p2"])
        db.add("prompt3", 2, 0, np.array([3.0]), ["p3"])

        results = db.query_by_position(0)
        assert len(results) == 2

    def test_query_by_layer_and_position(self):
        db = ActivationDatabase()
        db.add("prompt1", 1, 0, np.array([1.0]), ["p1"])
        db.add("prompt2", 1, 1, np.array([2.0]), ["p2"])
        db.add("prompt3", 1, 0, np.array([3.0]), ["p3"])

        results = db.query_by_layer_and_position(1, 0)
        assert len(results) == 2

    def test_query_by_prompt(self):
        db = ActivationDatabase()
        db.add("Hello world", 1, 0, np.array([1.0]), ["Hello", "world"])
        db.add("Goodbye world", 1, 0, np.array([2.0]), ["Goodbye", "world"])
        db.add("Hello there", 1, 0, np.array([3.0]), ["Hello", "there"])

        results = db.query_by_prompt("Hello")
        assert len(results) == 2

    def test_query_by_feature_threshold(self):
        db = ActivationDatabase()
        db.add("p1", 1, 0, np.array([1.0, 5.0]), ["p1"])
        db.add("p2", 1, 0, np.array([2.0, 3.0]), ["p2"])
        db.add("p3", 1, 0, np.array([3.0, 4.0]), ["p3"])

        results = db.query_by_feature_threshold(0, 1.5, "above")
        assert len(results) == 2

    def test_get_top_features(self):
        db = ActivationDatabase()
        db.add("p1", 1, 0, np.array([1.0, 5.0]), ["p1"])
        db.add("p2", 1, 0, np.array([2.0, 3.0]), ["p2"])

        top_features = db.get_top_features(1, 0, top_k=2)
        assert len(top_features) == 2

    def test_compute_feature_statistics(self):
        db = ActivationDatabase()
        db.add("p1", 1, 0, np.array([1.0, 2.0]), ["p1"])
        db.add("p2", 1, 0, np.array([3.0, 4.0]), ["p2"])

        stats = db.compute_feature_statistics(0)
        assert "mean" in stats
        assert stats["mean"] == 2.0

    def test_get_feature_correlation(self):
        db = ActivationDatabase()
        db.add("p1", 1, 0, np.array([1.0, 1.0]), ["p1"])
        db.add("p2", 1, 0, np.array([2.0, 2.0]), ["p2"])

        corr = db.get_feature_correlation(0, 1)
        assert abs(corr - 1.0) < 0.001

    def test_get_all_prompts(self):
        db = ActivationDatabase()
        db.add("Hello", 1, 0, np.array([1.0]), ["Hello"])
        db.add("Hello", 1, 1, np.array([2.0]), ["Hello"])

        prompts = db.get_all_prompts()
        assert len(prompts) == 1

    def test_get_all_layers(self):
        db = ActivationDatabase()
        db.add("p1", 1, 0, np.array([1.0]), ["p1"])
        db.add("p2", 3, 0, np.array([2.0]), ["p2"])

        layers = db.get_all_layers()
        assert layers == [1, 3]

    def test_clear(self):
        db = ActivationDatabase()
        db.add("p1", 1, 0, np.array([1.0]), ["p1"])
        db.clear()
        assert db.count_records() == 0


class TestActivationBatcher:
    def test_batcher_creation(self):
        db = ActivationDatabase()
        batcher = ActivationBatcher(db, batch_size=10)
        assert batcher.batch_size == 10

    def test_get_batches_by_layer(self):
        db = ActivationDatabase()
        for i in range(25):
            db.add(f"p{i}", 1, i % 5, np.array([float(i)]), [f"p{i}"])

        batcher = ActivationBatcher(db, batch_size=10)
        batches = batcher.get_batches_by_layer(1)
        assert len(batches) == 3
        assert len(batches[0]) == 10


class TestActivationAggregator:
    def test_aggregate_mean(self):
        records = [
            ActivationRecord("p1", 1, 0, np.array([1.0, 2.0]), ["p1"]),
            ActivationRecord("p2", 1, 0, np.array([3.0, 4.0]), ["p2"]),
        ]
        result = ActivationAggregator.aggregate_mean(records)
        assert result[0] == 2.0
        assert result[1] == 3.0

    def test_aggregate_max(self):
        records = [
            ActivationRecord("p1", 1, 0, np.array([1.0, 5.0]), ["p1"]),
            ActivationRecord("p2", 1, 0, np.array([3.0, 4.0]), ["p2"]),
        ]
        result = ActivationAggregator.aggregate_max(records)
        assert result[0] == 3.0
        assert result[1] == 5.0

    def test_aggregate_sum(self):
        records = [
            ActivationRecord("p1", 1, 0, np.array([1.0, 2.0]), ["p1"]),
            ActivationRecord("p2", 1, 0, np.array([3.0, 4.0]), ["p2"]),
        ]
        result = ActivationAggregator.aggregate_sum(records)
        assert result[0] == 4.0
        assert result[1] == 6.0

    def test_aggregate_by_position(self):
        records = [
            ActivationRecord("p1", 1, 0, np.array([1.0, 2.0]), ["p1"]),
            ActivationRecord("p2", 1, 1, np.array([3.0, 4.0]), ["p2"]),
        ]
        result = ActivationAggregator.aggregate_by_position(records)
        assert 0 in result
        assert 1 in result
