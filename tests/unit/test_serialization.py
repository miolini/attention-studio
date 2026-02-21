import pytest
import tempfile
import os
from pathlib import Path
from attention_studio.utils.serialization import (
    JsonSerializer,
    PickleSerializer,
    CompressedSerializer,
    StateSerializer,
    FeatureState,
    GraphState,
    ModelCheckpoint,
    ExperimentLogger,
    compute_checksum,
    verify_checksum,
    dataclass_to_dict,
)


class TestJsonSerializer:
    def test_serialize_deserialize(self):
        serializer = JsonSerializer()
        data = {"key": "value", "number": 42}
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)
        deserialized = serializer.deserialize(serialized)
        assert deserialized == data


class TestPickleSerializer:
    def test_serialize_deserialize(self):
        serializer = PickleSerializer()
        data = {"key": [1, 2, 3], "nested": {"a": "b"}}
        serialized = serializer.serialize(data)
        assert isinstance(serialized, bytes)
        deserialized = serializer.deserialize(serialized)
        assert deserialized == data


class TestCompressedSerializer:
    def test_compress_decompress(self):
        serializer = CompressedSerializer(JsonSerializer())
        data = {"key": "value"}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)
        assert deserialized == data


class TestStateSerializer:
    def test_save_load_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = Path(f.name)
        try:
            serializer = StateSerializer(JsonSerializer())
            data = {"features": [1, 2, 3]}
            serializer.save_to_file(data, path)
            loaded_data, metadata = serializer.load_from_file(path)
            assert loaded_data == data
            assert "version" in metadata
        finally:
            os.unlink(path)

    def test_save_load_bytes(self):
        serializer = StateSerializer(JsonSerializer())
        data = {"key": "value"}
        serialized = serializer.save_to_bytes(data)
        loaded_data, metadata = serializer.load_from_bytes(serialized)
        assert loaded_data == data


class TestFeatureState:
    def test_save_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            features = [
                {"idx": 0, "layer": 1, "activation": 0.5},
                {"idx": 1, "layer": 2, "activation": 0.8},
            ]
            FeatureState.save(features, path)
            loaded = FeatureState.load(path)
            assert loaded == features
        finally:
            os.unlink(path)

    def test_compressed_save_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = Path(f.name)
        try:
            features = [{"idx": i, "activation": i * 0.1} for i in range(10)]
            FeatureState.save(features, path, compressed=True)
            loaded = FeatureState.load(path)
            assert loaded == features
        finally:
            os.unlink(path)


class TestGraphState:
    def test_save_load_adjacency(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as path:
            path = Path(path.name)
        try:
            import networkx as nx
            graph = nx.DiGraph()
            graph.add_edge("a", "b", weight=1.0)
            graph.add_edge("b", "c", weight=2.0)
            GraphState.save_adjacency(graph, path)
            loaded = GraphState.load_adjacency(path)
            assert loaded.has_edge("a", "b")
            assert loaded.has_edge("b", "c")
        finally:
            os.unlink(path)


class TestModelCheckpoint:
    def test_save_load(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = Path(f.name)
        try:
            model_state = {"layer1.weight": [1, 2, 3]}
            optimizer_state = {"state": {}}
            metadata = {"learning_rate": 0.001}
            ModelCheckpoint.save(model_state, optimizer_state, 10, metadata, path)
            checkpoint = ModelCheckpoint.load(path)
            assert checkpoint["model_state"] == model_state
            assert checkpoint["optimizer_state"] == optimizer_state
            assert checkpoint["epoch"] == 10
        finally:
            os.unlink(path)


class TestExperimentLogger:
    def test_experiment_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = ExperimentLogger(log_dir)
            assert logger.experiment_dir.exists()

    def test_start_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = ExperimentLogger(log_dir)
            run_dir = logger.start_run("test_run")
            assert run_dir.exists()
            assert run_dir.name == "test_run"

    def test_log_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = ExperimentLogger(log_dir)
            run_dir = logger.start_run()
            logger.log_metrics({"loss": 0.5}, 1, run_dir)
            metrics_file = run_dir / "metrics.jsonl"
            assert metrics_file.exists()

    def test_log_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            logger = ExperimentLogger(log_dir)
            run_dir = logger.start_run()
            config = {"batch_size": 32, "lr": 0.001}
            logger.log_config(config, run_dir)
            config_file = run_dir / "config.json"
            assert config_file.exists()


class TestChecksum:
    def test_compute_checksum(self):
        data = b"hello world"
        checksum = compute_checksum(data)
        assert isinstance(checksum, str)
        assert len(checksum) == 32

    def test_verify_checksum(self):
        data = b"hello world"
        checksum = compute_checksum(data)
        assert verify_checksum(data, checksum) is True
        assert verify_checksum(b"different", checksum) is False


class TestDataclassConversion:
    def test_dataclass_to_dict(self):
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int

        point = Point(1, 2)
        result = dataclass_to_dict(point)
        assert result == {"x": 1, "y": 2}

    def test_nested_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int

        @dataclass
        class Segment:
            start: Point
            end: Point

        segment = Segment(Point(0, 0), Point(1, 1))
        result = dataclass_to_dict(segment)
        assert result == {"start": {"x": 0, "y": 0}, "end": {"x": 1, "y": 1}}
