from __future__ import annotations

import gzip
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional


@dataclass
class SerializationMetadata:
    version: str
    created_at: str
    format: str
    checksum: Optional[str] = None


class Serializer(ABC):
    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        pass


class JsonSerializer(Serializer):
    def __init__(self, indent: Optional[int] = 2, ensure_ascii: bool = False):
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def serialize(self, obj: Any) -> bytes:
        json_str = json.dumps(obj, indent=self.indent, ensure_ascii=self.ensure_ascii)
        return json_str.encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


class PickleSerializer(Serializer):
    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol

    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=self.protocol)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


class CompressedSerializer(Serializer):
    def __init__(self, serializer: Serializer, compress_level: int = 6):
        self.serializer = serializer
        self.compress_level = compress_level

    def serialize(self, obj: Any) -> bytes:
        data = self.serializer.serialize(obj)
        return gzip.compress(data, compresslevel=self.compress_level)

    def deserialize(self, data: bytes) -> Any:
        decompressed = gzip.decompress(data)
        return self.serializer.deserialize(decompressed)


class StateSerializer:
    def __init__(self, serializer: Optional[Serializer] = None):
        self.serializer = serializer or JsonSerializer()

    def save_to_file(self, obj: Any, path: Path, metadata: Optional[dict] = None) -> None:
        wrapper = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                **(metadata or {}),
            },
            "data": obj,
        }
        data = self.serializer.serialize(wrapper)
        path.write_bytes(data)

    def load_from_file(self, path: Path) -> tuple[Any, dict]:
        data = path.read_bytes()
        wrapper = self.serializer.deserialize(data)
        return wrapper.get("data", {}), wrapper.get("metadata", {})

    def save_to_bytes(self, obj: Any, metadata: Optional[dict] = None) -> bytes:
        wrapper = {
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                **(metadata or {}),
            },
            "data": obj,
        }
        return self.serializer.serialize(wrapper)

    def load_from_bytes(self, data: bytes) -> tuple[Any, dict]:
        wrapper = self.serializer.deserialize(data)
        return wrapper.get("data", {}), wrapper.get("metadata", {})


def dataclass_to_dict(obj: Any) -> dict:
    import dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(dataclass_to_dict(item) for item in obj)
    else:
        return obj


def dict_to_dataclass(cls: type, data: dict) -> Any:
    import dataclasses
    if dataclasses.is_dataclass(cls) and not isinstance(cls, type):
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key in field_types:
                field_type = field_types[key]
                if isinstance(value, dict) and not isinstance(value, (list, tuple)):
                    kwargs[key] = dict_to_dataclass(field_type, value)
                else:
                    kwargs[key] = value
        return cls(**kwargs)
    return data


class FeatureState:
    @staticmethod
    def save(features: list[dict], path: Path, compressed: bool = False) -> None:
        serializer = CompressedSerializer(JsonSerializer()) if compressed else JsonSerializer()
        state_serializer = StateSerializer(serializer)
        state_serializer.save_to_file(features, path, {"type": "feature_state"})

    @staticmethod
    def load(path: Path) -> list[dict]:
        suffix = path.suffix
        compressed = suffix in (".gz", ".pgz")
        serializer = CompressedSerializer(JsonSerializer()) if compressed else JsonSerializer()
        state_serializer = StateSerializer(serializer)
        features, _ = state_serializer.load_from_file(path)
        return features


class GraphState:
    @staticmethod
    def save_graphml(graph: Any, path: Path) -> None:
        try:
            import networkx as nx
            nx.write_graphml(graph, path)
        except ImportError:
            raise RuntimeError("networkx required for graph export")

    @staticmethod
    def load_graphml(path: Path) -> Any:
        try:
            import networkx as nx
            return nx.read_graphml(path)
        except ImportError:
            raise RuntimeError("networkx required for graph import")

    @staticmethod
    def save_adjacency(graph: Any, path: Path) -> None:
        import networkx as nx
        data = nx.to_dict_of_dicts(graph)
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load_adjacency(path: Path) -> Any:
        import networkx as nx
        data = json.loads(path.read_text())
        return nx.from_dict_of_dicts(data)


class ModelCheckpoint:
    @staticmethod
    def save(
        model_state: dict,
        optimizer_state: Optional[dict],
        epoch: int,
        metadata: dict,
        path: Path,
    ) -> None:
        checkpoint = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "epoch": epoch,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }
        serializer = CompressedSerializer(PickleSerializer())
        state_serializer = StateSerializer(serializer)
        state_serializer.save_to_file(checkpoint, path, {"type": "model_checkpoint"})

    @staticmethod
    def load(path: Path) -> dict:
        serializer = CompressedSerializer(PickleSerializer())
        state_serializer = StateSerializer(serializer)
        checkpoint, _ = state_serializer.load_from_file(path)
        return checkpoint


class ExperimentLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True)
        self.run_id = 0

    def start_run(self, run_name: Optional[str] = None) -> Path:
        run_name = run_name or f"run_{self.run_id}"
        run_dir = self.experiment_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id += 1
        return run_dir

    def log_metrics(self, metrics: dict, step: int, run_dir: Path) -> None:
        metrics_file = run_dir / "metrics.jsonl"
        entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
        with metrics_file.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_config(self, config: dict, run_dir: Path) -> None:
        config_file = run_dir / "config.json"
        config_file.write_text(json.dumps(config, indent=2))

    def save_checkpoint(self, checkpoint: dict, step: int, run_dir: Path) -> Path:
        ckpt_file = run_dir / f"checkpoint_{step}.pt"
        ModelCheckpoint.save(
            model_state=checkpoint.get("model_state", {}),
            optimizer_state=checkpoint.get("optimizer_state"),
            epoch=step,
            metadata=checkpoint.get("metadata", {}),
            path=ckpt_file,
        )
        return ckpt_file


def compute_checksum(data: bytes) -> str:
    import hashlib
    return hashlib.md5(data).hexdigest()


def verify_checksum(data: bytes, expected_checksum: str) -> bool:
    return compute_checksum(data) == expected_checksum
