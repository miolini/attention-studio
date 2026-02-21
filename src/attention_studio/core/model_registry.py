from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelVersion:
    version_id: str
    model_path: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_version: str | None = None
    checksum: str | None = None


@dataclass
class ModelRegistry:
    name: str
    versions: list[ModelVersion] = field(default_factory=list)
    current_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRegistryManager:
    def __init__(self, registry_name: str, storage_path: Path | None = None):
        self.registry_name = registry_name
        self.storage_path = storage_path or Path.home() / ".attention_studio" / "registries"
        self.registry = self._load_registry()

    def _load_registry(self) -> ModelRegistry:
        registry_file = self.storage_path / f"{self.registry_name}.json"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    data = json.load(f)
                    return ModelRegistry(
                        name=data["name"],
                        versions=[
                            ModelVersion(
                                version_id=v["version_id"],
                                model_path=v["model_path"],
                                created_at=v["created_at"],
                                metadata=v.get("metadata", {}),
                                parent_version=v.get("parent_version"),
                                checksum=v.get("checksum"),
                            )
                            for v in data.get("versions", [])
                        ],
                        current_version=data.get("current_version"),
                        metadata=data.get("metadata", {}),
                    )
            except (json.JSONDecodeError, KeyError):
                pass
        return ModelRegistry(name=self.registry_name)

    def _save_registry(self) -> None:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        registry_file = self.storage_path / f"{self.registry_name}.json"
        data = {
            "name": self.registry.name,
            "versions": [
                {
                    "version_id": v.version_id,
                    "model_path": v.model_path,
                    "created_at": v.created_at,
                    "metadata": v.metadata,
                    "parent_version": v.parent_version,
                    "checksum": v.checksum,
                }
                for v in self.registry.versions
            ],
            "current_version": self.registry.current_version,
            "metadata": self.registry.metadata,
        }
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_checksum(self, model_path: str) -> str:
        path = Path(model_path)
        if not path.exists():
            return ""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def register_model(
        self,
        model_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        version_id = f"v{len(self.registry.versions) + 1:04d}"
        created_at = datetime.now().isoformat()
        checksum = self._compute_checksum(model_path)

        parent_version = self.registry.current_version

        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            created_at=created_at,
            metadata=metadata or {},
            parent_version=parent_version,
            checksum=checksum,
        )

        self.registry.versions.append(version)
        self.registry.current_version = version_id
        self._save_registry()

        return version_id

    def get_version(self, version_id: str) -> ModelVersion | None:
        for version in self.registry.versions:
            if version.version_id == version_id:
                return version
        return None

    def get_current_version(self) -> ModelVersion | None:
        if self.registry.current_version:
            return self.get_version(self.registry.current_version)
        return None

    def list_versions(self) -> list[ModelVersion]:
        return list(self.registry.versions)

    def set_current_version(self, version_id: str) -> bool:
        if self.get_version(version_id):
            self.registry.current_version = version_id
            self._save_registry()
            return True
        return False

    def verify_version(self, version_id: str) -> bool:
        version = self.get_version(version_id)
        if not version or not version.checksum:
            return False
        current_checksum = self._compute_checksum(version.model_path)
        return current_checksum == version.checksum

    def delete_version(self, version_id: str) -> bool:
        for i, version in enumerate(self.registry.versions):
            if version.version_id == version_id:
                if self.registry.current_version == version_id:
                    return False
                self.registry.versions.pop(i)
                self._save_registry()
                return True
        return False

    def get_lineage(self, version_id: str) -> list[ModelVersion]:
        lineage = []
        current = self.get_version(version_id)
        while current:
            lineage.append(current)
            if current.parent_version:
                current = self.get_version(current.parent_version)
            else:
                break
        return lineage
