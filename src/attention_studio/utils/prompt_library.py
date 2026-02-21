from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptEntry:
    text: str
    category: str = "default"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptLibrary:
    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".attention_studio" / "prompts.json"
        self.prompts: list[PromptEntry] = []
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    for item in data.get("prompts", []):
                        self.prompts.append(
                            PromptEntry(
                                text=item["text"],
                                category=item.get("category", "default"),
                                tags=item.get("tags", []),
                                metadata=item.get("metadata", {}),
                            )
                        )
            except (json.JSONDecodeError, KeyError):
                self.prompts = []

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "prompts": [
                {
                    "text": p.text,
                    "category": p.category,
                    "tags": p.tags,
                    "metadata": p.metadata,
                }
                for p in self.prompts
            ]
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, text: str, category: str = "default", tags: list[str] | None = None) -> None:
        entry = PromptEntry(
            text=text,
            category=category,
            tags=tags or [],
        )
        self.prompts.append(entry)
        self._save()

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.prompts):
            self.prompts.pop(index)
            self._save()

    def get(self, index: int) -> PromptEntry | None:
        if 0 <= index < len(self.prompts):
            return self.prompts[index]
        return None

    def get_by_category(self, category: str) -> list[PromptEntry]:
        return [p for p in self.prompts if p.category == category]

    def get_by_tag(self, tag: str) -> list[PromptEntry]:
        return [p for p in self.prompts if tag in p.tags]

    def search(self, query: str) -> list[PromptEntry]:
        query_lower = query.lower()
        return [
            p for p in self.prompts
            if query_lower in p.text.lower()
        ]

    def list_categories(self) -> list[str]:
        return list({p.category for p in self.prompts})

    def list_tags(self) -> list[str]:
        tags = set()
        for p in self.prompts:
            tags.update(p.tags)
        return sorted(tags)

    def get_all(self) -> list[PromptEntry]:
        return list(self.prompts)

    def clear(self) -> None:
        self.prompts = []
        self._save()


class PromptDataset:
    def __init__(self, prompts: list[str] | None = None):
        self.prompts = prompts or []

    def add(self, prompt: str) -> None:
        self.prompts.append(prompt)

    def add_batch(self, prompts: list[str]) -> None:
        self.prompts.extend(prompts)

    def get(self, index: int) -> str | None:
        if 0 <= index < len(self.prompts):
            return self.prompts[index]
        return None

    def get_batch(self, indices: list[int]) -> list[str]:
        return [self.prompts[i] for i in indices if 0 <= i < len(self.prompts)]

    def filter_by_length(self, min_length: int = 0, max_length: int | None = None) -> list[str]:
        filtered = [p for p in self.prompts if len(p) >= min_length]
        if max_length is not None:
            filtered = [p for p in filtered if len(p) <= max_length]
        return filtered

    def shuffle(self, seed: int | None = None) -> None:
        import random
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.prompts)

    def split(self, ratio: float = 0.8) -> tuple[list[str], list[str]]:
        split_idx = int(len(self.prompts) * ratio)
        return self.prompts[:split_idx], self.prompts[split_idx:]

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self):
        return iter(self.prompts)
