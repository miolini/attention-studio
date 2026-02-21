from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import os

from attention_studio.utils.prompt_library import (
    PromptEntry,
    PromptLibrary,
    PromptDataset,
)


class TestPromptEntry:
    def test_entry_creation(self):
        entry = PromptEntry(
            text="Hello world",
            category="greeting",
            tags=["test", "hello"],
        )
        assert entry.text == "Hello world"
        assert entry.category == "greeting"
        assert "test" in entry.tags

    def test_entry_default_values(self):
        entry = PromptEntry(text="Simple prompt")
        assert entry.category == "default"
        assert entry.tags == []
        assert entry.metadata == {}


class TestPromptLibrary:
    def test_library_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            lib = PromptLibrary(storage_path=path)
            assert len(lib.prompts) == 0

    def test_add_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            lib = PromptLibrary(storage_path=path)
            lib.add("Test prompt", category="test", tags=["tag1"])
            assert len(lib.prompts) == 1
            assert lib.prompts[0].text == "Test prompt"

    def test_remove_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            lib = PromptLibrary(storage_path=path)
            lib.add("Prompt 1")
            lib.add("Prompt 2")
            lib.remove(0)
            assert len(lib.prompts) == 1
            assert lib.prompts[0].text == "Prompt 2"

    def test_get_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            lib = PromptLibrary(storage_path=path)
            lib.add("First")
            lib.add("Second")
            assert lib.get(1).text == "Second"
            assert lib.get(5) is None

    def test_get_by_category(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            lib = PromptLibrary(storage_path=path)
            lib.add("Prompt A", category="math")
            lib.add("Prompt B", category="math")
            lib.add("Prompt C", category="science")
            math_prompts = lib.get_by_category("math")
            assert len(math_prompts) == 2

    def test_get_by_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            lib = PromptLibrary(storage_path=path)
            lib.add("Prompt 1", tags=["ai", "ml"])
            lib.add("Prompt 2", tags=["ml"])
            lib.add("Prompt 3", tags=["ai"])
            ml_prompts = lib.get_by_tag("ml")
            assert len(ml_prompts) == 2

    def test_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            lib = PromptLibrary(storage_path=path)
            lib.add("The cat sat on the mat")
            lib.add("The dog ran in the park")
            lib.add("A cat is a feline")
            results = lib.search("cat")
            assert len(results) == 2

    def test_list_categories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            lib = PromptLibrary(storage_path=path)
            lib.add("P1", category="a")
            lib.add("P2", category="b")
            lib.add("P3", category="a")
            cats = lib.list_categories()
            assert set(cats) == {"a", "b"}

    def test_list_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            lib = PromptLibrary(storage_path=path)
            lib.add("P1", tags=["x", "y"])
            lib.add("P2", tags=["y", "z"])
            tags = lib.list_tags()
            assert tags == ["x", "y", "z"]

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.json"
            lib = PromptLibrary(storage_path=path)
            lib.add("P1")
            lib.add("P2")
            lib.clear()
            assert len(lib.prompts) == 0


class TestPromptDataset:
    def test_dataset_creation(self):
        dataset = PromptDataset(prompts=["a", "b", "c"])
        assert len(dataset) == 3

    def test_add(self):
        dataset = PromptDataset()
        dataset.add("test")
        assert len(dataset) == 1

    def test_add_batch(self):
        dataset = PromptDataset()
        dataset.add_batch(["a", "b", "c"])
        assert len(dataset) == 3

    def test_get(self):
        dataset = PromptDataset(["first", "second", "third"])
        assert dataset.get(1) == "second"
        assert dataset.get(10) is None

    def test_get_batch(self):
        dataset = PromptDataset(["a", "b", "c", "d", "e"])
        batch = dataset.get_batch([0, 2, 4])
        assert batch == ["a", "c", "e"]

    def test_filter_by_length(self):
        dataset = PromptDataset(["a", "ab", "abc", "abcd", "abcde"])
        filtered = dataset.filter_by_length(min_length=3, max_length=4)
        assert len(filtered) == 2

    def test_split(self):
        dataset = PromptDataset(list(range(10)))
        train, test = dataset.split(0.8)
        assert len(train) == 8
        assert len(test) == 2

    def test_iter(self):
        dataset = PromptDataset(["a", "b", "c"])
        items = list(dataset)
        assert items == ["a", "b", "c"]
