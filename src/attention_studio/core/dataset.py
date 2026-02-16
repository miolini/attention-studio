from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


@dataclass
class DatasetConfig:
    source: str = "c4"
    token_limit: int = 100000
    custom_path: Path | None = None
    format: str = "jsonl"
    shuffle: bool = True
    seed: int = 42


class TextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class DatasetManager:
    TOKEN_LIMIT_PRESETS = [10000, 50000, 100000]
    FORMAT_PRESETS = ["jsonl", "csv", "txt"]

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "attention_studio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer: AutoTokenizer | None = None
        self._texts: list[str] = []
        self._config: DatasetConfig | None = None

    @property
    def texts(self) -> list[str]:
        return self._texts

    @property
    def num_tokens(self) -> int:
        if not self._texts or self._tokenizer is None:
            return 0
        total = 0
        for text in self._texts[:1000]:
            total += len(self._tokenizer.encode(text))
        return (total // 1000) * len(self._texts)

    async def load(
        self,
        config: DatasetConfig,
        tokenizer: AutoTokenizer,
        progress: Any | None = None,
    ) -> None:
        self._config = config
        self._tokenizer = tokenizer

        logger.info(f"Loading dataset: source={config.source}, tokens={config.token_limit}")

        if config.source == "c4":
            self._texts = await self._load_c4(config.token_limit, progress)
        elif config.source == "custom":
            self._texts = self._load_custom(config.custom_path, config.format)
        else:
            raise ValueError(f"Unknown dataset source: {config.source}")

        if config.shuffle:
            random.seed(config.seed)
            random.shuffle(self._texts)

        logger.info(f"Loaded {len(self._texts)} texts, ~{self.num_tokens} tokens")

    async def _load_c4(self, token_limit: int, progress: Any | None = None) -> list[str]:
        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning("datasets not installed, using mock data")
            return self._generate_mock_data(token_limit)

        logger.info("Loading C4 dataset...")
        try:
            ds = load_dataset(
                "c4",
                "realnews",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception:
            ds = load_dataset(
                "c4",
                "en",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )

        texts = []
        total_tokens = 0
        target_tokens = token_limit

        for i, example in enumerate(ds):
            if total_tokens >= target_tokens:
                break

            text = example.get("text", "")
            if not text or len(text) < 50:
                continue

            texts.append(text)
            total_tokens += len(text.split())

            if progress and i % 100 == 0:
                progress(min(total_tokens / target_tokens, 1.0))

        return texts

    def _load_custom(self, path: Path | None, fmt: str) -> list[str]:
        if path is None or not path.exists():
            logger.warning("Custom path not found, using mock data")
            return self._generate_mock_data(10000)

        texts = []

        if fmt == "jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            text = obj.get("text", obj.get("content", ""))
                        else:
                            text = str(obj)
                        if text:
                            texts.append(text)
                    except json.JSONDecodeError:
                        continue
        elif fmt == "txt":
            with open(path, encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        elif fmt == "csv":
            import csv
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text", row.get("content", ""))
                    if text:
                        texts.append(text)

        logger.info(f"Loaded {len(texts)} texts from {path}")
        return texts

    def _generate_mock_data(self, token_limit: int) -> list[str]:
        templates = [
            "The quick brown fox jumps over the lazy dog.",
            "In a world where artificial intelligence continues to evolve, researchers are exploring new ways to understand neural networks.",
            "Machine learning models consist of many parameters that work together to process information.",
            "Natural language processing enables computers to understand human language.",
            "Transformers have revolutionized the field of deep learning.",
            "Attention mechanisms allow models to focus on relevant parts of the input.",
            "Sparse autoencoders can extract interpretable features from neural networks.",
            "Mechanistic interpretability seeks to understand how models compute their outputs.",
            "The model processes tokens and generates predictions one at a time.",
            "Understanding the internal representations of language models is an important research area.",
        ]
        texts = []
        while sum(len(t.split()) for t in texts) < token_limit:
            texts.extend(templates)
        return texts[:len(texts) // 10]

    def create_dataloader(
        self,
        batch_size: int = 4,
        max_length: int = 512,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        if not self._texts:
            raise RuntimeError("No texts loaded. Call load() first.")

        dataset = TextDataset(self._texts, self._tokenizer, max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_batch_iterator(
        self,
        batch_size: int = 4,
        max_length: int = 512,
    ) -> Iterator[dict[str, torch.Tensor]]:
        if not self._texts or self._tokenizer is None:
            raise RuntimeError("No texts loaded. Call load() first.")

        for i in range(0, len(self._texts), batch_size):
            batch_texts = self._texts[i:i + batch_size]
            encodings = self._tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            yield {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
            }

    def clear(self) -> None:
        self._texts = []
        self._config = None
        logger.info("Dataset cleared")

    def get_stats(self) -> dict[str, Any]:
        return {
            "num_texts": len(self._texts),
            "num_tokens_approx": self.num_tokens,
            "config": self._config.__dict__ if self._config else None,
        }
