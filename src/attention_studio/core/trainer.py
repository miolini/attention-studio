from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_studio.core.crm import (
    Lorsa,
    LorsaConfig,
    Transcoder,
    TranscoderConfig,
)


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 4
    epochs: int = 10
    warmup_steps: int = 100
    gradient_accumulation: int = 1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "cuda"
    mixed_precision: bool = True
    save_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    checkpoint_interval: int = 1
    log_interval: int = 10


class CRMTrainer:
    DICTIONARY_SIZE_PRESETS = [8192, 16384, 32768, 65536]
    TOP_K_PRESETS = [32, 64, 128, 256, 512]
    BATCH_SIZE_PRESETS = [2, 4, 8, 16]

    def __init__(
        self,
        model_manager: Any,
        config: TrainingConfig,
    ):
        self.model_manager = model_manager
        self.config = config
        if torch.backends.mps.is_available():
            self.device = "cpu"
            logger.warning("MPS detected but using CPU for training (MPS has stability issues with training)")
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        model_name = getattr(model_manager.model.config, "name_or_path", None) or getattr(model_manager.model.config, "model_name", "model")
        model_name = model_name.replace("/", "_")
        self.checkpoint_dir = self.config.save_dir / model_name
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        self.transcoders: list[Transcoder] = []
        self.lorsas: list[Lorsa] = []
        self.optimizers: list[torch.optim.Optimizer] = []
        self.schedulers: list[Any] = []
        self.current_epoch = 0
        self.global_step = 0
        self.history: list[dict[str, float]] = []

    def build_transcoders(
        self,
        layer_indices: list[int],
        config: TranscoderConfig,
    ) -> None:
        if self.model_manager is None or not self.model_manager.is_loaded:
            raise RuntimeError("Model not loaded")

        model_config = self.model_manager.model.config
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot determine hidden size from model config")

        logger.info(f"Building {len(layer_indices)} transcoders with hidden_size={hidden_size}")

        self.transcoders = nn.ModuleList([
            Transcoder(hidden_size, config) for _ in layer_indices
        ]).to(self.device)

        self.layer_indices = layer_indices

        self.optimizers.append(torch.optim.AdamW(
            self.transcoders.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        ))

        self.schedulers.append(torch.optim.lr_scheduler.LinearLR(
            self.optimizers[-1],
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        ))

    def build_lorsas(
        self,
        layer_indices: list[int],
        config: LorsaConfig,
    ) -> None:
        if self.model_manager is None or not self.model_manager.is_loaded:
            raise RuntimeError("Model not loaded")

        model_config = self.model_manager.model.config
        hidden_size = getattr(model_config, "n_embd", None) or getattr(model_config, "hidden_size", None)
        num_heads = getattr(model_config, "n_head", None) or getattr(model_config, "num_attention_heads", None)

        if hidden_size is None or num_heads is None:
            raise ValueError("Cannot determine hidden size or num heads from model config")

        config.num_heads = num_heads

        logger.info(f"Building {len(layer_indices)} Lorsa modules with hidden_size={hidden_size}, heads={num_heads}")

        self.lorsas = nn.ModuleList([
            Lorsa(hidden_size, config) for _ in layer_indices
        ]).to(self.device)

        self.lorsa_layer_indices = layer_indices

        self.optimizers.append(torch.optim.AdamW(
            self.lorsas.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        ))

        self.schedulers.append(torch.optim.lr_scheduler.LinearLR(
            self.optimizers[-1],
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        ))

    def train_transcoder(
        self,
        dataloader: DataLoader,
        layer_idx: int,
        resume: bool = False,
        progress_callback: Callable[[float, float], None] | None = None,
    ) -> dict[str, list[float]]:
        if not self.transcoders:
            raise RuntimeError("No transcoders built")

        tc_idx = self.layer_indices.index(layer_idx)
        transcoder = self.transcoders[tc_idx]
        optimizer = self.optimizers[0]
        scheduler = self.schedulers[0]

        if resume and self.current_epoch > 0:
            logger.info(f"Resuming training from epoch {self.current_epoch}")

        losses = []
        scaler = torch.amp.GradScaler("cuda") if self.config.mixed_precision and self.device == "cuda" else None

        model_device = self.model_manager.model.device
        if self.device != model_device:
            logger.info(f"Moving model from {model_device} to {self.device} for training")
            self.model_manager._model = self.model_manager._model.to(self.device).to(torch.float32)
            for tc in self.transcoders:
                tc.to(self.device).to(torch.float32)

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_losses = []

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.model_manager.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = outputs.hidden_states[layer_idx]

                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        reconstructed, features = transcoder(hidden_states)
                        loss = nn.functional.mse_loss(reconstructed, hidden_states)
                else:
                    reconstructed, features = transcoder(hidden_states)
                    loss = nn.functional.mse_loss(reconstructed, hidden_states)

                if self.config.gradient_accumulation > 1:
                    loss = loss / self.config.gradient_accumulation

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            transcoder.parameters(),
                            self.config.max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(
                            transcoder.parameters(),
                            self.config.max_grad_norm
                        )
                        optimizer.step()
                        optimizer.zero_grad()

                scheduler.step()

                epoch_losses.append(loss.item())
                self.global_step += 1

                pbar.set_postfix({"loss": loss.item()})

                if progress_callback and batch_idx % self.config.log_interval == 0:
                    progress_callback(
                        (epoch * len(dataloader) + batch_idx) / (self.config.epochs * len(dataloader)),
                        loss.item(),
                    )

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            self.history.append({"epoch": epoch, "loss": avg_loss, "layer": layer_idx})

            logger.info(f"Epoch {epoch+1}: loss = {avg_loss:.6f}")

            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(layer_idx, epoch)

        return {"losses": losses}

    def train_all_transcoders(
        self,
        dataloader: DataLoader,
        progress_callback: Callable[[float, float], None] | None = None,
    ) -> dict[int, dict[str, list[float]]]:
        results = {}

        for layer_idx in self.layer_indices:
            logger.info(f"Training transcoder for layer {layer_idx}")
            results[layer_idx] = self.train_transcoder(
                dataloader,
                layer_idx,
                progress_callback=progress_callback,
            )

        return results

    def save_checkpoint(self, layer_idx: int, epoch: int) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "layer_idx": layer_idx,
            "config": {
                "dictionary_size": self.transcoders[0].config.dictionary_size,
                "top_k": self.transcoders[0].config.top_k,
            },
            "transcoder_state": self.transcoders[0].state_dict(),
        }

        if len(self.optimizers) > 0:
            checkpoint["optimizer_state"] = self.optimizers[0].state_dict()

        path = self.checkpoint_dir / f"transcoder_l{layer_idx}_e{epoch}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        return path

    def load_checkpoint(self, path: Path, layer_idx: int) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        if "transcoder_state" in checkpoint:
            tc_idx = self.layer_indices.index(layer_idx)
            self.transcoders[tc_idx].load_state_dict(checkpoint["transcoder_state"])

        if "optimizer_state" in checkpoint and len(self.optimizers) > 0:
            self.optimizers[0].load_state_dict(checkpoint["optimizer_state"])

        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)

        logger.info(f"Checkpoint loaded from {path}")

    def get_transcoder(self, layer_idx: int) -> Transcoder | None:
        if layer_idx in self.layer_indices:
            tc_idx = self.layer_indices.index(layer_idx)
            return self.transcoders[tc_idx]
        return None

    def list_checkpoints(self, layer_idx: int | None = None) -> list[Path]:
        if not self.checkpoint_dir.exists():
            return []

        pattern = f"transcoder_l{layer_idx}_*.pt" if layer_idx is not None else "transcoder_l*.pt"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints

    def get_latest_checkpoint(self, layer_idx: int) -> Path | None:
        checkpoints = self.list_checkpoints(layer_idx)
        return checkpoints[0] if checkpoints else None

    def load_latest_checkpoint(self, layer_idx: int) -> bool:
        checkpoint_path = self.get_latest_checkpoint(layer_idx)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path, layer_idx)
            return True
        return False

    def get_training_history(self) -> list[dict[str, float]]:
        return self.history
