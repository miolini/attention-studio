from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


@dataclass
class ModelConfig:
    name: str = "gpt2"
    path: Path | None = None
    device: str = "auto"
    dtype: str = "float16"
    trust_remote_code: bool = True
    revision: str = "main"
    use_cache: bool = True


@dataclass
class LayerInfo:
    idx: int
    name: str
    type: str
    hidden_size: int
    num_heads: int = 0
    intermediate_size: int = 0


@dataclass
class InferenceResult:
    logits: torch.Tensor | None
    hidden_states: list[torch.Tensor] | None
    tokens: torch.Tensor
    token_strs: list[str]


class HiddenStatesHook:
    def __init__(self):
        self.states: list[torch.Tensor] | None = None

    def __call__(self, module, input, output):  # noqa: A002
        if isinstance(output, tuple):
            self.states = [output[0]]
        elif isinstance(output, torch.Tensor):
            self.states = [output]


class ModelManager:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface"
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._config: ModelConfig | None = None
        self._layer_info: list[LayerInfo] = []
        self._hooks: list[Any] = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        return self._tokenizer

    @property
    def layer_info(self) -> list[LayerInfo]:
        return self._layer_info

    def get_device(self) -> str:
        if self._model is None:
            return "cpu"
        if hasattr(self._model, "device"):
            return str(self._model.device)
        return "cuda" if torch.cuda.is_available() else "cpu"

    async def load_model(
        self,
        config: ModelConfig,
        progress: Callable[[float], None] | None = None,
    ) -> None:
        self._config = config

        logger.info(f"Loading model: {config.name}")

        source = config.path if config.path else config.name

        if progress:
            progress(0.1)

        logger.info(f"Loading tokenizer from {source}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=config.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if progress:
            progress(0.3)

        logger.info(f"Loading model weights from {source}")
        dtype = self._parse_dtype(config.dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=dtype,
            device_map=config.device,
            trust_remote_code=config.trust_remote_code,
            use_cache=config.use_cache,
            ignore_mismatched_sizes=True,
        )

        if progress:
            progress(0.8)

        self._extract_layer_info()

        if progress:
            progress(1.0)

        logger.info(f"Model loaded successfully. Device: {self.get_device()}")

    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float16)

    def _extract_layer_info(self) -> None:
        if self._model is None:
            return

        self._layer_info = []

        config = self._model.config
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size
        num_heads = config.n_head if hasattr(config, "n_head") else config.num_attention_heads
        num_layers = config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers

        if hasattr(self._model, "transformer"):
            for i in range(num_layers):
                layer = self._model.transformer.h[i]
                intermediate_size = 0
                if hasattr(layer, "mlp"):
                    mlp = layer.mlp
                    if hasattr(mlp, "c_fc"):
                        if hasattr(mlp.c_fc, "out_features"):
                            intermediate_size = mlp.c_fc.out_features
                        elif hasattr(mlp.c_fc, "weight"):
                            intermediate_size = mlp.c_fc.weight.shape[0]
                    elif hasattr(mlp, "fc1"):
                        if hasattr(mlp.fc1, "out_features"):
                            intermediate_size = mlp.fc1.out_features
                        elif hasattr(mlp.fc1, "weight"):
                            intermediate_size = mlp.fc1.weight.shape[0]
                self._layer_info.append(LayerInfo(
                    idx=i,
                    name=f"layer_{i}",
                    type="attention",
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                ))
            self._layer_info.append(LayerInfo(
                idx=num_layers,
                name="output",
                type="lm_head",
                hidden_size=hidden_size,
            ))
        elif hasattr(self._model, "model"):
            if hasattr(self._model.model, "layers"):
                for i, _layer in enumerate(self._model.model.layers):
                    self._layer_info.append(LayerInfo(
                        idx=i,
                        name=f"layer_{i}",
                        type="decoder_layer",
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                    ))

    def run_inference(
        self,
        prompt: str,
        return_logits: bool = True,
        return_hidden_states: bool = True,
        layer_indices: list[int] | None = None,
    ) -> InferenceResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        hook = HiddenStatesHook()

        with torch.no_grad():
            outputs = self._model(
                **inputs,
                output_hidden_states=return_hidden_states,
                return_dict=True,
            )

        return InferenceResult(
            logits=outputs.logits if return_logits else None,
            hidden_states=hook.states if return_hidden_states else None,
            tokens=inputs["input_ids"],
            token_strs=self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def get_attention_weights(
        self,
        prompt: str,
        layer_idx: int,
    ) -> dict[str, torch.Tensor]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        attention_weights: dict[str, torch.Tensor] = {}

        def hook_fn(module, input, output):  # noqa: A002
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights["attn_weights"] = output[0]
                attention_weights["attn_probs"] = output[1] if len(output) > 1 else None

        layer = self._model.transformer.h[layer_idx]
        handle = layer.attn.register_forward_hook(hook_fn)

        try:
            self.run_inference(prompt, return_hidden_states=False)
        finally:
            handle.remove()

        return attention_weights

    def get_layer_output(
        self,
        prompt: str,
        layer_idx: int,
    ) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        outputs = self._model(
            self._tokenizer(prompt, return_tensors="pt")["input_ids"].to(self._model.device),
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Hidden states not available")

        return hidden_states[layer_idx]

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._layer_info = []
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Model unloaded")

    def get_model_info(self) -> dict[str, Any]:
        if self._model is None:
            return {}

        config = self._model.config
        return {
            "name": self._config.name if self._config else "unknown",
            "num_parameters": sum(p.numel() for p in self._model.parameters()),
            "num_layers": config.n_layer if hasattr(config, "n_layer") else config.num_hidden_layers,
            "hidden_size": config.n_embd if hasattr(config, "n_embd") else config.hidden_size,
            "num_heads": config.n_head if hasattr(config, "n_head") else config.num_attention_heads,
            "vocab_size": config.vocab_size,
            "dtype": str(self._model.dtype),
            "device": self.get_device(),
        }
