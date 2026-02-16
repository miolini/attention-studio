from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def top_k_activation(x: torch.Tensor, k: int) -> torch.Tensor:
    if k >= x.shape[-1]:
        return x
    top_k_values, top_k_indices = torch.topk(x, k, dim=-1)
    output = torch.full_like(x, 0.0)
    output.scatter_(-1, top_k_indices, top_k_values)
    return output


@dataclass
class TranscoderConfig:
    dictionary_size: int = 32768
    top_k: int = 128
    encoder_bias: bool = True
    decoder_bias: bool = True
    activation: str = "relu"
    layernorm_after_encoder: bool = True
    layernorm_after_decoder: bool = False
    zero_init: bool = False


class Transcoder(nn.Module):
    def __init__(self, input_dim: int, config: TranscoderConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        self.encoder = nn.Linear(input_dim, config.dictionary_size, bias=config.encoder_bias)
        self.decoder = nn.Linear(config.dictionary_size, input_dim, bias=config.decoder_bias)

        if config.layernorm_after_encoder:
            self.encoder_ln = nn.LayerNorm(config.dictionary_size)
        if config.layernorm_after_decoder:
            self.decoder_ln = nn.LayerNorm(input_dim)

        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "prelu":
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        if self.config.zero_init:
            nn.init.zeros_(self.decoder.weight)
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias)
        else:
            nn.init.orthogonal_(self.encoder.weight, gain=1.0)
            nn.init.orthogonal_(self.decoder.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, dim = x.shape

        z = self.encoder(x)

        if hasattr(self, "encoder_ln"):
            z = self.encoder_ln(z)

        z = self.activation(z)
        features = top_k_activation(z, k=self.config.top_k)

        output = self.decoder(features)

        if hasattr(self, "decoder_ln"):
            output = self.decoder_ln(output)

        return output, features

    def get_virtual_weights(self) -> dict[str, torch.Tensor]:
        return {
            "W_enc": self.encoder.weight,
            "W_dec": self.decoder.weight,
            "b_enc": self.encoder.bias if self.encoder.bias is not None else None,
            "b_dec": self.decoder.bias if self.decoder.bias is not None else None,
        }

    def get_feature_norms(self) -> torch.Tensor:
        return torch.norm(self.decoder.weight, dim=1)

    def get_encoder_norms(self) -> torch.Tensor:
        return torch.norm(self.encoder.weight, dim=1)


@dataclass
class LorsaConfig:
    num_heads: int = 12
    top_k: int = 128
    qk_layernorm: bool = True
    use_rope: bool = False
    grouped_query_attention: bool = False
    num_kv_heads: int | None = None


class Lorsa(nn.Module):
    def __init__(self, hidden_size: int, config: LorsaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // config.num_heads
        self.num_heads = config.num_heads

        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_O = nn.Linear(hidden_size, hidden_size, bias=False)

        self.sparse_W_V = nn.Parameter(torch.randn(config.num_heads, self.head_dim, config.top_k) * 0.02)
        self.sparse_W_O = nn.Parameter(torch.randn(config.num_heads, config.top_k, self.head_dim) * 0.02)

        if config.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(self.head_dim)
            self.k_layernorm = nn.LayerNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x)  # noqa: N806
        K = self.W_K(x)  # noqa: N806
        V = self.W_V(x)  # noqa: N806

        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # noqa: N806
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # noqa: N806
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # noqa: N806

        if self.config.qk_layernorm:
            Q = self.q_layernorm(Q)  # noqa: N806
            K = self.k_layernorm(K)  # noqa: N806

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_probs = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_probs, V)

        batch_h = batch * self.num_heads
        context_flat = context.transpose(1, 2).reshape(batch_h, seq_len, self.head_dim)

        v_weights = self.sparse_W_V.mean(2)
        z = torch.einsum("bnd,hd->bhn", context_flat, v_weights)
        z_sparse = top_k_activation(z, k=self.config.top_k)

        o_weights = self.sparse_W_O.mean(1)
        y_flat = torch.einsum("bhn,hd->bnd", z_sparse, o_weights)
        y = y_flat.reshape(batch, self.num_heads, seq_len, self.head_dim)
        y = y.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
        y = self.W_O(y)

        info = {
            "attention_pattern": attn_probs,
            "z_pattern": z,
            "z_sparse": z_sparse,
            "Q": Q,
            "K": K,
            "V": V,
        }

        return y, info

    def get_qk_circuit(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.W_Q.weight, self.W_K.weight

    def get_ov_circuit(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sparse_W_V, self.sparse_W_O


class CompleteReplacementModel(nn.Module):
    def __init__(
        self,
        transcoders: nn.ModuleList,
        lorsas: nn.ModuleList | None = None,
        layer_indices: list | None = None,
    ):
        super().__init__()
        self.transcoders = transcoders
        self.lorsas = lorsas
        self.layer_indices = layer_indices or list(range(len(transcoders)))

    def forward(self, hidden_states: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, dict]:
        if layer_idx in self.layer_indices:
            tc_idx = self.layer_indices.index(layer_idx)
            if tc_idx < len(self.transcoders):
                return self.transcoders[tc_idx](hidden_states)

        return hidden_states, {}

    def get_crm_info(self) -> dict[str, Any]:
        info = {
            "num_transcoders": len(self.transcoders),
            "num_lorsas": len(self.lorsas) if self.lorsas else 0,
            "layer_indices": self.layer_indices,
        }
        return info
