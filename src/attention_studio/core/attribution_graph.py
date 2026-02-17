from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import torch
import torch.nn as nn


@dataclass
class AttributionNode:
    node_id: str
    node_type: str
    layer: int
    position: int
    feature_idx: int | None
    token: str | None
    activation: float
    encoder_vec: torch.Tensor | None
    decoder_vec: torch.Tensor | None


@dataclass
class AttributionEdge:
    source_id: str
    target_id: str
    weight: float
    edge_type: str
    attention_pattern: torch.Tensor | None = None
    virtual_weight: torch.Tensor | None = None


@dataclass
class CompleteAttributionGraph:
    graph: nx.DiGraph
    nodes: dict[str, AttributionNode] = field(default_factory=dict)
    edges: dict[tuple[str, str], AttributionEdge] = field(default_factory=dict)
    prompt: str = ""
    tokens: list[str] = field(default_factory=list)


@dataclass
class QKTracingResult:
    source_pos: int
    target_pos: int
    attention_score: float
    feature_contributions: list[dict[str, Any]]
    pairwise_contributions: list[dict[str, Any]]


@dataclass
class CircuitPath:
    source: tuple[int, int, int]
    target: tuple[int, int, int]
    path: list[str]
    total_weight: float
    path_type: str


class AttributionGraphBuilder:
    def __init__(
        self,
        model_manager: Any,
        transcoders: nn.ModuleList,
        lorsas: nn.ModuleList | None = None,
        layer_indices: list[int] | None = None,
    ):
        self.model_manager = model_manager
        self.transcoders = transcoders
        self.lorsas = lorsas
        self.layer_indices = layer_indices or list(range(len(transcoders)))
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

    def build_complete_attribution_graph(
        self,
        prompt: str,
        threshold: float = 0.01,
    ) -> CompleteAttributionGraph:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        seq_len = len(tokens)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        graph = nx.DiGraph()
        nodes = {}
        edges = {}

        node_id = 0
        for pos in range(seq_len):
            emb_node_id = f"emb_{pos}"
            nodes[emb_node_id] = AttributionNode(
                node_id=emb_node_id,
                node_type="embedding",
                layer=0,
                position=pos,
                feature_idx=None,
                token=tokens[pos],
                activation=1.0,
                encoder_vec=None,
                decoder_vec=None,
            )
            graph.add_node(emb_node_id)

        all_activations = []
        for tc_idx, layer in enumerate(self.layer_indices):
            transcoder = self.transcoders[tc_idx]
            hidden_states = outputs.hidden_states[layer]
            _, features = transcoder(hidden_states)
            features = features.squeeze(0)

            all_activations.append((layer, features))

        for tc_idx, layer in enumerate(self.layer_indices):
            _, features = all_activations[tc_idx]

            for pos in range(seq_len):
                feature_activations = features[pos]

                for feat_idx in range(len(feature_activations)):
                    activation = feature_activations[feat_idx].item()
                    if abs(activation) < threshold:
                        continue

                    node_id = f"tc_{layer}_{pos}_{feat_idx}"
                    transcoder = self.transcoders[tc_idx]

                    decoder_weight = transcoder.decoder.weight
                    encoder_weight = transcoder.encoder.weight

                    nodes[node_id] = AttributionNode(
                        node_id=node_id,
                        node_type="transcoder",
                        layer=layer,
                        position=pos,
                        feature_idx=feat_idx,
                        token=tokens[pos] if pos < len(tokens) else None,
                        activation=activation,
                        encoder_vec=encoder_weight[:, feat_idx].detach().clone(),
                        decoder_vec=decoder_weight[feat_idx].detach().clone(),
                    )
                    graph.add_node(node_id)

        if self.lorsas:
            for lc_idx, layer in enumerate(self.layer_indices):
                if lc_idx >= len(self.lorsas):
                    continue

                lorsa = self.lorsas[lc_idx]
                hidden_states = outputs.hidden_states[layer]

                batch, seq_len, hidden_dim = hidden_states.shape
                q = lorsa.W_Q(hidden_states)  # noqa: N806
                k = lorsa.W_K(hidden_states)  # noqa: N806
                v = lorsa.W_V(hidden_states)  # noqa: N806

                q = q.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
                k = k.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
                v = v.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806

                if hasattr(lorsa, 'q_layernorm'):
                    q = lorsa.q_layernorm(q)  # noqa: N806
                    k = lorsa.k_layernorm(k)  # noqa: N806

                scores = torch.matmul(q, k.transpose(-2, -1)) / (lorsa.head_dim ** 0.5)
                attn_probs = torch.softmax(scores, dim=-1)

                sparse_v = lorsa.sparse_W_V
                sparse_o = lorsa.sparse_W_O

                for head in range(lorsa.num_heads):
                    attn_pattern = attn_probs[0, head]

                    for pos in range(seq_len):
                        max_attn, max_pos = attn_pattern[pos].max(dim=0)
                        if max_attn.item() < threshold:
                            continue

                        node_id = f"lorsa_{layer}_{pos}_{head}"
                        nodes[node_id] = AttributionNode(
                            node_id=node_id,
                            node_type="lorsa",
                            layer=layer,
                            position=pos,
                            feature_idx=head,
                            token=tokens[pos] if pos < len(tokens) else None,
                            activation=max_attn.item(),
                            encoder_vec=sparse_v[head].detach().clone() if head < len(sparse_v) else None,
                            decoder_vec=sparse_o[head].detach().clone() if head < len(sparse_o) else None,
                        )
                        graph.add_node(node_id)

        for tc_idx, layer in enumerate(self.layer_indices):
            _, features = all_activations[tc_idx]

            for pos in range(seq_len):
                feature_activations = features[pos]

                for feat_idx in range(len(feature_activations)):
                    source_node_id = f"tc_{layer}_{pos}_{feat_idx}"
                    if source_node_id not in nodes:
                        continue

                    source_activation = feature_activations[feat_idx].item()

                    if tc_idx + 1 < len(self.layer_indices):
                        next_layer = self.layer_indices[tc_idx + 1]
                        next_tc_idx = tc_idx + 1

                        transcoder = self.transcoders[next_tc_idx]
                        decoder_weight = transcoder.decoder.weight
                        encoder_weight = transcoder.encoder.weight

                        for next_pos in range(seq_len):
                            for next_feat_idx in range(decoder_weight.shape[0]):
                                if source_node_id not in nodes:
                                    continue

                                source_decoder = nodes[source_node_id].decoder_vec
                                if source_decoder is None:
                                    continue

                                target_encoder = encoder_weight[:, next_feat_idx]

                                virtual_weight = torch.dot(source_decoder, target_encoder).item()

                                if abs(virtual_weight * source_activation) < threshold:
                                    continue

                                target_node_id = f"tc_{next_layer}_{next_pos}_{next_feat_idx}"

                                edge = AttributionEdge(
                                    source_id=source_node_id,
                                    target_id=target_node_id,
                                    weight=virtual_weight * source_activation,
                                    edge_type="mlp",
                                    attention_pattern=None,
                                    virtual_weight=torch.tensor(virtual_weight),
                                )
                                edges[(source_node_id, target_node_id)] = edge
                                graph.add_edge(source_node_id, target_node_id, weight=edge.weight)

        for lc_idx, layer in enumerate(self.layer_indices):
            if lc_idx >= len(self.lorsas):
                continue

            lorsa = self.lorsas[lc_idx]
            hidden_states = outputs.hidden_states[layer]

            batch, seq_len, hidden_dim = hidden_states.shape
            q = lorsa.W_Q(hidden_states)  # noqa: N806
            k = lorsa.W_K(hidden_states)  # noqa: N806
            v = lorsa.W_V(hidden_states)  # noqa: N806

            q = q.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
            k = k.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
            v = v.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806

            if hasattr(lorsa, 'q_layernorm'):
                q = lorsa.q_layernorm(q)  # noqa: N806
                k = lorsa.k_layernorm(k)  # noqa: N806

            scores = torch.matmul(q, k.transpose(-2, -1)) / (lorsa.head_dim ** 0.5)
            attn_probs = torch.softmax(scores, dim=-1)

            sparse_v = lorsa.sparse_W_V
            sparse_o = lorsa.sparse_W_O

            for head in range(lorsa.num_heads):
                attn_pattern = attn_probs[0, head]

                for source_pos in range(seq_len):
                    source_node_id = f"lorsa_{layer}_{source_pos}_{head}"
                    if source_node_id not in nodes:
                        continue

                    if head >= len(sparse_v):
                        continue
                    source_decoder = sparse_v[head]

                    for target_pos in range(source_pos, seq_len):
                        attn_score = attn_pattern[source_pos, target_pos].item()
                        if attn_score < threshold:
                            continue

                        transcoder_layer = layer
                        if transcoder_layer in self.layer_indices:
                            tc_idx = self.layer_indices.index(transcoder_layer)
                            if tc_idx < len(self.transcoders):
                                transcoder = self.transcoders[tc_idx]
                                decoder_weight = transcoder.decoder.weight

                                for feat_idx in range(min(10, decoder_weight.shape[0])):
                                    target_encoder = decoder_weight[feat_idx]

                                    virtual_weight = torch.dot(source_decoder, target_encoder).item()

                                    if abs(virtual_weight * attn_score) < threshold:
                                        continue

                                    target_node_id = f"tc_{transcoder_layer}_{target_pos}_{feat_idx}"
                                    if target_node_id not in nodes:
                                        continue

                                    edge = AttributionEdge(
                                        source_id=source_node_id,
                                        target_id=target_node_id,
                                        weight=virtual_weight * attn_score,
                                        edge_type="attention",
                                        attention_pattern=attn_pattern[source_pos, target_pos].detach().clone(),
                                        virtual_weight=torch.tensor(virtual_weight),
                                    )
                                    edges[(source_node_id, target_node_id)] = edge
                                    graph.add_edge(source_node_id, target_node_id, weight=edge.weight)

        for pos in range(seq_len):
            emb_node_id = f"emb_{pos}"

            if pos + 1 < len(self.layer_indices):
                next_layer = self.layer_indices[0]
                if next_layer in self.layer_indices:
                    tc_idx = self.layer_indices.index(next_layer)
                    if tc_idx < len(self.transcoders):
                        transcoder = self.transcoders[tc_idx]
                        encoder_weight = transcoder.encoder.weight

                        for feat_idx in range(min(10, encoder_weight.shape[1])):
                            target_encoder = encoder_weight[:, feat_idx]
                            embedding = self.model.embed_tokens(input_ids[0][pos])

                            virtual_weight = torch.dot(embedding, target_encoder).item()

                            if abs(virtual_weight) < threshold:
                                continue

                            target_node_id = f"tc_{next_layer}_{pos + 1}_{feat_idx}"
                            if target_node_id not in nodes:
                                continue

                            edge = AttributionEdge(
                                source_id=emb_node_id,
                                target_id=target_node_id,
                                weight=virtual_weight,
                                edge_type="embedding",
                                attention_pattern=None,
                                virtual_weight=torch.tensor(virtual_weight),
                            )
                            edges[(emb_node_id, target_node_id)] = edge
                            graph.add_edge(emb_node_id, target_node_id, weight=edge.weight)

        return CompleteAttributionGraph(
            graph=graph,
            nodes=nodes,
            edges=edges,
            prompt=prompt,
            tokens=tokens,
        )

    def compute_qk_tracing(
        self,
        prompt: str,
        layer_idx: int,
        head_idx: int | None = None,
    ) -> list[QKTracingResult]:
        if not self.lorsas:
            return []

        lc_idx = None
        for i, layer in enumerate(self.layer_indices):
            if layer == layer_idx:
                lc_idx = i
                break

        if lc_idx is None or lc_idx >= len(self.lorsas):
            return []

        lorsa = self.lorsas[lc_idx]

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        seq_len = len(tokens)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states[layer_idx]

        batch, seq_len, hidden_dim = hidden_states.shape
        Q = lorsa.W_Q(hidden_states)  # noqa: N806
        K = lorsa.W_K(hidden_states)  # noqa: N806

        Q = Q.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
        K = K.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806

        if hasattr(lorsa, 'q_layernorm'):
            Q = lorsa.q_layernorm(Q)  # noqa: N806
            K = lorsa.k_layernorm(K)  # noqa: N806

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (lorsa.head_dim ** 0.5)

        results = []
        heads_to_analyze = [head_idx] if head_idx is not None else range(lorsa.num_heads)

        for head in heads_to_analyze:
            for target_pos in range(seq_len):
                query_vector = Q[0, head, target_pos]

                tc_idx = lc_idx
                if tc_idx < len(self.transcoders):
                    transcoder = self.transcoders[tc_idx]
                    encoder_weight = transcoder.encoder.weight

                    feature_contributions = []
                    for feat_idx in range(min(20, encoder_weight.shape[1])):
                        key_vector = encoder_weight[:, feat_idx]
                        contribution = torch.dot(query_vector, key_vector).item()
                        if abs(contribution) > 0.01:
                            feature_contributions.append({
                                "feature_idx": feat_idx,
                                "contribution": contribution,
                                "token": tokens[target_pos],
                            })

                pairwise_contributions = []
                for source_pos in range(target_pos):
                    key_vector = K[0, head, source_pos]
                    score = scores[0, head, target_pos, source_pos].item()

                    pairwise_contributions.append({
                        "source_pos": source_pos,
                        "target_pos": target_pos,
                        "source_token": tokens[source_pos],
                        "target_token": tokens[target_pos],
                        "score": score,
                    })

                pairwise_contributions.sort(key=lambda x: abs(x["score"]), reverse=True)
                pairwise_contributions = pairwise_contributions[:10]

                results.append(QKTracingResult(
                    source_pos=target_pos,
                    target_pos=target_pos,
                    attention_score=scores[0, head, target_pos, :].mean().item(),
                    feature_contributions=feature_contributions,
                    pairwise_contributions=pairwise_contributions,
                ))

        return results

    def find_global_circuits(
        self,
        threshold: float = 0.1,
    ) -> dict[str, list[CircuitPath]]:
        circuits = {
            "induction": [],
            "copy": [],
            "prev_token": [],
            "duplicate_tokens": [],
        }

        if not self.lorsas:
            return circuits

        for lc_idx, layer in enumerate(self.layer_indices):
            if lc_idx >= len(self.lorsas):
                continue

            lorsa = self.lorsas[lc_idx]
            num_features = lorsa.sparse_W_V.shape[0] if hasattr(lorsa, 'sparse_W_V') else lorsa.num_heads

            for head in range(min(num_features, 4)):
                if head >= len(lorsa.sparse_W_V):
                    continue

                w_v = lorsa.sparse_W_V[head]
                w_o = lorsa.sparse_W_O[head] if head < len(lorsa.sparse_W_O) else lorsa.sparse_W_O[0]

                ov_strength = torch.norm(w_v) * torch.norm(w_o)
                ov_strength = ov_strength.item()

                if ov_strength > threshold:
                    circuits["induction"].append(CircuitPath(
                        source=(layer, head, 0),
                        target=(layer, head, 1),
                        path=[f"lorsa_{layer}_{head}"],
                        total_weight=ov_strength,
                        path_type="ov",
                    ))

        for lc_idx, layer in enumerate(self.layer_indices):
            if lc_idx >= len(self.lorsas):
                continue

            lorsa = self.lorsas[lc_idx]

            for head in range(lorsa.num_heads):
                w_v = lorsa.sparse_W_V[head] if head < len(lorsa.sparse_W_V) else None
                if w_v is None:
                    continue

                w_v_norm = torch.norm(w_v).item()

                if w_v_norm > threshold:
                    circuits["prev_token"].append(CircuitPath(
                        source=(layer, head, 0),
                        target=(layer, head, 0),
                        path=[f"lorsa_{layer}_{head}_prev"],
                        total_weight=w_v_norm,
                        path_type="prev_token",
                    ))

        for tc_idx, layer in enumerate(self.layer_indices):
            transcoder = self.transcoders[tc_idx]
            decoder_weight = transcoder.decoder.weight

            for feat_idx in range(min(decoder_weight.shape[0], 50)):
                feat_vec = decoder_weight[feat_idx]
                feat_norm = torch.norm(feat_vec).item()

                if feat_norm > threshold * 10:
                    circuits["duplicate_tokens"].append(CircuitPath(
                        source=(layer, feat_idx, 0),
                        target=(layer, feat_idx, 1),
                        path=[f"tc_{layer}_{feat_idx}"],
                        total_weight=feat_norm,
                        path_type="feature",
                    ))

        return circuits
