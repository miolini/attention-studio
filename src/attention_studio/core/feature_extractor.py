from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import torch


@dataclass
class FeatureInfo:
    idx: int
    layer: int
    activation: float
    norm: float
    top_contexts: list[dict[str, Any]] = None


@dataclass
class DLAResult:
    feature_logits: torch.Tensor
    feature_contributions: dict[int, float]


@dataclass
class CircuitInfo:
    source_feature: int
    target_feature: int
    source_layer: int
    target_layer: int
    circuit_type: str
    strength: float


@dataclass
class GlobalCircuit:
    name: str
    circuit_type: str
    features: list[tuple[int, int]]
    strength: float


class FeatureExtractor:
    def __init__(self, model_manager: Any, transcoder: Any, layer_idx: int):
        self.model_manager = model_manager
        self.transcoder = transcoder
        self.layer_idx = layer_idx

    def extract_features(
        self,
        prompt: str,
        top_k: int = 100,
    ) -> list[FeatureInfo]:
        if self.model_manager is None or not self.model_manager.is_loaded:
            raise RuntimeError("Model not loaded")

        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[self.layer_idx]

        _, features = self.transcoder(hidden_states)

        features = features.squeeze(0)
        feature_activations = features.mean(dim=0)

        top_indices = torch.topk(feature_activations, min(top_k, len(feature_activations)))

        feature_infos = []
        for _i, (idx, act) in enumerate(zip(top_indices.indices, top_indices.values, strict=True)):
            feature_idx_val = idx.item()
            norm = torch.norm(self.transcoder.decoder.weight[:, feature_idx_val]).item()
            feature_infos.append(FeatureInfo(
                idx=feature_idx_val,
                layer=self.layer_idx,
                activation=act.item(),
                norm=norm,
            ))

        return feature_infos

    def get_top_contexts(
        self,
        prompt: str,
        feature_idx: int,
        k: int = 10,
    ) -> list[dict[str, Any]]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[self.layer_idx]

        _, features = self.transcoder(hidden_states)

        feature_values = features[0, :, feature_idx]
        top_positions = torch.topk(feature_values, min(k, len(feature_values)))

        token_strs = tokenizer.convert_ids_to_tokens(input_ids[0])

        contexts = []
        for pos, val in zip(top_positions.indices, top_positions.values, strict=True):
            contexts.append({
                "position": pos.item(),
                "token": token_strs[pos.item()],
                "activation": val.item(),
            })

        return contexts

    def compute_dla(
        self,
        prompt: str,
        layer_idx: int | None = None,
    ) -> DLAResult:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model
        layer = layer_idx or self.layer_idx

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states[layer]

        feature_activations = hidden_states @ self.transcoder.encoder.weight.T

        feature_logits = feature_activations @ self.transcoder.decoder.weight @ model.lm_head.weight

        last_token_logits = feature_logits[0, -1, :]

        contributions = {}
        for i in range(last_token_logits.shape[0]):
            if last_token_logits[i] > 0:
                contributions[i] = last_token_logits[i].item()

        return DLAResult(
            feature_logits=feature_logits,
            feature_contributions=contributions,
        )


class GraphBuilder:
    def __init__(
        self,
        model_manager: Any,
        transcoders: list[Any],
        layer_indices: list[int],
    ):
        self.model_manager = model_manager
        self.transcoders = transcoders
        self.layer_indices = layer_indices

    def build_attribution_graph(
        self,
        prompt: str,
        threshold: float = 0.01,
    ) -> nx.DiGraph:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states_list = outputs.hidden_states

        graph = nx.DiGraph()

        for layer_idx in self.layer_indices:
            hidden_states = hidden_states_list[layer_idx]
            tc_idx = self.layer_indices.index(layer_idx)
            transcoder = self.transcoders[tc_idx]

            _, features = transcoder(hidden_states)

            features = features.squeeze(0)
            feature_activations = features.mean(dim=0)

            for feat_idx, act in enumerate(feature_activations):
                if act.item() > threshold:
                    node_id = f"L{layer_idx}_F{feat_idx}"
                    graph.add_node(
                        node_id,
                        layer=layer_idx,
                        feature=feat_idx,
                        activation=act.item(),
                    )

            virtual_weights = transcoder.decoder.weight

            for i in range(min(100, virtual_weights.shape[0])):
                if feature_activations[i].item() > threshold:
                    for j in range(min(100, virtual_weights.shape[1])):
                        weight = virtual_weights[i, j].item()
                        if abs(weight) > threshold:
                            src = f"L{layer_idx}_F{i}"
                            dst = f"L{layer_idx}_F{j}"
                            if graph.has_edge(src, dst):
                                graph[src][dst]["weight"] += weight
                            else:
                                graph.add_edge(src, dst, weight=weight)

        self._add_layer_edges(graph)

        return graph

    def _add_layer_edges(self, graph: nx.DiGraph):
        for i in range(len(self.layer_indices) - 1):
            src_layer = self.layer_indices[i]
            dst_layer = self.layer_indices[i + 1]

            for feat in range(min(100, self.transcoders[i].config.dictionary_size)):
                src_node = f"L{src_layer}_F{feat}"
                dst_node = f"L{dst_layer}_F{feat}"

                if src_node in graph.nodes and dst_node in graph.nodes:
                    graph.add_edge(src_node, dst_node, weight=1.0, type="layer")

    def find_paths(
        self,
        graph: nx.DiGraph,
        source: str,
        target: str,
        max_length: int = 10,
    ) -> list[list[str]]:
        try:
            paths = list(nx.all_simple_paths(
                graph,
                source,
                target,
                cutoff=max_length,
            ))
            return paths
        except nx.NetworkXNoPath:
            return []

    def extract_subgraph(
        self,
        graph: nx.DiGraph,
        node: str,
        depth: int = 2,
    ) -> nx.DiGraph:
        ancestors = set()
        descendants = set()

        if depth > 0:
            ancestors = nx.ancestors(graph, node)
            descendants = nx.descendants(graph, node)

        nodes = {node} | ancestors | descendants

        return graph.subgraph(nodes).copy()

    def get_graph_stats(self, graph: nx.DiGraph) -> dict[str, Any]:
        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_directed": graph.is_directed(),
        }


class GlobalCircuitAnalyzer:
    def __init__(
        self,
        model_manager: Any,
        transcoders: list[Any],
        lorsas: list[Any] | None = None,
        layer_indices: list[int] | None = None,
    ):
        self.model_manager = model_manager
        self.transcoders = transcoders
        self.lorsas = lorsas
        self.layer_indices = layer_indices or list(range(len(transcoders)))

    def compute_feature_circuits(
        self,
        layer_idx: int,
        feature_idx: int,
    ) -> dict[str, Any]:
        if layer_idx not in self.layer_indices:
            raise ValueError(f"Layer {layer_idx} not in layer_indices")

        tc_idx = self.layer_indices.index(layer_idx)
        transcoder = self.transcoders[tc_idx]

        decoder_weight = transcoder.decoder.weight
        encoder_weight = transcoder.encoder.weight

        feature_vec = decoder_weight[feature_idx]
        input_vec = encoder_weight[:, feature_idx]

        qk_circuit = None
        ov_circuit = None
        if self.lorsas and tc_idx < len(self.lorsas):
            lorsa = self.lorsas[tc_idx]
            qk_circuit = {
                "W_Q": lorsa.W_Q.weight,
                "W_K": lorsa.W_K.weight,
            }
            ov_circuit = {
                "w_V": lorsa.sparse_W_V,
                "w_O": lorsa.sparse_W_O,
            }

        return {
            "layer_idx": layer_idx,
            "feature_idx": feature_idx,
            "decoder_vec": feature_vec,
            "encoder_vec": input_vec,
            "qk_circuit": qk_circuit,
            "ov_circuit": ov_circuit,
            "norm": torch.norm(feature_vec).item(),
            "encoder_norm": torch.norm(input_vec).item(),
        }

    def find_induction_circuit(
        self,
        prompt: str,
        layer_idx: int,
        threshold: float = 0.1,
    ) -> list[GlobalCircuit]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        induction_features = []

        for i in range(1, len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                for tc_idx, layer in enumerate(self.layer_indices):
                    if layer != layer_idx:
                        continue

                    transcoder = self.transcoders[tc_idx]
                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        hidden_states = outputs.hidden_states[layer]
                        _, features = transcoder(hidden_states)

                    features = features.squeeze(0)
                    feature_activations = features[i].abs()

                    top_features = torch.topk(feature_activations, min(10, len(feature_activations)))
                    for feat_idx, act in zip(top_features.indices, top_features.values, strict=True):
                        if act.item() > threshold:
                            induction_features.append((
                                layer,
                                feat_idx.item(),
                                act.item(),
                            ))

        if not induction_features:
            return []

        unique_features = {}
        for layer, feat, act in induction_features:
            key = (layer, feat)
            if key not in unique_features or unique_features[key] < act:
                unique_features[key] = act

        sorted_features = sorted(unique_features.items(), key=lambda x: x[1], reverse=True)[:20]

        return [
            GlobalCircuit(
                name="Induction",
                circuit_type="induction",
                features=[(layer, feat) for (layer, feat), _ in sorted_features],
                strength=sum(act for _, act in sorted_features) / len(sorted_features) if sorted_features else 0.0,
            )
        ]

    def find_attention_circuits(
        self,
        prompt: str,
        layer_idx: int,
        head_idx: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.lorsas:
            return []

        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        if layer_idx not in self.layer_indices or not self.lorsas:
            return []

        tc_idx = self.layer_indices.index(layer_idx)
        if tc_idx >= len(self.lorsas):
            return []

        lorsa = self.lorsas[tc_idx]

        hidden_states = outputs.hidden_states[layer_idx]
        batch, seq_len, hidden_dim = hidden_states.shape

        Q = lorsa.W_Q(hidden_states)  # noqa: N806
        K = lorsa.W_K(hidden_states)  # noqa: N806
        V = lorsa.W_V(hidden_states)  # noqa: N806

        Q = Q.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
        K = K.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806
        V = V.view(batch, seq_len, lorsa.num_heads, lorsa.head_dim).transpose(1, 2)  # noqa: N806

        if lorsa.config.qk_layernorm:
            Q = lorsa.q_layernorm(Q)  # noqa: N806
            K = lorsa.k_layernorm(K)  # noqa: N806

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (lorsa.head_dim ** 0.5)
        attn_probs = torch.softmax(scores, dim=-1)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        circuits = []
        num_heads = lorsa.num_heads if head_idx is None else 1
        head_range = range(num_heads) if head_idx is None else [head_idx]

        for h in head_range:
            attn_pattern = attn_probs[0, h]

            max_attn, max_pos = attn_pattern.max(dim=1)

            for pos in range(seq_len):
                if max_attn[pos].item() > 0.3:
                    circuits.append({
                        "layer": layer_idx,
                        "head": h,
                        "from_pos": pos,
                        "from_token": tokens[pos],
                        "to_pos": max_pos[pos].item(),
                        "to_token": tokens[max_pos[pos].item()],
                        "strength": max_attn[pos].item(),
                    })

        return circuits

    def find_copy_circuits(
        self,
        prompt: str,
        layer_idx: int,
    ) -> list[GlobalCircuit]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        copy_features = []

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        for tc_idx, layer in enumerate(self.layer_indices):
            if layer != layer_idx:
                continue

            transcoder = self.transcoders[tc_idx]
            hidden_states = outputs.hidden_states[layer]
            _, features = transcoder(hidden_states)

            features = features.squeeze(0)

            for pos in range(1, len(tokens) - 1):
                if tokens[pos] == tokens[pos - 1]:
                    feature_activations = features[pos].abs()
                    top_feat = feature_activations.argmax().item()
                    copy_features.append((layer, top_feat, feature_activations[top_feat].item()))

        if not copy_features:
            return []

        unique_features = {}
        for layer, feat, act in copy_features:
            key = (layer, feat)
            if key not in unique_features or unique_features[key] < act:
                unique_features[key] = act

        sorted_features = sorted(unique_features.items(), key=lambda x: x[1], reverse=True)[:15]

        return [
            GlobalCircuit(
                name="Copying",
                circuit_type="copy",
                features=[(layer, feat) for (layer, feat), _ in sorted_features],
                strength=sum(act for _, act in sorted_features) / len(sorted_features) if sorted_features else 0.0,
            )
        ]

    def analyze_all_circuits(
        self,
        prompt: str,
    ) -> dict[str, list[GlobalCircuit]]:
        all_circuits = {}

        for layer_idx in self.layer_indices:
            induction = self.find_induction_circuit(prompt, layer_idx)
            if induction:
                all_circuits["induction"] = induction

            copy = self.find_copy_circuits(prompt, layer_idx)
            if copy:
                all_circuits["copy"] = copy

            if self.lorsas:
                attn_circuits = self.find_attention_circuits(prompt, layer_idx)
                if attn_circuits:
                    all_circuits[f"attention_layer_{layer_idx}"] = [
                        GlobalCircuit(
                            name=f"Attention L{layer_idx}",
                            circuit_type="attention",
                            features=[(layer_idx, 0)],
                            strength=sum(c["strength"] for c in attn_circuits) / len(attn_circuits),
                        )
                    ]

        return all_circuits
