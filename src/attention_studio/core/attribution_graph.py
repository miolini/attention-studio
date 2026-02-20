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

    def find_paths(
        self,
        graph: CompleteAttributionGraph,
        source_node: str,
        target_node: str,
        max_length: int = 10,
        threshold: float = 0.01,
    ) -> list[list[str]]:
        if source_node not in graph.graph or target_node not in graph.graph:
            return []

        try:
            paths = list(nx.all_simple_paths(
                graph.graph,
                source_node,
                target_node,
                cutoff=max_length,
            ))
            filtered_paths = []
            for path in paths:
                total_weight = 0.0
                for i in range(len(path) - 1):
                    edge = graph.edges.get((path[i], path[i + 1]))
                    if edge:
                        total_weight += abs(edge.weight)
                if total_weight >= threshold:
                    filtered_paths.append(path)
            return filtered_paths
        except nx.NetworkXNoPath:
            return []

    def find_shortest_path(
        self,
        graph: CompleteAttributionGraph,
        source_node: str,
        target_node: str,
        use_weights: bool = True,
    ) -> list[str] | None:
        if source_node not in graph.graph or target_node not in graph.graph:
            return None

        try:
            if use_weights:
                return nx.shortest_path(
                    graph.graph,
                    source_node,
                    target_node,
                    weight=lambda u, v, d: abs(d.get("weight", 1.0)),
                )
            else:
                return nx.shortest_path(graph.graph, source_node, target_node)
        except nx.NetworkXNoPath:
            return None

    def filter_edges(
        self,
        graph: CompleteAttributionGraph,
        min_weight: float = 0.0,
        max_weight: float | None = None,
        edge_types: list[str] | None = None,
    ) -> dict[tuple[str, str], AttributionEdge]:
        filtered = {}
        for edge_key, edge in graph.edges.items():
            if abs(edge.weight) < min_weight:
                continue
            if max_weight is not None and abs(edge.weight) > max_weight:
                continue
            if edge_types is not None and edge.edge_type not in edge_types:
                continue
            filtered[edge_key] = edge
        return filtered

    def filter_nodes(
        self,
        graph: CompleteAttributionGraph,
        node_types: list[str] | None = None,
        layers: list[int] | None = None,
        min_activation: float = 0.0,
    ) -> dict[str, AttributionNode]:
        filtered = {}
        for node_id, node in graph.nodes.items():
            if node_types is not None and node.node_type not in node_types:
                continue
            if layers is not None and node.layer not in layers:
                continue
            if abs(node.activation) < min_activation:
                continue
            filtered[node_id] = node
        return filtered

    def compute_graph_metrics(
        self,
        graph: CompleteAttributionGraph,
    ) -> dict[str, dict[str, float]]:
        if len(graph.graph.nodes) == 0:
            return {}

        metrics = {}

        try:
            degree_centrality = nx.degree_centrality(graph.graph)
            metrics["degree_centrality"] = degree_centrality
        except Exception:
            metrics["degree_centrality"] = {}

        try:
            betweenness = nx.betweenness_centrality(graph.graph, weight="weight")
            metrics["betweenness_centrality"] = betweenness
        except Exception:
            metrics["betweenness_centrality"] = {}

        try:
            pagerank = nx.pagerank(graph.graph, weight="weight")
            metrics["pagerank"] = pagerank
        except Exception:
            metrics["pagerank"] = {}

        try:
            in_degrees = dict(graph.graph.in_degree())
            out_degrees = dict(graph.graph.out_degree())
            metrics["in_degree"] = in_degrees
            metrics["out_degree"] = out_degrees
        except Exception:
            metrics["in_degree"] = {}
            metrics["out_degree"] = {}

        return metrics

    def find_cycles(
        self,
        graph: CompleteAttributionGraph,
        max_length: int = 10,
    ) -> list[list[str]]:
        try:
            cycles = list(nx.simple_cycles(graph.graph))
            return [c for c in cycles if len(c) <= max_length]
        except Exception:
            return []

    def find_edge_between_layers(
        self,
        graph: CompleteAttributionGraph,
        source_layer: int,
        target_layer: int,
    ) -> dict[tuple[str, str], AttributionEdge]:
        cross_layer_edges = {}
        for edge_key, edge in graph.edges.items():
            source_node = graph.nodes.get(edge_key[0])
            target_node = graph.nodes.get(edge_key[1])
            if source_node and target_node and source_node.layer == source_layer and target_node.layer == target_layer:
                cross_layer_edges[edge_key] = edge
        return cross_layer_edges


class LazyAttributionGraphBuilder:
    def __init__(
        self,
        model_manager: Any,
        transcoders: nn.ModuleList,
        lorsas: nn.ModuleList | None = None,
        layer_indices: list[int] | None = None,
        cache_size: int = 1000,
    ):
        self.model_manager = model_manager
        self.transcoders = transcoders
        self.lorsas = lorsas
        self.layer_indices = layer_indices or list(range(len(transcoders)))
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer
        self._node_cache: dict[str, AttributionNode] = {}
        self._edge_cache: dict[tuple[str, str], AttributionEdge] = {}
        self._cache_size = cache_size
        self._cached_outputs = None
        self._cached_prompt = None

    def _get_model_outputs(self, prompt: str):
        if self._cached_prompt == prompt and self._cached_outputs is not None:
            return self._cached_outputs

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        self._cached_outputs = outputs
        self._cached_prompt = prompt
        return outputs

    def get_embedding_nodes(self, prompt: str) -> dict[str, AttributionNode]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        nodes = {}
        for pos, token in enumerate(tokens):
            node_id = f"emb_{pos}"
            if node_id in self._node_cache:
                nodes[node_id] = self._node_cache[node_id]
            else:
                node = AttributionNode(
                    node_id=node_id,
                    node_type="embedding",
                    layer=0,
                    position=pos,
                    feature_idx=None,
                    token=token,
                    activation=1.0,
                    encoder_vec=None,
                    decoder_vec=None,
                )
                self._node_cache[node_id] = node
                nodes[node_id] = node

        return nodes

    def get_transcoder_nodes(
        self,
        prompt: str,
        layer_idx: int,
        top_k: int = 50,
    ) -> dict[str, AttributionNode]:
        if layer_idx not in self.layer_indices:
            return {}

        cache_key = f"tc_{layer_idx}"
        if cache_key in self._node_cache:
            return self._node_cache[cache_key]

        outputs = self._get_model_outputs(prompt)
        tc_idx = self.layer_indices.index(layer_idx)
        transcoder = self.transcoders[tc_idx]

        hidden_states = outputs.hidden_states[layer_idx]
        _, features = transcoder(hidden_states)
        features = features.squeeze(0)

        tokens = self.tokenizer.convert_ids_to_tokens(outputs.hidden_states[0].shape[1] * ["token"])

        nodes = {}
        for pos in range(features.shape[0]):
            pos_features = features[pos].abs()
            top_features = torch.topk(pos_features, min(top_k, len(pos_features)))

            for feat_idx, act in zip(top_features.indices, top_features.values, strict=True):
                if act.item() > 0.01:
                    node_id = f"tc_{layer_idx}_{pos}_{feat_idx.item()}"
                    node = AttributionNode(
                        node_id=node_id,
                        node_type="transcoder",
                        layer=layer_idx,
                        position=pos,
                        feature_idx=feat_idx.item(),
                        token=tokens[pos] if pos < len(tokens) else "",
                        activation=act.item(),
                        encoder_vec=transcoder.encoder.weight[feat_idx].detach().cpu() if transcoder.encoder.weight is not None else None,
                        decoder_vec=transcoder.decoder.weight[:, feat_idx].detach().cpu() if transcoder.decoder.weight is not None else None,
                    )
                    nodes[node_id] = node
                    self._node_cache[node_id] = node

        return nodes

    def get_layer_edges(
        self,
        prompt: str,
        source_layer: int,
        target_layer: int,
        threshold: float = 0.01,
    ) -> dict[tuple[str, str], AttributionEdge]:
        edges = {}

        source_nodes = self.get_transcoder_nodes(prompt, source_layer)
        target_nodes = self.get_transcoder_nodes(prompt, target_layer)

        for src_id, src_node in source_nodes.items():
            for tgt_id, tgt_node in target_nodes.items():
                if src_node.position == tgt_node.position and src_node.decoder_vec is not None and tgt_node.encoder_vec is not None:
                    weight = torch.dot(src_node.decoder_vec, tgt_node.encoder_vec).item()
                    if abs(weight) > threshold:
                        edge = AttributionEdge(
                            source_id=src_id,
                            target_id=tgt_id,
                            weight=weight,
                            edge_type="transcoder",
                        )
                        edges[(src_id, tgt_id)] = edge

        return edges

    def build_subgraph(
        self,
        prompt: str,
        layers: list[int] | None = None,
        top_k_per_layer: int = 20,
    ) -> CompleteAttributionGraph:
        layers = layers or self.layer_indices

        graph = nx.DiGraph()
        nodes = self.get_embedding_nodes(prompt)
        edges = {}

        for node_id, _node in nodes.items():
            graph.add_node(node_id)

        prev_layer = None
        for layer in sorted(layers):
            layer_nodes = self.get_transcoder_nodes(prompt, layer, top_k_per_layer)
            nodes.update(layer_nodes)

            for node_id in layer_nodes:
                graph.add_node(node_id)

            if prev_layer is not None:
                layer_edges = self.get_layer_edges(prompt, prev_layer, layer)
                edges.update(layer_edges)
                for (src, tgt), edge in layer_edges.items():
                    graph.add_edge(src, tgt, weight=edge.weight)

            prev_layer = layer

        tokens = list(self.tokenizer.convert_ids_to_tokens(
            self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        ))

        return CompleteAttributionGraph(
            graph=graph,
            nodes=nodes,
            edges=edges,
            prompt=prompt,
            tokens=tokens,
        )

    def clear_cache(self):
        self._node_cache.clear()
        self._edge_cache.clear()
        self._cached_outputs = None
        self._cached_prompt = None
