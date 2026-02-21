from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import numpy as np
import torch


@dataclass
class Circuit:
    name: str
    nodes: list[str]
    edges: list[tuple[str, str, dict]]
    metadata: dict = field(default_factory=dict)

    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        return graph

    def get_node_count(self) -> int:
        return len(self.nodes)

    def get_edge_count(self) -> int:
        return len(self.edges)


@dataclass
class Edge:
    source: str
    target: str
    weight: float
    layer: int


class CircuitDiscovery:
    def __init__(self, model_manager: Any, transcoders: list[Any]):
        self.model_manager = model_manager
        self.transcoders = transcoders

    def discover_by_correlation(
        self,
        prompts: list[str],
        layer_idx: int,
        feature_indices: list[int],
        threshold: float = 0.5,
    ) -> Circuit:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        activation_matrix = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states[layer_idx]
            tc_idx = layer_idx
            if tc_idx < len(self.transcoders):
                _, features = self.transcoders[tc_idx](hidden_states)
                features = features.squeeze(0)
                feature_activations = features.mean(dim=0)
                activation_matrix.append(feature_activations.cpu().numpy())

        if not activation_matrix:
            return Circuit(name="empty", nodes=[], edges=[])

        activation_matrix = np.array(activation_matrix)

        nodes = [f"L{layer_idx}_F{idx}" for idx in feature_indices]
        edges = []

        for i, feat_i in enumerate(feature_indices):
            for j, feat_j in enumerate(feature_indices):
                if i != j and feat_i < activation_matrix.shape[1] and feat_j < activation_matrix.shape[1]:
                    corr = np.corrcoef(activation_matrix[:, feat_i], activation_matrix[:, feat_j])[0, 1]
                    if not np.isnan(corr) and abs(corr) > threshold:
                        edges.append((nodes[i], nodes[j], {"weight": corr, "layer": layer_idx}))

        return Circuit(
            name=f"correlation_circuit_l{layer_idx}",
            nodes=nodes,
            edges=edges,
            metadata={"threshold": threshold, "num_prompts": len(prompts)},
        )

    def discover_by_ablations(
        self,
        prompt: str,
        layer_idx: int,
        feature_indices: list[int],
        top_k: int = 10,
    ) -> Circuit:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs_original = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            original_logits = outputs_original.logits[0, -1, :]

        feature_effects = {}

        for feature_idx in feature_indices:
            tc_idx = layer_idx
            if tc_idx >= len(self.transcoders):
                continue

            transcoder = self.transcoders[tc_idx]

            def create_hook(feat_idx):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    _, features = transcoder(hidden)
                    features = features.squeeze(0)
                    features[:, feat_idx] = 0.0
                    reconstructed = transcoder.decoder(features.view(-1, features.shape[-1]))
                    reconstructed = reconstructed.view(hidden.shape)
                    if isinstance(output, tuple):
                        return (reconstructed,) + output[1:]
                    return reconstructed
                return hook

            try:
                hook_handle = model.transformer.h[layer_idx].register_forward_pre_hook(
                    create_hook(feature_idx)
                )

                with torch.no_grad():
                    outputs_ablated = model(
                        input_ids=input_ids,
                        return_dict=True,
                    )
                    ablated_logits = outputs_ablated.logits[0, -1, :]

                effect = torch.nn.functional.cosine_similarity(
                    original_logits.unsqueeze(0),
                    ablated_logits.unsqueeze(0),
                ).item()

                feature_effects[feature_idx] = effect

            finally:
                hook_handle.remove()

        if not feature_effects:
            return Circuit(name="empty", nodes=[], edges=[])

        sorted_features = sorted(feature_effects.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = [f for f, _ in sorted_features[:top_k]]

        nodes = [f"L{layer_idx}_F{idx}" for idx in top_features]
        edges = []

        for i, feat_i in enumerate(top_features):
            for j, feat_j in enumerate(top_features):
                if i != j:
                    edge_weight = abs(feature_effects.get(feat_i, 0) - feature_effects.get(feat_j, 0))
                    if edge_weight > 0.1:
                        edges.append((nodes[i], nodes[j], {"weight": edge_weight, "layer": layer_idx}))

        return Circuit(
            name=f"ablation_circuit_l{layer_idx}",
            nodes=nodes,
            edges=edges,
            metadata={"prompt": prompt, "num_features": len(top_features)},
        )

    def discover_attention_circuits(
        self,
        prompt: str,
        layer_idx: int,
        head_threshold: float = 0.3,
    ) -> list[Circuit]:
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        circuits = []

        if not hasattr(outputs, "attentions") or outputs.attentions is None:
            return circuits

        attentions = outputs.attentions

        if layer_idx >= len(attentions):
            return circuits

        attn = attentions[layer_idx]
        num_heads = attn.shape[1]
        seq_len = attn.shape[2]

        for head_idx in range(num_heads):
            head_attn = attn[0, head_idx]

            edges = []
            nodes = [f"token_{i}_{tokens[i]}" for i in range(seq_len)]

            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j and head_attn[i, j].item() > head_threshold:
                        edges.append((
                            nodes[j],
                            nodes[i],
                            {"weight": head_attn[i, j].item(), "head": head_idx}
                        ))

            if edges:
                circuits.append(Circuit(
                    name=f"attention_L{layer_idx}_H{head_idx}",
                    nodes=nodes,
                    edges=edges,
                    metadata={"layer": layer_idx, "head": head_idx},
                ))

        return circuits


class PathFinder:
    @staticmethod
    def find_shortest_path(
        graph: nx.DiGraph,
        source: str,
        target: str,
    ) -> Optional[list[str]]:
        try:
            return nx.shortest_path(graph, source, target)
        except nx.NetworkXNoPath:
            return None

    @staticmethod
    def find_all_paths(
        graph: nx.DiGraph,
        source: str,
        target: str,
        max_length: int = 10,
    ) -> list[list[str]]:
        try:
            return list(nx.all_simple_paths(graph, source, target, cutoff=max_length))
        except nx.NetworkXNoPath:
            return []

    @staticmethod
    def find_paths_by_length(
        graph: nx.DiGraph,
        source: str,
        target: str,
    ) -> dict[int, list[list[str]]]:
        paths_by_length = {}
        for length in range(1, 11):
            paths = PathFinder.find_all_paths(graph, source, target, max_length=length)
            if paths:
                paths_by_length[length] = paths
        return paths_by_length

    @staticmethod
    def compute_betweenness(
        graph: nx.DiGraph,
    ) -> dict[str, float]:
        return nx.betweenness_centrality(graph)

    @staticmethod
    def compute_pagerank(
        graph: nx.DiGraph,
    ) -> dict[str, float]:
        return nx.pagerank(graph)

    @staticmethod
    def find_important_nodes(
        graph: nx.DiGraph,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        pagerank = PathFinder.compute_pagerank(graph)
        sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]


class SubgraphExtractor:
    @staticmethod
    def extract_ego_graph(
        graph: nx.DiGraph,
        node: str,
        radius: int = 1,
    ) -> nx.DiGraph:
        return nx.ego_graph(graph, node, radius=radius)

    @staticmethod
    def extract_by_edge_weight(
        graph: nx.DiGraph,
        threshold: float = 0.0,
    ) -> nx.DiGraph:
        filtered_graph = nx.DiGraph()
        filtered_graph.add_nodes_from(graph.nodes())

        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 0.0)
            if abs(weight) >= threshold:
                filtered_graph.add_edge(u, v, **data)

        return filtered_graph

    @staticmethod
    def extract_connected_components(
        graph: nx.DiGraph,
    ) -> list[nx.DiGraph]:
        components = list(nx.weakly_connected_components(graph))
        subgraphs = []
        for component in components:
            subgraph = graph.subgraph(component).copy()
            subgraphs.append(subgraph)
        return subgraphs

    @staticmethod
    def extract_dense_subgraph(
        graph: nx.DiGraph,
        min_density: float = 0.5,
    ) -> Optional[nx.DiGraph]:
        if graph.number_of_nodes() < 2:
            return graph

        for node in graph.nodes():
            ego = nx.ego_graph(graph, node, radius=1)
            if ego.number_of_nodes() > 1:
                density = nx.density(ego)
                if density >= min_density:
                    return ego
        return None


class CircuitAnalyzer:
    @staticmethod
    def analyze_connectivity(circuit: Circuit) -> dict[str, Any]:
        graph = circuit.to_networkx()

        if graph.number_of_nodes() == 0:
            return {"num_nodes": 0, "num_edges": 0, "density": 0.0}

        return {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_weakly_connected(graph),
            "num_components": nx.number_weakly_connected_components(graph),
        }

    @staticmethod
    def analyze_paths(circuit: Circuit) -> dict[str, Any]:
        graph = circuit.to_networkx()

        if graph.number_of_nodes() < 2:
            return {"avg_path_length": 0.0}

        try:
            avg_path_length = nx.average_shortest_path_length(graph)
        except nx.NetworkXError:
            avg_path_length = 0.0

        return {
            "avg_path_length": avg_path_length,
            "diameter": nx.diameter(graph) if nx.is_weakly_connected(graph) else -1,
        }

    @staticmethod
    def find_bridges(circuit: Circuit) -> list[tuple[str, str]]:
        graph = circuit.to_networkx()
        bridges = list(nx.bridges(graph.to_undirected()))
        return bridges

    @staticmethod
    def find_important_edges(
        circuit: Circuit,
        top_k: int = 10,
    ) -> list[tuple[str, str, float]]:
        graph = circuit.to_networkx()
        betweenness = nx.edge_betweenness_centrality(graph)
        sorted_edges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        result = []
        for (u, v), score in sorted_edges[:top_k]:
            if graph.has_edge(u, v):
                result.append((u, v, graph[u][v].get("weight", 1.0)))
        return result
