import pytest
import networkx as nx
import numpy as np
from attention_studio.core.circuit_discovery import (
    Circuit,
    CircuitDiscovery,
    PathFinder,
    SubgraphExtractor,
    CircuitAnalyzer,
)


class TestCircuit:
    def test_circuit_creation(self):
        circuit = Circuit(
            name="test_circuit",
            nodes=["a", "b", "c"],
            edges=[("a", "b", {"weight": 1.0})],
        )
        assert circuit.name == "test_circuit"
        assert circuit.get_node_count() == 3

    def test_to_networkx(self):
        circuit = Circuit(
            name="test",
            nodes=["a", "b"],
            edges=[("a", "b", {"weight": 1.0})],
        )
        graph = circuit.to_networkx()
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 2


class TestPathFinder:
    def test_shortest_path(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        path = PathFinder.find_shortest_path(graph, "a", "c")
        assert path == ["a", "b", "c"]

    def test_shortest_path_no_path(self):
        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        graph.add_edge("c", "d")
        path = PathFinder.find_shortest_path(graph, "a", "c")
        assert path is None

    def test_all_paths(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
        paths = PathFinder.find_all_paths(graph, "a", "c", max_length=3)
        assert len(paths) >= 1

    def test_compute_betweenness(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
        betweenness = PathFinder.compute_betweenness(graph)
        assert "b" in betweenness

    def test_compute_pagerank(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c")])
        pagerank = PathFinder.compute_pagerank(graph)
        assert len(pagerank) == 3

    def test_find_important_nodes(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a")])
        important = PathFinder.find_important_nodes(graph, top_k=2)
        assert len(important) == 2


class TestSubgraphExtractor:
    def test_extract_ego_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
        ego = SubgraphExtractor.extract_ego_graph(graph, "b", radius=1)
        assert "b" in ego.nodes()

    def test_extract_by_edge_weight(self):
        graph = nx.DiGraph()
        graph.add_edge("a", "b", weight=0.8)
        graph.add_edge("a", "c", weight=0.2)
        filtered = SubgraphExtractor.extract_by_edge_weight(graph, threshold=0.5)
        assert filtered.has_edge("a", "b")
        assert not filtered.has_edge("a", "c")

    def test_extract_connected_components(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("c", "d")])
        components = SubgraphExtractor.extract_connected_components(graph)
        assert len(components) == 2

    def test_extract_dense_subgraph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("d", "e")])
        dense = SubgraphExtractor.extract_dense_subgraph(graph, min_density=0.5)
        assert dense is not None


class TestCircuitAnalyzer:
    def test_analyze_connectivity(self):
        circuit = Circuit(
            name="test",
            nodes=["a", "b", "c"],
            edges=[("a", "b", {"weight": 1.0}), ("b", "c", {"weight": 1.0})],
        )
        result = CircuitAnalyzer.analyze_connectivity(circuit)
        assert result["num_nodes"] == 3
        assert result["num_edges"] == 2

    def test_analyze_paths(self):
        circuit = Circuit(
            name="test",
            nodes=["a", "b", "c"],
            edges=[
                ("a", "b", {"weight": 1.0}),
                ("b", "c", {"weight": 1.0}),
                ("c", "a", {"weight": 1.0}),
                ("a", "c", {"weight": 1.0}),
            ],
        )
        result = CircuitAnalyzer.analyze_paths(circuit)
        assert "avg_path_length" in result
        assert result["avg_path_length"] > 0

    def test_find_bridges(self):
        circuit = Circuit(
            name="test",
            nodes=["a", "b", "c"],
            edges=[("a", "b", {"weight": 1.0}), ("b", "c", {"weight": 1.0})],
        )
        bridges = CircuitAnalyzer.find_bridges(circuit)
        assert isinstance(bridges, list)

    def test_find_important_edges(self):
        circuit = Circuit(
            name="test",
            nodes=["a", "b", "c"],
            edges=[("a", "b", {"weight": 1.0}), ("b", "c", {"weight": 1.0}), ("a", "c", {"weight": 0.5})],
        )
        important = CircuitAnalyzer.find_important_edges(circuit, top_k=2)
        assert isinstance(important, list)


class TestEdgeCases:
    def test_empty_graph_betweenness(self):
        graph = nx.DiGraph()
        betweenness = PathFinder.compute_betweenness(graph)
        assert betweenness == {}

    def test_empty_graph_pagerank(self):
        graph = nx.DiGraph()
        pagerank = PathFinder.compute_pagerank(graph)
        assert pagerank == {}

    def test_single_node_graph(self):
        graph = nx.DiGraph()
        graph.add_node("a")
        pagerank = PathFinder.compute_pagerank(graph)
        assert pagerank["a"] == 1.0

    def test_empty_circuit(self):
        circuit = Circuit(name="empty", nodes=[], edges=[])
        assert circuit.get_node_count() == 0
        assert circuit.get_edge_count() == 0
