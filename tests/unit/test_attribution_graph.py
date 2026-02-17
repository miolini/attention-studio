import torch
import torch.nn as nn

from attention_studio.core.attribution_graph import (
    AttributionEdge,
    AttributionGraphBuilder,
    AttributionNode,
    CircuitPath,
    CompleteAttributionGraph,
    QKTracingResult,
)


class MockModelManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @property
    def is_loaded(self):
        return True


class TestAttributionNode:
    def test_attribution_node_creation(self):
        node = AttributionNode(
            node_id="test_1",
            node_type="transcoder",
            layer=5,
            position=3,
            feature_idx=10,
            token="test",
            activation=0.5,
            encoder_vec=torch.randn(768),
            decoder_vec=torch.randn(768),
        )
        assert node.node_id == "test_1"
        assert node.node_type == "transcoder"
        assert node.layer == 5
        assert node.position == 3
        assert node.feature_idx == 10
        assert node.token == "test"
        assert node.activation == 0.5

    def test_attribution_node_optional_fields(self):
        node = AttributionNode(
            node_id="emb_0",
            node_type="embedding",
            layer=0,
            position=0,
            feature_idx=None,
            token="test",
            activation=1.0,
            encoder_vec=None,
            decoder_vec=None,
        )
        assert node.feature_idx is None
        assert node.encoder_vec is None
        assert node.decoder_vec is None


class TestAttributionEdge:
    def test_attribution_edge_creation(self):
        edge = AttributionEdge(
            source_id="source_1",
            target_id="target_1",
            weight=0.8,
            edge_type="mlp",
            attention_pattern=torch.randn(10, 10),
            virtual_weight=torch.tensor(0.5),
        )
        assert edge.source_id == "source_1"
        assert edge.target_id == "target_1"
        assert edge.weight == 0.8
        assert edge.edge_type == "mlp"

    def test_attribution_edge_optional_fields(self):
        edge = AttributionEdge(
            source_id="emb_0",
            target_id="tc_0_0_0",
            weight=0.1,
            edge_type="embedding",
        )
        assert edge.attention_pattern is None
        assert edge.virtual_weight is None


class TestCompleteAttributionGraph:
    def test_complete_attribution_graph_creation(self):
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_node("test_node")

        nodes = {
            "test_node": AttributionNode(
                node_id="test_node",
                node_type="transcoder",
                layer=1,
                position=0,
                feature_idx=0,
                token="test",
                activation=0.5,
                encoder_vec=None,
                decoder_vec=None,
            )
        }
        edges = {}
        tokens = ["test", "token"]

        complete_graph = CompleteAttributionGraph(
            graph=graph,
            nodes=nodes,
            edges=edges,
            prompt="test prompt",
            tokens=tokens,
        )
        assert complete_graph.prompt == "test prompt"
        assert len(complete_graph.tokens) == 2
        assert len(complete_graph.nodes) == 1


class TestCircuitPath:
    def test_circuit_path_creation(self):
        path = CircuitPath(
            source=(0, 1, 2),
            target=(0, 1, 3),
            path=["node_1", "node_2", "node_3"],
            total_weight=0.75,
            path_type="induction",
        )
        assert path.source == (0, 1, 2)
        assert path.target == (0, 1, 3)
        assert len(path.path) == 3
        assert path.total_weight == 0.75
        assert path.path_type == "induction"


class TestQKTracingResult:
    def test_qk_tracing_result_creation(self):
        result = QKTracingResult(
            source_pos=5,
            target_pos=10,
            attention_score=0.3,
            feature_contributions=[{"feature_idx": 0, "contribution": 0.1}],
            pairwise_contributions=[
                {"source_pos": 0, "target_pos": 10, "score": 0.2}
            ],
        )
        assert result.source_pos == 5
        assert result.target_pos == 10
        assert result.attention_score == 0.3
        assert len(result.feature_contributions) == 1


class TestAttributionGraphBuilder:
    def test_attribution_graph_builder_initialization(self):
        class MockTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.randint(0, 50000, (1, 10))}

            def convert_ids_to_tokens(self, ids):
                return ["token"] * len(ids)

        class MockModel:
            device = torch.device("cpu")

            def embed_tokens(self, ids):
                return torch.randn(1, 768)

        class MockTranscoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(768, 512)
                self.decoder = nn.Linear(512, 768)

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded, encoded

        model_manager = MockModelManager(MockModel(), MockTokenizer())
        transcoder = MockTranscoder()
        transcoders = nn.ModuleList([transcoder])

        builder = AttributionGraphBuilder(
            model_manager=model_manager,
            transcoders=transcoders,
            lorsas=None,
            layer_indices=[0],
        )

        assert builder.model_manager is model_manager
        assert len(builder.transcoders) == 1
        assert builder.layer_indices == [0]
        assert builder.lorsas is None

    def test_attribution_graph_builder_with_lorsas(self):
        class MockTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.randint(0, 50000, (1, 10))}

            def convert_ids_to_tokens(self, ids):
                return ["token"] * len(ids)

        class MockModel:
            device = torch.device("cpu")

            def embed_tokens(self, ids):
                return torch.randn(1, 768)

        class MockTranscoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(768, 512)
                self.decoder = nn.Linear(512, 768)

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded, encoded

        class MockLorsa(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 4
                self.head_dim = 64
                self.W_Q = nn.Linear(768, 256)
                self.W_K = nn.Linear(768, 256)
                self.W_V = nn.Linear(768, 256)
                self.sparse_W_V = nn.Parameter(torch.randn(4, 64, 16))
                self.sparse_W_O = nn.Parameter(torch.randn(4, 64, 16))

        model_manager = MockModelManager(MockModel(), MockTokenizer())
        transcoder = MockTranscoder()
        transcoders = nn.ModuleList([transcoder])
        lorsa = MockLorsa()
        lorsas = nn.ModuleList([lorsa])

        builder = AttributionGraphBuilder(
            model_manager=model_manager,
            transcoders=transcoders,
            lorsas=lorsas,
            layer_indices=[0],
        )

        assert builder.lorsas is not None
        assert len(builder.lorsas) == 1

    def test_find_global_circuits_without_lorsas(self):
        class MockTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.randint(0, 50000, (1, 10))}

            def convert_ids_to_tokens(self, ids):
                return ["token"] * len(ids)

        class MockModel:
            device = torch.device("cpu")

            def embed_tokens(self, ids):
                return torch.randn(1, 768)

        class MockTranscoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(768, 512)
                self.decoder = nn.Linear(512, 768)

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded, encoded

        model_manager = MockModelManager(MockModel(), MockTokenizer())
        transcoder = MockTranscoder()
        transcoders = nn.ModuleList([transcoder])

        builder = AttributionGraphBuilder(
            model_manager=model_manager,
            transcoders=transcoders,
            lorsas=None,
            layer_indices=[0],
        )

        circuits = builder.find_global_circuits(threshold=0.1)

        assert "induction" in circuits
        assert "copy" in circuits
        assert "prev_token" in circuits
        assert "duplicate_tokens" in circuits
        assert isinstance(circuits["induction"], list)
        assert isinstance(circuits["copy"], list)
        assert isinstance(circuits["prev_token"], list)
        assert isinstance(circuits["duplicate_tokens"], list)

    def test_find_global_circuits_with_lorsas(self):
        class MockTokenizer:
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.randint(0, 50000, (1, 10))}

            def convert_ids_to_tokens(self, ids):
                return ["token"] * len(ids)

        class MockModel:
            device = torch.device("cpu")

            def embed_tokens(self, ids):
                return torch.randn(1, 768)

        class MockTranscoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(768, 512)
                self.decoder = nn.Linear(512, 768)

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded, encoded

        class MockLorsa(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 4
                self.head_dim = 64
                self.W_Q = nn.Linear(768, 256)
                self.W_K = nn.Linear(768, 256)
                self.W_V = nn.Linear(768, 256)
                self.sparse_W_V = nn.Parameter(torch.randn(4, 64, 16) * 0.5)
                self.sparse_W_O = nn.Parameter(torch.randn(4, 64, 16) * 0.5)

        model_manager = MockModelManager(MockModel(), MockTokenizer())
        transcoder = MockTranscoder()
        transcoders = nn.ModuleList([transcoder])
        lorsa = MockLorsa()
        lorsas = nn.ModuleList([lorsa])

        builder = AttributionGraphBuilder(
            model_manager=model_manager,
            transcoders=transcoders,
            lorsas=lorsas,
            layer_indices=[0],
        )

        circuits = builder.find_global_circuits(threshold=0.01)

        assert "induction" in circuits
        assert "prev_token" in circuits
        assert isinstance(circuits["induction"], list)
        assert isinstance(circuits["prev_token"], list)
