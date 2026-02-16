import torch

from attention_studio.core.crm import (
    Lorsa,
    LorsaConfig,
    Transcoder,
    TranscoderConfig,
    top_k_activation,
)


class TestTranscoder:
    def test_transcoder_forward(self):
        config = TranscoderConfig(dictionary_size=1024, top_k=128)
        transcoder = Transcoder(input_dim=768, config=config)

        x = torch.randn(2, 10, 768)
        output, features = transcoder(x)

        assert output.shape == x.shape
        assert features.shape == (2, 10, config.dictionary_size)

    def test_top_k_activation(self):
        x = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        result = top_k_activation(x, k=2)

        assert result.shape == x.shape
        assert (result == 0).sum() == 3

    def test_transcoder_sparsity(self):
        config = TranscoderConfig(dictionary_size=1024, top_k=32)
        transcoder = Transcoder(input_dim=256, config=config)

        x = torch.randn(1, 5, 256)
        _, features = transcoder(x)

        nonzero = (features != 0).sum().item()
        max_nonzeros = 1 * 5 * 32
        assert nonzero <= max_nonzeros


class TestLorsa:
    def test_lorsa_forward(self):
        config = LorsaConfig(num_heads=12, top_k=64)
        hidden_size = 768
        lorsa = Lorsa(hidden_size=hidden_size, config=config)

        x = torch.randn(2, 10, hidden_size)
        output, info = lorsa(x)

        assert output.shape == x.shape
        assert "attention_pattern" in info
        assert "z_sparse" in info

    def test_lorsa_attention_shape(self):
        config = LorsaConfig(num_heads=8, top_k=32)
        hidden_size = 256
        lorsa = Lorsa(hidden_size=hidden_size, config=config)

        x = torch.randn(1, 20, hidden_size)
        output, info = lorsa(x)

        batch, seq_len, _ = x.shape
        attn = info["attention_pattern"]

        assert attn.shape == (batch, config.num_heads, seq_len, seq_len)

    def test_lorsa_qk_ov_circuits(self):
        config = LorsaConfig(num_heads=4, top_k=16)
        hidden_size = 128
        lorsa = Lorsa(hidden_size=hidden_size, config=config)

        w_q, w_k = lorsa.get_qk_circuit()
        w_v, w_o = lorsa.get_ov_circuit()

        assert w_q.shape == (hidden_size, hidden_size)
        assert w_k.shape == (hidden_size, hidden_size)
        assert w_v.shape == (config.num_heads, hidden_size // config.num_heads, config.top_k)
        assert w_o.shape == (config.num_heads, config.top_k, hidden_size // config.num_heads)

    def test_lorsa_sparsity(self):
        config = LorsaConfig(num_heads=8, top_k=16)
        hidden_size = 256
        lorsa = Lorsa(hidden_size=hidden_size, config=config)

        x = torch.randn(2, 10, hidden_size)
        _, info = lorsa(x)

        z_sparse = info["z_sparse"]
        nonzero = (z_sparse != 0).sum().item()
        max_nonzeros = 2 * 10 * 8 * config.top_k

        assert nonzero <= max_nonzeros
