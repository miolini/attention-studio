import pytest
import torch

from attention_studio.core.model_manager import ModelConfig, ModelManager


def get_device():
    """Get the best available device (mps/cuda/cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TestModelLoading:
    @pytest.mark.integration
    def test_load_gpt2(self):
        """Test loading GPT-2 model from HuggingFace Hub."""
        manager = ModelManager()

        config = ModelConfig(
            name="gpt2",
            device=get_device(),
            dtype="float32",
        )

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(manager.load_model(config))

        assert manager.is_loaded
        assert manager.model is not None
        assert manager.tokenizer is not None

        model_info = manager.get_model_info()
        assert model_info["name"] == "gpt2"
        assert model_info["num_layers"] == 12
        assert model_info["hidden_size"] == 768
        assert model_info["num_heads"] == 12

        manager.unload()
        assert not manager.is_loaded

    @pytest.mark.integration
    def test_gpt2_inference(self):
        """Test running inference on GPT-2."""
        manager = ModelManager()

        config = ModelConfig(
            name="gpt2",
            device=get_device(),
            dtype="float32",
        )

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(manager.load_model(config))

        prompt = "The quick brown fox"
        result = manager.run_inference(prompt)

        assert result.logits is not None
        assert result.tokens is not None
        assert result.logits.shape[0] == 1

        manager.unload()

    @pytest.mark.integration
    def test_layer_info_extraction(self):
        """Test extracting layer information from GPT-2."""
        manager = ModelManager()

        config = ModelConfig(
            name="gpt2",
            device=get_device(),
            dtype="float32",
        )

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(manager.load_model(config))

        layer_info = manager.layer_info

        assert len(layer_info) > 0
        assert layer_info[0].hidden_size == 768
        assert layer_info[0].num_heads == 12

        manager.unload()

    @pytest.mark.integration
    def test_quantum_mechanics_song(self):
        """Test generating a song about quantum mechanics using inference."""
        manager = ModelManager()

        config = ModelConfig(
            name="gpt2",
            device=get_device(),
            dtype="float32",
        )

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(manager.load_model(config))

        prompt = "Explain quantum mechanics"
        result = manager.run_inference(prompt)

        assert result.logits is not None
        assert result.tokens is not None
        assert result.logits.shape[0] == 1

        output_ids = result.logits[0].argmax(dim=-1)
        generated_text = manager.tokenizer.decode(output_ids, skip_special_tokens=True)

        assert len(generated_text) > 0

        print(f"\nGenerated text:\n{generated_text}")

        manager.unload()
