import pytest
import torch

from attention_studio.core.dataset import DatasetConfig, DatasetManager
from attention_studio.core.model_manager import ModelConfig, ModelManager
from attention_studio.core.trainer import CRMTrainer, TrainingConfig, TranscoderConfig


def get_device():
    """Get the best available device (mps/cuda/cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TestCRMTraining:
    @pytest.mark.integration
    def test_build_transcoder(self):
        """Test building transcoder for GPT-2 model."""
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

        trainer = CRMTrainer(manager, TrainingConfig())

        tc_config = TranscoderConfig(
            dictionary_size=1024,
            top_k=64,
        )

        layer_indices = [0, 1, 2]
        trainer.build_transcoders(layer_indices, tc_config)

        assert len(trainer.transcoders) == 3
        assert trainer.layer_indices == layer_indices

        manager.unload()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_transcoder_training_step(self):
        """Test training a transcoder for one step."""
        manager = ModelManager()

        model_config = ModelConfig(
            name="gpt2",
            device=get_device(),
            dtype="float32",
        )

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(manager.load_model(model_config))

        dataset_manager = DatasetManager()
        ds_config = DatasetConfig(
            source="custom",
            token_limit=10000,
        )
        loop.run_until_complete(dataset_manager.load(ds_config, manager.tokenizer))

        trainer = CRMTrainer(manager, TrainingConfig())

        tc_config = TranscoderConfig(
            dictionary_size=512,
            top_k=32,
        )

        layer_indices = [0]
        trainer.build_transcoders(layer_indices, tc_config)

        dataloader = dataset_manager.create_dataloader(batch_size=2, max_length=128)

        import torch
        device = get_device()

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                outputs = manager.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[0]

            tc = trainer.transcoders[0]
            reconstructed, features = tc(hidden_states)

            loss = torch.nn.functional.mse_loss(reconstructed, hidden_states)

            loss.backward()

            optimizer = trainer.optimizers[0]
            optimizer.step()
            optimizer.zero_grad()

            assert loss.item() > 0
            print(f"Training step loss: {loss.item():.4f}")
            break

        manager.unload()
        dataset_manager.clear()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_crm_training_mps(self):
        """Test full CRM training on MPS device."""
        manager = ModelManager()

        device = get_device()
        assert device == "mps", f"Expected MPS, got {device}"

        model_config = ModelConfig(
            name="gpt2",
            device=device,
            dtype="float32",
        )

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(manager.load_model(model_config))

        assert manager.get_device() == "mps:0" or "mps" in manager.get_device()

        trainer = CRMTrainer(manager, TrainingConfig())

        tc_config = TranscoderConfig(
            dictionary_size=256,
            top_k=16,
        )

        layer_indices = [0]
        trainer.build_transcoders(layer_indices, tc_config)

        for tc in trainer.transcoders:
            assert next(tc.parameters()).device.type == "mps"

        print(f"Transcoders built on device: {next(trainer.transcoders.parameters()).device}")

        dataset_manager = DatasetManager()
        ds_config = DatasetConfig(
            source="custom",
            token_limit=5000,
        )
        loop.run_until_complete(dataset_manager.load(ds_config, manager.tokenizer))

        dataloader = dataset_manager.create_dataloader(batch_size=2, max_length=64)

        losses = []
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                outputs = manager.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[0]

            tc = trainer.transcoders[0]
            reconstructed, features = tc(hidden_states)

            loss = torch.nn.functional.mse_loss(reconstructed, hidden_states)
            losses.append(loss.item())

            loss.backward()

            optimizer = trainer.optimizers[0]
            optimizer.step()
            optimizer.zero_grad()

        assert len(losses) == 2
        print(f"Training losses: {losses}")

        manager.unload()
        dataset_manager.clear()
