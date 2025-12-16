"""
Unit tests for VoxFormer model components.
"""

import pytest
import torch

# Skip if dependencies not available
pytest.importorskip("transformers")


class TestWeightedLayerSum:
    """Tests for WeightedLayerSum module."""

    def test_forward_shape(self):
        from src.model.wavlm_frontend import WeightedLayerSum

        layer_sum = WeightedLayerSum(num_layers=12)

        # Simulate WavLM hidden states (13 tensors: CNN + 12 transformer layers)
        batch_size, seq_len, dim = 2, 100, 768
        hidden_states = tuple(
            torch.randn(batch_size, seq_len, dim) for _ in range(13)
        )

        output = layer_sum(hidden_states)

        assert output.shape == (batch_size, seq_len, dim)

    def test_weights_normalized(self):
        from src.model.wavlm_frontend import WeightedLayerSum

        layer_sum = WeightedLayerSum(num_layers=12, normalize=True)
        weights = layer_sum.get_layer_weights()

        assert weights.sum().item() == pytest.approx(1.0, abs=1e-5)


class TestWavLMAdapter:
    """Tests for WavLMAdapter module."""

    def test_forward_shape(self):
        from src.model.wavlm_frontend import WavLMAdapter

        adapter = WavLMAdapter(wavlm_dim=768, d_model=512)

        x = torch.randn(2, 100, 768)
        output = adapter(x)

        assert output.shape == (2, 100, 512)


class TestConformerBlock:
    """Tests for ConformerBlock module."""

    def test_forward_shape(self):
        from src.model.conformer import ConformerBlock

        block = ConformerBlock(
            d_model=512,
            num_heads=8,
            d_ff=2048,
            kernel_size=31,
        )

        x = torch.randn(2, 100, 512)
        output, cache = block(x)

        assert output.shape == x.shape

    def test_causal_mode(self):
        from src.model.conformer import ConformerBlock

        block = ConformerBlock(
            d_model=512,
            num_heads=8,
            d_ff=2048,
            kernel_size=31,
            causal=True,
        )

        x = torch.randn(2, 100, 512)
        output, _ = block(x)

        assert output.shape == x.shape


class TestZipformerEncoder:
    """Tests for ZipformerEncoder module."""

    def test_forward_shape(self):
        from src.model.zipformer import ZipformerEncoder

        encoder = ZipformerEncoder(
            d_model=512,
            num_heads=8,
            d_ff=2048,
            num_blocks=3,
            layers_per_block=2,
        )

        x = torch.randn(2, 100, 512)
        output, mask = encoder(x)

        # Output should have same shape as input (after U-Net)
        assert output.shape == x.shape
        assert mask.shape == (2, 100)


class TestTransformerDecoder:
    """Tests for TransformerDecoder module."""

    def test_forward_shape(self):
        from src.model.decoder import TransformerDecoder

        decoder = TransformerDecoder(
            vocab_size=2000,
            d_model=512,
            num_heads=8,
            num_layers=4,
        )

        encoder_output = torch.randn(2, 100, 512)
        input_ids = torch.randint(0, 2000, (2, 20))

        logits, cache = decoder(input_ids, encoder_output)

        assert logits.shape == (2, 20, 2000)

    def test_generate_greedy(self):
        from src.model.decoder import TransformerDecoder

        decoder = TransformerDecoder(
            vocab_size=2000,
            d_model=512,
            num_heads=8,
            num_layers=4,
        )

        encoder_output = torch.randn(2, 100, 512)

        generated = decoder.generate(
            encoder_output,
            max_length=10,
            beam_size=1,
        )

        assert generated.shape[0] == 2
        assert generated.shape[1] <= 10


class TestVoxFormer:
    """Tests for full VoxFormer model."""

    @pytest.fixture
    def model(self):
        from src.model import VoxFormer

        # Use smaller model for testing
        return VoxFormer(
            vocab_size=100,
            d_model=256,
            encoder_num_heads=4,
            encoder_num_blocks=2,
            encoder_layers_per_block=1,
            decoder_num_heads=4,
            decoder_num_layers=2,
            d_ff=512,
            wavlm_model_name="microsoft/wavlm-base",
        )

    def test_forward_shape(self, model):
        # Dummy input
        waveform = torch.randn(2, 16000)  # 1 second
        target_ids = torch.randint(0, 100, (2, 10))

        outputs = model(
            waveform=waveform,
            target_ids=target_ids,
        )

        assert "encoder_output" in outputs
        assert "ctc_logits" in outputs
        assert "decoder_logits" in outputs

    def test_transcribe(self, model):
        waveform = torch.randn(1, 16000)

        generated = model.transcribe(
            waveform=waveform,
            max_length=10,
            beam_size=1,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] <= 10

    def test_training_stages(self, model):
        # Stage 1: WavLM frozen
        model.set_training_stage(1)
        wavlm_trainable = sum(
            p.numel() for p in model.frontend.wavlm.parameters() if p.requires_grad
        )
        assert wavlm_trainable == 0

        # Stage 2: Top 3 layers unfrozen
        model.set_training_stage(2)
        wavlm_trainable = sum(
            p.numel() for p in model.frontend.wavlm.parameters() if p.requires_grad
        )
        assert wavlm_trainable > 0
