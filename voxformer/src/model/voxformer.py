"""
VoxFormer: Complete Speech-to-Text Model

This module combines all components into the full VoxFormer architecture:
    Audio → WavLM Frontend → Zipformer Encoder → Transformer Decoder → Text

Model Specifications:
- Total Parameters: ~142M (47M trainable with WavLM frozen)
- WavLM-Base: 95M params (frozen/partial)
- Zipformer Encoder: ~25M params
- Transformer Decoder: ~20M params
- Adapter + misc: ~2M params

Target Performance:
- WER: <3.5% (LibriSpeech), <8% (gaming domain)
- Latency: <200ms streaming
- RTF: <0.1 (GPU), <0.3 (CPU)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union
import logging

from src.model.wavlm_frontend import WavLMFrontend
from src.model.zipformer import ZipformerEncoder
from src.model.decoder import TransformerDecoder

logger = logging.getLogger(__name__)


class VoxFormer(nn.Module):
    """
    VoxFormer: Elite Speech-to-Text Transformer for Gaming AI.

    End-to-end ASR model combining:
    1. WavLM pretrained audio features (frozen/partial)
    2. Zipformer encoder with U-Net downsampling
    3. Transformer decoder with cross-attention

    Training Strategy:
    - Stage 1: WavLM frozen, train encoder + decoder on LibriSpeech
    - Stage 2: Unfreeze WavLM top 3 layers, fine-tune all
    - Stage 3: Domain adaptation for gaming vocabulary

    Args:
        vocab_size: BPE vocabulary size (default: 2000)
        d_model: Model dimension (default: 512)
        encoder_num_heads: Encoder attention heads (default: 8)
        encoder_num_blocks: Zipformer resolution stages (default: 3)
        encoder_layers_per_block: Conformer layers per block (default: 2)
        decoder_num_heads: Decoder attention heads (default: 8)
        decoder_num_layers: Decoder layers (default: 4)
        d_ff: Feed-forward dimension (default: 2048)
        kernel_size: Convolution kernel size (default: 31)
        dropout: Dropout probability (default: 0.1)
        wavlm_model_name: HuggingFace model ID (default: microsoft/wavlm-base)
        freeze_wavlm: Whether to freeze WavLM (default: True)
        unfreeze_top_k: Number of top WavLM layers to unfreeze (default: 0)
        pad_token_id: Padding token ID (default: 0)
        bos_token_id: BOS token ID (default: 1)
        eos_token_id: EOS token ID (default: 2)
        ctc_weight: CTC loss weight for joint training (default: 0.3)
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        d_model: int = 512,
        encoder_num_heads: int = 8,
        encoder_num_blocks: int = 3,
        encoder_layers_per_block: int = 2,
        decoder_num_heads: int = 8,
        decoder_num_layers: int = 4,
        d_ff: int = 2048,
        kernel_size: int = 31,
        dropout: float = 0.1,
        wavlm_model_name: str = "microsoft/wavlm-base",
        freeze_wavlm: bool = True,
        unfreeze_top_k: int = 0,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        ctc_weight: float = 0.3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ctc_weight = ctc_weight
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # WavLM Frontend
        self.frontend = WavLMFrontend(
            d_model=d_model,
            wavlm_model_name=wavlm_model_name,
            freeze_wavlm=freeze_wavlm,
            unfreeze_top_k=unfreeze_top_k,
            dropout=dropout,
        )

        # Zipformer Encoder
        self.encoder = ZipformerEncoder(
            d_model=d_model,
            num_heads=encoder_num_heads,
            d_ff=d_ff,
            num_blocks=encoder_num_blocks,
            layers_per_block=encoder_layers_per_block,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=False,  # Non-causal for training, causal for streaming
        )

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            d_ff=d_ff,
            dropout=dropout,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        # CTC projection for joint CTC-Attention training
        self.ctc_proj = nn.Linear(d_model, vocab_size)

        # Log model info
        self._log_model_info()

    def _log_model_info(self):
        """Log model parameter counts."""
        def count_params(module, trainable_only=False):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())

        frontend_total = count_params(self.frontend)
        frontend_trainable = count_params(self.frontend, trainable_only=True)
        encoder_params = count_params(self.encoder)
        decoder_params = count_params(self.decoder)
        ctc_params = count_params(self.ctc_proj)

        total = frontend_total + encoder_params + decoder_params + ctc_params
        trainable = frontend_trainable + encoder_params + decoder_params + ctc_params

        logger.info(f"VoxFormer Model Summary:")
        logger.info(f"  Frontend (WavLM): {frontend_total:,} ({frontend_trainable:,} trainable)")
        logger.info(f"  Encoder: {encoder_params:,}")
        logger.info(f"  Decoder: {decoder_params:,}")
        logger.info(f"  CTC Head: {ctc_params:,}")
        logger.info(f"  Total: {total:,} ({trainable:,} trainable)")

    def set_training_stage(self, stage: int):
        """
        Configure model for different training stages.

        Args:
            stage: Training stage (1, 2, or 3)
                1: WavLM frozen, train encoder + decoder
                2: Unfreeze WavLM top 3 layers
                3: Full fine-tuning (gaming domain)
        """
        if stage == 1:
            self.frontend.set_freeze_strategy(freeze_wavlm=True, unfreeze_top_k=0)
            logger.info("Stage 1: WavLM frozen")
        elif stage == 2:
            self.frontend.set_freeze_strategy(freeze_wavlm=True, unfreeze_top_k=3)
            logger.info("Stage 2: WavLM top 3 layers unfrozen")
        elif stage == 3:
            self.frontend.set_freeze_strategy(freeze_wavlm=False, unfreeze_top_k=0)
            logger.info("Stage 3: Full model trainable")
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            waveform: Raw audio tensor of shape (B, num_samples) at 16kHz
            waveform_lengths: Optional lengths of shape (B,)
            target_ids: Target token IDs of shape (B, T_text) for teacher forcing

        Returns:
            Dictionary containing:
            - encoder_output: Encoder features (B, T_enc, d_model)
            - encoder_mask: Encoder mask (B, T_enc)
            - ctc_logits: CTC logits (B, T_enc, vocab_size)
            - decoder_logits: Decoder logits (B, T_text, vocab_size) if target_ids provided
        """
        # Create attention mask from lengths
        if waveform_lengths is not None:
            max_len = waveform.shape[1]
            attention_mask = torch.arange(max_len, device=waveform.device).expand(
                waveform.shape[0], -1
            ) < waveform_lengths.unsqueeze(1)
        else:
            attention_mask = None

        # Frontend: Audio → Features
        features, feature_mask = self.frontend(waveform, attention_mask)

        # Encoder: Features → Encoded representations
        encoder_output, encoder_mask = self.encoder(features, feature_mask)

        # CTC projection for joint training
        ctc_logits = self.ctc_proj(encoder_output)

        result = {
            "encoder_output": encoder_output,
            "encoder_mask": encoder_mask,
            "ctc_logits": ctc_logits,
        }

        # Decoder: Encoded + Targets → Logits (teacher forcing)
        if target_ids is not None:
            decoder_logits, _ = self.decoder(
                target_ids,
                encoder_output,
                encoder_mask.float(),
            )
            result["decoder_logits"] = decoder_logits

        return result

    def transcribe(
        self,
        waveform: torch.Tensor,
        waveform_lengths: Optional[torch.Tensor] = None,
        max_length: int = 256,
        beam_size: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Transcribe audio to text tokens.

        Args:
            waveform: Raw audio tensor of shape (B, num_samples) at 16kHz
            waveform_lengths: Optional lengths
            max_length: Maximum generation length
            beam_size: Beam size (1 = greedy)
            temperature: Sampling temperature

        Returns:
            Generated token IDs of shape (B, gen_len)
        """
        # Create attention mask
        if waveform_lengths is not None:
            max_len = waveform.shape[1]
            attention_mask = torch.arange(max_len, device=waveform.device).expand(
                waveform.shape[0], -1
            ) < waveform_lengths.unsqueeze(1)
        else:
            attention_mask = None

        # Encode
        features, feature_mask = self.frontend(waveform, attention_mask)
        encoder_output, encoder_mask = self.encoder(features, feature_mask)

        # Generate
        generated = self.decoder.generate(
            encoder_output,
            encoder_mask.float(),
            max_length=max_length,
            beam_size=beam_size,
            temperature=temperature,
        )

        return generated

    def transcribe_streaming(
        self,
        chunk: torch.Tensor,
        cache: Optional[Dict] = None,
        max_new_tokens: int = 32,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Streaming transcription for real-time inference.

        Processes audio chunks incrementally, maintaining state across chunks.

        Args:
            chunk: Audio chunk of shape (B, chunk_samples) at 16kHz
            cache: State from previous chunks
            max_new_tokens: Maximum tokens to generate per chunk

        Returns:
            Tuple of:
            - Generated token IDs for this chunk
            - Updated cache for next chunk
        """
        if cache is None:
            cache = {
                "encoder_cache": None,
                "decoder_cache": None,
                "encoder_output_buffer": None,
            }

        # Process chunk through frontend
        features, feature_mask = self.frontend(chunk, None)

        # Process through encoder with caching
        encoder_output, encoder_mask, encoder_cache = self.encoder.forward_streaming(
            features, feature_mask, cache.get("encoder_cache")
        )

        # Accumulate encoder outputs
        if cache.get("encoder_output_buffer") is not None:
            encoder_output = torch.cat(
                [cache["encoder_output_buffer"], encoder_output], dim=1
            )
            encoder_mask = torch.cat(
                [cache.get("encoder_mask_buffer", encoder_mask), encoder_mask], dim=1
            )

        # Generate tokens for this chunk
        generated = self.decoder.generate(
            encoder_output,
            encoder_mask.float(),
            max_length=max_new_tokens,
            beam_size=1,  # Always greedy for streaming
            temperature=1.0,
        )

        # Update cache
        new_cache = {
            "encoder_cache": encoder_cache,
            "decoder_cache": cache.get("decoder_cache"),
            "encoder_output_buffer": encoder_output,
            "encoder_mask_buffer": encoder_mask,
        }

        return generated, new_cache

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "VoxFormer":
        """
        Load a pretrained VoxFormer model.

        Args:
            path: Path to checkpoint file
            device: Device to load to

        Returns:
            Loaded VoxFormer model
        """
        checkpoint = torch.load(path, map_location=device)

        # Get config from checkpoint
        config = checkpoint.get("config", {})

        # Create model
        model = cls(**config)

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Loaded VoxFormer from {path}")
        return model

    def save_pretrained(self, path: str, config: Optional[dict] = None):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            config: Optional config dict to save with checkpoint
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {},
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved VoxFormer to {path}")
