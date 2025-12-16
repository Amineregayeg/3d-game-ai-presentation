"""
Zipformer Encoder Module

Implementation of the Zipformer encoder with U-Net style downsampling.
The encoder progressively reduces temporal resolution while increasing
the receptive field:
    50 fps → 25 fps → 12.5 fps → 25 fps → 50 fps (symmetric)

Key features:
- Stack of Conformer blocks at each resolution
- Downsampling via strided convolution
- Upsampling via transposed convolution
- Skip connections between matching resolutions

Reference: "Zipformer: A faster and better encoder for automatic speech recognition"
           https://arxiv.org/abs/2310.11230
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from src.model.conformer import ConformerBlock


class DownsampleLayer(nn.Module):
    """
    Temporal downsampling layer using strided convolution.

    Reduces sequence length by factor of 2 while preserving dimension.

    Args:
        d_model: Model dimension
        kernel_size: Convolution kernel size (default: 3)
    """

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            bias=False,
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample input by factor of 2.

        Args:
            x: Input tensor of shape (B, T, D)

        Returns:
            Downsampled tensor of shape (B, T//2, D)
        """
        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # (B, D, T//2) -> (B, T//2, D)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x


class UpsampleLayer(nn.Module):
    """
    Temporal upsampling layer using transposed convolution.

    Increases sequence length by factor of 2 while preserving dimension.

    Args:
        d_model: Model dimension
        kernel_size: Convolution kernel size (default: 3)
    """

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
            bias=False,
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Upsample input by factor of 2.

        Args:
            x: Input tensor of shape (B, T, D)
            target_len: Target sequence length after upsampling

        Returns:
            Upsampled tensor of shape (B, target_len, D)
        """
        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Trim to exact target length
        x = x[:, :, :target_len]
        # (B, D, target_len) -> (B, target_len, D)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x


class ZipformerBlock(nn.Module):
    """
    Zipformer block consisting of multiple Conformer layers at one resolution.

    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 2048)
        num_layers: Number of Conformer blocks (default: 2)
        kernel_size: Convolution kernel size (default: 31)
        dropout: Dropout probability (default: 0.1)
        causal: Whether to use causal convolution (default: False)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                kernel_size=kernel_size,
                dropout=dropout,
                causal=causal,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through all Conformer layers.

        Args:
            x: Input tensor of shape (B, T, D)
            attention_mask: Optional mask of shape (B, T)
            cache: Optional list of KV caches for each layer

        Returns:
            Tuple of output tensor and updated caches
        """
        new_caches = [] if cache is not None else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, attention_mask, layer_cache)

            if new_caches is not None:
                new_caches.append(new_cache)

        return x, new_caches


class ZipformerEncoder(nn.Module):
    """
    Zipformer Encoder with U-Net style multi-scale processing.

    Architecture (6 blocks, ~25M params):
        Input (50fps) → Block1 → Down → Block2 → Down → Block3
                                                          ↓
        Output (50fps) ← Block6 ← Up ← Block5 ← Up ← Block4

    The encoder uses skip connections between blocks at matching resolutions.

    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 2048)
        num_blocks: Number of resolution stages (default: 3, means 6 total blocks)
        layers_per_block: Conformer layers per block (default: 2)
        kernel_size: Convolution kernel size (default: 31)
        dropout: Dropout probability (default: 0.1)
        causal: Whether to use causal convolution for streaming (default: False)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_blocks: int = 3,
        layers_per_block: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks

        # Encoder blocks (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        for i in range(num_blocks):
            self.encoder_blocks.append(
                ZipformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_layers=layers_per_block,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    causal=causal,
                )
            )
            if i < num_blocks - 1:
                self.downsample_layers.append(DownsampleLayer(d_model))

        # Decoder blocks (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        for i in range(num_blocks - 1):
            self.upsample_layers.append(UpsampleLayer(d_model))
            self.decoder_blocks.append(
                ZipformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_layers=layers_per_block,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    causal=causal,
                )
            )

        # Skip connection fusion layers
        self.skip_fusions = nn.ModuleList([
            nn.Linear(2 * d_model, d_model)
            for _ in range(num_blocks - 1)
        ])

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Zipformer encoder.

        Args:
            x: Input features of shape (B, T, d_model)
            attention_mask: Optional mask of shape (B, T)

        Returns:
            Tuple of:
            - Encoder output of shape (B, T, d_model)
            - Output mask of shape (B, T)
        """
        # Store skip connections and lengths for decoder path
        skip_connections = []
        skip_lengths = []

        # Encoder path (downsampling)
        for i, block in enumerate(self.encoder_blocks):
            x, _ = block(x, attention_mask)

            if i < self.num_blocks - 1:
                # Save for skip connection
                skip_connections.append(x)
                skip_lengths.append(x.shape[1])

                # Downsample
                x = self.downsample_layers[i](x)

                # Downsample attention mask
                if attention_mask is not None:
                    # Keep every other position
                    attention_mask = attention_mask[:, ::2]
                    # Ensure mask length matches
                    attention_mask = attention_mask[:, :x.shape[1]]

        # Decoder path (upsampling)
        for i in range(self.num_blocks - 1):
            # Upsample to match skip connection length
            target_len = skip_lengths[-(i + 1)]
            x = self.upsample_layers[i](x, target_len)

            # Upsample attention mask
            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(2, dim=1)
                attention_mask = attention_mask[:, :target_len]

            # Fuse with skip connection
            skip = skip_connections[-(i + 1)]
            x = self.skip_fusions[i](torch.cat([x, skip], dim=-1))

            # Process through decoder block
            x, _ = self.decoder_blocks[i](x, attention_mask)

        # Final normalization
        x = self.output_norm(x)

        # Create output mask
        if attention_mask is None:
            output_mask = torch.ones(
                x.shape[0], x.shape[1],
                dtype=torch.bool, device=x.device
            )
        else:
            output_mask = attention_mask

        return x, output_mask

    def forward_streaming(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Streaming forward pass with caching.

        For real-time inference, processes one chunk at a time while
        maintaining state across chunks.

        Args:
            x: Input chunk of shape (B, chunk_len, d_model)
            attention_mask: Optional mask
            cache: Dictionary of cached states from previous chunks

        Returns:
            Tuple of:
            - Output tensor
            - Output mask
            - Updated cache dictionary
        """
        # Initialize cache if needed
        if cache is None:
            cache = {
                "encoder_caches": [None] * len(self.encoder_blocks),
                "decoder_caches": [None] * len(self.decoder_blocks),
                "skip_buffers": [None] * (self.num_blocks - 1),
            }

        # Simplified streaming - process full chunk through encoder
        # Note: Full streaming with multi-scale would require more complex buffering
        skip_connections = []
        skip_lengths = []

        # Encoder path
        for i, block in enumerate(self.encoder_blocks):
            x, cache["encoder_caches"][i] = block(
                x, attention_mask, cache["encoder_caches"][i]
            )

            if i < self.num_blocks - 1:
                skip_connections.append(x)
                skip_lengths.append(x.shape[1])
                x = self.downsample_layers[i](x)
                if attention_mask is not None:
                    attention_mask = attention_mask[:, ::2][:, :x.shape[1]]

        # Decoder path
        for i in range(self.num_blocks - 1):
            target_len = skip_lengths[-(i + 1)]
            x = self.upsample_layers[i](x, target_len)

            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(2, dim=1)[:, :target_len]

            skip = skip_connections[-(i + 1)]
            x = self.skip_fusions[i](torch.cat([x, skip], dim=-1))

            x, cache["decoder_caches"][i] = self.decoder_blocks[i](
                x, attention_mask, cache["decoder_caches"][i]
            )

        x = self.output_norm(x)

        output_mask = attention_mask if attention_mask is not None else torch.ones(
            x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
        )

        return x, output_mask, cache

    @property
    def output_dim(self) -> int:
        """Return output dimension."""
        return self.d_model
