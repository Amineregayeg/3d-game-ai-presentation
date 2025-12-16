"""
WavLM Frontend Module

This module implements the WavLM-based feature extraction frontend:
1. WeightedLayerSum: Learnable combination of WavLM's 12 hidden layers
2. WavLMAdapter: Projects 768-dim WavLM features to 512-dim encoder input
3. WavLMFrontend: Complete frontend combining WavLM + Weighted Sum + Adapter

The WavLM backbone is frozen during Stage 1 training and partially unfrozen
(top 3 layers) during Stage 2 fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, WavLMConfig
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WeightedLayerSum(nn.Module):
    """
    Learnable weighted sum of WavLM hidden layer outputs.

    Different layers capture different features:
    - Lower layers (1-4): Acoustic features (pitch, energy)
    - Middle layers (5-8): Phonetic features (phonemes, pronunciation)
    - Upper layers (9-12): Semantic features (word boundaries, context)

    The learnable weights allow the model to optimally combine these features
    for ASR, typically improving WER by 0.2-0.5%.

    Args:
        num_layers: Number of hidden layers to combine (default: 12 for WavLM-Base)
        normalize: Whether to normalize weights with softmax (default: True)
    """

    def __init__(self, num_layers: int = 12, normalize: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.normalize = normalize

        # Learnable weights initialized uniformly
        # Using raw weights that get softmax-normalized during forward
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute weighted sum of hidden states.

        Args:
            hidden_states: Tuple of tensors from WavLM, each (B, T, 768)
                          First element is CNN output, rest are transformer layers

        Returns:
            Weighted sum tensor of shape (B, T, 768)
        """
        # Stack hidden states: (num_layers, B, T, D)
        # Skip the first element (CNN features) and use transformer outputs
        stacked = torch.stack(hidden_states[1:], dim=0)

        # Normalize weights if enabled
        if self.normalize:
            weights = F.softmax(self.layer_weights, dim=0)
        else:
            weights = self.layer_weights

        # Reshape weights for broadcasting: (num_layers, 1, 1, 1)
        weights = weights.view(-1, 1, 1, 1)

        # Weighted sum: (B, T, D)
        weighted_sum = (stacked * weights).sum(dim=0)

        return weighted_sum

    def get_layer_weights(self) -> torch.Tensor:
        """Return normalized layer weights for visualization/debugging."""
        if self.normalize:
            return F.softmax(self.layer_weights, dim=0).detach()
        return self.layer_weights.detach()

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}, normalize={self.normalize}"


class WavLMAdapter(nn.Module):
    """
    Adapter module to project WavLM features to encoder dimension.

    Architecture:
        LayerNorm → Linear(768 → d_model) → GELU → Dropout → Linear(d_model → d_model)

    This bridges the 768-dim WavLM output to the 512-dim Zipformer input
    while applying normalization and non-linearity.

    Args:
        wavlm_dim: WavLM hidden dimension (768 for Base, 1024 for Large)
        d_model: Target encoder dimension (default: 512)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        wavlm_dim: int = 768,
        d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.wavlm_dim = wavlm_dim
        self.d_model = d_model

        self.layer_norm = nn.LayerNorm(wavlm_dim)
        self.proj = nn.Linear(wavlm_dim, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable training."""
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project WavLM features to encoder dimension.

        Args:
            x: WavLM features of shape (B, T, wavlm_dim)

        Returns:
            Projected features of shape (B, T, d_model)
        """
        x = self.layer_norm(x)
        x = self.proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def extra_repr(self) -> str:
        return f"wavlm_dim={self.wavlm_dim}, d_model={self.d_model}"


class WavLMFrontend(nn.Module):
    """
    Complete WavLM-based frontend for VoxFormer.

    This module:
    1. Loads pretrained WavLM-Base (95M params)
    2. Extracts features with output_hidden_states=True
    3. Applies WeightedLayerSum to combine all 12 layers
    4. Projects to encoder dimension via WavLMAdapter

    Freezing Strategy:
    - Stage 1: Entire WavLM frozen (freeze_wavlm=True, unfreeze_top_k=0)
    - Stage 2: Top 3 layers unfrozen (freeze_wavlm=True, unfreeze_top_k=3)

    Args:
        d_model: Output dimension for encoder (default: 512)
        wavlm_model_name: HuggingFace model ID (default: microsoft/wavlm-base)
        freeze_wavlm: Whether to freeze WavLM weights (default: True)
        unfreeze_top_k: Number of top layers to unfreeze (default: 0)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int = 512,
        wavlm_model_name: str = "microsoft/wavlm-base",
        freeze_wavlm: bool = True,
        unfreeze_top_k: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.freeze_wavlm = freeze_wavlm
        self.unfreeze_top_k = unfreeze_top_k

        # Load pretrained WavLM
        logger.info(f"Loading WavLM from {wavlm_model_name}")
        self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)
        self.wavlm_config: WavLMConfig = self.wavlm.config

        # Get WavLM hidden dimension
        wavlm_dim = self.wavlm_config.hidden_size  # 768 for base
        num_layers = self.wavlm_config.num_hidden_layers  # 12 for base

        logger.info(f"WavLM: {wavlm_dim}d, {num_layers} layers")

        # Weighted layer sum
        self.weighted_layer_sum = WeightedLayerSum(num_layers=num_layers)

        # Adapter projection
        self.adapter = WavLMAdapter(
            wavlm_dim=wavlm_dim,
            d_model=d_model,
            dropout=dropout,
        )

        # Apply freezing strategy
        self._apply_freeze_strategy()

    def _apply_freeze_strategy(self):
        """Apply freezing to WavLM based on configuration."""
        if self.freeze_wavlm:
            # Freeze all WavLM parameters
            for param in self.wavlm.parameters():
                param.requires_grad = False

            # Unfreeze top-k layers if specified
            if self.unfreeze_top_k > 0:
                num_layers = self.wavlm_config.num_hidden_layers
                layers_to_unfreeze = list(range(num_layers - self.unfreeze_top_k, num_layers))

                for layer_idx in layers_to_unfreeze:
                    for param in self.wavlm.encoder.layers[layer_idx].parameters():
                        param.requires_grad = True

                logger.info(f"Unfroze WavLM layers: {layers_to_unfreeze}")

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"WavLM Frontend: {trainable:,} / {total:,} trainable params")

    def set_freeze_strategy(self, freeze_wavlm: bool, unfreeze_top_k: int = 0):
        """
        Update freezing strategy (for transitioning between training stages).

        Args:
            freeze_wavlm: Whether to freeze WavLM
            unfreeze_top_k: Number of top layers to unfreeze
        """
        self.freeze_wavlm = freeze_wavlm
        self.unfreeze_top_k = unfreeze_top_k
        self._apply_freeze_strategy()

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from raw audio waveform.

        Args:
            waveform: Raw audio tensor of shape (B, num_samples) at 16kHz
            attention_mask: Optional mask of shape (B, num_samples)

        Returns:
            Tuple of:
            - features: Encoder input features of shape (B, T, d_model)
            - feature_mask: Attention mask for features of shape (B, T)
        """
        # Extract WavLM features with all hidden states
        outputs = self.wavlm(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get all hidden states (tuple of 13 tensors: CNN + 12 transformer layers)
        hidden_states = outputs.hidden_states

        # Apply weighted layer sum
        features = self.weighted_layer_sum(hidden_states)

        # Project to encoder dimension
        features = self.adapter(features)

        # Compute feature-level attention mask
        if attention_mask is not None:
            # WavLM downsamples by factor of 320 (20ms @ 16kHz)
            # Then outputs at 50 fps (one frame per 320 samples)
            feature_lengths = self._compute_feature_lengths(attention_mask)
            feature_mask = self._make_feature_mask(features, feature_lengths)
        else:
            feature_mask = torch.ones(
                features.shape[0], features.shape[1],
                dtype=torch.bool, device=features.device
            )

        return features, feature_mask

    def _compute_feature_lengths(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute output lengths from input attention mask."""
        # Sum mask to get input lengths
        input_lengths = attention_mask.sum(dim=1)

        # WavLM conv layers downsample by factor of 320
        # Formula from WavLM config
        feature_lengths = self.wavlm._get_feat_extract_output_lengths(input_lengths)

        return feature_lengths.long()

    def _make_feature_mask(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Create boolean mask from lengths."""
        batch_size, max_len, _ = features.shape
        mask = torch.arange(max_len, device=features.device).expand(batch_size, -1)
        mask = mask < lengths.unsqueeze(1)
        return mask

    @property
    def output_dim(self) -> int:
        """Return output dimension."""
        return self.d_model

    @property
    def downsample_factor(self) -> int:
        """Return total downsampling factor from audio to features."""
        # WavLM: 320x downsample (16kHz → 50 fps)
        return 320

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"freeze_wavlm={self.freeze_wavlm}, "
            f"unfreeze_top_k={self.unfreeze_top_k}"
        )
