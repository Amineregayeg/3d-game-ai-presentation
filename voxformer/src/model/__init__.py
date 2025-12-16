"""
VoxFormer Model Components

This module contains all neural network components:
- WavLM feature extraction with weighted layer sum
- Zipformer encoder (Conformer-based)
- Transformer decoder with cross-attention
- Full VoxFormer model
"""

from src.model.wavlm_frontend import WeightedLayerSum, WavLMAdapter, WavLMFrontend
from src.model.conformer import ConformerBlock, ConvolutionModule, FeedForwardModule
from src.model.zipformer import ZipformerEncoder, ZipformerBlock
from src.model.decoder import TransformerDecoder, TransformerDecoderLayer
from src.model.voxformer import VoxFormer

__all__ = [
    # Frontend
    "WeightedLayerSum",
    "WavLMAdapter",
    "WavLMFrontend",
    # Encoder
    "ConformerBlock",
    "ConvolutionModule",
    "FeedForwardModule",
    "ZipformerEncoder",
    "ZipformerBlock",
    # Decoder
    "TransformerDecoder",
    "TransformerDecoderLayer",
    # Full model
    "VoxFormer",
]
