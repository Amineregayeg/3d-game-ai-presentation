"""
VoxFormer: Elite Speech-to-Text Transformer for Gaming AI

A high-performance ASR system combining WavLM pretrained features
with a custom Zipformer encoder and Transformer decoder.

Architecture:
    Audio → WavLM (frozen) → Adapter → Zipformer → Decoder → Text

Target Metrics:
    - WER: <3.5% (LibriSpeech), <8% (gaming)
    - Latency: <200ms streaming
    - Model: 142M params (47M trainable)
"""

__version__ = "1.0.0"
__author__ = "VoxFormer Team"

from src.model import VoxFormer

__all__ = ["VoxFormer", "__version__"]
