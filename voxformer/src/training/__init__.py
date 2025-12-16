"""
VoxFormer Training Module

This module contains training infrastructure:
- HybridCTCAttentionLoss: Joint CTC + Cross-Entropy loss
- Trainer: Training loop with mixed precision and gradient accumulation
- Evaluation metrics: WER calculation
"""

from src.training.loss import HybridCTCAttentionLoss
from src.training.trainer import Trainer
from src.training.metrics import compute_wer, WERMetric

__all__ = [
    "HybridCTCAttentionLoss",
    "Trainer",
    "compute_wer",
    "WERMetric",
]
