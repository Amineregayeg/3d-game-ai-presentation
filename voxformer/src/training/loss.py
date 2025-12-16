"""
Hybrid CTC + Attention Loss Module

Implements the joint CTC-Attention loss function for VoxFormer training:
    L = λ_ctc · L_CTC + λ_ce · L_CE

Where:
- L_CTC: Connectionist Temporal Classification loss (alignment-free)
- L_CE: Cross-Entropy loss with label smoothing (attention-based)

Benefits:
- CTC provides fast initial convergence and handles variable-length alignment
- Cross-Entropy provides better final WER through explicit token modeling

Training Schedule:
- Warmup (0-5K steps): λ_ctc=0.4, λ_ce=0.6
- Main training: λ_ctc=0.3, λ_ce=0.7
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LabelSmoothingLoss(nn.Module):
    """
    Cross-Entropy loss with label smoothing.

    Label smoothing prevents overconfident predictions and improves
    generalization by distributing a small probability mass to non-target tokens.

    Formula:
        target_dist = (1 - smoothing) * one_hot + smoothing / vocab_size
        loss = -sum(target_dist * log_probs)

    Args:
        vocab_size: Size of vocabulary
        smoothing: Label smoothing factor (default: 0.1)
        pad_token_id: Padding token ID to ignore (default: 0)
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_token_id = pad_token_id
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Predicted logits of shape (B, T, vocab_size)
            targets: Target token IDs of shape (B, T)

        Returns:
            Scalar loss value
        """
        B, T, V = logits.shape

        # Flatten for loss computation
        logits = logits.reshape(-1, V)
        targets = targets.reshape(-1)

        # Log softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed target distribution
        # smooth_target = confidence * one_hot + smoothing / V
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / V)
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Create padding mask
        pad_mask = targets != self.pad_token_id

        # Compute loss: -sum(smooth_targets * log_probs)
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Apply padding mask
        loss = loss * pad_mask.float()

        # Return mean over non-padded tokens
        num_tokens = pad_mask.sum()
        if num_tokens > 0:
            return loss.sum() / num_tokens
        else:
            return loss.sum()


class CTCLoss(nn.Module):
    """
    CTC Loss wrapper with proper handling of input/target lengths.

    CTC (Connectionist Temporal Classification) enables training without
    explicit alignment between audio frames and text tokens. It marginalizes
    over all valid alignments using the forward-backward algorithm.

    Key features:
    - Alignment-free: No need for frame-level labels
    - Variable length: Handles different input/output lengths
    - Blank token: Special token for representing "no output"

    Args:
        blank_token_id: ID of blank token (default: 0, same as padding)
        reduction: Loss reduction method (default: "mean")
    """

    def __init__(
        self,
        blank_token_id: int = 0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.blank_token_id = blank_token_id
        self.ctc_loss = nn.CTCLoss(
            blank=blank_token_id,
            reduction=reduction,
            zero_infinity=True,  # Clamp infinite losses
        )

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            log_probs: Log probabilities of shape (T, B, vocab_size)
                      Note: Time-first format required by CTC
            targets: Flattened target sequence (sum of all target_lengths)
            input_lengths: Length of each input sequence (B,)
            target_lengths: Length of each target sequence (B,)

        Returns:
            Scalar CTC loss value
        """
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


class HybridCTCAttentionLoss(nn.Module):
    """
    Hybrid CTC + Attention Loss for joint training.

    Combines CTC and cross-entropy losses with configurable weights:
        L = λ_ctc · L_CTC + λ_ce · L_CE

    The loss weights can be dynamically adjusted during training
    using the warmup schedule.

    Args:
        vocab_size: Size of vocabulary
        ctc_weight: Weight for CTC loss (default: 0.3)
        ce_weight: Weight for cross-entropy loss (default: 0.7)
        label_smoothing: Label smoothing factor (default: 0.1)
        blank_token_id: CTC blank token ID (default: 0)
        pad_token_id: Padding token ID (default: 0)
        warmup_ctc_weight: CTC weight during warmup (default: 0.4)
        warmup_ce_weight: CE weight during warmup (default: 0.6)
        warmup_steps: Number of warmup steps (default: 5000)
    """

    def __init__(
        self,
        vocab_size: int,
        ctc_weight: float = 0.3,
        ce_weight: float = 0.7,
        label_smoothing: float = 0.1,
        blank_token_id: int = 0,
        pad_token_id: int = 0,
        warmup_ctc_weight: float = 0.4,
        warmup_ce_weight: float = 0.6,
        warmup_steps: int = 5000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight
        self.warmup_ctc_weight = warmup_ctc_weight
        self.warmup_ce_weight = warmup_ce_weight
        self.warmup_steps = warmup_steps
        self.pad_token_id = pad_token_id

        # Loss components
        self.ctc_loss = CTCLoss(blank_token_id=blank_token_id)
        self.ce_loss = LabelSmoothingLoss(
            vocab_size=vocab_size,
            smoothing=label_smoothing,
            pad_token_id=pad_token_id,
        )

        # Track current step for warmup
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))

    def get_current_weights(self) -> Tuple[float, float]:
        """Get current CTC and CE weights based on training step."""
        step = self.current_step.item()

        if step < self.warmup_steps:
            return self.warmup_ctc_weight, self.warmup_ce_weight
        else:
            return self.ctc_weight, self.ce_weight

    def forward(
        self,
        ctc_logits: torch.Tensor,
        decoder_logits: torch.Tensor,
        targets: torch.Tensor,
        encoder_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute hybrid CTC + Attention loss.

        Args:
            ctc_logits: CTC branch logits (B, T_enc, vocab_size)
            decoder_logits: Decoder logits (B, T_dec, vocab_size)
            targets: Target token IDs (B, T_dec) with BOS prepended
            encoder_lengths: Encoder output lengths (B,)
            target_lengths: Target sequence lengths (B,) without BOS

        Returns:
            Tuple of:
            - Total loss (scalar)
            - Dictionary with individual losses for logging
        """
        # Get current loss weights
        ctc_w, ce_w = self.get_current_weights()

        # ========== CTC Loss ==========
        # Convert to log probabilities
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)

        # Transpose to (T, B, V) for CTC
        ctc_log_probs = ctc_log_probs.transpose(0, 1)

        # Flatten targets for CTC (remove BOS, concat all sequences)
        # targets[:, 1:] removes BOS token
        target_flat = []
        for i, length in enumerate(target_lengths):
            # Skip BOS token (position 0), take 'length' tokens
            target_flat.append(targets[i, 1:length + 1])
        target_flat = torch.cat(target_flat)

        # Compute CTC loss
        ctc_loss = self.ctc_loss(
            ctc_log_probs,
            target_flat,
            encoder_lengths,
            target_lengths,
        )

        # ========== Cross-Entropy Loss ==========
        # decoder_logits: (B, T, V) - predictions for positions 0 to T-1
        # targets: (B, T) - where targets[:, 1:] are the expected outputs

        # Shift for autoregressive: predict target[t] from input[t-1]
        # decoder_logits[:, :-1] predicts targets[:, 1:]
        ce_loss = self.ce_loss(
            decoder_logits[:, :-1, :],  # Predictions up to T-1
            targets[:, 1:],              # Targets from 1 to T
        )

        # ========== Combined Loss ==========
        total_loss = ctc_w * ctc_loss + ce_w * ce_loss

        # Update step counter
        self.current_step += 1

        # Return loss and components for logging
        loss_dict = {
            "loss": total_loss.item(),
            "ctc_loss": ctc_loss.item(),
            "ce_loss": ce_loss.item(),
            "ctc_weight": ctc_w,
            "ce_weight": ce_w,
        }

        return total_loss, loss_dict

    def reset_step(self):
        """Reset step counter (for resuming training)."""
        self.current_step.zero_()

    def set_step(self, step: int):
        """Set current step (for resuming training)."""
        self.current_step.fill_(step)
