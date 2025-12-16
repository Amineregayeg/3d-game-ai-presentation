"""
VoxFormer Trainer

Production-grade training loop with:
- Mixed precision training (AMP)
- Gradient accumulation
- Gradient clipping
- Learning rate scheduling
- Checkpoint saving/loading
- Wandb/TensorBoard logging
- Multi-stage training support
- Evaluation and WER tracking
"""

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.model import VoxFormer
from src.training.loss import HybridCTCAttentionLoss
from src.training.metrics import WERMetric
from src.data.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create cosine learning rate schedule with warmup.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of peak LR

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return current_step / max(1, num_warmup_steps)
        else:
            # Cosine decay
            progress = (current_step - num_warmup_steps) / max(
                1, num_training_steps - num_warmup_steps
            )
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    VoxFormer Trainer with production-grade features.

    Handles the complete training loop including:
    - Mixed precision training
    - Gradient accumulation
    - Multi-stage training (freezing/unfreezing)
    - Checkpointing
    - Evaluation

    Args:
        model: VoxFormer model
        tokenizer: BPETokenizer
        train_dataloader: Training DataLoader
        eval_dataloader: Optional evaluation DataLoader
        learning_rate: Peak learning rate (default: 1e-4)
        weight_decay: Weight decay (default: 0.01)
        max_grad_norm: Gradient clipping norm (default: 1.0)
        warmup_steps: Number of warmup steps (default: 1000)
        num_epochs: Number of training epochs (default: 10)
        gradient_accumulation_steps: Gradient accumulation steps (default: 1)
        mixed_precision: Whether to use AMP (default: True)
        checkpoint_dir: Directory for checkpoints (default: "checkpoints")
        log_interval: Steps between logging (default: 100)
        eval_interval: Steps between evaluation (default: 1000)
        save_interval: Steps between checkpoints (default: 5000)
        device: Training device (default: "cuda")
        wandb_project: Optional Wandb project name
        wandb_run_name: Optional Wandb run name
    """

    def __init__(
        self,
        model: VoxFormer,
        tokenizer: BPETokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        num_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        device: str = "cuda",
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.device = device

        # Loss function
        self.loss_fn = HybridCTCAttentionLoss(
            vocab_size=model.vocab_size,
            ctc_weight=model.ctc_weight,
            ce_weight=1.0 - model.ctc_weight,
        ).to(device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None

        # Metrics
        self.wer_metric = WERMetric()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_wer = float("inf")

        # Wandb logging
        self.wandb_run = None
        if wandb_project:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "max_grad_norm": max_grad_norm,
                        "warmup_steps": warmup_steps,
                        "num_epochs": num_epochs,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "mixed_precision": mixed_precision,
                    },
                )
            except ImportError:
                logger.warning("wandb not installed, skipping logging")

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay exclusion for certain params."""
        # Don't apply weight decay to bias and LayerNorm parameters
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        optimizer_groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(optimizer_groups, lr=self.learning_rate, betas=(0.9, 0.98))

    def train(self):
        """Run full training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Total steps: {len(self.train_dataloader) * self.num_epochs}")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self._train_epoch()

            # Evaluate at end of epoch
            if self.eval_dataloader:
                eval_wer = self.evaluate()
                logger.info(f"Epoch {epoch} - Eval WER: {eval_wer:.2f}%")

                if eval_wer < self.best_wer:
                    self.best_wer = eval_wer
                    self.save_checkpoint("best.pt")

        logger.info(f"Training complete. Best WER: {self.best_wer:.2f}%")

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            waveform = batch["waveform"].to(self.device)
            waveform_lengths = batch["waveform_lengths"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.mixed_precision):
                outputs = self.model(
                    waveform=waveform,
                    waveform_lengths=waveform_lengths,
                    target_ids=input_ids,
                )

                # Compute encoder lengths (after WavLM downsampling)
                encoder_lengths = waveform_lengths // self.model.frontend.downsample_factor

                # Compute loss
                loss, loss_dict = self.loss_fn(
                    ctc_logits=outputs["ctc_logits"],
                    decoder_logits=outputs["decoder_logits"],
                    targets=input_ids,
                    encoder_lengths=encoder_lengths,
                    target_lengths=target_lengths,
                )

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        "loss": f"{loss_dict['loss']:.4f}",
                        "ctc": f"{loss_dict['ctc_loss']:.4f}",
                        "ce": f"{loss_dict['ce_loss']:.4f}",
                        "lr": f"{lr:.2e}",
                    })

                    if self.wandb_run:
                        import wandb

                        wandb.log({
                            "train/loss": loss_dict["loss"],
                            "train/ctc_loss": loss_dict["ctc_loss"],
                            "train/ce_loss": loss_dict["ce_loss"],
                            "train/learning_rate": lr,
                            "train/grad_norm": grad_norm.item(),
                            "train/step": self.global_step,
                        })

                # Evaluation
                if self.eval_dataloader and self.global_step % self.eval_interval == 0:
                    eval_wer = self.evaluate()
                    logger.info(f"Step {self.global_step} - Eval WER: {eval_wer:.2f}%")
                    self.model.train()  # Back to training mode

                # Checkpointing
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")

            epoch_loss += loss_dict["loss"]
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        logger.info(f"Epoch {self.epoch} - Avg Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate model on validation set.

        Returns:
            WER percentage
        """
        self.model.eval()
        self.wer_metric.reset()

        for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
            waveform = batch["waveform"].to(self.device)
            waveform_lengths = batch["waveform_lengths"].to(self.device)
            texts = batch["texts"]

            # Generate transcriptions
            with autocast(enabled=self.mixed_precision):
                generated = self.model.transcribe(
                    waveform=waveform,
                    waveform_lengths=waveform_lengths,
                    max_length=256,
                    beam_size=4,
                )

            # Decode tokens to text
            hypotheses = []
            for gen in generated:
                hyp = self.tokenizer.decode(gen.tolist(), skip_special_tokens=True)
                hypotheses.append(hyp)

            # Update WER
            self.wer_metric.update(texts, hypotheses)

        wer = self.wer_metric.compute()

        if self.wandb_run:
            import wandb

            wandb.log({
                "eval/wer": wer,
                "eval/step": self.global_step,
            })

        return wer

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_wer": self.best_wer,
            "config": {
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
            },
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_wer = checkpoint.get("best_wer", float("inf"))

        # Sync loss function step
        self.loss_fn.set_step(self.global_step)

        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")

    def set_training_stage(self, stage: int, learning_rate: Optional[float] = None):
        """
        Configure for multi-stage training.

        Args:
            stage: Training stage (1, 2, or 3)
            learning_rate: Optional new learning rate (e.g., 10x lower for Stage 2)
        """
        self.model.set_training_stage(stage)

        if learning_rate:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate
            logger.info(f"Set learning rate to {learning_rate}")

        # Recreate optimizer to include newly unfrozen parameters
        self.optimizer = self._create_optimizer()
        logger.info(f"Trainer configured for Stage {stage}")
