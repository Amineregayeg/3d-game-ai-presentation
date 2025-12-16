#!/usr/bin/env python3
"""
VoxFormer Training Script

Usage:
    python scripts/train.py --config configs/stage1.yaml
    python scripts/train.py --config configs/stage2.yaml --resume checkpoints/stage1/best.pt
    python scripts/train.py --config configs/stage3_gaming.yaml --resume checkpoints/stage2/best.pt

Arguments:
    --config: Path to YAML configuration file
    --resume: Path to checkpoint to resume from (optional)
    --stage: Training stage override (1, 2, or 3)
    --device: Device to train on (cuda, cpu)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import VoxFormer
from src.data import BPETokenizer, ASRDataset, ASRCollator
from src.data.dataset import create_librispeech_dataloader, create_manifest_dataloader
from src.training import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def setup_tokenizer(config: OmegaConf) -> BPETokenizer:
    """Setup or train tokenizer."""
    tokenizer_path = config.tokenizer.model_path

    if os.path.exists(tokenizer_path):
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.from_pretrained(Path(tokenizer_path).parent)
    else:
        logger.info("Tokenizer not found. You need to train it first.")
        logger.info("Run: python scripts/train_tokenizer.py --config <config>")
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    return tokenizer


def setup_dataloaders(config: OmegaConf, tokenizer: BPETokenizer):
    """Create train and eval dataloaders."""
    data_config = config.data
    training_config = config.training

    # Create dataloaders based on format
    if data_config.data_format == "librispeech":
        train_loader = create_librispeech_dataloader(
            data_dir=data_config.train_data,
            tokenizer=tokenizer,
            batch_size=training_config.batch_size,
            num_workers=data_config.num_workers,
            shuffle=True,
            max_audio_len=data_config.max_audio_len,
            max_text_len=data_config.max_text_len,
        )
        eval_loader = create_librispeech_dataloader(
            data_dir=data_config.eval_data,
            tokenizer=tokenizer,
            batch_size=training_config.batch_size,
            num_workers=data_config.num_workers,
            shuffle=False,
            max_audio_len=data_config.max_audio_len,
            max_text_len=data_config.max_text_len,
        )
    else:
        train_loader = create_manifest_dataloader(
            manifest_path=data_config.train_data,
            tokenizer=tokenizer,
            batch_size=training_config.batch_size,
            num_workers=data_config.num_workers,
            shuffle=True,
            max_audio_len=data_config.max_audio_len,
            max_text_len=data_config.max_text_len,
        )
        eval_loader = create_manifest_dataloader(
            manifest_path=data_config.eval_data,
            tokenizer=tokenizer,
            batch_size=training_config.batch_size,
            num_workers=data_config.num_workers,
            shuffle=False,
            max_audio_len=data_config.max_audio_len,
            max_text_len=data_config.max_text_len,
        )

    return train_loader, eval_loader


def setup_model(config: OmegaConf) -> VoxFormer:
    """Create VoxFormer model from config."""
    model_config = config.model

    model = VoxFormer(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        encoder_num_heads=model_config.encoder_num_heads,
        encoder_num_blocks=model_config.encoder_num_blocks,
        encoder_layers_per_block=model_config.encoder_layers_per_block,
        decoder_num_heads=model_config.decoder_num_heads,
        decoder_num_layers=model_config.decoder_num_layers,
        d_ff=model_config.d_ff,
        kernel_size=model_config.kernel_size,
        dropout=model_config.dropout,
        wavlm_model_name=model_config.wavlm_model_name,
        freeze_wavlm=model_config.freeze_wavlm,
        unfreeze_top_k=model_config.unfreeze_top_k,
        ctc_weight=model_config.ctc_weight,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Train VoxFormer model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Training stage override (1, 2, or 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Config: {OmegaConf.to_yaml(config)}")

    # Setup device
    device = args.device
    logger.info(f"Using device: {device}")

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    # Setup tokenizer
    tokenizer = setup_tokenizer(config)

    # Setup dataloaders
    train_loader, eval_loader = setup_dataloaders(config, tokenizer)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Eval samples: {len(eval_loader.dataset)}")

    # Setup model
    model = setup_model(config)

    # Override stage if specified
    if args.stage:
        model.set_training_stage(args.stage)

    # Setup trainer
    training_config = config.training
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        warmup_steps=training_config.warmup_steps,
        num_epochs=training_config.num_epochs,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        mixed_precision=training_config.mixed_precision,
        checkpoint_dir=training_config.checkpoint_dir,
        log_interval=training_config.log_interval,
        eval_interval=training_config.eval_interval,
        save_interval=training_config.save_interval,
        device=device,
        wandb_project=training_config.get("wandb_project"),
        wandb_run_name=training_config.get("wandb_run_name"),
    )

    # Resume from checkpoint if specified
    resume_path = args.resume or training_config.get("resume_from")
    if resume_path and os.path.exists(resume_path):
        trainer.load_checkpoint(resume_path)

    # Train
    trainer.train()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
