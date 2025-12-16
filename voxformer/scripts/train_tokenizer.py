#!/usr/bin/env python3
"""
Tokenizer Training Script

Trains a BPE tokenizer on LibriSpeech transcriptions.

Usage:
    python scripts/train_tokenizer.py --data /data/librispeech/train-clean-100 --output tokenizer
"""

import argparse
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import BPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def collect_librispeech_transcripts(data_dir: str) -> list:
    """Collect all transcriptions from LibriSpeech directory."""
    texts = []
    root = Path(data_dir)

    for trans_file in root.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    texts.append(parts[1].lower())

    logger.info(f"Collected {len(texts)} transcriptions")
    return texts


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to LibriSpeech data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer",
        help="Output directory for tokenizer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2000,
        help="Vocabulary size",
    )
    args = parser.parse_args()

    # Collect transcripts
    logger.info(f"Collecting transcripts from {args.data}")
    texts = collect_librispeech_transcripts(args.data)

    if not texts:
        logger.error("No transcripts found!")
        return

    # Train tokenizer
    logger.info(f"Training tokenizer with vocab_size={args.vocab_size}")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(texts, args.output)

    # Save config
    tokenizer.save_config(os.path.join(args.output, "config.json"))

    # Test tokenizer
    test_text = "hello world this is a test"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    logger.info(f"Test: '{test_text}' -> {encoded} -> '{decoded}'")

    logger.info(f"Tokenizer saved to {args.output}")


if __name__ == "__main__":
    main()
