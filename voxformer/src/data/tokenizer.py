"""
BPE Tokenizer for VoxFormer

Implements a Byte Pair Encoding tokenizer using the SentencePiece library.
BPE provides a good balance between vocabulary size and sequence length.

Vocabulary:
- Size: 2000 tokens (configurable)
- Special tokens: <pad>, <bos>, <eos>, <unk>
- Subword units learned from training data

Reference: "Neural Machine Translation of Rare Words with Subword Units"
           https://arxiv.org/abs/1508.07909
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Try to import sentencepiece, with fallback for systems without it
try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not installed. Using fallback tokenizer.")


class BPETokenizer:
    """
    BPE Tokenizer using SentencePiece.

    Special tokens:
    - <pad> (ID: 0): Padding token
    - <bos> (ID: 1): Beginning of sequence
    - <eos> (ID: 2): End of sequence
    - <unk> (ID: 3): Unknown token

    Args:
        vocab_size: Target vocabulary size (default: 2000)
        model_path: Path to pretrained SentencePiece model
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(
        self,
        vocab_size: int = 2000,
        model_path: Optional[str] = None,
    ):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self._sp = None

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def is_trained(self) -> bool:
        """Check if tokenizer is trained/loaded."""
        return self._sp is not None

    def train(
        self,
        texts: List[str],
        output_dir: str,
        model_prefix: str = "tokenizer",
    ):
        """
        Train BPE tokenizer on text corpus.

        Args:
            texts: List of training texts
            output_dir: Directory to save model
            model_prefix: Prefix for model files
        """
        if not HAS_SENTENCEPIECE:
            raise RuntimeError("sentencepiece is required for training. Install with: pip install sentencepiece")

        os.makedirs(output_dir, exist_ok=True)

        # Write texts to temporary file
        text_file = os.path.join(output_dir, "train_text.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text.strip() + "\n")

        # Train SentencePiece model
        model_prefix_path = os.path.join(output_dir, model_prefix)

        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix_path,
            vocab_size=self.vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            num_threads=os.cpu_count(),
            # Special tokens
            pad_id=self.PAD_ID,
            bos_id=self.BOS_ID,
            eos_id=self.EOS_ID,
            unk_id=self.UNK_ID,
            pad_piece=self.PAD_TOKEN,
            bos_piece=self.BOS_TOKEN,
            eos_piece=self.EOS_TOKEN,
            unk_piece=self.UNK_TOKEN,
            # Training parameters
            max_sentence_length=16384,
            shuffle_input_sentence=True,
        )

        # Load trained model
        self.model_path = model_prefix_path + ".model"
        self.load(self.model_path)

        # Clean up temp file
        os.remove(text_file)

        logger.info(f"Trained BPE tokenizer with vocab_size={self.vocab_size}")
        logger.info(f"Model saved to: {self.model_path}")

    def load(self, model_path: str):
        """
        Load pretrained SentencePiece model.

        Args:
            model_path: Path to .model file
        """
        if not HAS_SENTENCEPIECE:
            raise RuntimeError("sentencepiece is required. Install with: pip install sentencepiece")

        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)
        self.model_path = model_path
        self.vocab_size = self._sp.get_piece_size()

        logger.info(f"Loaded tokenizer from {model_path}, vocab_size={self.vocab_size}")

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_bos: Whether to prepend BOS token
            add_eos: Whether to append EOS token

        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer not trained/loaded")

        ids = self._sp.encode(text)

        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer not trained/loaded")

        if skip_special_tokens:
            ids = [i for i in ids if i not in [self.PAD_ID, self.BOS_ID, self.EOS_ID]]

        return self._sp.decode(ids)

    def batch_encode(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[List[int]]:
        """
        Encode multiple texts.

        Args:
            texts: List of input texts
            add_bos: Whether to prepend BOS token
            add_eos: Whether to append EOS token

        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_bos, add_eos) for text in texts]

    def batch_decode(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode multiple token ID sequences.

        Args:
            batch_ids: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded texts
        """
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    def get_vocab(self) -> dict:
        """Get vocabulary as dict mapping token to ID."""
        if not self.is_trained:
            raise RuntimeError("Tokenizer not trained/loaded")

        return {self._sp.id_to_piece(i): i for i in range(self.vocab_size)}

    def save_config(self, path: str):
        """Save tokenizer configuration."""
        config = {
            "vocab_size": self.vocab_size,
            "model_path": self.model_path,
            "pad_token": self.PAD_TOKEN,
            "bos_token": self.BOS_TOKEN,
            "eos_token": self.EOS_TOKEN,
            "unk_token": self.UNK_TOKEN,
            "pad_id": self.PAD_ID,
            "bos_id": self.BOS_ID,
            "eos_id": self.EOS_ID,
            "unk_id": self.UNK_ID,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "BPETokenizer":
        """
        Load tokenizer from directory.

        Args:
            model_dir: Directory containing tokenizer.model and config.json

        Returns:
            Loaded BPETokenizer
        """
        model_path = os.path.join(model_dir, "tokenizer.model")
        config_path = os.path.join(model_dir, "config.json")

        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            vocab_size = config.get("vocab_size", 2000)
        else:
            vocab_size = 2000

        tokenizer = cls(vocab_size=vocab_size)
        tokenizer.load(model_path)

        return tokenizer

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID

    @property
    def bos_token_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID

    @property
    def unk_token_id(self) -> int:
        return self.UNK_ID
