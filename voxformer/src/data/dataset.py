"""
ASR Dataset Module

Implements PyTorch datasets for audio-text pairs:
- ASRDataset: Loads audio and transcriptions
- ASRCollator: Collates samples into batched tensors

Supports:
- LibriSpeech format
- Common Voice format
- Custom audio-transcript pairs
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Callable
import logging

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from src.data.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


class ASRDataset(Dataset):
    """
    Dataset for audio-transcription pairs.

    Supports multiple data formats:
    - LibriSpeech: .flac files with .trans.txt transcriptions
    - Manifest: JSON/JSONL files with audio_path and text fields
    - Custom: List of (audio_path, transcription) tuples

    Args:
        manifest_path: Path to manifest file (JSON/JSONL) or directory
        tokenizer: BPETokenizer for text encoding
        sample_rate: Target sample rate (default: 16000)
        max_audio_len: Maximum audio length in seconds (default: 30)
        max_text_len: Maximum text length in tokens (default: 256)
        data_format: Data format ("manifest", "librispeech", "custom")
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer: BPETokenizer,
        sample_rate: int = 16000,
        max_audio_len: float = 30.0,
        max_text_len: int = 256,
        data_format: str = "manifest",
    ):
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_audio_samples = int(max_audio_len * sample_rate)
        self.max_text_len = max_text_len
        self.data_format = data_format

        # Load samples
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def _load_samples(self) -> List[Dict]:
        """Load samples based on data format."""
        if self.data_format == "manifest":
            return self._load_manifest()
        elif self.data_format == "librispeech":
            return self._load_librispeech()
        else:
            raise ValueError(f"Unknown data format: {self.data_format}")

    def _load_manifest(self) -> List[Dict]:
        """Load samples from manifest file."""
        samples = []

        path = Path(self.manifest_path)

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = data.get("samples", [])

        elif path.suffix in [".jsonl", ".manifest"]:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

        else:
            raise ValueError(f"Unknown manifest format: {path.suffix}")

        # Validate samples
        validated = []
        for s in samples:
            if "audio_path" in s and "text" in s:
                validated.append(s)
            elif "audio" in s and "text" in s:
                validated.append({"audio_path": s["audio"], "text": s["text"]})

        return validated

    def _load_librispeech(self) -> List[Dict]:
        """Load samples from LibriSpeech directory structure."""
        samples = []
        root = Path(self.manifest_path)

        # Find all .trans.txt files
        for trans_file in root.rglob("*.trans.txt"):
            audio_dir = trans_file.parent

            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        audio_id, text = parts

                        # Find audio file (could be .flac or .wav)
                        audio_path = None
                        for ext in [".flac", ".wav"]:
                            candidate = audio_dir / f"{audio_id}{ext}"
                            if candidate.exists():
                                audio_path = str(candidate)
                                break

                        if audio_path:
                            samples.append({
                                "audio_path": audio_path,
                                "text": text.lower(),
                                "audio_id": audio_id,
                            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - waveform: Audio tensor (num_samples,)
            - waveform_length: Audio length
            - input_ids: Token IDs with BOS (seq_len,)
            - target_ids: Token IDs without BOS for loss (seq_len,)
            - text: Original text string
        """
        sample = self.samples[idx]
        audio_path = sample["audio_path"]
        text = sample["text"]

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Remove channel dimension
        waveform = waveform.squeeze(0)

        # Truncate if too long
        if waveform.shape[0] > self.max_audio_samples:
            waveform = waveform[:self.max_audio_samples]

        # Tokenize text
        input_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)

        # Truncate if too long
        if len(input_ids) > self.max_text_len:
            input_ids = input_ids[:self.max_text_len - 1] + [self.tokenizer.EOS_ID]

        return {
            "waveform": waveform,
            "waveform_length": waveform.shape[0],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "text": text,
        }


class ASRCollator:
    """
    Collator for batching ASR samples.

    Handles variable-length audio and text sequences with padding.

    Args:
        pad_token_id: Padding token ID for text (default: 0)
        sample_rate: Sample rate for audio (default: 16000)
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        sample_rate: int = 16000,
    ):
        self.pad_token_id = pad_token_id
        self.sample_rate = sample_rate

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched dictionary with padded tensors
        """
        # Get max lengths
        max_audio_len = max(s["waveform"].shape[0] for s in batch)
        max_text_len = max(s["input_ids"].shape[0] for s in batch)

        batch_size = len(batch)

        # Initialize padded tensors
        waveforms = torch.zeros(batch_size, max_audio_len)
        waveform_lengths = torch.zeros(batch_size, dtype=torch.long)
        input_ids = torch.full((batch_size, max_text_len), self.pad_token_id, dtype=torch.long)
        target_lengths = torch.zeros(batch_size, dtype=torch.long)
        texts = []

        # Fill tensors
        for i, sample in enumerate(batch):
            wav_len = sample["waveform"].shape[0]
            text_len = sample["input_ids"].shape[0]

            waveforms[i, :wav_len] = sample["waveform"]
            waveform_lengths[i] = wav_len
            input_ids[i, :text_len] = sample["input_ids"]
            target_lengths[i] = text_len - 1  # Exclude BOS for target length
            texts.append(sample["text"])

        return {
            "waveform": waveforms,
            "waveform_lengths": waveform_lengths,
            "input_ids": input_ids,
            "target_lengths": target_lengths,
            "texts": texts,
        }


def create_librispeech_dataloader(
    data_dir: str,
    tokenizer: BPETokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create DataLoader for LibriSpeech dataset.

    Args:
        data_dir: Path to LibriSpeech split (e.g., train-clean-100)
        tokenizer: BPETokenizer instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional ASRDataset arguments

    Returns:
        PyTorch DataLoader
    """
    dataset = ASRDataset(
        manifest_path=data_dir,
        tokenizer=tokenizer,
        data_format="librispeech",
        **dataset_kwargs,
    )

    collator = ASRCollator(pad_token_id=tokenizer.PAD_ID)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )


def create_manifest_dataloader(
    manifest_path: str,
    tokenizer: BPETokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create DataLoader from manifest file.

    Args:
        manifest_path: Path to JSON/JSONL manifest
        tokenizer: BPETokenizer instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional ASRDataset arguments

    Returns:
        PyTorch DataLoader
    """
    dataset = ASRDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        data_format="manifest",
        **dataset_kwargs,
    )

    collator = ASRCollator(pad_token_id=tokenizer.PAD_ID)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
