"""
Streaming ASR Inference

Real-time speech recognition with chunked processing.
Supports <200ms latency for gaming voice commands.

Features:
- Chunked audio processing (160-200ms chunks)
- Left context buffer (0.5-1.0s)
- KV-cache for decoder
- Greedy decoding for real-time
- Beam search for final refinement
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from collections import deque
import logging

from src.model import VoxFormer
from src.data import BPETokenizer

logger = logging.getLogger(__name__)


class StreamingASR:
    """
    Streaming ASR wrapper for real-time inference.

    Processes audio in chunks while maintaining state across chunks
    for continuous speech recognition.

    Args:
        model: VoxFormer model
        tokenizer: BPETokenizer
        chunk_size_ms: Chunk size in milliseconds (default: 160)
        left_context_ms: Left context in milliseconds (default: 500)
        sample_rate: Audio sample rate (default: 16000)
        device: Inference device (default: "cuda")
    """

    def __init__(
        self,
        model: VoxFormer,
        tokenizer: BPETokenizer,
        chunk_size_ms: int = 160,
        left_context_ms: int = 500,
        sample_rate: int = 16000,
        device: str = "cuda",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.device = device

        # Chunk parameters
        self.chunk_size_samples = int(chunk_size_ms * sample_rate / 1000)
        self.left_context_samples = int(left_context_ms * sample_rate / 1000)

        # State
        self.audio_buffer = deque(maxlen=self.left_context_samples + self.chunk_size_samples)
        self.encoder_cache = None
        self.decoder_cache = None
        self.generated_tokens = []

        logger.info(
            f"StreamingASR initialized: chunk={chunk_size_ms}ms, context={left_context_ms}ms"
        )

    def reset(self):
        """Reset streaming state for new utterance."""
        self.audio_buffer.clear()
        self.encoder_cache = None
        self.decoder_cache = None
        self.generated_tokens = []

    @torch.no_grad()
    def process_chunk(self, audio_chunk: torch.Tensor) -> str:
        """
        Process a chunk of audio and return transcription.

        Args:
            audio_chunk: Audio samples (num_samples,)

        Returns:
            Transcribed text for this chunk
        """
        # Add chunk to buffer
        self.audio_buffer.extend(audio_chunk.tolist())

        # Check if we have enough samples
        if len(self.audio_buffer) < self.chunk_size_samples:
            return ""

        # Prepare input with left context
        audio_with_context = torch.tensor(
            list(self.audio_buffer), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # Forward through model
        generated, self.encoder_cache = self.model.transcribe_streaming(
            chunk=audio_with_context,
            cache=self.encoder_cache,
            max_new_tokens=16,
        )

        # Decode new tokens
        new_tokens = generated[0].tolist()

        # Filter out special tokens and duplicates
        filtered_tokens = []
        for tok in new_tokens:
            if tok not in [
                self.tokenizer.PAD_ID,
                self.tokenizer.BOS_ID,
                self.tokenizer.EOS_ID,
            ]:
                if not self.generated_tokens or tok != self.generated_tokens[-1]:
                    filtered_tokens.append(tok)

        self.generated_tokens.extend(filtered_tokens)

        # Decode to text
        text = self.tokenizer.decode(filtered_tokens, skip_special_tokens=True)

        return text

    @torch.no_grad()
    def finalize(self, beam_size: int = 4) -> str:
        """
        Finalize transcription with beam search.

        Called at end of utterance for higher quality.

        Args:
            beam_size: Beam size for final decoding

        Returns:
            Final transcribed text
        """
        if not self.audio_buffer:
            return ""

        # Process remaining audio with beam search
        audio = torch.tensor(
            list(self.audio_buffer), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        generated = self.model.transcribe(
            waveform=audio,
            max_length=256,
            beam_size=beam_size,
        )

        text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

        return text

    def transcribe_file(self, audio_path: str, beam_size: int = 4) -> str:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            beam_size: Beam size for decoding

        Returns:
            Transcribed text
        """
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0).to(self.device)

        # Full transcription (not streaming)
        generated = self.model.transcribe(
            waveform=waveform.unsqueeze(0),
            max_length=256,
            beam_size=beam_size,
        )

        text = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

        return text


class StreamingASRSession:
    """
    Manages a streaming ASR session with audio buffering.

    Useful for handling audio from microphone or network stream.
    """

    def __init__(self, asr: StreamingASR):
        self.asr = asr
        self.is_active = False
        self.accumulated_text = []

    def start(self):
        """Start new session."""
        self.asr.reset()
        self.is_active = True
        self.accumulated_text = []

    def add_audio(self, audio: torch.Tensor) -> str:
        """Add audio and get incremental transcription."""
        if not self.is_active:
            return ""

        text = self.asr.process_chunk(audio)
        if text:
            self.accumulated_text.append(text)

        return text

    def end(self) -> str:
        """End session and get final transcription."""
        if not self.is_active:
            return ""

        final_text = self.asr.finalize()
        self.is_active = False

        return final_text

    @property
    def current_text(self) -> str:
        """Get current accumulated text."""
        return " ".join(self.accumulated_text)
