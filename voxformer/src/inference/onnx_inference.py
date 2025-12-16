"""
ONNX Inference Module

Fast inference using ONNX Runtime for production deployment.
Supports CPU and GPU execution with INT8 quantization.
"""

from __future__ import annotations

import os
from typing import Optional, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False
    logger.warning("onnxruntime not installed")


class ONNXInference:
    """
    ONNX Runtime inference for VoxFormer.

    Provides fast CPU/GPU inference for production deployment.
    Supports FP32, FP16, and INT8 models.

    Args:
        encoder_path: Path to encoder ONNX model
        decoder_path: Path to decoder ONNX model (optional)
        device: Execution device ("cpu" or "cuda")
        num_threads: Number of CPU threads (default: 4)
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: Optional[str] = None,
        device: str = "cpu",
        num_threads: int = 4,
    ):
        if not HAS_ONNXRUNTIME:
            raise RuntimeError("onnxruntime is required. Install with: pip install onnxruntime-gpu")

        self.device = device
        self.num_threads = num_threads

        # Setup execution providers
        if device == "cuda":
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                }),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load encoder
        logger.info(f"Loading encoder from {encoder_path}")
        self.encoder_session = ort.InferenceSession(
            encoder_path,
            sess_options=sess_options,
            providers=providers,
        )

        # Load decoder if provided
        self.decoder_session = None
        if decoder_path and os.path.exists(decoder_path):
            logger.info(f"Loading decoder from {decoder_path}")
            self.decoder_session = ort.InferenceSession(
                decoder_path,
                sess_options=sess_options,
                providers=providers,
            )

        # Get input/output names
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_names = [o.name for o in self.encoder_session.get_outputs()]

        logger.info(f"ONNX Inference ready on {device}")

    def encode(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode audio waveform.

        Args:
            waveform: Audio array of shape (batch, num_samples)

        Returns:
            Tuple of (encoder_output, encoder_mask, ctc_logits)
        """
        # Ensure float32
        waveform = waveform.astype(np.float32)

        # Run encoder
        outputs = self.encoder_session.run(
            self.encoder_output_names,
            {self.encoder_input_name: waveform},
        )

        return tuple(outputs)

    def transcribe_ctc(
        self,
        waveform: np.ndarray,
        blank_id: int = 0,
    ) -> List[List[int]]:
        """
        Transcribe using CTC greedy decoding.

        Args:
            waveform: Audio array of shape (batch, num_samples)
            blank_id: CTC blank token ID

        Returns:
            List of token ID lists for each batch item
        """
        # Encode
        encoder_output, encoder_mask, ctc_logits = self.encode(waveform)

        # CTC greedy decode
        predictions = np.argmax(ctc_logits, axis=-1)  # (batch, seq_len)

        results = []
        for pred in predictions:
            # Remove blanks and consecutive duplicates
            tokens = []
            prev = None
            for token in pred:
                if token != blank_id and token != prev:
                    tokens.append(int(token))
                prev = token
            results.append(tokens)

        return results

    def benchmark(
        self,
        num_iterations: int = 100,
        audio_length_seconds: float = 5.0,
        sample_rate: int = 16000,
    ) -> dict:
        """
        Benchmark inference performance.

        Args:
            num_iterations: Number of iterations
            audio_length_seconds: Audio length for benchmark
            sample_rate: Sample rate

        Returns:
            Dictionary with benchmark results
        """
        import time

        # Create dummy input
        audio_samples = int(audio_length_seconds * sample_rate)
        dummy_input = np.random.randn(1, audio_samples).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.encode(dummy_input)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.encode(dummy_input)
            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times)

        results = {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "min_latency_ms": np.min(times) * 1000,
            "max_latency_ms": np.max(times) * 1000,
            "rtf": np.mean(times) / audio_length_seconds,
            "throughput_samples_per_sec": audio_samples / np.mean(times),
        }

        logger.info(f"Benchmark results:")
        logger.info(f"  Mean latency: {results['mean_latency_ms']:.2f} ms")
        logger.info(f"  RTF: {results['rtf']:.4f}")

        return results


def create_grpc_server():
    """
    Create gRPC server for VoxFormer inference.

    This is a placeholder - implement with grpcio for production.
    """
    raise NotImplementedError(
        "gRPC server implementation requires additional setup. "
        "See docs/deployment.md for instructions."
    )
