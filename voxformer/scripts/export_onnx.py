#!/usr/bin/env python3
"""
ONNX Export Script

Exports VoxFormer model to ONNX format for production deployment.
Supports FP32, FP16, and INT8 quantization.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/stage2/best.pt --output exports/
    python scripts/export_onnx.py --checkpoint checkpoints/stage2/best.pt --output exports/ --fp16
    python scripts/export_onnx.py --checkpoint checkpoints/stage2/best.pt --output exports/ --int8
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import VoxFormer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VoxFormerEncoder(torch.nn.Module):
    """Encoder-only wrapper for ONNX export."""

    def __init__(self, model: VoxFormer):
        super().__init__()
        self.frontend = model.frontend
        self.encoder = model.encoder
        self.ctc_proj = model.ctc_proj

    def forward(self, waveform: torch.Tensor):
        features, feature_mask = self.frontend(waveform, None)
        encoder_output, encoder_mask = self.encoder(features, feature_mask)
        ctc_logits = self.ctc_proj(encoder_output)
        return encoder_output, encoder_mask, ctc_logits


def export_encoder(model: VoxFormer, output_path: str, opset_version: int = 17):
    """Export encoder to ONNX."""
    encoder = VoxFormerEncoder(model)
    encoder.eval()

    # Dummy input
    dummy_waveform = torch.randn(1, 16000 * 5)  # 5 seconds

    # Export
    torch.onnx.export(
        encoder,
        dummy_waveform,
        output_path,
        opset_version=opset_version,
        input_names=["waveform"],
        output_names=["encoder_output", "encoder_mask", "ctc_logits"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "audio_len"},
            "encoder_output": {0: "batch", 1: "seq_len"},
            "encoder_mask": {0: "batch", 1: "seq_len"},
            "ctc_logits": {0: "batch", 1: "seq_len"},
        },
    )
    logger.info(f"Exported encoder to {output_path}")


def convert_to_fp16(input_path: str, output_path: str):
    """Convert ONNX model to FP16."""
    try:
        from onnxconverter_common import float16
        import onnx

        model = onnx.load(input_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, output_path)
        logger.info(f"Converted to FP16: {output_path}")
    except ImportError:
        logger.error("onnxconverter-common required for FP16 conversion")
        logger.error("Install with: pip install onnxconverter-common")


def quantize_int8(input_path: str, output_path: str, calibration_data_path: str = None):
    """Quantize ONNX model to INT8."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8,
        )
        logger.info(f"Quantized to INT8: {output_path}")
    except ImportError:
        logger.error("onnxruntime required for INT8 quantization")


def main():
    parser = argparse.ArgumentParser(description="Export VoxFormer to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports",
        help="Output directory",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert to FP16",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Quantize to INT8",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = VoxFormer.from_pretrained(args.checkpoint)
    model.eval()

    # Export FP32
    fp32_path = os.path.join(args.output, "voxformer_encoder.onnx")
    export_encoder(model, fp32_path)

    # Get file size
    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    logger.info(f"FP32 model size: {fp32_size:.1f} MB")

    # Convert to FP16 if requested
    if args.fp16:
        fp16_path = os.path.join(args.output, "voxformer_encoder_fp16.onnx")
        convert_to_fp16(fp32_path, fp16_path)
        if os.path.exists(fp16_path):
            fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
            logger.info(f"FP16 model size: {fp16_size:.1f} MB")

    # Quantize to INT8 if requested
    if args.int8:
        int8_path = os.path.join(args.output, "voxformer_encoder_int8.onnx")
        quantize_int8(fp32_path, int8_path)
        if os.path.exists(int8_path):
            int8_size = os.path.getsize(int8_path) / (1024 * 1024)
            logger.info(f"INT8 model size: {int8_size:.1f} MB")

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
