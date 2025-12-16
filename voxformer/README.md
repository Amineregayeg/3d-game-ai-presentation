# VoxFormer

**Elite Speech-to-Text Transformer for Gaming AI**

VoxFormer is a high-performance ASR system designed for real-time voice command recognition in gaming applications. It combines WavLM pretrained features with a custom Zipformer encoder and Transformer decoder.

## Architecture

```
Audio → WavLM (95M frozen) → Adapter → Zipformer Encoder (25M) → Transformer Decoder (20M) → Text
```

### Key Components

- **WavLM Frontend**: Pretrained audio feature extractor with weighted layer sum
- **Zipformer Encoder**: Conformer-based encoder with U-Net downsampling (50→25→12.5 fps)
- **Transformer Decoder**: Autoregressive decoder with cross-attention and KV-cache
- **Hybrid CTC-Attention Loss**: Joint training with 0.3 CTC + 0.7 Cross-Entropy

### Model Specifications

| Metric | Target |
|--------|--------|
| Total Parameters | ~142M (47M trainable) |
| WER (LibriSpeech) | <3.5% |
| WER (Gaming) | <8% |
| Latency | <200ms streaming |
| RTF (GPU) | <0.1 |
| Training Cost | ~$20 |

## Installation

```bash
cd voxformer
pip install -e .
```

## Quick Start

### 1. Train Tokenizer

```bash
python scripts/train_tokenizer.py \
    --data /path/to/librispeech/train-clean-100 \
    --output tokenizer \
    --vocab-size 2000
```

### 2. Stage 1 Training (WavLM Frozen)

```bash
python scripts/train.py --config configs/stage1.yaml
```

### 3. Stage 2 Fine-tuning (WavLM Partial Unfreeze)

```bash
python scripts/train.py \
    --config configs/stage2.yaml \
    --resume checkpoints/stage1/best.pt
```

### 4. Stage 3 Gaming Domain Adaptation

```bash
python scripts/train.py \
    --config configs/stage3_gaming.yaml \
    --resume checkpoints/stage2/best.pt
```

### 5. Export to ONNX

```bash
python scripts/export_onnx.py \
    --checkpoint checkpoints/stage2/best.pt \
    --output exports/ \
    --fp16 --int8
```

## Project Structure

```
voxformer/
├── src/
│   ├── model/
│   │   ├── wavlm_frontend.py   # WavLM + Adapter + WeightedLayerSum
│   │   ├── conformer.py        # Conformer block with RoPE
│   │   ├── zipformer.py        # Zipformer encoder
│   │   ├── decoder.py          # Transformer decoder
│   │   └── voxformer.py        # Full model
│   ├── data/
│   │   ├── tokenizer.py        # BPE tokenizer
│   │   └── dataset.py          # ASR dataset
│   └── training/
│       ├── loss.py             # Hybrid CTC-Attention loss
│       ├── metrics.py          # WER calculation
│       └── trainer.py          # Training loop
├── configs/
│   ├── stage1.yaml             # LibriSpeech (WavLM frozen)
│   ├── stage2.yaml             # Fine-tuning (partial unfreeze)
│   └── stage3_gaming.yaml      # Gaming domain
├── scripts/
│   ├── train.py                # Training script
│   ├── train_tokenizer.py      # Tokenizer training
│   └── export_onnx.py          # ONNX export
└── tests/
    └── ...
```

## Training Schedule

| Stage | Duration | GPU Hours | Description |
|-------|----------|-----------|-------------|
| 1 | Day 4 | 30h | LibriSpeech with WavLM frozen |
| 2 | Day 5 | 5h | Unfreeze WavLM top 3 layers |
| 3 | Day 5 | 10h | Gaming domain fine-tuning |
| **Total** | | **45h** | **~$18 on A100** |

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- CUDA 12.0+ (recommended)
- 8GB+ VRAM for inference
- 24GB+ VRAM for training (or use gradient accumulation)

## License

MIT License
