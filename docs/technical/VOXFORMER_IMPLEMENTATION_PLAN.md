# VoxFormer Elite Implementation Plan
## Custom Speech-to-Text Transformer for Game AI Assistant

**Document Version:** 1.0 | December 2025
**Status:** Production-Ready Blueprint
**Budget:** $20 (50 A100 GPU hours) + RTX 2070 local development
**Timeline:** 7 DAYS (AI-Accelerated Development)

---

## EXECUTIVE SUMMARY

### What We're Building

VoxFormer is a **custom Speech-to-Text transformer** optimized for game AI assistants. It uses WavLM-Base as a frozen feature extractor while building a fully custom Zipformer encoder and Transformer decoder from scratch.

### Key Specifications

| Specification | Target |
|---------------|--------|
| WER (LibriSpeech test-clean) | < 3.5% |
| WER (Gaming domain) | < 8% |
| Real-Time Factor (RTF) | < 0.1 |
| Streaming Latency | < 200ms |
| Model Parameters | ~140M total (~45M trainable) |
| Training Budget | 50 A100 hours ($20) |
| Local Development | RTX 2070 8GB |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW AUDIO (16kHz)                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WavLM-Base (FROZEN)                          │
│                    95M params, pretrained                       │
│                    Output: 768-dim @ 50fps                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              WEIGHTED LAYER SUM (TRAINABLE)                     │
│              Learnable weights across 12 WavLM layers           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ADAPTER (TRAINABLE)                          │
│         LN → Linear(768→512) → GELU → Dropout → Linear          │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              ZIPFORMER ENCODER (TRAINABLE)                      │
│              6 blocks, d_model=512, heads=8                     │
│              ~25M params, FROM SCRATCH                          │
│                                                                 │
│              Blocks 1-2: stride 1 (50 fps)                      │
│              Blocks 3-4: stride 2 (25 fps)                      │
│              Blocks 5-6: stride 2 (12.5 fps)                    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│      CTC HEAD            │    │  TRANSFORMER DECODER     │
│   Linear(512→vocab)      │    │  4 layers, d=512         │
│   For alignment          │    │  ~20M params             │
│   Weight: 0.3            │    │  Weight: 0.7             │
└──────────────────────────┘    └──────────────────────────┘
                    │                         │
                    └────────────┬────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSCRIBED TEXT                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## PART 1: DETAILED ARCHITECTURE SPECIFICATIONS

### 1.1 WavLM Feature Extractor (Frozen)

```python
# WavLM-Base Configuration (DO NOT MODIFY)
WAVLM_CONFIG = {
    "model": "microsoft/wavlm-base",
    "params": 94_700_000,
    "output_dim": 768,
    "num_layers": 12,
    "output_fps": 50,  # 20ms hop
    "input_sample_rate": 16000,
    "trainable": False,  # Frozen in Stage 1
    "partial_unfreeze_layers": [10, 11, 12],  # Unfreeze in Stage 2
}
```

### 1.2 Weighted Layer Sum

```python
class WeightedLayerSum(nn.Module):
    """
    Learnable weighted combination of all WavLM layers.
    Better than last-layer-only by +0.2-0.5 WER.
    """
    def __init__(self, num_layers: int = 13, dim: int = 768):
        super().__init__()
        # +1 for embedding layer
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states: List of [B, T, 768] tensors (13 total)
        Returns:
            Combined features [B, T, 768]
        """
        w = torch.softmax(self.weights, dim=0)
        x = sum(w[i] * h for i, h in enumerate(hidden_states))
        return self.layer_norm(x)
```

### 1.3 Adapter Module

```python
class WavLMAdapter(nn.Module):
    """
    Projects WavLM features to encoder dimension with regularization.
    """
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

### 1.4 Zipformer Encoder Configuration

```python
ENCODER_CONFIG = {
    "num_blocks": 6,
    "d_model": 512,
    "num_heads": 8,
    "ffn_dim": 2048,
    "conv_kernel_size": 31,
    "dropout": 0.1,
    "attention_dropout": 0.1,

    # Downsampling schedule (U-Net style)
    "strides": [1, 1, 2, 2, 2, 2],  # Blocks 1-2: 50fps, 3-4: 25fps, 5-6: 12.5fps

    # Per-block configuration
    "blocks": [
        {"id": 1, "stride": 1, "fps": 50.0},
        {"id": 2, "stride": 1, "fps": 50.0},
        {"id": 3, "stride": 2, "fps": 25.0},
        {"id": 4, "stride": 1, "fps": 25.0},
        {"id": 5, "stride": 2, "fps": 12.5},
        {"id": 6, "stride": 1, "fps": 12.5},
    ],

    # Normalization
    "norm_type": "layer_norm",  # Can upgrade to BiasNorm

    # Activation
    "activation": "swish",  # Swish/SiLU activation

    # Estimated parameters
    "estimated_params": 25_000_000,
}
```

### 1.5 Transformer Decoder Configuration

```python
DECODER_CONFIG = {
    "num_layers": 4,
    "d_model": 512,
    "num_heads": 8,
    "ffn_dim": 2048,
    "dropout": 0.1,

    # Cross-attention
    "cross_attention_every_layer": True,

    # Vocabulary
    "vocab_size": 2000,  # BPE tokens
    "tokenizer_type": "sentencepiece_bpe",

    # Special tokens
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "blank_token_id": 0,  # CTC blank = pad

    # Estimated parameters
    "estimated_params": 20_000_000,
}
```

### 1.6 Loss Configuration

```python
LOSS_CONFIG = {
    "type": "hybrid_ctc_attention",

    # Loss weights
    "ctc_weight": 0.3,
    "attention_weight": 0.7,

    # Warmup schedule (CTC slightly higher early)
    "warmup_ctc_weight": 0.4,
    "warmup_attention_weight": 0.6,
    "warmup_steps": 5000,

    # Label smoothing
    "label_smoothing": 0.1,

    # CTC configuration
    "ctc_blank_id": 0,
    "ctc_reduction": "mean",
}
```

### 1.7 Complete Model Summary

| Component | Parameters | Trainable | Memory (FP16) |
|-----------|------------|-----------|---------------|
| WavLM-Base | 95M | No (Stage 1) / Partial (Stage 2) | ~190MB |
| Weighted Layer Sum | 13 | Yes | <1KB |
| Adapter | ~790K | Yes | ~1.5MB |
| Zipformer Encoder | ~25M | Yes | ~50MB |
| Transformer Decoder | ~20M | Yes | ~40MB |
| CTC Head | ~1M | Yes | ~2MB |
| **Total** | **~142M** | **~47M trainable** | **~285MB** |

---

## PART 2: GPU HOUR ALLOCATION

### 2.1 Budget Breakdown (50 A100 Hours = $20)

| Phase | A100 Hours | Cost | Purpose |
|-------|------------|------|---------|
| Stage 1: LibriSpeech Training | 30 | $12 | Main ASR training (frozen WavLM) |
| Stage 2: Partial Unfreeze | 5 | $2 | Unfreeze WavLM top layers |
| Stage 3: Gaming Fine-tuning | 10 | $4 | Domain adaptation |
| Buffer for failures | 5 | $2 | Safety margin |
| **Total** | **50** | **$20** | |

### 2.2 Training Step Calculations

```python
# Stage 1: LibriSpeech 960h Training
STAGE1_CONFIG = {
    "dataset_hours": 960,
    "avg_utterance_seconds": 12,
    "total_utterances": 281_241,  # LibriSpeech train-960

    "batch_size_per_gpu": 16,
    "gradient_accumulation": 4,
    "effective_batch_size": 64,

    "steps_per_epoch": 281_241 // 64,  # ~4,394 steps
    "target_epochs": 1.5,
    "total_steps": 6_591,

    "estimated_time_per_step": 16,  # seconds on A100
    "total_time_hours": (6_591 * 16) / 3600,  # ~29.3 hours
}

# Stage 2: Partial WavLM Unfreeze (continued training)
STAGE2_CONFIG = {
    "continue_from": "stage1_checkpoint",
    "additional_steps": 1_500,
    "estimated_hours": 5,
}

# Stage 3: Gaming Domain Fine-tuning
STAGE3_CONFIG = {
    "dataset_hours": 50,  # Pseudo-labeled gaming audio
    "epochs": 2,
    "estimated_hours": 10,
}
```

### 2.3 Local Development (RTX 2070 - FREE)

| Task | Time Estimate | Memory Usage |
|------|---------------|--------------|
| Full codebase development | 40-60 hours | N/A |
| Unit testing all components | 10-15 hours | 2-4 GB |
| Mini training (10h subset) | 8-12 hours | 6-7 GB |
| Debugging & profiling | 10-20 hours | 4-6 GB |
| ONNX export & quantization | 5-10 hours | 4-6 GB |
| Streaming inference testing | 5-10 hours | 3-4 GB |
| **Total Local** | **80-130 hours** | **$0** |

---

## PART 3: 7-DAY AGGRESSIVE DEVELOPMENT SCHEDULE

> **AI-Accelerated Development:** With Claude as coding partner, we can parallelize and accelerate all implementation tasks. No compromises on quality.

---

### DAY 1: Foundation & Core Architecture (LOCAL - RTX 2070)

| Hour Block | Task | AI Assistance | Deliverable |
|------------|------|---------------|-------------|
| 0-2 | Environment setup | Generate requirements.txt, setup scripts | Conda env ready |
| 2-4 | Download LibriSpeech (start background) | Parallel task | Download initiated |
| 2-4 | Implement WeightedLayerSum | AI generates code | Tested module |
| 4-6 | Implement WavLMAdapter | AI generates code | Tested module |
| 6-8 | Implement ConformerBlock | AI generates code | Single block working |
| 8-10 | Implement full ZipformerEncoder | AI generates code | 6-block encoder |
| 10-12 | Implement TransformerDecoder | AI generates code | 4-layer decoder |

**Day 1 Checkpoint:**
- ✓ Full model architecture implemented
- ✓ Forward pass working on dummy input
- ✓ LibriSpeech downloading in background

---

### DAY 2: Training Infrastructure (LOCAL - RTX 2070)

| Hour Block | Task | AI Assistance | Deliverable |
|------------|------|---------------|-------------|
| 0-2 | Implement HybridCTCAttentionLoss | AI generates + debug | Loss function ready |
| 2-4 | Implement Dataset + DataLoader | AI generates code | Data pipeline |
| 4-5 | Train BPE tokenizer | Script execution | 2000-token vocab |
| 5-7 | Implement training loop | AI generates code | Trainer class |
| 7-8 | Add gradient checkpointing | AI optimizes code | Memory optimized |
| 8-10 | Implement LR scheduler + optimizer | AI generates code | AdamW + cosine |
| 10-12 | Test training on LibriSpeech-10h | Run locally | Loss decreasing ✓ |

**Day 2 Checkpoint:**
- ✓ Complete training pipeline
- ✓ Verified loss decreases on small subset
- ✓ Memory fits on RTX 2070 with batch=2

---

### DAY 3: Local Validation & Cloud Prep (LOCAL - RTX 2070)

| Hour Block | Task | AI Assistance | Deliverable |
|------------|------|---------------|-------------|
| 0-3 | Mini training run (100h subset) | Monitor training | WER < 25% |
| 3-5 | Implement WER evaluation | AI generates code | Metrics pipeline |
| 5-6 | Debug any issues found | AI debugging | Stable code |
| 6-8 | Prepare cloud training configs | AI generates YAML | A100 configs |
| 8-10 | Set up checkpoint saving to cloud | Configure S3/GCS | Backup ready |
| 10-12 | Final local validation run | Execute | Ready for A100 |

**Day 3 Checkpoint:**
- ✓ Code 100% validated locally
- ✓ Cloud configs prepared
- ✓ Ready to launch A100 training

---

### DAY 4: Cloud Training Stage 1 (A100 - 30 hours)

| Hour Block | Task | GPU Hours | Deliverable |
|------------|------|-----------|-------------|
| 0-2 | Launch A100 training | 2 | Training started |
| 2-8 | Monitor Stage 1 (steps 0-3000) | 6 | Loss < 2.5 |
| 8-16 | Continue Stage 1 (steps 3000-5000) | 8 | Loss < 2.0 |
| 16-24 | Complete Stage 1 | 8 | WER < 5% dev |

**Parallel (while A100 training):**
- Collect gaming audio from Twitch (no GPU needed)
- Pseudo-label with Whisper API or local Whisper-small
- Prepare gaming dataset

**Day 4 Checkpoint:**
- ✓ Stage 1 model trained (30 A100 hours used)
- ✓ Gaming data collected and pseudo-labeled

---

### DAY 5: Cloud Training Stage 2 & 3 (A100 - 15 hours)

| Hour Block | Task | GPU Hours | Deliverable |
|------------|------|-----------|-------------|
| 0-5 | Stage 2: Unfreeze WavLM top layers | 5 | Improved model |
| 5-15 | Stage 3: Gaming fine-tuning | 10 | Domain-adapted |

**Parallel (while A100 training):**
- Start implementing streaming inference locally
- Prepare ONNX export scripts

**Day 5 Checkpoint:**
- ✓ Stage 2 complete (WER < 3.5%)
- ✓ Stage 3 complete (Gaming WER < 10%)
- ✓ Total A100 hours: 45 (5 buffer remaining)

---

### DAY 6: Inference & Optimization (LOCAL - RTX 2070)

| Hour Block | Task | AI Assistance | Deliverable |
|------------|------|---------------|-------------|
| 0-2 | Implement streaming WavLM wrapper | AI generates code | Chunked processing |
| 2-4 | Implement streaming decoder | AI generates code | Cached inference |
| 4-6 | Build end-to-end streaming pipeline | AI integrates | Full streaming |
| 6-8 | Implement greedy + beam decoding | AI generates code | Decoding options |
| 8-10 | ONNX export | AI generates script | .onnx file |
| 10-12 | INT8 quantization | AI generates code | Quantized model |

**Day 6 Checkpoint:**
- ✓ Streaming inference working
- ✓ Latency < 200ms verified
- ✓ ONNX + INT8 models exported

---

### DAY 7: Integration, Testing & Release (LOCAL)

| Hour Block | Task | AI Assistance | Deliverable |
|------------|------|---------------|-------------|
| 0-2 | Benchmark latency (RTF, TTFT) | Run benchmarks | Performance report |
| 2-4 | Final WER evaluation | Run evaluation | Metrics documented |
| 4-6 | gRPC server implementation | AI generates code | Serving endpoint |
| 6-8 | Unity/Unreal client stubs | AI generates code | Integration ready |
| 8-10 | Documentation | AI assists | README, API docs |
| 10-12 | Code cleanup + release | Final polish | v1.0 tagged |

**Day 7 Checkpoint:**
- ✓ All metrics verified
- ✓ Deployment artifacts ready
- ✓ Documentation complete
- ✓ **VoxFormer v1.0 Released**

---

### 7-DAY SUMMARY

| Day | Focus | GPU | Key Output |
|-----|-------|-----|------------|
| 1 | Core architecture | RTX 2070 | Model forward pass |
| 2 | Training infra | RTX 2070 | Training pipeline |
| 3 | Validation + prep | RTX 2070 | Cloud-ready code |
| 4 | Stage 1 training | A100 (30h) | Base ASR model |
| 5 | Stage 2 + 3 | A100 (15h) | Final model |
| 6 | Inference | RTX 2070 | Streaming + ONNX |
| 7 | Release | RTX 2070 | v1.0 complete |

**Total A100 Hours:** 45 ($18) + 5 buffer ($2) = **$20**

---

### AI-ACCELERATED DEVELOPMENT PRINCIPLES

1. **Parallel Implementation:** AI generates multiple modules simultaneously
2. **Instant Debugging:** AI identifies and fixes issues in real-time
3. **No Wasted Iterations:** Code is validated before cloud deployment
4. **Documentation on the Fly:** AI documents as we build
5. **Quality Maintained:** Every component tested before integration

---

### CRITICAL PATH (Must Not Slip)

```
Day 1-2: Architecture + Training Pipeline (BLOCKING)
    ↓
Day 3: Local Validation (BLOCKING for cloud)
    ↓
Day 4-5: A100 Training (45 hours, continuous)
    ↓
Day 6-7: Optimization + Release
```

**Risk Mitigation:**
- Day 3 buffer absorbs Day 1-2 delays
- 5 A100 hours buffer for training issues
- Day 7 has slack for unexpected problems

---

## PART 4: TRAINING CONFIGURATIONS

### 4.1 Stage 1: LibriSpeech Training (Frozen WavLM)

```yaml
# config/stage1_librispeech.yaml

# Model
model:
  wavlm_frozen: true
  encoder_blocks: 6
  decoder_layers: 4
  d_model: 512
  num_heads: 8
  ffn_dim: 2048
  vocab_size: 2000

# Data
data:
  train_manifest: "data/librispeech/train-960.json"
  valid_manifest: "data/librispeech/dev-clean.json"
  max_audio_length: 20.0  # seconds
  min_audio_length: 0.5
  sample_rate: 16000

# Training
training:
  epochs: 2
  max_steps: 7000
  batch_size: 16
  gradient_accumulation: 4
  effective_batch_size: 64

  # Optimizer
  optimizer: "adamw"
  learning_rate: 3.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.98]
  eps: 1.0e-8

  # Scheduler
  scheduler: "cosine_with_warmup"
  warmup_steps: 500

  # Loss
  ctc_weight: 0.3
  attention_weight: 0.7
  label_smoothing: 0.1

  # Regularization
  dropout: 0.1
  gradient_clip: 1.0

  # Mixed precision
  mixed_precision: "bf16"

  # Checkpointing
  save_every_steps: 1000
  eval_every_steps: 500
  gradient_checkpointing: true

# Hardware
hardware:
  device: "cuda"
  num_workers: 8
  pin_memory: true
```

### 4.2 Stage 2: Partial WavLM Unfreeze

```yaml
# config/stage2_unfreeze.yaml

# Inherits from stage1, with modifications:
model:
  wavlm_frozen: false
  wavlm_unfreeze_layers: [10, 11, 12]  # Top 3 layers

training:
  # Load from Stage 1
  resume_from: "checkpoints/stage1_final.pt"

  # Reduced steps
  max_steps: 1500

  # Different LR for WavLM vs rest
  learning_rate_groups:
    - name: "wavlm_unfrozen"
      params: "model.wavlm.encoder.layers.10-12"
      lr: 3.0e-5  # 10x smaller
    - name: "encoder_decoder"
      params: "model.encoder,model.decoder"
      lr: 1.0e-4  # Reduced from 3e-4
    - name: "adapter"
      params: "model.adapter"
      lr: 1.0e-4

  warmup_steps: 100
```

### 4.3 Stage 3: Gaming Domain Fine-tuning

```yaml
# config/stage3_gaming.yaml

# Load from Stage 2
model:
  wavlm_frozen: false
  wavlm_unfreeze_layers: [10, 11, 12]

data:
  train_manifest: "data/gaming/train.json"
  valid_manifest: "data/gaming/dev.json"
  max_audio_length: 15.0  # Gaming commands shorter

training:
  resume_from: "checkpoints/stage2_final.pt"

  epochs: 2
  max_steps: 3000
  batch_size: 16
  gradient_accumulation: 2
  effective_batch_size: 32

  # Lower LR for fine-tuning
  learning_rate_groups:
    - name: "wavlm_unfrozen"
      lr: 1.0e-5
    - name: "encoder_decoder"
      lr: 5.0e-5

  warmup_steps: 100

  # Early stopping
  early_stopping:
    patience: 3
    metric: "wer"
    mode: "min"
```

### 4.4 RTX 2070 Local Training Config

```yaml
# config/local_rtx2070.yaml

# Aggressive memory optimization for 8GB VRAM
model:
  wavlm_frozen: true  # Must be frozen locally
  encoder_blocks: 6
  decoder_layers: 4
  d_model: 512

data:
  train_manifest: "data/librispeech/train-clean-100.json"
  max_audio_length: 15.0  # Shorter for memory

training:
  batch_size: 2  # Minimum
  gradient_accumulation: 16  # Effective: 32

  mixed_precision: "fp16"
  gradient_checkpointing: true

  # Memory optimization
  empty_cache_freq: 10  # Clear CUDA cache every 10 steps

hardware:
  device: "cuda"
  num_workers: 4
  pin_memory: false  # Save CPU memory
```

---

## PART 5: FILE STRUCTURE

```
VoxFormer/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── pyproject.toml
│
├── configs/
│   ├── stage1_librispeech.yaml
│   ├── stage2_unfreeze.yaml
│   ├── stage3_gaming.yaml
│   ├── local_rtx2070.yaml
│   └── inference.yaml
│
├── voxformer/
│   ├── __init__.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── voxformer.py          # Main model class
│   │   ├── wavlm_wrapper.py      # WavLM + WeightedLayerSum
│   │   ├── adapter.py            # WavLM → Encoder adapter
│   │   ├── encoder.py            # Zipformer encoder
│   │   ├── decoder.py            # Transformer decoder
│   │   ├── attention.py          # Multi-head attention
│   │   ├── conformer_block.py    # Conformer block
│   │   ├── convolution.py        # Convolution module
│   │   ├── feed_forward.py       # FFN with SwiGLU
│   │   └── positional.py         # Positional encodings
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # ASR Dataset class
│   │   ├── collator.py           # Batch collation
│   │   ├── tokenizer.py          # BPE tokenizer wrapper
│   │   ├── augmentation.py       # Audio augmentation
│   │   └── preprocessing.py      # Audio preprocessing
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Main training loop
│   │   ├── loss.py               # Hybrid CTC + CE loss
│   │   ├── optimizer.py          # Optimizer factory
│   │   ├── scheduler.py          # LR schedulers
│   │   └── callbacks.py          # Training callbacks
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── recognizer.py         # High-level ASR interface
│   │   ├── streaming.py          # Streaming inference
│   │   ├── decoding.py           # Greedy/beam decoding
│   │   └── vad.py                # Voice activity detection
│   │
│   ├── export/
│   │   ├── __init__.py
│   │   ├── onnx_export.py        # ONNX export
│   │   ├── quantization.py       # INT8 quantization
│   │   └── tensorrt.py           # TensorRT optimization
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── metrics.py            # WER, CER, RTF
│       ├── logging.py            # Logging utilities
│       └── checkpointing.py      # Checkpoint save/load
│
├── scripts/
│   ├── prepare_librispeech.py    # Download & prepare data
│   ├── prepare_gaming_data.py    # Gaming data pipeline
│   ├── train_tokenizer.py        # BPE tokenizer training
│   ├── train.py                  # Main training script
│   ├── evaluate.py               # Evaluation script
│   ├── export_onnx.py            # Export to ONNX
│   ├── quantize.py               # Quantization script
│   └── benchmark.py              # Latency benchmarking
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_architecture.ipynb
│   ├── 03_training_analysis.ipynb
│   ├── 04_inference_demo.ipynb
│   └── 05_streaming_demo.ipynb
│
├── tests/
│   ├── test_model.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   ├── test_loss.py
│   ├── test_streaming.py
│   └── test_export.py
│
├── deployment/
│   ├── server/
│   │   ├── grpc_server.py
│   │   ├── proto/
│   │   │   └── voxformer.proto
│   │   └── Dockerfile
│   │
│   ├── clients/
│   │   ├── python_client.py
│   │   ├── unity/
│   │   │   └── VoxFormerClient.cs
│   │   └── unreal/
│   │       └── VoxFormerClient.cpp
│   │
│   └── kubernetes/
│       ├── deployment.yaml
│       └── service.yaml
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TRAINING.md
│   ├── INFERENCE.md
│   ├── DEPLOYMENT.md
│   └── API.md
│
└── checkpoints/
    └── .gitkeep
```

---

## PART 6: RISK MITIGATION

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CTC loss explosion | Medium | High | Use warmup, gradient clipping, blank biasing |
| OOM on RTX 2070 | High | Medium | Gradient checkpointing, batch=1-2, shorter audio |
| WavLM integration issues | Low | High | Test thoroughly locally before cloud |
| Slow convergence | Medium | Medium | Curriculum learning, proper LR schedule |
| A100 job failure | Medium | High | Checkpoint every 1000 steps, 5h buffer |
| Poor gaming domain WER | Medium | Medium | More pseudo-labeled data, vocabulary biasing |

### 6.2 Checkpoint Strategy

```python
CHECKPOINT_STRATEGY = {
    "save_frequency": 1000,  # steps
    "keep_last_n": 5,
    "save_best": True,
    "best_metric": "wer",

    # Backup to cloud every N checkpoints
    "cloud_backup_frequency": 3,
    "cloud_storage": "s3://voxformer-checkpoints/",

    # Resume strategy
    "auto_resume": True,
    "resume_from_latest": True,
}
```

### 6.3 Validation Milestones

| Milestone | Expected Metric | Action if Missed |
|-----------|-----------------|------------------|
| Week 3: Loss decreasing locally | Loss < 5.0 | Debug gradient flow |
| Week 5: 100h subset WER | WER < 20% | Check data pipeline |
| Week 6: Stage 1 step 2000 | Loss < 2.0 | Continue training |
| Week 6: Stage 1 complete | WER < 5% test-clean | Consider more epochs |
| Week 8: Gaming fine-tune | WER < 12% gaming | More gaming data |
| Week 9: Streaming latency | TTFT < 300ms | Optimize chunking |
| Week 10: Quantized model | WER degradation < 0.5% | Adjust quantization |

---

## PART 7: EVALUATION METRICS

### 7.1 Target Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| WER (LibriSpeech test-clean) | < 3.5% | Standard WER calculation |
| WER (LibriSpeech test-other) | < 7.0% | Standard WER calculation |
| WER (Gaming domain) | < 8.0% | Custom gaming test set |
| RTF (GPU) | < 0.1 | inference_time / audio_duration |
| RTF (CPU) | < 0.3 | inference_time / audio_duration |
| TTFT (Time to First Token) | < 300ms | First output latency |
| EoU (End of Utterance) | < 500ms | Final transcription latency |
| Model Size (FP16) | < 300MB | Checkpoint size |
| Model Size (INT8) | < 150MB | Quantized size |

### 7.2 Evaluation Script

```python
# scripts/evaluate.py

def evaluate_model(model, test_loader, device):
    """
    Comprehensive evaluation with all metrics.
    """
    results = {
        "wer": [],
        "cer": [],
        "rtf": [],
        "ttft": [],
    }

    model.eval()

    for batch in tqdm(test_loader):
        audio = batch["audio"].to(device)
        transcripts = batch["transcript"]

        # Measure latency
        start = time.perf_counter()
        with torch.no_grad():
            predictions = model.transcribe(audio)
        end = time.perf_counter()

        # Calculate metrics
        for pred, ref, audio_len in zip(predictions, transcripts, batch["audio_length"]):
            results["wer"].append(compute_wer(pred, ref))
            results["cer"].append(compute_cer(pred, ref))
            results["rtf"].append((end - start) / audio_len)

    return {
        "wer": np.mean(results["wer"]) * 100,
        "cer": np.mean(results["cer"]) * 100,
        "rtf": np.mean(results["rtf"]),
    }
```

---

## PART 8: QUICK START COMMANDS

### 8.1 Environment Setup

```bash
# Create conda environment
conda create -n voxformer python=3.10 -y
conda activate voxformer

# Install PyTorch with CUDA
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install flash-attention (optional, for speed)
pip install flash-attn --no-build-isolation
```

### 8.2 Data Preparation

```bash
# Download LibriSpeech
python scripts/prepare_librispeech.py --output-dir data/librispeech

# Train BPE tokenizer
python scripts/train_tokenizer.py \
    --input data/librispeech/train-960.json \
    --output tokenizers/bpe_2000.model \
    --vocab-size 2000
```

### 8.3 Local Training (RTX 2070)

```bash
# Train on small subset for validation
python scripts/train.py \
    --config configs/local_rtx2070.yaml \
    --experiment-name local_debug
```

### 8.4 Cloud Training (A100)

```bash
# Stage 1: LibriSpeech training
python scripts/train.py \
    --config configs/stage1_librispeech.yaml \
    --experiment-name stage1_librispeech

# Stage 2: Unfreeze WavLM
python scripts/train.py \
    --config configs/stage2_unfreeze.yaml \
    --resume checkpoints/stage1_final.pt \
    --experiment-name stage2_unfreeze

# Stage 3: Gaming fine-tuning
python scripts/train.py \
    --config configs/stage3_gaming.yaml \
    --resume checkpoints/stage2_final.pt \
    --experiment-name stage3_gaming
```

### 8.5 Evaluation

```bash
# Evaluate on LibriSpeech
python scripts/evaluate.py \
    --checkpoint checkpoints/stage3_final.pt \
    --test-set data/librispeech/test-clean.json \
    --output results/librispeech_eval.json

# Evaluate on gaming domain
python scripts/evaluate.py \
    --checkpoint checkpoints/stage3_final.pt \
    --test-set data/gaming/test.json \
    --output results/gaming_eval.json
```

### 8.6 Export & Deployment

```bash
# Export to ONNX
python scripts/export_onnx.py \
    --checkpoint checkpoints/stage3_final.pt \
    --output exports/voxformer.onnx

# INT8 quantization
python scripts/quantize.py \
    --input exports/voxformer.onnx \
    --output exports/voxformer_int8.onnx \
    --calibration-data data/librispeech/dev-clean.json

# Benchmark
python scripts/benchmark.py \
    --model exports/voxformer_int8.onnx \
    --test-audio test_samples/
```

---

## PART 9: EXPECTED OUTCOMES

### 9.1 Performance Projections

| Stage | Dataset | Expected WER |
|-------|---------|--------------|
| Stage 1 Complete | LibriSpeech test-clean | 3.5-4.5% |
| Stage 2 Complete | LibriSpeech test-clean | 3.0-3.5% |
| Stage 3 Complete | LibriSpeech test-clean | 3.0-3.5% |
| Stage 3 Complete | Gaming domain | 6-8% |

### 9.2 Cost Summary

| Item | Cost |
|------|------|
| A100 GPU (50 hours @ $0.40/hr) | $20 |
| Cloud storage | ~$2 |
| **Total** | **~$22** |

### 9.3 Deliverables

1. **Trained Models**
   - `stage1_librispeech.pt` - Base ASR model
   - `stage2_unfreeze.pt` - Improved with WavLM fine-tuning
   - `stage3_gaming.pt` - Gaming domain optimized

2. **Deployment Artifacts**
   - `voxformer.onnx` - ONNX export (FP16)
   - `voxformer_int8.onnx` - Quantized model
   - `voxformer.trt` - TensorRT optimized

3. **Integration Code**
   - gRPC server for production
   - Unity C# client
   - Unreal C++ client

4. **Documentation**
   - Architecture documentation
   - Training guide
   - Deployment guide
   - API reference

---

## APPENDIX A: DEPENDENCIES

```
# requirements.txt

# Core
torch>=2.1.0
torchaudio>=2.1.0
transformers>=4.35.0
sentencepiece>=0.1.99

# Training
pytorch-lightning>=2.1.0
wandb>=0.16.0
tensorboard>=2.15.0

# Data
librosa>=0.10.0
soundfile>=0.12.0
datasets>=2.15.0

# Inference
onnx>=1.15.0
onnxruntime>=1.16.0
onnxruntime-gpu>=1.16.0

# Utilities
hydra-core>=1.3.0
omegaconf>=2.3.0
tqdm>=4.66.0
numpy>=1.24.0
jiwer>=3.0.0  # WER calculation

# Optional: Flash Attention
# flash-attn>=2.3.0

# Development
pytest>=7.4.0
black>=23.0.0
isort>=5.12.0
```

---

## APPENDIX B: HARDWARE REQUIREMENTS

### Minimum (Local Development)
- GPU: NVIDIA RTX 2070 8GB or equivalent
- RAM: 32GB
- Storage: 200GB SSD (for LibriSpeech)
- CPU: 8+ cores

### Recommended (Cloud Training)
- GPU: NVIDIA A100 40GB
- RAM: 64GB+
- Storage: 500GB+ SSD
- CPU: 16+ cores

### Inference (Production)
- GPU: NVIDIA RTX 3060+ (or CPU-only with INT8)
- RAM: 16GB
- Low-latency storage for model files

---

**Document Status:** Complete
**Ready for Implementation:** Yes
**Next Step:** Begin DAY 1 immediately

---

## APPENDIX C: DAY-BY-DAY DELIVERABLES CHECKLIST

### DAY 1 Checklist
- [ ] Conda environment created with all dependencies
- [ ] LibriSpeech download started (background)
- [ ] `WeightedLayerSum` class implemented and tested
- [ ] `WavLMAdapter` class implemented and tested
- [ ] `ConformerBlock` class implemented and tested
- [ ] `ZipformerEncoder` (6 blocks) implemented
- [ ] `TransformerDecoder` (4 layers) implemented
- [ ] Full `VoxFormer` model forward pass verified
- [ ] Unit tests passing for all components

### DAY 2 Checklist
- [ ] `HybridCTCAttentionLoss` implemented and tested
- [ ] `ASRDataset` class for LibriSpeech
- [ ] `Collator` for batch padding
- [ ] BPE tokenizer trained (2000 vocab)
- [ ] `Trainer` class with training loop
- [ ] Gradient checkpointing enabled
- [ ] LR scheduler (warmup + cosine)
- [ ] Training verified on 10h subset (loss decreasing)
- [ ] Memory usage < 7GB on RTX 2070

### DAY 3 Checklist
- [ ] Training on 100h subset complete
- [ ] WER evaluation pipeline working
- [ ] WER < 25% on 100h subset (validates pipeline)
- [ ] All bugs fixed
- [ ] A100 config files ready (stage1, stage2, stage3)
- [ ] Cloud checkpoint saving configured
- [ ] Final local validation passed
- [ ] Code ready for A100 (no local dependencies)

### DAY 4 Checklist
- [ ] A100 instance launched
- [ ] Stage 1 training started
- [ ] Loss < 2.5 at step 3000
- [ ] Loss < 2.0 at step 5000
- [ ] Stage 1 complete (30 hours)
- [ ] Dev WER < 5%
- [ ] Gaming audio collected (50h)
- [ ] Gaming audio pseudo-labeled

### DAY 5 Checklist
- [ ] Stage 2 training complete (5 hours)
- [ ] WavLM top layers fine-tuned
- [ ] WER < 3.5% (test-clean)
- [ ] Stage 3 gaming fine-tuning complete (10 hours)
- [ ] Gaming domain WER < 10%
- [ ] All checkpoints saved to cloud
- [ ] A100 instance terminated

### DAY 6 Checklist
- [ ] `StreamingWavLMWrapper` implemented
- [ ] `StreamingDecoder` with KV-cache
- [ ] End-to-end streaming pipeline working
- [ ] Greedy decoding implemented
- [ ] Beam search decoding implemented
- [ ] ONNX export successful
- [ ] INT8 quantization complete
- [ ] Streaming latency < 200ms verified

### DAY 7 Checklist
- [ ] RTF < 0.1 on GPU
- [ ] RTF < 0.3 on CPU
- [ ] TTFT < 300ms
- [ ] Final WER documented
- [ ] gRPC server implemented
- [ ] Unity client stub ready
- [ ] Unreal client stub ready
- [ ] README.md complete
- [ ] API documentation complete
- [ ] Code cleanup done
- [ ] v1.0 tagged and released

---

## APPENDIX D: EMERGENCY PROCEDURES

### If Training Diverges
1. Stop training immediately
2. Reduce learning rate by 50%
3. Increase warmup steps to 1000
4. Resume from last good checkpoint

### If OOM on A100
1. Reduce batch_size from 16 to 12
2. Increase gradient_accumulation from 4 to 6
3. Enable gradient checkpointing on WavLM
4. Reduce max_audio_length to 15s

### If WER Plateau
1. Increase CTC weight from 0.3 to 0.4
2. Add more SpecAugment
3. Check for data loading issues
4. Verify tokenizer coverage

### If Streaming Latency Too High
1. Reduce chunk size from 200ms to 160ms
2. Use greedy decoding (no beam search)
3. Quantize to INT8
4. Reduce context window

---

## APPENDIX E: SUCCESS CRITERIA

### Minimum Viable Product (MVP)
| Metric | Threshold | Status |
|--------|-----------|--------|
| WER LibriSpeech test-clean | < 5% | Required |
| WER Gaming domain | < 15% | Required |
| RTF (GPU) | < 0.2 | Required |
| Streaming latency | < 500ms | Required |

### Target (Full Success)
| Metric | Threshold | Status |
|--------|-----------|--------|
| WER LibriSpeech test-clean | < 3.5% | Target |
| WER Gaming domain | < 8% | Target |
| RTF (GPU) | < 0.1 | Target |
| Streaming latency | < 200ms | Target |
| Model size (INT8) | < 150MB | Target |

### Stretch Goals
| Metric | Threshold | Status |
|--------|-----------|--------|
| WER LibriSpeech test-clean | < 3.0% | Stretch |
| WER Gaming domain | < 6% | Stretch |
| CPU real-time inference | RTF < 0.3 | Stretch |
| Mobile deployment ready | ONNX-mobile | Stretch |

---

*This plan was synthesized from the VoxFormer Elite Guide (December 2025) and Perplexity research on WavLM integration best practices.*

**LET'S BUILD THIS.**
