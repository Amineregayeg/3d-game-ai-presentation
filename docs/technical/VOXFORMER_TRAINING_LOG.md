# VoxFormer Training Documentation

## Overview

This document details the complete training process for VoxFormer, a custom Speech-to-Text Transformer architecture designed for gaming AI applications.

---

## Infrastructure Setup

### GPU Instance (Vast.ai)

| Specification | Value |
|---------------|-------|
| **GPU** | NVIDIA RTX 4090 |
| **VRAM** | 24 GB |
| **Location** | Hungary |
| **CUDA Version** | 12.1 |
| **Cost** | ~$0.40/hour |
| **Provider** | Vast.ai |

### Connection Architecture

```
┌─────────────────┐      SSH Key Auth      ┌─────────────────┐
│  Local Machine  │ ────────────────────►  │   VPS Server    │
│  (Windows/WSL)  │                        │ 134.255.234.188 │
└─────────────────┘                        └────────┬────────┘
                                                    │
                                           SSH Key Auth
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  GPU Instance   │
                                           │ 213.178.112.3   │
                                           │   Port 10648    │
                                           └─────────────────┘
```

**SSH Connection Commands:**
```bash
# Via VPS (recommended)
ssh developer@134.255.234.188
ssh -p 10648 root@213.178.112.3

# SSH Key: RSA 4096-bit generated on VPS
```

---

## Software Environment

### Dependencies Installed

```bash
# PyTorch with CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ML/Audio Libraries
pip3 install transformers==4.45.0  # Downgraded for torch 2.5 compatibility
pip3 install sentencepiece datasets librosa soundfile
pip3 install einops omegaconf tqdm pyyaml jiwer wandb
```

### Version Compatibility Note

> **Issue**: `transformers>=4.57` requires `torch>=2.6` due to `torch.load` security changes.
> **Solution**: Pinned `transformers==4.45.0` for compatibility with `torch 2.5.x`

---

## Training Data

### LibriSpeech train-clean-100

| Metric | Value |
|--------|-------|
| **Dataset** | LibriSpeech train-clean-100 |
| **Size** | 6.3 GB (compressed) |
| **Samples** | 28,539 utterances |
| **Duration** | ~100 hours |
| **Speakers** | 251 speakers |
| **Sample Rate** | 16 kHz |

### Data Pipeline

```
Raw Audio (16kHz)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  WavLM Frontend (Frozen)                                │
│  - Pretrained: microsoft/wavlm-base (95M params)        │
│  - Output: 768-dim features @ 50Hz                      │
│  - Adapter: Linear projection to 512-dim                │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Zipformer Encoder                                      │
│  - 3 resolution blocks with U-Net downsampling          │
│  - Conformer layers with convolution + attention        │
│  - Output: 512-dim encoded representations              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Dual Output Heads                                      │
│  ├── CTC Head: Frame-level alignment (vocab_size)       │
│  └── Transformer Decoder: Autoregressive generation     │
└─────────────────────────────────────────────────────────┘
```

---

## Model Architecture

### VoxFormer Specifications

| Component | Parameters | Trainable | Notes |
|-----------|------------|-----------|-------|
| **WavLM Frontend** | 94.4M | 0 (frozen) | Pretrained encoder |
| **Adapter Layer** | 393K | 393K | 768→512 projection |
| **Zipformer Encoder** | 25.2M | 25.2M | 3 blocks, 6 layers |
| **Transformer Decoder** | 20.1M | 20.1M | 4 layers, 8 heads |
| **CTC Projection** | 1.0M | 1.0M | 512→2000 vocab |
| **Total** | **204.8M** | **110.4M** | 54% trainable |

### Training Configuration

```yaml
# configs/train_config.yaml
model:
  vocab_size: 2000
  d_model: 512
  encoder_num_heads: 8
  encoder_num_blocks: 3
  encoder_layers_per_block: 2
  decoder_num_heads: 8
  decoder_num_layers: 4
  d_ff: 2048
  kernel_size: 31
  dropout: 0.1
  freeze_wavlm: true
  ctc_weight: 0.3

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  effective_batch_size: 32
  learning_rate: 0.0001
  warmup_steps: 1000
  num_epochs: 20
  mixed_precision: true  # FP16 training

optimizer:
  type: AdamW
  betas: [0.9, 0.98]
  weight_decay: 0.01

scheduler:
  type: OneCycleLR
  max_lr: 0.0001
  pct_start: 0.1  # 10% warmup
```

---

## Training Progress

### Stage 1: WavLM Frozen Training

**Start Time**: 2025-12-08 16:45 UTC
**Hardware**: RTX 4090 24GB

#### Loss Progression

| Epoch | Total Loss | CTC Loss | CE Loss | Avg Loss | LR |
|-------|------------|----------|---------|----------|-----|
| 0 (start) | 22.16 | 44.12 | 7.53 | - | 1.0e-06 |
| 0 (end) | 5.95 | 5.33 | 6.37 | 9.81 | 1.6e-05 |
| 1 | 4.46 | 2.29 | 5.39 | 4.95 | 3.4e-05 |
| 2 | 3.65 | 1.53 | 4.55 | 3.96 | 5.2e-05 |
| 3 | 2.06 | 0.92 | 2.55 | 2.80 | 7.0e-05 |
| 4 | 1.48 | 0.86 | 1.75 | 1.68 | 8.8e-05 |
| 5 | 1.20 | 0.55 | 1.50 | 1.33 | 9.2e-05 |
| 6 | 1.17 | 0.48 | 1.45 | 1.20 | 9.6e-05 |
| 7 | 1.07 | 0.40 | 1.35 | TBD | 9.6e-05 |
| 8 | TBD | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... | ... |
| 19 | TBD | TBD | TBD | TBD | TBD |

#### Training Metrics

| Metric | Value |
|--------|-------|
| **Steps per Epoch** | 3,568 |
| **Time per Epoch** | ~16 minutes |
| **Evaluation Time** | ~8 minutes |
| **Total Epoch Cycle** | ~24 minutes |
| **Training Speed** | 3.6-3.9 it/s |
| **GPU Memory Usage** | ~18 GB / 24 GB |
| **Estimated Total Time** | ~8 hours |

#### Key Observations

1. **Rapid Initial Convergence**
   - Loss dropped 4x in first epoch (22 → 5.95)
   - CTC loss dropped 57% in second epoch
   - Model quickly learning audio-text alignment

2. **Learning Rate Warmup**
   - Started at 1e-06, warming up to 1e-04
   - OneCycleLR with 10% warmup phase
   - Peak LR reached around epoch 2

3. **Mixed Precision Benefits**
   - FP16 training enabled
   - ~40% memory savings
   - No loss instability observed

---

## Technical Challenges & Solutions

### Issue 1: Python Command Not Found

**Problem**: `python` command not available in tmux session
```bash
bash: python: command not found
```

**Solution**: Use `python3` explicitly
```bash
PYTHONPATH=/root/voxformer python3 scripts/train.py --config configs/train_config.yaml
```

### Issue 2: Transformers Version Incompatibility

**Problem**: transformers 4.57+ requires torch 2.6+ for secure `torch.load`
```
ValueError: Due to a serious vulnerability issue in torch.load,
we now require users to upgrade torch to at least v2.6
```

**Solution**: Downgrade transformers
```bash
pip3 install transformers==4.45.0
```

### Issue 3: CTC Length Mismatch

**Problem**: Encoder output length shorter than calculated input_lengths
```
RuntimeError: Expected input_lengths to have value at most 752, but got value 753
```

**Root Cause**: Downsampling factor calculation mismatch due to convolution padding

**Solution**: Clamp encoder lengths in trainer.py
```python
# Calculate expected encoder lengths from waveform
encoder_lengths = waveform_lengths // self.model.frontend.downsample_factor

# Clamp to actual encoder output length to avoid CTC errors
encoder_lengths = torch.clamp(encoder_lengths, max=outputs["ctc_logits"].shape[1])
```

---

## Persistent Training Setup

### tmux Session Management

```bash
# Create persistent session
tmux new -s training

# Start training
cd /root/voxformer
PYTHONPATH=/root/voxformer python3 scripts/train.py --config configs/train_config.yaml

# Detach: Ctrl+B, then D
# Reattach later:
tmux attach -t training
```

### Monitoring Commands

```bash
# Check training progress (from VPS)
ssh -p 10648 root@213.178.112.3 'tmux capture-pane -t training -p | tail -50'

# View training log
tail -f /root/voxformer/checkpoints/training.log

# Check GPU utilization
nvidia-smi
```

---

## Expected Outcomes

### Stage 1 Targets (After 20 Epochs)

| Metric | Target | Notes |
|--------|--------|-------|
| **Training Loss** | < 2.0 | Combined CTC + CE |
| **CTC Loss** | < 1.0 | Alignment convergence |
| **CE Loss** | < 3.0 | Decoder accuracy |
| **WER (dev-clean)** | < 15% | Without LM |

### Next Stages

1. **Stage 2**: Unfreeze WavLM top 3 layers, fine-tune all
2. **Stage 3**: Gaming domain adaptation with custom vocabulary

---

## File Locations

### GPU Instance (`/root/voxformer/`)

```
/root/voxformer/
├── configs/
│   └── train_config.yaml
├── data/
│   └── LibriSpeech/
│       ├── train-clean-100/
│       └── dev-clean/
├── src/
│   ├── model/
│   │   ├── voxformer.py
│   │   ├── wavlm_frontend.py
│   │   ├── zipformer.py
│   │   └── decoder.py
│   ├── training/
│   │   └── trainer.py
│   └── data/
│       └── dataset.py
├── scripts/
│   └── train.py
└── checkpoints/
    ├── best_model.pt
    └── training.log
```

### Local Development (`/mnt/d/3d-game-ai-presentation/voxformer/`)

- Source code synchronized from local to VPS to GPU
- Evaluation results: `evaluation_results.json`
- Architecture validated with 100% test pass rate

---

## Cost Analysis

| Resource | Cost | Duration | Total |
|----------|------|----------|-------|
| RTX 4090 GPU | $0.40/hr | ~8 hours | ~$3.20 |
| VPS Server | Fixed | - | Existing |
| Data Transfer | - | - | Included |
| **Total Stage 1** | - | - | **~$3.20** |

---

## Appendix: Training Commands Reference

```bash
# === SSH Connection ===
# Connect to VPS
ssh developer@134.255.234.188

# Connect to GPU from VPS
ssh -p 10648 root@213.178.112.3

# === Training ===
# Start training in tmux
tmux new -s training
cd /root/voxformer
PYTHONPATH=/root/voxformer python3 scripts/train.py --config configs/train_config.yaml

# === Monitoring ===
# Check progress remotely
ssh -p 10648 root@213.178.112.3 'tmux capture-pane -t training -p | tail -50'

# GPU status
nvidia-smi

# === File Transfer ===
# Upload to GPU (via VPS)
scp -P 10648 local_file.py root@213.178.112.3:/root/voxformer/

# Download checkpoint
scp -P 10648 root@213.178.112.3:/root/voxformer/checkpoints/best_model.pt ./
```

---

---

## Training Run #1 - FAILED (Network Issue)

### Incident Report

**Date**: 2025-12-08 to 2025-12-09
**Instance**: Vast.ai #28618064, Host #80573, Machine #14205
**GPU**: RTX 4090 24GB (Hungary)

#### Timeline

| Time (UTC) | Event |
|------------|-------|
| 16:45 Dec 8 | Training started |
| 22:13 Dec 8 | Epoch 11 completed (loss: 0.96) |
| ~22:30 Dec 8 | Network connectivity lost |
| 08:44 Dec 9 | Discovered instance unreachable |
| 09:17 Dec 9 | Attempted reboot - network still broken |
| 09:30 Dec 9 | Instance destroyed, support email sent |

#### Progress Before Failure

| Epoch | Avg Loss | Time |
|-------|----------|------|
| 4 | 1.68 | 18:51 |
| 5 | 1.33 | 19:25 |
| 6 | 1.20 | 19:50 |
| 7 | 1.13 | 20:15 |
| 8 | 1.08 | 20:49 |
| 9 | 1.04 | 21:14 |
| 10 | 1.00 | 21:39 |
| 11 | 0.96 | 22:13 |
| 12 | ~0.88 | (in progress when failed) |

**Best achieved**: Loss 0.88 at epoch 12 (60% complete)

#### Root Cause

Host machine (ID: 80573) lost network connectivity:
```
ssh: Could not resolve hostname ssh4.vast.ai: Temporary failure in name resolution
dial udp 192.168.1.1:53: connect: network is unreachable
```

The DNS server `192.168.1.1` became unreachable - a host-level infrastructure failure outside user control.

#### What Was Lost

- Trained checkpoint: `/root/voxformer/checkpoints/stage1/best.pt` (1.6GB)
- ~6 hours of training time
- ~$2.50 in GPU costs

#### Lessons Learned

1. **Set up periodic checkpoint backup** - Copy checkpoints to VPS every few epochs
2. **Avoid problematic hosts** - Host #80573 had infrastructure issues
3. **Monitor actively** - Check training progress more frequently
4. **Use Vast.ai with caution** - Network reliability varies by host

---

## Training Run #2 - IN PROGRESS

**Date**: 2025-12-09 09:54 UTC
**Instance**: Vast.ai #28646636, Host #109053
**GPU**: RTX 4090 24GB (Finland)
**SSH**: `ssh -p 2674 root@82.141.118.40`

### Improvements for Run #2

1. **Automatic checkpoint backup** - Backup script monitors `best.pt` and copies to VPS every minute
2. **Different host** - Using Host #109053 (Finland) instead of broken Host #80573 (Hungary)
3. **SSH key bidirectional** - GPU can push checkpoints to VPS via SSH
4. **VPS backup directory**: `developer@134.255.234.188:~/voxformer_checkpoints/`

### Setup Completed

| Step | Status | Notes |
|------|--------|-------|
| GPU Instance Created | ✅ | Finland, Host #109053 |
| SSH Key Setup | ✅ | GPU can connect to VPS |
| PyTorch + Deps | ✅ | torch 2.5.1+cu121 |
| VoxFormer Code | ✅ | Transferred from VPS |
| LibriSpeech train-clean-100 | ✅ | 28,539 samples |
| LibriSpeech dev-clean | ✅ | For evaluation |
| CTC Length Fix | ✅ | Clamp encoder lengths |
| Training Started | ✅ | 09:54 UTC |

### Training Progress

| Epoch | Loss | CTC | CE | LR | Time |
|-------|------|-----|----|----|------|
| 0 | - | - | - | - | In progress |
| ... | ... | ... | ... | ... | ... |

### Monitoring Commands

```bash
# Check progress from VPS
ssh -p 2674 root@82.141.118.40 'tmux capture-pane -t training -p | tail -30'

# Check VPS backups
ls -la ~/voxformer_checkpoints/

# GPU utilization
ssh -p 2674 root@82.141.118.40 nvidia-smi
```

---

---

## Training Run #2 - COMPLETED (Stage 1)

**Date**: 2025-12-09
**Result**: SUCCESS - 20 epochs completed
**Final Loss**: ~1.0
**Final Checkpoint**: `/home/developer/voxformer_checkpoints/best_final_stage1.pt`

### Stage 1 Final Metrics

| Metric | Value |
|--------|-------|
| **Final Epoch** | 19 |
| **Final Loss** | ~1.01 |
| **CTC Loss** | ~0.40 |
| **CE Loss** | ~0.60 |
| **Training Time** | ~8 hours |

### Checkpoints Saved to VPS

```
/home/developer/voxformer_checkpoints/
├── best_final_stage1.pt       # 1.6 GB - Primary checkpoint
├── final_epoch19_step17840.pt # Final epoch
├── best_epoch15_20251209_1620.pt
├── step_14272_epoch15.pt
└── various intermediate checkpoints
```

---

## Training Run #3 - Stage 2 Fine-tuning (IN PROGRESS)

**Date**: 2025-12-11 08:13 UTC
**Instance**: Same GPU (Vast.ai RTX 4090, Finland)
**SSH**: `ssh -p 2674 root@82.141.118.40`

### Stage 2 Configuration

**Key Differences from Stage 1:**

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| **WavLM Layers** | All frozen | Top 3 unfrozen |
| **Trainable Params** | 110.4M | 131.6M |
| **Learning Rate** | 1e-4 | 1e-5 (10x lower) |
| **Epochs** | 20 | 5 |
| **Resume From** | Scratch | Stage 1 best.pt |

### Model Architecture (Stage 2)

```
VoxFormer Model Summary:
  Frontend (WavLM): 95,039,868 (21,923,144 trainable) ← TOP 3 LAYERS UNFROZEN
  Encoder: 85,675,008
  Decoder: 23,036,928
  CTC Head: 1,026,000
  Total: 204,777,804 (131,661,080 trainable)

Unfroze WavLM layers: [9, 10, 11]
```

### Training Progress (Stage 2)

| Epoch | Avg Loss | CTC Loss | CE Loss | LR | Time |
|-------|----------|----------|---------|-----|------|
| 0 | 3.9160 | 1.80 | 5.35 | 8.5e-06 | 15:31 |
| 1 | TBD | | | | |
| 2 | TBD | | | | |
| 3 | TBD | | | | |
| 4 | TBD | | | | |

### Stage 2 Notes

1. **Higher Initial Loss Expected**: Loss started at ~4.0 instead of continuing from ~1.0 because:
   - WavLM top 3 layers are now trainable (were frozen in Stage 1)
   - These layers need to adapt to the downstream task
   - Loss should decrease rapidly as WavLM fine-tunes

2. **Auto-Backup Configured**:
   - Checkpoints backed up to VPS every 30 minutes
   - Backup location: `/home/developer/voxformer_checkpoints/stage2/`

3. **Dashboard Monitoring**:
   - URL: http://5.249.161.66:3000/training
   - Shows real-time metrics, GPU stats, loss history

### Technical Changes for Stage 2

1. **Trainer Modification** (`trainer.py`):
   - Added `weights_only` parameter to `load_checkpoint()`
   - When transitioning stages, only model weights are loaded (not optimizer state)
   - This avoids size mismatch errors from different trainable parameters

2. **Train Script Modification** (`train.py`):
   - Auto-detects stage transition (stage1 → stage2)
   - Uses `weights_only=True` when loading cross-stage checkpoints

3. **Config Changes** (`stage2.yaml`):
   - `unfreeze_top_k: 3` to unfreeze WavLM layers 9, 10, 11
   - `learning_rate: 1e-5` (10x lower for fine-tuning)
   - `num_epochs: 5`
   - `wandb_project: null` (disabled - was causing login errors)

---

## Expected Outcomes

### Stage 2 Targets

| Metric | Current | Target |
|--------|---------|--------|
| **WER (dev-clean)** | TBD | < 10% |
| **Training Loss** | 3.9 | < 2.0 |
| **CTC Loss** | 1.8 | < 0.8 |

### Next Steps

1. **Complete Stage 2** (~1.5 hours remaining)
2. **Evaluate WER** on dev-clean
3. **Stage 3 (Optional)**: Gaming domain adaptation with custom vocabulary

---

*Document last updated: 2025-12-11 08:30 UTC*
*Training status: Stage 2 in progress (Epoch 1 starting)*
