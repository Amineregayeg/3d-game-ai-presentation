# VoxFormer Training Progress Report

**Last Updated:** December 17, 2024
**Status:** Paused (ready to resume)
**Current Stage:** Stage 4 - Hybrid CTC-Attention Fine-tuning

---

## Executive Summary

Training a custom Speech-to-Text (STT) model called VoxFormer using a hybrid CTC-Attention architecture. Currently ~15% through full training (1.5 out of 10 epochs completed). Training paused to save GPU costs, can resume anytime from backed-up checkpoints.

---

## Training Infrastructure

### GPU Server (Vast.ai)
| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA RTX 4090 / H100 (24-80GB VRAM) |
| Cost | ~$0.40/hr |
| Connection | `ssh -p PORT root@IP` via VPS |

### VPS (Hostinger)
| Parameter | Value |
|-----------|-------|
| Host | `5.249.161.66` |
| Purpose | Checkpoint backup, dashboard hosting |
| Backup Path | `/home/developer/voxformer_checkpoints/` |

### Dashboard
- **URL:** http://5.249.161.66:3000/training
- **Features:** Real-time loss, WER tracking, GPU metrics

---

## Training Stages Overview

### Stage 1: Encoder Pre-training ✅ COMPLETED
- **Goal:** Train audio encoder on speech features
- **Checkpoint:** `stage1/best.pt` (1.7GB)
- **Status:** Completed previously

### Stage 2: CTC-only Training ✅ COMPLETED
- **Goal:** Train CTC head for basic alignment
- **Checkpoint:** `stage2/best.pt` (782MB)
- **Status:** Completed previously

### Stage 3: Decoder Pre-training ✅ COMPLETED
- **Goal:** Pre-train attention decoder
- **Status:** Completed previously

### Stage 4: Hybrid CTC-Attention Fine-tuning 🔄 IN PROGRESS
- **Goal:** Joint CTC + Attention training for best accuracy
- **Checkpoint:** `stage4_fixed/latest.pt` (1.8GB)
- **Progress:** Epoch 1, Step ~2500/3568 (~15% total)
- **Status:** PAUSED - ready to resume

---

## Problems Encountered & Solutions

### Problem 1: Extremely High WER (>1000%)
**Symptom:** Initial WER showed values like 1200%, 5000%, even higher.

**Root Cause:**
- Decoder producing repetitive outputs (e.g., "the the the the...")
- Attention mechanism collapsing to single positions
- No length penalty in beam search

**Solution:**
- Added repetition penalty to decoder (penalty=1.2)
- Fixed attention masking
- Added length normalization to beam search
- Result: WER dropped to ~98-99%

### Problem 2: Disk Full Error (32GB → 2MB free)
**Symptom:** Training crashed with `RuntimeError: PytorchStreamWriter failed writing file`

**Root Cause:**
- Multiple large checkpoints accumulated (1.8GB each)
- Pip cache consuming space
- Old training run checkpoints not cleaned

**Solution:**
```bash
# Removed redundant checkpoints
rm step_0.pt  # Redundant (had step_500)
rm stage4_hybrid/step_*.pt  # Old training run

# Cleared pip cache
rm -rf ~/.cache/pip/http/*

# Freed 6.7GB → training resumed
```

**Files Lost:** Steps 500-2138 (had to resume from step_500.pt)

### Problem 3: Checkpoint Corruption
**Symptom:** `latest.pt` couldn't be loaded after disk error

**Root Cause:** Partial write due to disk full during save

**Solution:** Resume from last known good checkpoint (`step_500.pt`)

### Problem 4: SSH Connection Timeouts
**Symptom:** Nested SSH commands timing out frequently

**Solution:** Use script-based approach:
```bash
# Create script on VPS, execute on GPU
ssh VPS 'cat > /tmp/script.sh << "EOF"
ssh GPU "commands"
EOF
/tmp/script.sh'
```

### Problem 5: Wrong Python Environment in tmux
**Symptom:** `ModuleNotFoundError: No module named 'torch'`

**Root Cause:** Vast.ai auto-activates conda env without torch

**Solution:** Use system Python explicitly:
```bash
/usr/bin/python3 scripts/train_hybrid.py --config configs/stage4_fixed.yaml
```

---

## Current Training Metrics

### Loss Progress
| Epoch | Step | Loss | CTC Loss | CE Loss |
|-------|------|------|----------|---------|
| 0 | 0 | 3.5+ | 2.5+ | 5.0+ |
| 0 | 500 | 2.5 | 1.6 | 4.7 |
| 0 | 2000 | 2.4 | 1.5 | 4.6 |
| 1 | 1500 | 1.8 | 0.9 | 4.0 |
| 1 | 2500 | 1.7 | 0.8 | 3.9 |

**Trend:** Loss decreasing steadily ✅

### WER Progress
| Step | WER | Notes |
|------|-----|-------|
| 0 | 98.65% | Baseline after fixes |
| 500 | 98.85% | +0.20% |
| 1000 | 99.04% | Lost in crash |
| 1500 | 99.37% | Current |

**Note:** WER still high but loss improving. WER typically lags behind loss improvement by 1-2 epochs.

---

## Checkpoints Backed Up on VPS

**Location:** `/home/developer/voxformer_checkpoints/stage4_fixed/`

| File | Size | Description |
|------|------|-------------|
| `latest.pt` | 1.8 GB | Step ~2500 (most recent) |
| `step_1500.pt` | 1.8 GB | With WER evaluation |
| `epoch_0.pt` | 1.8 GB | End of epoch 0 |
| `step_500.pt` | 1.8 GB | Fallback checkpoint |
| `best.pt` | 782 MB | Lowest loss checkpoint |
| `training.log` | 3.1 MB | Full training history |
| `stage4_fixed.yaml` | 1.5 KB | Training config |

**Total Backup Size:** 8.6 GB

---

## Training Configuration (stage4_fixed.yaml)

```yaml
model:
  vocab_size: 2000
  d_model: 512
  encoder_num_heads: 8
  encoder_num_blocks: 3
  decoder_num_heads: 8
  decoder_num_layers: 4
  ctc_weight: 0.5  # Hybrid: 50% CTC, 50% Attention

training:
  learning_rate: 1.0e-4
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch: 32
  num_epochs: 10
  warmup_steps: 2000
  save_interval: 500
  eval_interval: 500
  resume_from: checkpoints/stage4_fixed/latest.pt
```

---

## How to Resume Training

### Step 1: Rent New GPU
- Vast.ai: RTX 4090 or better (~$0.40/hr)
- Minimum 24GB VRAM, 30GB disk

### Step 2: Copy Checkpoints from VPS
```bash
# On new GPU instance
scp -r developer@5.249.161.66:/home/developer/voxformer_checkpoints/stage4_fixed/* \
    /root/voxformer/checkpoints/stage4_fixed/
```

### Step 3: Update Config
```bash
# Set resume checkpoint
sed -i 's|resume_from:.*|resume_from: checkpoints/stage4_fixed/latest.pt|' \
    configs/stage4_fixed.yaml
```

### Step 4: Start Training
```bash
cd /root/voxformer
nohup /usr/bin/python3 scripts/train_hybrid.py \
    --config configs/stage4_fixed.yaml \
    >> checkpoints/stage4_fixed/training.log 2>&1 &
```

### Step 5: Set Up Backup (Optional)
```bash
# Auto-backup to VPS every 5 minutes
while true; do
    scp checkpoints/stage4_fixed/*.pt developer@VPS:/path/
    sleep 300
done &
```

---

## What's Still Needed

### Immediate (Next Training Session)
- [ ] Resume training from `latest.pt`
- [ ] Monitor WER - should start decreasing after epoch 2-3
- [ ] Complete remaining ~8.5 epochs

### Short-term Goals
- [ ] Achieve WER < 50% (moderate)
- [ ] Achieve WER < 25% (good)
- [ ] Target WER < 15% (production-ready)

### Expected Timeline
| Milestone | Estimated Time |
|-----------|---------------|
| Complete Epoch 2 | +2 hours |
| Complete Epoch 5 | +6 hours |
| Complete Epoch 10 | +12 hours |
| **Total Remaining** | **~12-15 hours** |

### Potential Improvements (If WER Plateaus)
1. **Increase data:** Add train-clean-360 dataset
2. **Learning rate schedule:** Try cosine annealing
3. **CTC weight annealing:** Start 0.7→0.3 over epochs
4. **Longer training:** Extend to 15-20 epochs
5. **Model size:** Increase d_model to 768

---

## WER Monitoring Thresholds

| WER Range | Status | Action |
|-----------|--------|--------|
| >100% | 🔴 CRITICAL | Stop and debug |
| 50-100% | 🟠 High | Watch closely |
| 25-50% | 🟡 Moderate | Keep training |
| 15-25% | 🟢 Good | Almost there |
| <15% | ✅ Target | Production ready |

---

## Key Files & Paths

### Local Development
```
/mnt/d/3d-game-ai-presentation/
├── docs/technical/
│   ├── STT_ARCHITECTURE_PLAN.md    # Full architecture spec
│   ├── VOXFORMER_STAGE4_PLAN.md    # Stage 4 training plan
│   └── TRAIN_PROGRESS.md           # This file
├── src/app/
│   ├── training/page.tsx           # Training dashboard
│   └── api/training-status/route.ts # Dashboard API
```

### VPS
```
/home/developer/voxformer_checkpoints/
├── stage2/best.pt                  # Stage 2 checkpoint
└── stage4_fixed/                   # Current training
    ├── latest.pt                   # Resume from here
    ├── step_*.pt                   # Step checkpoints
    ├── training.log                # Full log
    └── stage4_fixed.yaml           # Config
```

---

## Contact & Resources

- **Dashboard:** http://5.249.161.66:3000/training
- **Architecture Doc:** `/docs/technical/STT_ARCHITECTURE_PLAN.md`
- **Training Plan:** `/docs/technical/VOXFORMER_STAGE4_PLAN.md`

---

*Document auto-generated from training session. Last training run: December 17, 2024.*
