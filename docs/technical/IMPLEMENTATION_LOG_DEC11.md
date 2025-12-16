# Implementation Log - December 11, 2025

## Session Overview

This document records all implementation work completed during the December 11, 2025 session, covering VoxFormer Stage 2 training setup and SadTalker avatar integration.

---

## Part 1: VoxFormer Stage 2 Training Setup

### Background

VoxFormer Stage 1 training was completed on the previous session with:
- **Final Loss:** ~1.0
- **Epochs:** 20
- **Checkpoint:** `/home/developer/voxformer_checkpoints/best_final_stage1.pt`

Stage 2 involves **unfreezing the top 3 WavLM layers** for fine-tuning with a 10x lower learning rate.

### Infrastructure

| Component | Details |
|-----------|---------|
| **GPU Server** | Vast.ai RTX 4090 24GB |
| **GPU IP** | 82.141.118.40:2674 |
| **VPS** | 5.249.161.66 (Debian 13) |
| **Connection** | Local → VPS → GPU (SSH chain) |

### Files Modified/Created on GPU Server

#### 1. Stage 2 Configuration (`/root/voxformer/configs/stage2.yaml`)

```yaml
# Key changes from Stage 1:
model:
  freeze_wavlm: true
  unfreeze_top_k: 3  # Unfreeze top 3 WavLM layers

training:
  learning_rate: 1.0e-5  # 10x lower than Stage 1
  num_epochs: 5
  checkpoint_dir: "checkpoints/stage2"
  resume_from: "checkpoints/stage1/best.pt"
  wandb_project: null  # Disabled (was causing login error)

data:
  train_data: "/root/voxformer/data/LibriSpeech/train-clean-100"
  eval_data: "/root/voxformer/data/LibriSpeech/dev-clean"
```

#### 2. Training Script (`/root/voxformer/train_stage2.sh`)

```bash
#!/bin/bash
# VoxFormer Stage 2 Training Script
# - Auto-backup to VPS every 30 minutes
# - Runs in tmux session named 'training'
# - Dashboard monitoring at http://5.249.161.66:3000/training

VPS_HOST="5.249.161.66"
VPS_USER="developer"
VPS_PASS="    "  # 4 spaces
VPS_BACKUP_DIR="/home/developer/voxformer_checkpoints/stage2"

# Background backup loop (every 30 minutes)
# Backs up: best.pt, latest step checkpoint, epoch_history.log
```

#### 3. Trainer Patch (`/root/voxformer/src/training/trainer.py`)

**Problem:** Stage 2 requires different trainable parameters (WavLM layers unfrozen), causing optimizer state mismatch when resuming from Stage 1.

**Solution:** Added `weights_only` parameter to `load_checkpoint()`:
```python
def load_checkpoint(self, path: str, weights_only: bool = False):
    """Load training checkpoint.

    Args:
        path: Path to checkpoint
        weights_only: If True, only load model weights (for stage transitions)
    """
    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
    self.model.load_state_dict(checkpoint["model_state_dict"])

    if weights_only:
        logger.info(f"Loaded model weights only from {path}")
        return
    # ... rest of optimizer/scheduler loading
```

#### 4. Train Script Patch (`/root/voxformer/scripts/train.py`)

Auto-detect stage transitions:
```python
# Detect stage transition (loading from different stage checkpoint)
is_stage_transition = "stage1" in resume_path and "stage2" in str(training_config.get("checkpoint_dir", ""))
trainer.load_checkpoint(resume_path, weights_only=is_stage_transition)
```

### Dependencies Installed on GPU

```bash
apt-get install -y sshpass  # For automated VPS backups
```

### Training Progress (Stage 2)

| Epoch | Avg Loss | CTC Loss | CE Loss | LR | Time |
|-------|----------|----------|---------|-----|------|
| 0 | 3.9160 | 1.80 | 5.35 | 8.5e-06 | 15:31 |
| 1 | In progress... | | | | |

**Notes:**
- Loss is higher than Stage 1 end because WavLM layers are now trainable
- Expected to decrease as training progresses
- Speed: ~3.8-4.0 it/s

---

## Part 2: Training Dashboard Updates

### Files Modified on VPS

#### 1. Frontend Page (`/home/developer/3d-game-ai/frontend/src/app/training/page.tsx`)

Changed hardcoded epoch count from 20 to 5:
```typescript
// Before
const totalProgress = ((metrics.epoch * metrics.totalSteps + metrics.step) / (20 * metrics.totalSteps)) * 100;
const epochsRemaining = 20 - metrics.epoch;

// After
const totalProgress = ((metrics.epoch * metrics.totalSteps + metrics.step) / (5 * metrics.totalSteps)) * 100;
const epochsRemaining = 5 - metrics.epoch;
```

#### 2. API Route (`/home/developer/3d-game-ai/frontend/src/app/api/training-status/route.ts`)

Fixed ETA calculation:
```typescript
// Before
const remainingSteps = (20 - epoch) * totalSteps - step;

// After
const remainingSteps = (5 - epoch) * totalSteps - step;
```

### Dashboard URL

**http://5.249.161.66:3000/training**

Shows real-time:
- Current epoch/step
- Loss values (total, CTC, CE)
- Learning rate
- GPU utilization and memory
- Estimated time remaining
- Loss history graph

---

## Part 3: SadTalker Avatar Integration

### Background

Replaced MuseTalk with SadTalker for lip-sync video generation due to:
- Better compatibility with Python 3.12
- Natural eye blinking and head movements
- GFPGAN face enhancement

### Files Modified

#### 1. Backend Avatar API (`/mnt/d/3d-game-ai-presentation/backend/avatar_api.py`)

**Key Changes:**
```python
# Changed MuseTalk path to SadTalker
MUSETALK_PATH = os.environ.get('MUSETALK_PATH', '/root/SadTalker')

# Increased timeout for long videos (GFPGAN enhancement is slow)
], check=True, timeout=600)  # Was 300s, now 600s (10 min)
```

#### 2. SadTalker Wrapper (`/root/SadTalker/run_inference.py` on GPU)

Created wrapper script to translate backend API interface to SadTalker CLI:

```python
#!/usr/bin/env python3
"""
SadTalker Wrapper Script
Translates: --audio --image --output --version
To SadTalker: --source_image --driven_audio --result_dir
"""

cmd = [
    sys.executable,
    str(script_dir / "inference.py"),
    "--source_image", args.image,
    "--driven_audio", args.audio,
    "--result_dir", str(result_dir),
    "--size", str(args.size),      # 256 or 512
    "--preprocess", "crop",
    "--enhancer", args.enhancer,   # gfpgan or RestoreFormer
]
```

#### 3. SadTalker Compatibility Fixes (on GPU)

Fixed Python 3.12 compatibility issues:

```bash
# Fix deprecated np.float
sed -i "s/np.float,/float,/g" src/face3d/extract_kp_videos_safe.py

# Fix deprecated np.VisibleDeprecationWarning
sed -i "s/np.VisibleDeprecationWarning/DeprecationWarning/g" src/utils/face_enhancer.py

# Fix torchvision import path
sed -i "s/from torchvision.transforms.functional_tensor/from torchvision.transforms._functional_tensor/g" \
    /usr/local/lib/python3.12/dist-packages/basicsr/data/degradations.py

# Fix trans_params array issue
# In preprocess.py, changed to explicit float() casting
```

### Avatar Selection Feature

Added 6 avatar options from SadTalker examples:

| Avatar ID | Description |
|-----------|-------------|
| default | Default avatar |
| happy | Happy person |
| people_0 | Person style 0 |
| art_0 | Artistic style 0 |
| art_5 | Artistic style 5 |
| art_18 | Artistic style 18 |

**Copied to:** `/home/developer/3d-game-ai/backend/static/avatars/`

#### 4. Frontend Avatar Display Fix

Fixed avatar images not showing in frontend:

```typescript
// Before - relative path didn't work
<AvatarImage src={avatar.url} />

// After - prepend API base URL
<AvatarImage src={`${API_BASE}${avatar.url}`} />
```

**File:** `/home/developer/3d-game-ai/frontend/src/app/avatar_demo/page.tsx`

### Avatar Demo URL

**http://5.249.161.66:3000/avatar_demo**

Features:
- Text input for speech
- Voice selection (ElevenLabs voices)
- Avatar selection (6 options)
- Video generation toggle
- Audio/video playback

---

## Part 4: Backup Configuration

### VPS Backup Directory Structure

```
/home/developer/voxformer_checkpoints/
├── best_final_stage1.pt          # Stage 1 final checkpoint
├── best_epoch15_20251209_1620.pt # Intermediate backups
├── step_14272_epoch15.pt
├── final_epoch19_step17840.pt
└── stage2/                       # Stage 2 backups (new)
    ├── best.pt                   # Auto-backed up every 30 min
    ├── step_*.pt                 # Latest step checkpoint
    └── epoch_history.log         # Training progress log
```

### Backup Script on GPU (`/root/voxformer/backup_to_vps.sh`)

```bash
#!/bin/bash
# Backs up:
# - best.pt
# - Latest step_*.pt checkpoint
# - epoch_history.log

sshpass -p "    " scp -o StrictHostKeyChecking=no \
    "$CHECKPOINT_DIR/best.pt" \
    "$VPS_USER@$VPS_HOST:$VPS_BACKUP_DIR/"
```

### Automatic Backup (in train_stage2.sh)

- Runs in background every 30 minutes
- Triggered by `backup_loop()` function
- Final backup on training completion or exit

---

## Quick Reference Commands

### SSH to GPU via VPS
```bash
sshpass -p '<VPS_PASSWORD>' ssh root@5.249.161.66
ssh -p 2674 root@82.141.118.40
```

### Start/Monitor Stage 2 Training
```bash
# On GPU server
cd /root/voxformer
tmux new -s training
./train_stage2.sh
# Detach: Ctrl+B, D

# Check progress
tmux capture-pane -t training -p -S -20
```

### Deploy Frontend Changes
```bash
sshpass -p '<VPS_PASSWORD>' ssh root@5.249.161.66 \
  'cd /home/developer/3d-game-ai/frontend && npm run build && pm2 restart frontend'
```

### Test Avatar API
```bash
curl -X POST http://5.249.161.66:5000/api/avatar/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "generate_video": true, "avatar_id": "art_0"}'
```

---

## Summary of Changes

### Local Files Modified
| File | Change |
|------|--------|
| `backend/avatar_api.py` | Updated MUSETALK_PATH, increased timeout |
| `backend/sadtalker_wrapper.py` | New file - SadTalker wrapper script |

### VPS Files Modified
| File | Change |
|------|--------|
| `frontend/src/app/training/page.tsx` | Changed 20 epochs to 5 |
| `frontend/src/app/api/training-status/route.ts` | Fixed ETA calculation |
| `frontend/src/app/avatar_demo/page.tsx` | Fixed avatar image URLs |

### GPU Server Files Modified/Created
| File | Change |
|------|--------|
| `/root/voxformer/configs/stage2.yaml` | Updated paths, disabled wandb |
| `/root/voxformer/train_stage2.sh` | New file - Stage 2 training launcher |
| `/root/voxformer/backup_to_vps.sh` | New file - Backup script |
| `/root/voxformer/src/training/trainer.py` | Added weights_only parameter |
| `/root/voxformer/scripts/train.py` | Added stage transition detection |
| `/root/SadTalker/run_inference.py` | New file - API wrapper |
| `/root/SadTalker/src/face3d/*.py` | Python 3.12 compatibility fixes |

---

*Document created: December 11, 2025*
*Training status: Stage 2 Epoch 0 completed, continuing...*
