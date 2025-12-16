# VoxFormer Stage 4+ Training Plan

## Executive Summary

This plan outlines a multi-stage approach to achieve production-quality Speech-to-Text with target **WER < 5%** on LibriSpeech and **< 10%** on gaming domain.

**Key Insight**: Stage 2 CTC branch was working well but WER metric was measuring the broken AR decoder. Stage 3 made things worse by continuing to train on corrupted state.

**Strategy**: Revert to Stage 2, implement proper CTC-only evaluation, then systematically improve.

---

## Infrastructure Requirements

### GPU Server (Vast.ai - NEW INSTANCE)
- **GPU**: RTX 4090 24GB
- **Cost**: ~$0.40/hour
- **Estimated total**: ~$15-20 for full pipeline
- **Note**: Instance will be rented fresh - all setup from scratch

### VPS Backup Server (ALWAYS AVAILABLE)
- **Host**: 5.249.161.66
- **SSH**: `sshpass -p '<VPS_PASSWORD>' ssh root@5.249.161.66`
- **Backup Dir**: `/home/developer/voxformer_checkpoints/stage4/`
- **Metrics File**: `/home/developer/voxformer_checkpoints/stage4/metrics.json`
- **Dashboard**: http://5.249.161.66:3000/training

### Critical Backup Assets on VPS
```
/home/developer/voxformer_checkpoints/
├── stage2/
│   └── best.pt (1.8GB) ← CRITICAL: Starting point for Stage 4
├── stage4/              ← Will be created
│   ├── metrics.json     ← Real-time dashboard data
│   ├── best.pt          ← Best checkpoint
│   ├── latest.pt        ← Most recent checkpoint
│   └── backup.log       ← Backup history
```

---

## Part 1: GPU Instance Setup Script

Run this immediately after renting a new Vast.ai RTX 4090 instance.

### `setup_stage4.sh` (Run on GPU Server)

```bash
#!/bin/bash
# VoxFormer Stage 4 - Complete Setup Script
# Run this on a fresh Vast.ai RTX 4090 instance

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"; exit 1; }

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          VoxFormer Stage 4 - Complete Setup                  ║"
echo "║          GPU: RTX 4090 24GB | Target: WER < 15%              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================
# Step 1: System Setup
# ============================================
log "Step 1/7: Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq sshpass tmux htop tree jq

# ============================================
# Step 2: Python Environment
# ============================================
log "Step 2/7: Setting up Python environment..."
pip install --quiet torch torchaudio transformers datasets sentencepiece \
    tqdm tensorboard jiwer soundfile librosa

# ============================================
# Step 3: Download VoxFormer Code
# ============================================
log "Step 3/7: Setting up VoxFormer codebase..."
mkdir -p /root/voxformer
cd /root/voxformer

# Download code from VPS backup
sshpass -p '    ' scp -r -o StrictHostKeyChecking=no \
    developer@5.249.161.66:/home/developer/voxformer_backup/* /root/voxformer/ 2>/dev/null || {
    warn "Could not download from VPS, using git clone..."
    # Fallback: Clone from repository if available
    git clone https://github.com/YOUR_REPO/voxformer.git /root/voxformer 2>/dev/null || true
}

mkdir -p /root/voxformer/{checkpoints/stage4,configs,data,tokenizer,logs}

# ============================================
# Step 4: Download LibriSpeech Data
# ============================================
log "Step 4/7: Downloading LibriSpeech train-clean-100..."
cd /root/voxformer/data
mkdir -p LibriSpeech
cd LibriSpeech

if [ ! -d "train-clean-100" ]; then
    wget -q --show-progress https://www.openslr.org/resources/12/train-clean-100.tar.gz
    tar -xzf train-clean-100.tar.gz
    rm train-clean-100.tar.gz
    log "train-clean-100 downloaded (28,539 samples)"
else
    log "train-clean-100 already exists"
fi

if [ ! -d "dev-clean" ]; then
    wget -q --show-progress https://www.openslr.org/resources/12/dev-clean.tar.gz
    tar -xzf dev-clean.tar.gz
    rm dev-clean.tar.gz
    log "dev-clean downloaded (evaluation set)"
else
    log "dev-clean already exists"
fi

# ============================================
# Step 5: Download Stage 2 Checkpoint from VPS
# ============================================
log "Step 5/7: Downloading Stage 2 checkpoint from VPS..."
cd /root/voxformer/checkpoints
mkdir -p stage2

sshpass -p '    ' scp -o StrictHostKeyChecking=no \
    developer@5.249.161.66:/home/developer/voxformer_checkpoints/stage2/best.pt \
    /root/voxformer/checkpoints/stage2/best.pt

if [ -f "stage2/best.pt" ]; then
    SIZE=$(du -h stage2/best.pt | cut -f1)
    log "Stage 2 checkpoint downloaded: ${SIZE}"
else
    error "Failed to download Stage 2 checkpoint!"
fi

# ============================================
# Step 6: Download Tokenizer
# ============================================
log "Step 6/7: Setting up tokenizer..."
cd /root/voxformer
sshpass -p '    ' scp -r -o StrictHostKeyChecking=no \
    developer@5.249.161.66:/home/developer/voxformer_backup/tokenizer/* \
    /root/voxformer/tokenizer/ 2>/dev/null || {
    warn "Could not download tokenizer, will use default BPE"
}

# ============================================
# Step 7: Create Configuration Files
# ============================================
log "Step 7/7: Creating Stage 4 configuration..."

cat > /root/voxformer/configs/stage4.yaml << 'EOF'
# VoxFormer Stage 4 Configuration
# CTC-Only Training from Stage 2 Checkpoint

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
  wavlm_model_name: microsoft/wavlm-base
  freeze_wavlm: true
  unfreeze_top_k: 3
  ctc_weight: 1.0

training:
  learning_rate: 5.0e-6
  weight_decay: 0.01
  max_grad_norm: 1.0
  warmup_steps: 200
  batch_size: 8
  gradient_accumulation_steps: 4
  num_epochs: 5
  mixed_precision: true
  checkpoint_dir: checkpoints/stage4
  save_interval: 500
  keep_last_n: 3
  resume_from: checkpoints/stage2/best.pt
  log_interval: 50
  eval_interval: 500
  wandb_project: null

data:
  train_data: /root/voxformer/data/LibriSpeech/train-clean-100
  eval_data: /root/voxformer/data/LibriSpeech/dev-clean
  data_format: librispeech
  sample_rate: 16000
  max_audio_len: 30.0
  max_text_len: 256
  num_workers: 4

loss:
  ctc_weight: 1.0
  ce_weight: 0.0
  label_smoothing: 0.0
  warmup_steps: 0

evaluation:
  use_ctc_decoding: true
  use_ar_decoding: false
  beam_size: 1
  compute_wer: true

tokenizer:
  vocab_size: 2000
  model_path: tokenizer/tokenizer.model
EOF

# ============================================
# Setup Complete
# ============================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    SETUP COMPLETE!                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Directory structure:"
tree -L 2 /root/voxformer
echo ""
echo "Next steps:"
echo "  1. Run: ./train_stage4.sh"
echo "  2. Monitor: http://5.249.161.66:3000/training"
echo ""
```

---

## Part 2: Robust Backup System

### Backup Architecture

```
GPU Server                          VPS (5.249.161.66)
┌─────────────────┐                ┌─────────────────────────┐
│ /root/voxformer │   Every 5min   │ /home/developer/        │
│ ├─ checkpoints/ │ ──────────────►│   voxformer_checkpoints/│
│ │  └─ stage4/   │                │   └─ stage4/            │
│ │     ├─ best.pt│                │       ├─ best.pt        │
│ │     └─ *.pt   │                │       ├─ latest.pt      │
│ └─ metrics.json │                │       ├─ metrics.json   │ ◄── Dashboard reads this
└─────────────────┘                │       └─ backup.log     │
                                   └─────────────────────────┘
```

### `/root/voxformer/auto_backup.sh`

```bash
#!/bin/bash
# VoxFormer Robust Backup System
# - Backs up every 5 minutes
# - Verifies successful transfers
# - Updates metrics.json for dashboard
# - Logs all operations

set -e

# ============================================
# Configuration
# ============================================
VPS_HOST="5.249.161.66"
VPS_USER="developer"
VPS_PASS="    "  # 4 spaces
STAGE="stage4"
LOCAL_CHECKPOINT_DIR="/root/voxformer/checkpoints/${STAGE}"
LOCAL_METRICS="/root/voxformer/metrics.json"
VPS_BACKUP_DIR="/home/developer/voxformer_checkpoints/${STAGE}"
LOG_FILE="/root/voxformer/backup.log"
BACKUP_INTERVAL=300  # 5 minutes

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================
# Logging Functions
# ============================================
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1"
    echo -e "${YELLOW}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo -e "${RED}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

# ============================================
# SSH Helper
# ============================================
vps_ssh() {
    sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "${VPS_USER}@${VPS_HOST}" "$@" 2>/dev/null
}

vps_scp() {
    sshpass -p "$VPS_PASS" scp -o StrictHostKeyChecking=no -o ConnectTimeout=30 \
        "$@" 2>/dev/null
}

# ============================================
# Backup Functions
# ============================================
backup_file() {
    local src="$1"
    local dst_name="$2"
    local dst="${VPS_BACKUP_DIR}/${dst_name}"

    if [ ! -f "$src" ]; then
        return 1
    fi

    local src_size=$(stat -c%s "$src" 2>/dev/null || echo "0")

    # Upload file
    if vps_scp "$src" "${VPS_USER}@${VPS_HOST}:${dst}"; then
        # Verify upload
        local remote_size=$(vps_ssh "stat -c%s ${dst}" 2>/dev/null || echo "0")

        if [ "$src_size" == "$remote_size" ]; then
            log "✓ Backed up ${dst_name} ($(numfmt --to=iec $src_size))"
            return 0
        else
            error "✗ Size mismatch for ${dst_name}: local=$src_size remote=$remote_size"
            return 1
        fi
    else
        error "✗ Failed to upload ${dst_name}"
        return 1
    fi
}

update_metrics_on_vps() {
    # Update the metrics file on VPS with backup status
    local backup_time=$(date '+%Y-%m-%d %H:%M:%S')
    local backup_status="success"

    if [ -f "$LOCAL_METRICS" ]; then
        # Add backup info to metrics
        local temp_metrics="/tmp/metrics_with_backup.json"
        jq --arg bt "$backup_time" --arg bs "$backup_status" \
            '. + {backup: {last_backup: $bt, status: $bs}}' \
            "$LOCAL_METRICS" > "$temp_metrics" 2>/dev/null || cp "$LOCAL_METRICS" "$temp_metrics"

        vps_scp "$temp_metrics" "${VPS_USER}@${VPS_HOST}:${VPS_BACKUP_DIR}/metrics.json"
        rm -f "$temp_metrics"
    fi
}

# ============================================
# Main Backup Cycle
# ============================================
run_backup() {
    log "━━━━━━━━━━ Starting backup cycle ━━━━━━━━━━"

    # Ensure remote directory exists
    vps_ssh "mkdir -p ${VPS_BACKUP_DIR}"

    local success_count=0
    local total_count=0

    # 1. Backup best checkpoint
    if [ -f "${LOCAL_CHECKPOINT_DIR}/best.pt" ]; then
        ((total_count++))
        if backup_file "${LOCAL_CHECKPOINT_DIR}/best.pt" "best.pt"; then
            ((success_count++))
        fi
    fi

    # 2. Backup latest step checkpoint (find most recent step_*.pt)
    local latest_step=$(ls -t ${LOCAL_CHECKPOINT_DIR}/step_*.pt 2>/dev/null | head -1)
    if [ -n "$latest_step" ] && [ -f "$latest_step" ]; then
        ((total_count++))
        if backup_file "$latest_step" "latest.pt"; then
            ((success_count++))
        fi
    fi

    # 3. Backup metrics.json (CRITICAL for dashboard)
    if [ -f "$LOCAL_METRICS" ]; then
        ((total_count++))
        if backup_file "$LOCAL_METRICS" "metrics.json"; then
            ((success_count++))
        fi
    fi

    # 4. Backup training log
    if [ -f "${LOCAL_CHECKPOINT_DIR}/training.log" ]; then
        ((total_count++))
        if backup_file "${LOCAL_CHECKPOINT_DIR}/training.log" "training.log"; then
            ((success_count++))
        fi
    fi

    # 5. Backup epoch history
    if [ -f "/root/voxformer/epoch_history.log" ]; then
        ((total_count++))
        if backup_file "/root/voxformer/epoch_history.log" "epoch_history.log"; then
            ((success_count++))
        fi
    fi

    # Update backup status on VPS
    update_metrics_on_vps

    log "Backup complete: ${success_count}/${total_count} files"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ============================================
# Main Loop
# ============================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          VoxFormer Backup Service Started                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Stage: ${STAGE}                                               ║"
echo "║  Interval: ${BACKUP_INTERVAL} seconds (5 minutes)                        ║"
echo "║  VPS: ${VPS_HOST}                                         ║"
echo "║  Remote Dir: ${VPS_BACKUP_DIR}                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

log "Backup service started"

while true; do
    run_backup
    sleep $BACKUP_INTERVAL
done
```

---

## Part 3: Training Script with Metrics Export

### `/root/voxformer/train_stage4.sh`

```bash
#!/bin/bash
# VoxFormer Stage 4 Training Launcher
# - Exports real-time metrics to JSON for dashboard
# - Automatic backup service
# - Graceful shutdown handling

set -e

STAGE="stage4"
CONFIG="configs/${STAGE}.yaml"
CHECKPOINT_DIR="checkpoints/${STAGE}"
LOG_FILE="${CHECKPOINT_DIR}/training.log"
METRICS_FILE="/root/voxformer/metrics.json"
EPOCH_HISTORY="/root/voxformer/epoch_history.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          VoxFormer Stage 4 - CTC Recovery Training           ║"
echo "║          From Stage 2 Checkpoint | Target WER < 15%          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Create directories
mkdir -p "$CHECKPOINT_DIR"
touch "$EPOCH_HISTORY"

# Initialize metrics file
cat > "$METRICS_FILE" << EOF
{
  "stage": "${STAGE}",
  "total_epochs": 5,
  "total_steps_per_epoch": 3568,
  "status": "initializing",
  "epoch": 0,
  "step": 0,
  "loss": 0,
  "ctc_loss": 0,
  "wer": null,
  "learning_rate": "5e-6",
  "gpu_memory": 0,
  "gpu_util": 0,
  "speed": 0,
  "last_update": "$(date -Iseconds)",
  "training_start": "$(date -Iseconds)",
  "backup": {
    "last_backup": null,
    "status": "pending"
  }
}
EOF

# Verify Stage 2 checkpoint exists
if [ ! -f "checkpoints/stage2/best.pt" ]; then
    echo -e "${RED}ERROR: Stage 2 checkpoint not found!${NC}"
    echo "Expected: checkpoints/stage2/best.pt"
    echo ""
    echo "Download from VPS:"
    echo "  sshpass -p '    ' scp developer@5.249.161.66:/home/developer/voxformer_checkpoints/stage2/best.pt checkpoints/stage2/"
    exit 1
fi

echo -e "${GREEN}✓ Stage 2 checkpoint found${NC}"
echo ""

# Start backup service in background
echo -e "${YELLOW}Starting backup service...${NC}"
./auto_backup.sh &
BACKUP_PID=$!
echo "Backup PID: $BACKUP_PID"

# Start metrics updater in background
./metrics_updater.sh &
METRICS_PID=$!
echo "Metrics PID: $METRICS_PID"

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"

    # Update metrics to show stopped
    jq '.status = "stopped"' "$METRICS_FILE" > /tmp/metrics_tmp.json && mv /tmp/metrics_tmp.json "$METRICS_FILE"

    # Kill background processes
    kill $BACKUP_PID 2>/dev/null || true
    kill $METRICS_PID 2>/dev/null || true

    # Final backup
    echo -e "${YELLOW}Running final backup...${NC}"
    ./auto_backup.sh &
    FINAL_BACKUP_PID=$!
    sleep 10
    kill $FINAL_BACKUP_PID 2>/dev/null || true

    echo -e "${GREEN}Cleanup complete${NC}"
}
trap cleanup EXIT INT TERM

# Update metrics to running
jq '.status = "running"' "$METRICS_FILE" > /tmp/metrics_tmp.json && mv /tmp/metrics_tmp.json "$METRICS_FILE"

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo "  Config: $CONFIG"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Log file: $LOG_FILE"
echo "  Dashboard: http://5.249.161.66:3000/training"
echo ""

# Start training
cd /root/voxformer
PYTHONPATH=/root/voxformer python3 scripts/train.py \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_FILE"

# Update metrics to completed
jq '.status = "completed"' "$METRICS_FILE" > /tmp/metrics_tmp.json && mv /tmp/metrics_tmp.json "$METRICS_FILE"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    TRAINING COMPLETE!                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
```

### `/root/voxformer/metrics_updater.sh`

```bash
#!/bin/bash
# Real-time metrics updater for dashboard
# Parses training output and updates metrics.json every 10 seconds

METRICS_FILE="/root/voxformer/metrics.json"
LOG_FILE="/root/voxformer/checkpoints/stage4/training.log"
EPOCH_HISTORY="/root/voxformer/epoch_history.log"

update_metrics() {
    # Get GPU stats
    local gpu_stats=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0, 0")
    local gpu_memory=$(echo "$gpu_stats" | cut -d',' -f1 | tr -d ' ')
    local gpu_util=$(echo "$gpu_stats" | cut -d',' -f2 | tr -d ' ')

    # Convert GPU memory to GB
    gpu_memory=$(echo "scale=1; $gpu_memory / 1024" | bc 2>/dev/null || echo "0")

    # Get training progress from tmux
    local tmux_output=$(tmux capture-pane -t training -p -S -50 2>/dev/null || echo "")

    # Parse epoch and step: "Epoch 0:   5%|▍         | 165/3568"
    local epoch=$(echo "$tmux_output" | grep -oP 'Epoch \K\d+' | tail -1 || echo "0")
    local step=$(echo "$tmux_output" | grep -oP '\| \K\d+(?=/)' | tail -1 || echo "0")
    local total_steps=$(echo "$tmux_output" | grep -oP '/\K\d+(?= \[)' | tail -1 || echo "3568")
    local speed=$(echo "$tmux_output" | grep -oP '\d+\.\d+(?=it/s)' | tail -1 || echo "0")

    # Parse loss values
    local loss=$(echo "$tmux_output" | grep -oP 'loss[=:]\K\d+\.\d+' | tail -1 || echo "0")
    local ctc_loss=$(echo "$tmux_output" | grep -oP 'ctc[=:]\K\d+\.\d+' | tail -1 || echo "0")
    local lr=$(echo "$tmux_output" | grep -oP 'lr[=:]\K[0-9.e-]+' | tail -1 || echo "5e-6")

    # Parse WER if available (from evaluation)
    local wer=$(echo "$tmux_output" | grep -oP 'WER[=:]\s*\K\d+\.\d+' | tail -1 || echo "null")

    # Check status
    local status="running"
    if echo "$tmux_output" | grep -q "Training complete"; then
        status="completed"
    elif echo "$tmux_output" | grep -q "Error\|Exception\|Traceback"; then
        status="error"
    fi

    # Calculate ETA
    local eta="Calculating..."
    if [ "$speed" != "0" ] && [ "$speed" != "" ]; then
        local remaining_steps=$(( (5 - ${epoch:-0}) * ${total_steps:-3568} - ${step:-0} ))
        local remaining_secs=$(echo "scale=0; $remaining_steps / $speed" | bc 2>/dev/null || echo "0")
        local hours=$((remaining_secs / 3600))
        local mins=$(((remaining_secs % 3600) / 60))
        eta="${hours}h ${mins}m"
    fi

    # Update metrics.json
    cat > "$METRICS_FILE" << EOF
{
  "stage": "stage4",
  "total_epochs": 5,
  "total_steps_per_epoch": ${total_steps:-3568},
  "status": "${status}",
  "epoch": ${epoch:-0},
  "step": ${step:-0},
  "loss": ${loss:-0},
  "ctc_loss": ${ctc_loss:-0},
  "ce_loss": 0,
  "wer": ${wer},
  "learning_rate": "${lr:-5e-6}",
  "gpu_memory": ${gpu_memory:-0},
  "gpu_util": ${gpu_util:-0},
  "speed": ${speed:-0},
  "eta": "${eta}",
  "last_update": "$(date -Iseconds)"
}
EOF
}

# Main loop
while true; do
    update_metrics
    sleep 10
done
```

---

## Part 4: Dashboard Updates

### Updated API Route: `/src/app/api/training-status/route.ts`

The API should now read from the VPS metrics file (more reliable than SSH to GPU):

```typescript
import { NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

interface Metrics {
  stage: string;
  total_epochs: number;
  total_steps_per_epoch: number;
  status: "running" | "completed" | "error" | "idle" | "initializing";
  epoch: number;
  step: number;
  loss: number;
  ctc_loss: number;
  ce_loss: number;
  wer: number | null;
  learning_rate: string;
  gpu_memory: number;
  gpu_util: number;
  speed: number;
  eta: string;
  last_update: string;
  backup?: {
    last_backup: string | null;
    status: string;
  };
}

// Cache for epoch history
let epochHistory: Array<{
  epoch: number;
  avgLoss: number;
  ctcLoss: number;
  ceLoss: number;
  wer: number | null;
  time: string;
}> = [];

let lossHistory: Array<{ step: number; loss: number }> = [];

export async function GET() {
  try {
    // Primary: Read metrics from VPS (backed up from GPU)
    const { stdout: metricsJson } = await execAsync(
      `sshpass -p '<VPS_PASSWORD>' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@5.249.161.66 "cat /home/developer/voxformer_checkpoints/stage4/metrics.json 2>/dev/null"`,
      { timeout: 10000 }
    ).catch(() => ({ stdout: "{}" }));

    let metrics: Metrics;
    try {
      metrics = JSON.parse(metricsJson);
    } catch {
      metrics = {
        stage: "stage4",
        total_epochs: 5,
        total_steps_per_epoch: 3568,
        status: "idle",
        epoch: 0,
        step: 0,
        loss: 0,
        ctc_loss: 0,
        ce_loss: 0,
        wer: null,
        learning_rate: "5e-6",
        gpu_memory: 0,
        gpu_util: 0,
        speed: 0,
        eta: "Not started",
        last_update: new Date().toISOString(),
      };
    }

    // Read epoch history from VPS
    const { stdout: epochLog } = await execAsync(
      `sshpass -p '<VPS_PASSWORD>' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@5.249.161.66 "cat /home/developer/voxformer_checkpoints/stage4/epoch_history.log 2>/dev/null"`,
      { timeout: 10000 }
    ).catch(() => ({ stdout: "" }));

    // Parse epoch history: "epoch:0,loss:5.13,wer:25.4,time:2025-12-16 10:18:00"
    if (epochLog) {
      const lines = epochLog.trim().split("\n");
      for (const line of lines) {
        const match = line.match(/epoch:(\d+),loss:(\d+\.?\d*),(?:wer:(\d+\.?\d*),)?time:(.+)/);
        if (match) {
          const epoch = parseInt(match[1]);
          const avgLoss = parseFloat(match[2]);
          const wer = match[3] ? parseFloat(match[3]) : null;
          const time = match[4].split(" ")[1] || match[4];

          if (!epochHistory.some((h) => h.epoch === epoch)) {
            epochHistory.push({
              epoch,
              avgLoss,
              ctcLoss: avgLoss, // CTC-only in Stage 4
              ceLoss: 0,
              wer,
              time: time.substring(0, 5),
            });
          }
        }
      }
      epochHistory.sort((a, b) => a.epoch - b.epoch);
    }

    // Build loss history
    if (metrics.loss > 0) {
      const currentStep = metrics.epoch * metrics.total_steps_per_epoch + metrics.step;
      if (!lossHistory.some((l) => l.step === currentStep)) {
        lossHistory.push({ step: currentStep, loss: metrics.loss });
        if (lossHistory.length > 200) lossHistory.shift();
      }
    }

    return NextResponse.json({
      metrics: {
        epoch: metrics.epoch,
        step: metrics.step,
        totalSteps: metrics.total_steps_per_epoch,
        totalEpochs: metrics.total_epochs,
        loss: metrics.loss,
        ctcLoss: metrics.ctc_loss,
        ceLoss: metrics.ce_loss,
        wer: metrics.wer,
        learningRate: metrics.learning_rate,
        speed: metrics.speed,
        gpuMemory: metrics.gpu_memory,
        gpuUtil: metrics.gpu_util,
        eta: metrics.eta,
        status: metrics.status,
        lastUpdate: metrics.last_update,
        stage: metrics.stage,
        backup: metrics.backup,
      },
      history: epochHistory,
      lossHistory,
    });
  } catch (error) {
    console.error("Error fetching training status:", error);
    return NextResponse.json(
      {
        metrics: {
          epoch: 0,
          step: 0,
          totalSteps: 3568,
          totalEpochs: 5,
          loss: 0,
          ctcLoss: 0,
          ceLoss: 0,
          wer: null,
          learningRate: "0",
          speed: 0,
          gpuMemory: 0,
          gpuUtil: 0,
          eta: "Unknown",
          status: "error" as const,
          lastUpdate: new Date().toISOString(),
          stage: "stage4",
          backup: null,
        },
        history: epochHistory,
        lossHistory,
        error: "Failed to fetch training status",
      },
      { status: 500 }
    );
  }
}
```

### Dashboard UI Updates Required

Update `/src/app/training/page.tsx`:

1. **Change total epochs from 20 to dynamic** (Stage 4 = 5 epochs)
2. **Add WER display card**
3. **Add backup status with last backup time**
4. **Update stage name display**

Key changes needed:
```typescript
// Line 99: Change from hardcoded 20 epochs
const totalProgress = ((metrics.epoch * metrics.totalSteps + metrics.step) / (metrics.totalEpochs * metrics.totalSteps)) * 100;

// Add to TrainingMetrics interface:
wer: number | null;
totalEpochs: number;
stage: string;
backup: { last_backup: string | null; status: string } | null;

// Add WER card in the Key Stats Row
// Add real backup status instead of static
```

---

## Part 5: Stage 4 Configuration

### `configs/stage4.yaml`

```yaml
# VoxFormer Stage 4 Configuration
# CTC-Only Training - Recovery from Stage 2

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
  wavlm_model_name: microsoft/wavlm-base
  freeze_wavlm: true
  unfreeze_top_k: 3
  ctc_weight: 1.0

training:
  learning_rate: 5.0e-6     # Very conservative
  weight_decay: 0.01
  max_grad_norm: 1.0
  warmup_steps: 200
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch = 32
  num_epochs: 5
  mixed_precision: true
  checkpoint_dir: checkpoints/stage4
  save_interval: 500        # Save every 500 steps (~7 min)
  keep_last_n: 3
  resume_from: checkpoints/stage2/best.pt
  log_interval: 50          # Log every 50 steps
  eval_interval: 500        # Evaluate every 500 steps
  wandb_project: null

data:
  train_data: /root/voxformer/data/LibriSpeech/train-clean-100
  eval_data: /root/voxformer/data/LibriSpeech/dev-clean
  data_format: librispeech
  sample_rate: 16000
  max_audio_len: 30.0
  max_text_len: 256
  num_workers: 4

loss:
  ctc_weight: 1.0
  ce_weight: 0.0            # Disabled - CTC only
  label_smoothing: 0.0
  warmup_steps: 0

evaluation:
  use_ctc_decoding: true
  use_ar_decoding: false
  beam_size: 1
  compute_wer: true         # Compute WER at each eval

tokenizer:
  vocab_size: 2000
  model_path: tokenizer/tokenizer.model
```

---

## Part 6: Expected Results & Timeline

### Stage 4 Checkpoints (Guaranteed Saves)

| Checkpoint | Step | Approx Time | Expected Loss |
|------------|------|-------------|---------------|
| step_500.pt | 500 | +7 min | ~4.5 |
| step_1000.pt | 1000 | +14 min | ~3.8 |
| step_1500.pt | 1500 | +21 min | ~3.2 |
| step_2000.pt | 2000 | +28 min | ~2.8 |
| step_2500.pt | 2500 | +35 min | ~2.5 |
| step_3000.pt | 3000 | +42 min | ~2.3 |
| **Epoch 1** | 3568 | +50 min | ~2.1 |
| **Epoch 2** | 7136 | +1h 40m | ~1.5 |
| **Epoch 3** | 10704 | +2h 30m | ~1.0 |
| **Epoch 4** | 14272 | +3h 20m | ~0.7 |
| **Epoch 5** | 17840 | +4h 10m | ~0.5 |

### WER Milestones

| Stage | Time | Target CTC WER |
|-------|------|----------------|
| Epoch 1 complete | +50 min | < 25% |
| Epoch 3 complete | +2.5h | < 18% |
| Epoch 5 complete | +4h | < 15% |

---

## Part 7: Quick Start Commands

### On Your Local Machine

```bash
# 1. Rent RTX 4090 on Vast.ai
#    - Go to vast.ai, select RTX 4090, ~$0.40/hr
#    - Note the SSH command (e.g., ssh -p XXXX root@IP)

# 2. Connect and run setup
ssh -p PORT root@GPU_IP
wget -O setup_stage4.sh "URL_TO_SCRIPT"
chmod +x setup_stage4.sh
./setup_stage4.sh

# 3. Start training in tmux
tmux new -s training
./train_stage4.sh
# Detach: Ctrl+B, D

# 4. Monitor
# Dashboard: http://5.249.161.66:3000/training
# Or: tmux attach -t training
```

### Recovery Commands

```bash
# If GPU crashes, on new instance:
./setup_stage4.sh

# Download latest checkpoint from VPS
sshpass -p '    ' scp developer@5.249.161.66:/home/developer/voxformer_checkpoints/stage4/latest.pt \
    /root/voxformer/checkpoints/stage4/

# Update config to resume from latest
sed -i 's|resume_from:.*|resume_from: checkpoints/stage4/latest.pt|' configs/stage4.yaml

# Continue training
./train_stage4.sh
```

---

## Part 8: Approval Checklist

Before starting Stage 4:

- [ ] New GPU instance rented on Vast.ai (RTX 4090)
- [ ] SSH connection verified
- [ ] Stage 2 checkpoint exists on VPS (`/home/developer/voxformer_checkpoints/stage2/best.pt`)
- [ ] VPS backup directory created (`stage4/`)
- [ ] Dashboard API updated for Stage 4
- [ ] Dashboard UI updated (5 epochs, WER display)
- [ ] ~$2-3 budget approved for Stage 4 (~4-5 hours)

---

## Success Criteria

### Minimum Viable (Stage 4 Complete)
- CTC WER < 15% on dev-clean
- Training completes without crashes
- All checkpoints backed up to VPS
- Dashboard shows accurate metrics

### Ready for Production
- CTC WER < 10% (requires Stage 5)
- Real-time transcription working
- < 0.3 RTF (Real-Time Factor)

---

*Plan updated: December 16, 2025*
*Status: Ready for implementation*
