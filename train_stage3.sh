#!/bin/bash
# VoxFormer Stage 3 Training Script
# Focus: Decoder improvement with scheduled sampling
# Auto-backup to VPS every 20 minutes

set -e

echo "========================================"
echo "VoxFormer Stage 3 Training"
echo "Started: $(date)"
echo "========================================"

# VPS backup configuration
VPS_HOST="5.249.161.66"
VPS_USER="developer"
VPS_PASS="    "  # 4 spaces
VPS_BACKUP_DIR="/home/developer/voxformer_checkpoints/stage3"

# Create backup directory on VPS
sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "mkdir -p $VPS_BACKUP_DIR" || true

# Checkpoint directory
CHECKPOINT_DIR="checkpoints/stage3"
mkdir -p "$CHECKPOINT_DIR"

# Backup function
backup_to_vps() {
    echo "[$(date)] Running backup to VPS..."

    # Backup best.pt
    if [ -f "$CHECKPOINT_DIR/best.pt" ]; then
        sshpass -p "$VPS_PASS" scp -o StrictHostKeyChecking=no \
            "$CHECKPOINT_DIR/best.pt" \
            "$VPS_USER@$VPS_HOST:$VPS_BACKUP_DIR/" && \
            echo "[$(date)] Backed up best.pt"
    fi

    # Backup latest step checkpoint
    LATEST_STEP=$(ls -t $CHECKPOINT_DIR/step_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_STEP" ]; then
        sshpass -p "$VPS_PASS" scp -o StrictHostKeyChecking=no \
            "$LATEST_STEP" \
            "$VPS_USER@$VPS_HOST:$VPS_BACKUP_DIR/" && \
            echo "[$(date)] Backed up $LATEST_STEP"
    fi

    # Backup training log
    if [ -f "$CHECKPOINT_DIR/training.log" ]; then
        sshpass -p "$VPS_PASS" scp -o StrictHostKeyChecking=no \
            "$CHECKPOINT_DIR/training.log" \
            "$VPS_USER@$VPS_HOST:$VPS_BACKUP_DIR/"
    fi
}

# Background backup loop (every 20 minutes)
backup_loop() {
    while true; do
        sleep 1200  # 20 minutes
        backup_to_vps
    done
}

# Start backup loop in background
backup_loop &
BACKUP_PID=$!

# Cleanup on exit
cleanup() {
    echo "Cleaning up..."
    kill $BACKUP_PID 2>/dev/null || true
    backup_to_vps  # Final backup
    echo "Training stopped at $(date)"
}
trap cleanup EXIT

# Run training
cd /root/voxformer
python3 scripts/train.py --config configs/stage3.yaml 2>&1 | tee "$CHECKPOINT_DIR/training.log"

echo "========================================"
echo "Training Complete!"
echo "Finished: $(date)"
echo "========================================"
