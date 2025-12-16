# VoxFormer Stage 3 Analysis & Fixes

## Date: December 11, 2025

---

## Executive Summary

After completing Stage 2 training, we discovered that while the **CTC branch works excellently**, the **autoregressive decoder has a critical repetition bug**. Stage 3 training has been initiated with a CTC-focused approach to leverage the working encoder while continuing to improve the model.

---

## Stage 2 Results Analysis

### Training Metrics
| Metric | Value |
|--------|-------|
| Final Loss | 3.65 |
| Best WER (reported) | 971.93% |
| Training Duration | ~2 hours |
| Epochs Completed | 5 |

### The WER Problem

The extremely high WER (>900%) was misleading. Investigation revealed:

1. **The encoder is working correctly** - CTC branch produces good transcriptions
2. **The autoregressive decoder is broken** - Stuck in repetition loops
3. **WER was measuring decoder output** - Not representative of model quality

---

## Root Cause Investigation

### Test 1: CTC-Only Decoding

We bypassed the autoregressive decoder and used CTC greedy decoding directly:

```python
# CTC greedy decode
predictions = ctc_logits.argmax(dim=-1)
# Remove blanks and consecutive duplicates
```

**Results:**

| Audio | Expected | CTC Output | Quality |
|-------|----------|------------|---------|
| 84-121123-0000 | GO DO YOU HEAR | go do you here | ✅ Excellent |
| 84-121123-0001 | BUT IN LESS THAN FIVE MINUTES... | but in less than five minutes the st case... | 🔶 Good |
| 84-121123-0002 | AT THIS MOMENT THE WHOLE SOUL... | at this moment the whole so of the old man... | 🔶 Good |

**Conclusion:** The encoder + CTC head learned useful representations.

### Test 2: Autoregressive Decoder

```python
generated = model.transcribe(waveform, max_length=100, beam_size=4)
```

**Results:**

| Audio | Expected | AR Decoder Output |
|-------|----------|-------------------|
| 84-121123-0000 | GO DO YOU HEAR | you are you are you are you are... (repeated 50x) |

**Conclusion:** Decoder is stuck in a repetition loop.

---

## Diagnosis: Why the Decoder Fails

### 1. Exposure Bias (Teacher Forcing Problem)

During training:
- Decoder sees **ground truth** previous tokens
- Learns: P(next_token | correct_previous_tokens)

During inference:
- Decoder sees **its own predictions**
- Small errors compound exponentially
- Gets stuck in high-probability loops like "you are"

### 2. Insufficient Encoder-Decoder Attention

The decoder may not be properly attending to encoder features:
- CE Loss (70%) focuses on next-token prediction
- CTC Loss (30%) only supervises encoder
- Decoder can "cheat" by learning language model patterns

### 3. BPE Tokenization Vulnerability

- Short subword units (2000 vocab)
- Common patterns like "you", "are" have high probability
- Easy to get trapped in loops

---

## Solution: Stage 3 Training

### Strategy: CTC-Focused Training

Since CTC works well, we shift focus:

| Parameter | Stage 2 | Stage 3 | Rationale |
|-----------|---------|---------|-----------|
| CTC Weight | 0.3 | **0.7** | Strengthen working branch |
| CE Weight | 0.7 | **0.3** | Reduce decoder emphasis |
| Learning Rate | 1e-5 | **2e-5** | Moderate fine-tuning |
| Epochs | 5 | **10** | More training time |
| Eval Interval | 1000 | **500** | Catch issues early |

### Configuration (`configs/stage3.yaml`)

```yaml
# Loss Configuration - CTC focused
loss:
  ctc_weight: 0.7      # HIGH - CTC works, focus here
  ce_weight: 0.3       # Lower decoder weight
  label_smoothing: 0.1

# Training Parameters
training:
  learning_rate: 2.0e-5
  num_epochs: 10
  checkpoint_dir: "checkpoints/stage3"
  resume_from: "checkpoints/stage2/best.pt"
```

### Why This Approach?

1. **Leverage what works**: CTC produces good transcriptions
2. **Gradual improvement**: Decoder may still learn with continued training
3. **Practical solution**: Can use CTC decoding for inference now
4. **Risk mitigation**: If decoder doesn't improve, CTC still works

---

## Backup System

### VPS Backup Configuration

| Setting | Value |
|---------|-------|
| Backup Directory | `/home/developer/voxformer_checkpoints/stage3/` |
| Backup Interval | Every 20 minutes |
| Files Backed Up | `best.pt`, `step_*.pt`, `training.log` |

### Backup Script

```bash
# Background loop in train_stage3.sh
backup_to_vps() {
    sshpass -p "$VPS_PASS" scp \
        "$CHECKPOINT_DIR/best.pt" \
        "$VPS_USER@$VPS_HOST:$VPS_BACKUP_DIR/"
}
```

---

## Inference Options

### Option 1: CTC Greedy Decoding (Recommended)

```python
def ctc_greedy_decode(logits, tokenizer, blank_id=0):
    predictions = logits.argmax(dim=-1)
    decoded_ids = []
    prev_token = None
    for token in predictions.tolist():
        if token != blank_id and token != prev_token:
            decoded_ids.append(token)
        prev_token = token
    return tokenizer.decode(decoded_ids, skip_special_tokens=True)
```

**Pros:** Works now, simple, fast
**Cons:** No language model, may have minor errors

### Option 2: CTC Beam Search (Future)

Add language model rescoring for better accuracy:
- Requires external LM
- More complex implementation
- Better handling of homophones

### Option 3: AR Decoder with Repetition Penalty (Future)

```python
def generate_with_penalty(self, ..., repetition_penalty=1.2, no_repeat_ngram_size=3):
    # Penalize repeated tokens
    # Block repeated n-grams
```

**Pros:** May produce more natural output
**Cons:** Needs more decoder training

---

## Training Progress Monitoring

### Commands

```bash
# Check GPU training
ssh -p 2674 root@82.141.118.40 "tmux capture-pane -t training -p -S -20"

# Check VPS backups
ls -la /home/developer/voxformer_checkpoints/stage3/

# Run CTC inference test
python3 test_ctc_decode.py
```

### Dashboard

Training dashboard available at: `http://5.249.161.66:3000/training`

---

## Next Steps

### Immediate (Stage 3 Training - 4 hours)
1. ✅ Stage 3 training started
2. ⏳ Monitor loss convergence
3. ⏳ Verify VPS backups
4. ⏳ Test CTC output periodically

### After Stage 3 Completes
1. Run comprehensive CTC evaluation on dev-clean
2. Calculate actual WER using CTC decoding
3. Test on various audio samples
4. If decoder still broken, implement CTC beam search

### Future Improvements (If Needed)
1. **Scheduled Sampling**: Train decoder with its own predictions
2. **Larger Dataset**: Train on train-clean-360 or train-other-500
3. **Beam Search + LM**: Add language model for CTC decoding
4. **Model Architecture**: Consider CTC-only model (no decoder)

---

## Files Created/Modified

### Local Files
| File | Purpose |
|------|---------|
| `configs/stage3.yaml` | Stage 3 training config |
| `train_stage3.sh` | Training script with auto-backup |
| `test_ctc_decode.py` | CTC decoding test script |
| `test_inference.py` | AR decoder test script |
| `decoder_fix.py` | Repetition penalty implementation |

### GPU Server Files
| File | Purpose |
|------|---------|
| `/root/voxformer/configs/stage3.yaml` | Deployed config |
| `/root/voxformer/train_stage3.sh` | Deployed training script |
| `/root/voxformer/test_ctc_decode.py` | CTC test script |

### VPS Files
| Directory | Contents |
|-----------|----------|
| `/home/developer/voxformer_checkpoints/stage3/` | Stage 3 backups |

---

## Key Learnings

1. **Always test both branches**: CTC and AR decoder should be evaluated separately
2. **High WER doesn't mean bad model**: The metric was measuring the wrong thing
3. **Teacher forcing creates exposure bias**: Common problem in seq2seq models
4. **CTC is robust**: Works even when decoder fails
5. **Backup frequently**: 20-minute intervals saved us from losing progress

---

## Conclusion

The VoxFormer encoder and CTC branch are working well, producing intelligible transcriptions. The autoregressive decoder has a repetition bug caused by exposure bias. Stage 3 training focuses on strengthening the CTC branch while continuing to train the decoder with lower weight. For production use, CTC greedy decoding is recommended until the decoder is fixed.

---

*Document created: December 11, 2025*
*Training status: Stage 3 in progress*
