# VoxFormer Deployment Guide

## Overview

VoxFormer is a custom Speech-to-Text (STT) Transformer trained on LibriSpeech. This document covers deployment for the Full Demo pipeline.

## Available Checkpoints (on VPS)

| Stage | Path | Size | Description | Recommended |
|-------|------|------|-------------|-------------|
| Stage 1 | `/home/developer/voxformer_checkpoints/best_final_stage1.pt` | 1.6GB | Initial training | No |
| Stage 2 | `/home/developer/voxformer_checkpoints/stage2/best.pt` | 1.8GB | Stable baseline | **YES** |
| Stage 3 | `/home/developer/voxformer_checkpoints/stage3/step_8000.pt` | 1.8GB | Continued training | No |
| Stage 4 | `/home/developer/voxformer_checkpoints/stage4/best.pt` | 1.8GB | CTC-focused | Test later |
| Stage 4 Fixed | `/home/developer/voxformer_checkpoints/stage4_fixed/best.pt` | 819MB | Hybrid loss (possibly incomplete) | No |

### Why Stage 2?

1. **Completed training** - Full training cycle with stable loss convergence
2. **Consistent size** - 1.8GB matches expected full model architecture
3. **Verified outputs** - Stage 4 metrics show `"wer": null` indicating incomplete evaluation
4. **Lower risk** - Stage 2 is a known-good baseline

## VPS File Locations

```
/home/developer/
├── voxformer_backup/
│   ├── src/                    # Model source code
│   │   ├── model/              # VoxFormer architecture
│   │   ├── data/               # Data loading
│   │   └── utils/              # Utilities
│   ├── tokenizer/
│   │   ├── tokenizer.model     # SentencePiece model
│   │   ├── tokenizer.vocab     # Vocabulary (2000 tokens)
│   │   └── config.json         # Tokenizer config
│   ├── inference_server.py     # Flask inference server
│   ├── transcribe_audio.py     # CLI transcription script
│   └── configs/                # Training configs
├── voxformer_checkpoints/
│   ├── stage2/best.pt          # RECOMMENDED checkpoint
│   └── ...
```

## Deployment Architecture

### Current Demo Flow (with mock):
```
User Audio → Frontend → /api/gpu/stt (Whisper) or Mock
```

### Target Demo Flow (with VoxFormer):
```
User Audio → Frontend → Flask Backend → VoxFormer Inference → Transcription
                        (/api/voxformer/transcribe)
```

## Implementation Plan

### 1. Create Flask Endpoint

Add to Flask backend (`/home/developer/3d-game-ai/backend/`):

```python
# voxformer_api.py
from flask import Blueprint, request, jsonify
import torch
import sys
sys.path.insert(0, '/home/developer/voxformer_backup')

voxformer_bp = Blueprint('voxformer', __name__)

# Lazy-load model to save memory
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        # Load model and tokenizer
        checkpoint = torch.load('/home/developer/voxformer_checkpoints/stage2/best.pt')
        # Initialize model architecture...
        # Load tokenizer...
    return model, tokenizer

@voxformer_bp.route('/api/voxformer/transcribe', methods=['POST'])
def transcribe():
    # Load model lazily
    model, tokenizer = load_model()
    # Process audio and return transcription
    ...
```

### 2. Memory Considerations

- VoxFormer model: ~1.8GB in memory
- VPS RAM: Limited (~16GB shared)
- Strategy: **Lazy loading** - only load when first request comes in
- Alternative: Run on GPU server if VPS memory is insufficient

### 3. Audio Processing Pipeline

1. Receive base64 WebM audio from frontend
2. Convert to WAV (16kHz, mono)
3. Extract mel spectrogram features
4. Run through VoxFormer encoder
5. CTC decode to text
6. Return transcription

## Integration Points

### Frontend (full_demo/page.tsx)

The frontend already supports VoxFormer selection:

```typescript
const endpoint = sttEngine === "whisper"
  ? `${API_URL}/api/gpu/stt`
  : `${API_URL}/api/voxformer/transcribe`;  // NEW endpoint
```

### Backend Registration

In `app.py`:
```python
from voxformer_api import voxformer_bp
app.register_blueprint(voxformer_bp)
```

## Testing

### Health Check
```bash
curl http://5.249.161.66:5000/api/voxformer/health
```

### Transcription Test
```bash
curl -X POST http://5.249.161.66:5000/api/voxformer/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio": "<base64-audio>"}'
```

## Rollback Plan

If VoxFormer causes issues:
1. The endpoint is separate (`/api/voxformer/transcribe`)
2. Frontend can switch back to Whisper (`sttEngine === "whisper"`)
3. No changes to SadTalker or Blender MCP

## Performance Expectations

- First request: ~5-10s (model loading)
- Subsequent requests: ~1-3s (inference only)
- Memory usage: ~2GB when loaded

## Related Documentation

- Architecture: `/docs/technical/STT_ARCHITECTURE_PLAN.md`
- Training: `/docs/technical/TRAIN_PROGRESS.md`
- Full Demo: `/docs/technical/FULL_DEMO_PLAN.md`
