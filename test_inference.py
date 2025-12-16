#!/usr/bin/env python3
"""Quick inference test for VoxFormer"""

import torch
import torchaudio
import yaml
from src.model.voxformer import VoxFormer
from src.data.tokenizer import BPETokenizer

# Load tokenizer
tokenizer = BPETokenizer.from_pretrained("tokenizer")
print(f"Tokenizer vocab size: {len(tokenizer)}")

# Load config from yaml
with open("configs/stage2.yaml") as f:
    config = yaml.safe_load(f)

# Load checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
checkpoint = torch.load("checkpoints/stage2/best.pt", map_location=device, weights_only=False)

# Build model
model = VoxFormer(
    vocab_size=config["model"]["vocab_size"],
    d_model=config["model"]["d_model"],
    encoder_num_heads=config["model"]["encoder_num_heads"],
    encoder_num_blocks=config["model"]["encoder_num_blocks"],
    encoder_layers_per_block=config["model"]["encoder_layers_per_block"],
    decoder_num_heads=config["model"]["decoder_num_heads"],
    decoder_num_layers=config["model"]["decoder_num_layers"],
    d_ff=config["model"]["d_ff"],
    kernel_size=config["model"]["kernel_size"],
    dropout=0.0,
    wavlm_model_name=config["model"]["wavlm_model_name"],
    freeze_wavlm=True,
    ctc_weight=config["model"]["ctc_weight"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print("Model loaded successfully")

# Load a test audio
test_audio_path = "data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"
waveform, sr = torchaudio.load(test_audio_path)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
waveform = waveform.to(device)
print(f"Audio shape: {waveform.shape}, duration: {waveform.shape[1]/16000:.2f}s")

# Get expected transcription
with open("data/LibriSpeech/dev-clean/84/121123/84-121123.trans.txt") as f:
    for line in f:
        if line.startswith("84-121123-0000"):
            expected = line.strip().split(" ", 1)[1]
            break
print(f"Expected: {expected}")

# Run inference
with torch.no_grad():
    # Beam search
    generated = model.transcribe(waveform, max_length=100, beam_size=4)
    print(f"Generated tokens shape: {generated.shape}")
    print(f"Generated tokens (first 30): {generated[0][:30].tolist()}")

    # Decode
    hyp = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    print(f"Hypothesis (beam=4): {hyp}")

    # Greedy
    generated_greedy = model.transcribe(waveform, max_length=100, beam_size=1)
    hyp_greedy = tokenizer.decode(generated_greedy[0].tolist(), skip_special_tokens=True)
    print(f"Hypothesis (greedy): {hyp_greedy}")
