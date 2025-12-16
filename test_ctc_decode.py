#!/usr/bin/env python3
"""Test CTC-only decoding to check encoder quality"""

import torch
import torchaudio
import yaml
from src.model.voxformer import VoxFormer
from src.data.tokenizer import BPETokenizer

def ctc_greedy_decode(logits, tokenizer, blank_id=0):
    """
    Greedy CTC decoding with blank removal and deduplication.

    Args:
        logits: CTC output logits (T, vocab_size)
        tokenizer: BPE tokenizer
        blank_id: CTC blank token ID (usually 0 = <pad>)

    Returns:
        Decoded text string
    """
    # Get most likely tokens
    predictions = logits.argmax(dim=-1)  # (T,)

    # Remove consecutive duplicates and blanks
    decoded_ids = []
    prev_token = None
    for token in predictions.tolist():
        if token != blank_id and token != prev_token:
            decoded_ids.append(token)
        prev_token = token

    # Decode to text
    text = tokenizer.decode(decoded_ids, skip_special_tokens=True)
    return text, decoded_ids

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

# Test multiple audio files
test_files = [
    ("data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac", "84-121123-0000"),
    ("data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac", "84-121123-0001"),
    ("data/LibriSpeech/dev-clean/84/121123/84-121123-0002.flac", "84-121123-0002"),
]

# Load transcripts
transcripts = {}
with open("data/LibriSpeech/dev-clean/84/121123/84-121123.trans.txt") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            transcripts[parts[0]] = parts[1]

print("\n" + "="*60)
print("CTC GREEDY DECODING TEST")
print("="*60)

for audio_path, utt_id in test_files:
    print(f"\n--- {utt_id} ---")

    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.to(device)

    expected = transcripts.get(utt_id, "N/A")
    print(f"Expected: {expected}")
    print(f"Duration: {waveform.shape[1]/16000:.2f}s")

    # Run model forward pass
    with torch.no_grad():
        # Get CTC logits directly from model forward
        outputs = model(waveform)
        ctc_logits = outputs["ctc_logits"]

        # CTC decode
        ctc_text, ctc_ids = ctc_greedy_decode(ctc_logits[0], tokenizer, blank_id=0)
        print(f"CTC Output: {ctc_text}")
        print(f"CTC Token IDs (first 20): {ctc_ids[:20]}")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)
print("If CTC output is reasonable -> Encoder is working, decoder needs fixing")
print("If CTC output is garbage -> Encoder/frontend has issues")
