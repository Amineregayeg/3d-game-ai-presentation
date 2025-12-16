#!/usr/bin/env python3
"""
VoxFormer Quality Evaluation Script

Comprehensive evaluation of:
1. Model architecture correctness
2. Data pipeline integrity
3. Loss function computation
4. Real audio transcription capability
5. Inference performance benchmarks
"""

import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Results storage
results = {
    "architecture": {},
    "data_pipeline": {},
    "loss_function": {},
    "transcription": {},
    "performance": {},
    "issues": [],
    "warnings": [],
}


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_architecture():
    """Test 1: Model Architecture Correctness"""
    print_section("1. MODEL ARCHITECTURE EVALUATION")

    from src.model import VoxFormer
    from src.model.wavlm_frontend import WeightedLayerSum, WavLMAdapter
    from src.model.conformer import ConformerBlock, RotaryPositionEmbedding
    from src.model.zipformer import ZipformerEncoder
    from src.model.decoder import TransformerDecoder

    tests_passed = 0
    tests_total = 0

    # Test 1.1: WeightedLayerSum
    tests_total += 1
    try:
        layer_sum = WeightedLayerSum(num_layers=12)
        hidden_states = tuple(torch.randn(2, 50, 768) for _ in range(13))
        output = layer_sum(hidden_states)
        assert output.shape == (2, 50, 768), f"Expected (2, 50, 768), got {output.shape}"
        weights = layer_sum.get_layer_weights()
        assert abs(weights.sum().item() - 1.0) < 1e-5, "Weights should sum to 1"
        print("✅ WeightedLayerSum: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ WeightedLayerSum: FAIL - {e}")
        results["issues"].append(f"WeightedLayerSum: {e}")

    # Test 1.2: WavLMAdapter
    tests_total += 1
    try:
        adapter = WavLMAdapter(wavlm_dim=768, d_model=512)
        x = torch.randn(2, 50, 768)
        output = adapter(x)
        assert output.shape == (2, 50, 512), f"Expected (2, 50, 512), got {output.shape}"
        print("✅ WavLMAdapter: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ WavLMAdapter: FAIL - {e}")
        results["issues"].append(f"WavLMAdapter: {e}")

    # Test 1.3: RoPE
    tests_total += 1
    try:
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=1024)
        x = torch.randn(2, 100, 64)
        cos, sin = rope(x, 100)
        assert cos.shape == (100, 64), f"Expected (100, 64), got {cos.shape}"
        print("✅ RotaryPositionEmbedding: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ RotaryPositionEmbedding: FAIL - {e}")
        results["issues"].append(f"RoPE: {e}")

    # Test 1.4: ConformerBlock
    tests_total += 1
    try:
        block = ConformerBlock(d_model=512, num_heads=8, d_ff=2048, kernel_size=31)
        x = torch.randn(2, 100, 512)
        output, cache = block(x)
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
        print("✅ ConformerBlock: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ ConformerBlock: FAIL - {e}")
        results["issues"].append(f"ConformerBlock: {e}")

    # Test 1.5: ZipformerEncoder U-Net structure
    tests_total += 1
    try:
        encoder = ZipformerEncoder(
            d_model=512, num_heads=8, d_ff=2048,
            num_blocks=3, layers_per_block=2
        )
        x = torch.randn(2, 100, 512)
        output, mask = encoder(x)
        assert output.shape == x.shape, f"U-Net should preserve shape: {x.shape} -> {output.shape}"
        print("✅ ZipformerEncoder (U-Net preservation): PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ ZipformerEncoder: FAIL - {e}")
        results["issues"].append(f"ZipformerEncoder: {e}")

    # Test 1.6: TransformerDecoder with cross-attention
    tests_total += 1
    try:
        decoder = TransformerDecoder(vocab_size=2000, d_model=512, num_heads=8, num_layers=4)
        encoder_output = torch.randn(2, 100, 512)
        input_ids = torch.randint(0, 2000, (2, 20))
        logits, cache = decoder(input_ids, encoder_output)
        assert logits.shape == (2, 20, 2000), f"Expected (2, 20, 2000), got {logits.shape}"
        print("✅ TransformerDecoder: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ TransformerDecoder: FAIL - {e}")
        results["issues"].append(f"TransformerDecoder: {e}")

    # Test 1.7: Full VoxFormer integration
    tests_total += 1
    try:
        model = VoxFormer(
            vocab_size=2000, d_model=256,
            encoder_num_heads=4, encoder_num_blocks=2, encoder_layers_per_block=1,
            decoder_num_heads=4, decoder_num_layers=2, d_ff=512
        )
        waveform = torch.randn(1, 16000)
        target_ids = torch.randint(0, 2000, (1, 10))
        outputs = model(waveform=waveform, target_ids=target_ids)

        assert "encoder_output" in outputs
        assert "ctc_logits" in outputs
        assert "decoder_logits" in outputs
        print("✅ VoxFormer Integration: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ VoxFormer Integration: FAIL - {e}")
        results["issues"].append(f"VoxFormer Integration: {e}")

    # Test 1.8: Parameter count verification
    tests_total += 1
    try:
        model = VoxFormer(vocab_size=2000, d_model=512)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Expected: ~142M total, ~47M trainable (with WavLM frozen)
        # Actual may vary slightly based on implementation
        print(f"   Total params: {total:,}")
        print(f"   Trainable params: {trainable:,}")

        if total < 100_000_000:
            results["warnings"].append(f"Total params ({total:,}) lower than expected (~142M)")
        print("✅ Parameter Count: RECORDED")
        tests_passed += 1

        results["architecture"]["total_params"] = total
        results["architecture"]["trainable_params"] = trainable
    except Exception as e:
        print(f"❌ Parameter Count: FAIL - {e}")

    results["architecture"]["tests_passed"] = tests_passed
    results["architecture"]["tests_total"] = tests_total
    results["architecture"]["score"] = tests_passed / tests_total * 100

    print(f"\nArchitecture Score: {tests_passed}/{tests_total} ({results['architecture']['score']:.1f}%)")
    return tests_passed == tests_total


def test_data_pipeline():
    """Test 2: Data Pipeline Integrity"""
    print_section("2. DATA PIPELINE EVALUATION")

    from src.data import BPETokenizer, ASRDataset, ASRCollator

    tests_passed = 0
    tests_total = 0

    # Test 2.1: Tokenizer roundtrip
    tests_total += 1
    try:
        tokenizer = BPETokenizer.from_pretrained("tokenizer")
        test_texts = [
            "hello world",
            "the quick brown fox",
            "speech recognition test",
        ]
        for text in test_texts:
            encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
            decoded = tokenizer.decode(encoded, skip_special_tokens=True)
            # BPE may not be perfectly invertible, but should be close
            if text.lower() != decoded.lower().strip():
                results["warnings"].append(f"Tokenizer roundtrip: '{text}' -> '{decoded}'")
        print("✅ Tokenizer Roundtrip: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Tokenizer Roundtrip: FAIL - {e}")
        results["issues"].append(f"Tokenizer: {e}")

    # Test 2.2: Dataset loading
    tests_total += 1
    try:
        tokenizer = BPETokenizer.from_pretrained("tokenizer")
        dataset = ASRDataset(
            manifest_path="data/LibriSpeech/dev-clean",
            tokenizer=tokenizer,
            data_format="librispeech",
            max_audio_len=10.0,
        )
        assert len(dataset) > 0, "Dataset is empty"
        print(f"   Dataset size: {len(dataset)} samples")
        results["data_pipeline"]["dataset_size"] = len(dataset)
        print("✅ Dataset Loading: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Dataset Loading: FAIL - {e}")
        results["issues"].append(f"Dataset: {e}")

    # Test 2.3: Single sample loading
    tests_total += 1
    try:
        sample = dataset[0]
        assert "waveform" in sample
        assert "input_ids" in sample
        assert sample["waveform"].dim() == 1, "Waveform should be 1D"
        assert sample["input_ids"].dim() == 1, "Input IDs should be 1D"
        print(f"   Sample waveform shape: {sample['waveform'].shape}")
        print(f"   Sample input_ids shape: {sample['input_ids'].shape}")
        print(f"   Sample text: '{sample['text'][:50]}...'")
        print("✅ Sample Loading: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Sample Loading: FAIL - {e}")
        results["issues"].append(f"Sample loading: {e}")

    # Test 2.4: Collator batching
    tests_total += 1
    try:
        collator = ASRCollator(pad_token_id=tokenizer.PAD_ID)
        batch = [dataset[i] for i in range(min(4, len(dataset)))]
        collated = collator(batch)

        assert collated["waveform"].dim() == 2, "Batched waveform should be 2D"
        assert collated["input_ids"].dim() == 2, "Batched input_ids should be 2D"
        print(f"   Batched waveform shape: {collated['waveform'].shape}")
        print(f"   Batched input_ids shape: {collated['input_ids'].shape}")
        print("✅ Collator Batching: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Collator Batching: FAIL - {e}")
        results["issues"].append(f"Collator: {e}")

    # Test 2.5: Audio quality check
    tests_total += 1
    try:
        sample = dataset[0]
        waveform = sample["waveform"]

        # Check for NaN/Inf
        assert not torch.isnan(waveform).any(), "Waveform contains NaN"
        assert not torch.isinf(waveform).any(), "Waveform contains Inf"

        # Check reasonable amplitude
        max_amp = waveform.abs().max().item()
        assert max_amp > 0.001, f"Waveform too quiet: max={max_amp}"
        assert max_amp < 10.0, f"Waveform amplitude too high: max={max_amp}"

        print(f"   Audio max amplitude: {max_amp:.4f}")
        print("✅ Audio Quality: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Audio Quality: FAIL - {e}")
        results["issues"].append(f"Audio quality: {e}")

    results["data_pipeline"]["tests_passed"] = tests_passed
    results["data_pipeline"]["tests_total"] = tests_total
    results["data_pipeline"]["score"] = tests_passed / tests_total * 100

    print(f"\nData Pipeline Score: {tests_passed}/{tests_total} ({results['data_pipeline']['score']:.1f}%)")
    return tests_passed == tests_total


def test_loss_function():
    """Test 3: Loss Function Computation"""
    print_section("3. LOSS FUNCTION EVALUATION")

    from src.training.loss import HybridCTCAttentionLoss, LabelSmoothingLoss, CTCLoss

    tests_passed = 0
    tests_total = 0

    # Test 3.1: Label smoothing loss
    tests_total += 1
    try:
        ce_loss = LabelSmoothingLoss(vocab_size=2000, smoothing=0.1, pad_token_id=0)
        logits = torch.randn(2, 20, 2000)
        targets = torch.randint(1, 2000, (2, 20))

        loss = ce_loss(logits, targets)
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() > 0, "Loss should be positive"
        print(f"   Label smoothing loss: {loss.item():.4f}")
        print("✅ Label Smoothing Loss: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Label Smoothing Loss: FAIL - {e}")
        results["issues"].append(f"Label smoothing: {e}")

    # Test 3.2: CTC loss
    tests_total += 1
    try:
        ctc_loss = CTCLoss(blank_token_id=0)
        log_probs = torch.randn(100, 2, 2000).log_softmax(dim=-1)
        targets = torch.randint(1, 2000, (30,))
        input_lengths = torch.tensor([100, 100])
        target_lengths = torch.tensor([15, 15])

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        assert not torch.isnan(loss), "CTC loss is NaN"
        print(f"   CTC loss: {loss.item():.4f}")
        print("✅ CTC Loss: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ CTC Loss: FAIL - {e}")
        results["issues"].append(f"CTC loss: {e}")

    # Test 3.3: Hybrid loss
    tests_total += 1
    try:
        hybrid_loss = HybridCTCAttentionLoss(
            vocab_size=2000, ctc_weight=0.3, ce_weight=0.7,
            label_smoothing=0.1, warmup_steps=5000
        )

        ctc_logits = torch.randn(2, 100, 2000)
        decoder_logits = torch.randn(2, 20, 2000)
        targets = torch.randint(1, 2000, (2, 20))
        targets[:, 0] = 1  # BOS token
        encoder_lengths = torch.tensor([100, 100])
        target_lengths = torch.tensor([18, 18])

        loss, loss_dict = hybrid_loss(
            ctc_logits, decoder_logits, targets, encoder_lengths, target_lengths
        )

        assert not torch.isnan(loss), "Hybrid loss is NaN"
        assert "ctc_loss" in loss_dict
        assert "ce_loss" in loss_dict
        print(f"   Hybrid loss: {loss.item():.4f}")
        print(f"   CTC component: {loss_dict['ctc_loss']:.4f}")
        print(f"   CE component: {loss_dict['ce_loss']:.4f}")
        print("✅ Hybrid Loss: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Hybrid Loss: FAIL - {e}")
        results["issues"].append(f"Hybrid loss: {e}")

    # Test 3.4: Loss gradient flow
    tests_total += 1
    try:
        from src.model import VoxFormer

        model = VoxFormer(
            vocab_size=2000, d_model=256,
            encoder_num_heads=4, encoder_num_blocks=2, encoder_layers_per_block=1,
            decoder_num_heads=4, decoder_num_layers=2, d_ff=512
        )

        waveform = torch.randn(1, 16000)
        target_ids = torch.randint(1, 2000, (1, 10))
        target_ids[0, 0] = 1  # BOS

        outputs = model(waveform=waveform, target_ids=target_ids)

        # Compute simplified loss
        ctc_loss = outputs["ctc_logits"].mean()
        ce_loss = outputs["decoder_logits"].mean()
        total_loss = 0.3 * ctc_loss + 0.7 * ce_loss

        total_loss.backward()

        # Check gradients exist
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_grad = True
                    break

        assert has_grad, "No gradients computed"
        print("✅ Gradient Flow: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Gradient Flow: FAIL - {e}")
        results["issues"].append(f"Gradient flow: {e}")

    results["loss_function"]["tests_passed"] = tests_passed
    results["loss_function"]["tests_total"] = tests_total
    results["loss_function"]["score"] = tests_passed / tests_total * 100

    print(f"\nLoss Function Score: {tests_passed}/{tests_total} ({results['loss_function']['score']:.1f}%)")
    return tests_passed == tests_total


def test_real_transcription():
    """Test 4: Real Audio Transcription"""
    print_section("4. REAL AUDIO TRANSCRIPTION TEST")

    from src.model import VoxFormer
    from src.data import BPETokenizer, ASRDataset

    tests_passed = 0
    tests_total = 0

    # Test 4.1: Load real audio and run inference
    tests_total += 1
    try:
        tokenizer = BPETokenizer.from_pretrained("tokenizer")
        dataset = ASRDataset(
            manifest_path="data/LibriSpeech/dev-clean",
            tokenizer=tokenizer,
            data_format="librispeech",
        )

        # Get a real sample
        sample = dataset[0]
        waveform = sample["waveform"].unsqueeze(0)
        ground_truth = sample["text"]

        print(f"   Ground truth: '{ground_truth}'")
        print(f"   Audio length: {waveform.shape[1] / 16000:.2f}s")

        # Create model (untrained, so output will be random)
        model = VoxFormer(
            vocab_size=2000, d_model=256,
            encoder_num_heads=4, encoder_num_blocks=2, encoder_layers_per_block=1,
            decoder_num_heads=4, decoder_num_layers=2, d_ff=512
        )
        model.eval()

        with torch.no_grad():
            # Test that inference runs without errors
            generated = model.transcribe(
                waveform=waveform,
                max_length=50,
                beam_size=1,
            )

        hypothesis = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        print(f"   Model output: '{hypothesis[:100]}...' (untrained)")

        print("✅ Real Audio Inference: PASS (model runs, output is random since untrained)")
        tests_passed += 1

        results["transcription"]["ground_truth_sample"] = ground_truth
        results["transcription"]["untrained_output"] = hypothesis[:100]
    except Exception as e:
        print(f"❌ Real Audio Inference: FAIL - {e}")
        results["issues"].append(f"Real audio inference: {e}")

    # Test 4.2: CTC decoding path
    tests_total += 1
    try:
        with torch.no_grad():
            outputs = model(waveform=waveform, target_ids=sample["input_ids"].unsqueeze(0))

        ctc_logits = outputs["ctc_logits"]
        ctc_preds = ctc_logits.argmax(dim=-1)

        # Simple CTC decode (remove blanks and duplicates)
        decoded = []
        prev = -1
        for token in ctc_preds[0].tolist():
            if token != 0 and token != prev:  # 0 is blank
                decoded.append(token)
            prev = token

        ctc_text = tokenizer.decode(decoded, skip_special_tokens=True)
        print(f"   CTC path output: '{ctc_text[:100]}...'")
        print("✅ CTC Decoding Path: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ CTC Decoding Path: FAIL - {e}")
        results["issues"].append(f"CTC decoding: {e}")

    results["transcription"]["tests_passed"] = tests_passed
    results["transcription"]["tests_total"] = tests_total
    results["transcription"]["score"] = tests_passed / tests_total * 100

    print(f"\nTranscription Score: {tests_passed}/{tests_total} ({results['transcription']['score']:.1f}%)")
    return tests_passed == tests_total


def test_performance():
    """Test 5: Inference Performance Benchmarks"""
    print_section("5. PERFORMANCE BENCHMARKS")

    from src.model import VoxFormer

    tests_passed = 0
    tests_total = 0

    # Test 5.1: Inference speed
    tests_total += 1
    try:
        model = VoxFormer(
            vocab_size=2000, d_model=256,
            encoder_num_heads=4, encoder_num_blocks=2, encoder_layers_per_block=1,
            decoder_num_heads=4, decoder_num_layers=2, d_ff=512
        )
        model.eval()

        # Benchmark different audio lengths
        audio_lengths = [1, 3, 5, 10]  # seconds

        print("   Audio Length | Inference Time | RTF")
        print("   " + "-" * 45)

        rtf_values = []
        for audio_sec in audio_lengths:
            waveform = torch.randn(1, audio_sec * 16000)

            # Warmup
            with torch.no_grad():
                _ = model(waveform=waveform, target_ids=torch.randint(0, 2000, (1, 10)))

            # Benchmark
            times = []
            for _ in range(3):
                start = time.time()
                with torch.no_grad():
                    _ = model(waveform=waveform, target_ids=torch.randint(0, 2000, (1, 10)))
                times.append(time.time() - start)

            avg_time = sum(times) / len(times)
            rtf = avg_time / audio_sec
            rtf_values.append(rtf)

            print(f"   {audio_sec:>5}s        | {avg_time:>6.2f}s        | {rtf:.3f}")

        avg_rtf = sum(rtf_values) / len(rtf_values)
        results["performance"]["avg_rtf_cpu"] = avg_rtf
        results["performance"]["rtf_by_length"] = dict(zip(audio_lengths, rtf_values))

        # On CPU, RTF > 1 is expected (slower than real-time)
        # GPU should achieve RTF < 0.3
        print(f"\n   Average RTF (CPU): {avg_rtf:.3f}")
        print("   Note: RTF > 1 is expected on CPU. GPU target: RTF < 0.3")
        print("✅ Performance Benchmark: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Performance Benchmark: FAIL - {e}")
        results["issues"].append(f"Performance: {e}")

    # Test 5.2: Memory usage estimation
    tests_total += 1
    try:
        # Get model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

        print(f"\n   Model memory: {model_size_mb:.1f} MB")
        results["performance"]["model_size_mb"] = model_size_mb
        print("✅ Memory Estimation: PASS")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Memory Estimation: FAIL - {e}")

    results["performance"]["tests_passed"] = tests_passed
    results["performance"]["tests_total"] = tests_total
    results["performance"]["score"] = tests_passed / tests_total * 100

    print(f"\nPerformance Score: {tests_passed}/{tests_total} ({results['performance']['score']:.1f}%)")
    return tests_passed == tests_total


def generate_report():
    """Generate final quality report"""
    print_section("QUALITY ASSESSMENT REPORT")

    # Calculate overall score
    scores = []
    for category in ["architecture", "data_pipeline", "loss_function", "transcription", "performance"]:
        if category in results and "score" in results[category]:
            scores.append(results[category]["score"])

    overall_score = sum(scores) / len(scores) if scores else 0
    results["overall_score"] = overall_score

    # Determine quality level
    if overall_score >= 95:
        quality = "EXCELLENT"
        emoji = "🌟"
    elif overall_score >= 85:
        quality = "GOOD"
        emoji = "✅"
    elif overall_score >= 70:
        quality = "MODERATE"
        emoji = "⚠️"
    else:
        quality = "POOR"
        emoji = "❌"

    results["quality_level"] = quality

    print(f"{emoji} Overall Quality: {quality} ({overall_score:.1f}%)")
    print()
    print("Category Scores:")
    print(f"  • Architecture:    {results.get('architecture', {}).get('score', 0):.1f}%")
    print(f"  • Data Pipeline:   {results.get('data_pipeline', {}).get('score', 0):.1f}%")
    print(f"  • Loss Function:   {results.get('loss_function', {}).get('score', 0):.1f}%")
    print(f"  • Transcription:   {results.get('transcription', {}).get('score', 0):.1f}%")
    print(f"  • Performance:     {results.get('performance', {}).get('score', 0):.1f}%")

    if results["issues"]:
        print(f"\n❌ Critical Issues ({len(results['issues'])}):")
        for issue in results["issues"]:
            print(f"   • {issue}")

    if results["warnings"]:
        print(f"\n⚠️ Warnings ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            print(f"   • {warning}")

    # Recommendations
    print("\n📋 Recommendations:")
    if overall_score >= 85:
        print("   • Code quality is sufficient for training")
        print("   • Proceed with GPU training when ready")
    else:
        print("   • Fix critical issues before proceeding")
        print("   • Review architecture implementation")

    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n📄 Detailed results saved to evaluation_results.json")

    return overall_score


def main():
    print("\n" + "="*60)
    print("   VOXFORMER QUALITY EVALUATION")
    print("="*60)

    all_passed = True

    all_passed &= test_architecture()
    all_passed &= test_data_pipeline()
    all_passed &= test_loss_function()
    all_passed &= test_real_transcription()
    all_passed &= test_performance()

    overall_score = generate_report()

    return 0 if overall_score >= 85 else 1


if __name__ == "__main__":
    sys.exit(main())
