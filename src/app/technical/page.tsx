"use client";

import { Suspense } from "react";
import {
  VoxFormerTitleSlide,
  ArchitectureOverviewSlide,
  WavLMIntegrationSlide,
  DSPPipelineSlide,
  ConformerBlockSlide,
  AttentionMechanismSlide,
  HybridLossSlide,
  TrainingStrategySlide,
  TrainingProgressSlide,
  StreamingInferenceSlide,
  DeploymentSlide,
  ImplementationRoadmapSlide
} from "@/components/tech-slides";
import Link from "next/link";
import { useSlideNavigation } from "@/hooks/useSlideNavigation";
import { SlideNavigation, PresenterNotes } from "@/components/presentation";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

const slides = [
  VoxFormerTitleSlide,
  ArchitectureOverviewSlide,
  WavLMIntegrationSlide,
  DSPPipelineSlide,
  ConformerBlockSlide,
  AttentionMechanismSlide,
  HybridLossSlide,
  TrainingStrategySlide,
  TrainingProgressSlide,
  StreamingInferenceSlide,
  DeploymentSlide,
  ImplementationRoadmapSlide
];

// Slide metadata for navigation and presenter notes
const slideInfo = [
  {
    title: "VoxFormer Introduction",
    subtitle: "WavLM + Custom Transformer",
    notes: [
      "VoxFormer: Elite STT system for game AI with <3.5% WER",
      "WavLM backbone (95M params frozen) + custom Zipformer encoder + Transformer decoder",
      "Total: 142M params, only 47M trainable, $20 training budget",
      "7-day AI-accelerated development timeline",
    ],
  },
  {
    title: "Architecture Overview",
    subtitle: "WavLM → Zipformer → Decoder",
    notes: [
      "WavLM-Base: 95M params, pretrained on 94K hours, outputs 768-dim @ 50fps",
      "Adapter: Projects 768→512 with LayerNorm + Linear + GELU",
      "Zipformer Encoder: 6 Conformer blocks, 25M params, U-Net downsampling (50→25→12.5 fps)",
      "Transformer Decoder: 4 layers, 20M params, cross-attention + KV-cache",
    ],
  },
  {
    title: "WavLM Integration",
    subtitle: "Pretrained feature extractor",
    notes: [
      "WavLM-Base frozen in Stage 1, top 3 layers unfrozen in Stage 2",
      "Weighted Layer Sum: learnable combination of all 12 layers (+0.2-0.5% WER)",
      "Different layers capture acoustic (lower), phonetic (middle), semantic (upper) features",
      "Adapter bridges WavLM 768-dim to encoder 512-dim",
    ],
  },
  {
    title: "DSP Pipeline",
    subtitle: "Voice isolation preprocessing",
    notes: [
      "6-stage DSP pipeline for robust voice extraction from noisy game audio",
      "Signal conditioning, VAD, noise estimation, noise reduction, echo cancellation",
      "Custom FFT (Cooley-Tukey), FIR/IIR filters, adaptive NLMS",
      "Target: <10ms latency, >-20dB noise reduction",
    ],
  },
  {
    title: "Conformer Block",
    subtitle: "Zipformer encoder architecture",
    notes: [
      "Conformer = Convolution + Transformer for speech recognition",
      "Macaron-style: FFN(½) → Attention → Conv → FFN(½) → LayerNorm",
      "Depthwise separable convolution with kernel size 31",
      "8 attention heads, 512 model dim, 2048 FFN dim",
    ],
  },
  {
    title: "Attention Mechanism",
    subtitle: "Multi-head attention + RoPE",
    notes: [
      "Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V",
      "RoPE: Rotary Position Embeddings encode relative positions via 2D rotation",
      "8 heads × 64 dim per head = 512 model dimension",
      "Flash Attention optional: O(N²) → O(N) memory, 2x faster training",
    ],
  },
  {
    title: "Hybrid Loss Function",
    subtitle: "CTC + Cross-Entropy",
    notes: [
      "Hybrid loss: 0.3 CTC + 0.7 Cross-Entropy",
      "CTC: alignment-free, fast convergence, handles variable-length I/O",
      "Cross-Entropy: better final WER, token-level accuracy, label smoothing 0.1",
      "Warmup: 0.4 CTC + 0.6 CE for first 5K steps, then 0.3 + 0.7",
    ],
  },
  {
    title: "Training Strategy",
    subtitle: "3-stage training approach",
    notes: [
      "Stage 1: LibriSpeech 960h, WavLM frozen (30h GPU, target WER <5%)",
      "Stage 2: Unfreeze WavLM top 3 layers, 10x lower LR (5h GPU, target WER <3.5%)",
      "Stage 3: Gaming domain fine-tuning (10h GPU, target gaming WER <10%)",
      "Total: 45h A100 = $18, plus 5h buffer = $20 total",
    ],
  },
  {
    title: "Training Progress",
    subtitle: "Live training metrics",
    notes: [
      "RTX 4090 24GB on Vast.ai ($0.40/hr) - 204.8M params, 110.4M trainable",
      "LibriSpeech train-clean-100: 28,539 samples, ~100 hours audio",
      "Loss dropped 4x in first epoch (22 → 5.95), CTC dropped 57% by epoch 2",
      "Training speed: 3.7 it/s, ~16 min/epoch, ~8 hours total for 20 epochs",
    ],
  },
  {
    title: "Streaming Inference",
    subtitle: "Real-time processing",
    notes: [
      "Chunked audio processing: 160-200ms chunks with 0.5-1.0s left context",
      "Latency breakdown: 20ms capture + 40ms WavLM + 60ms encoder + 40ms decoder = ~160ms",
      "Decoder KV-cache for efficient autoregressive generation",
      "Greedy decoding for real-time, beam search (size=4) after utterance ends",
    ],
  },
  {
    title: "Deployment",
    subtitle: "ONNX + INT8 + gRPC",
    notes: [
      "Export pipeline: PyTorch → ONNX → FP16 → INT8 quantization",
      "Size reduction: 285MB (FP32) → 145MB (FP16) → 75MB (INT8)",
      "Performance: RTF <0.1 (GPU), <0.3 (CPU), <200ms TTFT, <0.3% WER loss",
      "gRPC server with Transcribe and StreamASR endpoints, Unity/Unreal clients",
    ],
  },
  {
    title: "Implementation Roadmap",
    subtitle: "7-day development plan",
    notes: [
      "Days 1-3: Local development on RTX 2070 (architecture, training infra, validation)",
      "Days 4-5: A100 cloud training (Stage 1, 2, 3 = 45h GPU, $18)",
      "Days 6-7: Streaming inference, ONNX export, INT8 quantization, gRPC server",
      "Total budget: $20 | Final deliverable: VoxFormer v1.0",
    ],
  },
];

function TechnicalPresentationContent() {
  const totalSlides = slides.length;

  const {
    currentSlide,
    goToSlide,
    goToNextSlide,
    goToPrevSlide,
    isFirstSlide,
    isLastSlide,
  } = useSlideNavigation({ totalSlides });

  const CurrentSlideComponent = slides[currentSlide];
  const currentSlideInfo = slideInfo[currentSlide];

  return (
    <div className="dark">
      {/* Back to Overview Link */}
      <Link
        href="/"
        className="fixed top-4 left-4 z-50 px-3 py-1.5 bg-slate-800/80 border border-slate-700 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-700 transition-all backdrop-blur-sm flex items-center gap-2"
      >
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        Overview
      </Link>

      {/* Slide Navigation */}
      <SlideNavigation
        currentSlide={currentSlide}
        totalSlides={totalSlides}
        slides={slideInfo}
        onSlideChange={goToSlide}
        onPrev={goToPrevSlide}
        onNext={goToNextSlide}
        isFirstSlide={isFirstSlide}
        isLastSlide={isLastSlide}
        presentationTitle="VoxFormer Technical"
        accentGradient="from-cyan-500 to-purple-500"
      />

      {/* Presenter Notes */}
      <PresenterNotes
        notes={currentSlideInfo.notes}
        slideNumber={currentSlide + 1}
        totalSlides={totalSlides}
        slideTitle={currentSlideInfo.title}
      />

      {/* Current Slide */}
      <CurrentSlideComponent slideNumber={currentSlide + 1} totalSlides={totalSlides} />

      {/* Navigation Dock */}
      <PresentationDock items={dockItems} />
    </div>
  );
}

export default function TechnicalPresentation() {
  return (
    <Suspense fallback={
      <div className="dark min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-slate-400">Loading presentation...</div>
      </div>
    }>
      <TechnicalPresentationContent />
    </Suspense>
  );
}
