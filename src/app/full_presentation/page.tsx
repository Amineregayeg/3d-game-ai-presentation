"use client";

import { Suspense } from "react";
import {
  CoverSlide,
  ProblemHookSlide,
  PipelineOverviewSlide,
  DeploymentArchSlide,
  WhySTTSlide,
  VoxFormerOverviewSlide,
  DSPSlide,
  EncoderSlide,
  DecoderSlide,
  TrainingLossSlide,
  RAGHighLevelSlide,
  AvatarParallelSlide,
  BlenderMCPSlide,
  DemoResultSlide,
} from "@/components/full-presentation-slides";
import Link from "next/link";
import { useSlideNavigation } from "@/hooks/useSlideNavigation";
import { SlideNavigation, PresenterNotes } from "@/components/presentation";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

const slides = [
  CoverSlide,              // 1. Cover
  ProblemHookSlide,        // 2. Problem Hook
  PipelineOverviewSlide,   // 3. System Overview Pipeline
  DeploymentArchSlide,     // 4. Deployment Architecture
  WhySTTSlide,             // 5. Why STT Core DL
  VoxFormerOverviewSlide,  // 6. VoxFormer Overview
  DSPSlide,                // 7. DSP & Audio Preprocessing
  EncoderSlide,            // 8. Encoder Design
  DecoderSlide,            // 9. Decoder & Attention
  TrainingLossSlide,       // 10. Training Strategy & Loss
  RAGHighLevelSlide,       // 11. Advanced RAG (High-Level)
  AvatarParallelSlide,     // 12. Avatar & Parallel Execution
  BlenderMCPSlide,         // 13. Blender Execution & MCP
  DemoResultSlide,         // 14. Demo Result & Observability
];

// Slide metadata for navigation and presenter notes
const slideInfo = [
  {
    title: "Cover",
    subtitle: "3D Game Generation AI Assistant",
    notes: [
      "Deep Learning Final Project",
      "Team: Amine Regaieg, Firas Bajjar, Med Salim Soussi, Ons Ouenniche",
      "Professor: Hichem Kallel | Lab Instructor: Med Iheb Hergli",
    ],
  },
  {
    title: "Problem Hook",
    subtitle: "Why Blender Is Hard to Use",
    notes: [
      "14+ million downloads vs ~1-3 million active users",
      "Steep learning curve leads to high drop-off",
      "Powerful tool but complex UI and workflows",
    ],
  },
  {
    title: "System Overview",
    subtitle: "End-to-End Pipeline",
    notes: [
      "Voice Input → STT → RAG → Avatar → Blender Execution",
      "Parallel execution of Avatar and Blender",
      "Real-time constraints, GPU accelerated",
    ],
  },
  {
    title: "Deployment Architecture",
    subtitle: "Component Layers",
    notes: [
      "Browser: Mic, UI, Three.js, SSE",
      "VPS Backend: API, orchestration, RAG",
      "GPU Server: STT inference, Blender headless, avatar processing",
    ],
  },
  {
    title: "Why STT Is Core DL",
    subtitle: "Deep Learning Focus",
    notes: [
      "First AI component in pipeline - errors propagate downstream",
      "VoxFormer: custom built and trained by our team",
      "Whisper used only as demo fallback",
    ],
  },
  {
    title: "VoxFormer Overview",
    subtitle: "Seq2Seq ASR Architecture",
    notes: [
      "Raw Audio → WavLM → Zipformer Encoder → Transformer Decoder → Text",
      "Transformer-based ASR designed for low latency",
      "Hybrid CTC + Cross-Entropy training",
    ],
  },
  {
    title: "DSP & Preprocessing",
    subtitle: "Audio Input Processing",
    notes: [
      "Mic → Resampling (16kHz) → Normalization → Silence Segmentation → Model",
      "Cleaner input improves convergence",
      "Reduces model burden and stabilizes alignment",
    ],
  },
  {
    title: "Encoder Design",
    subtitle: "WavLM + Zipformer + RoPE",
    notes: [
      "WavLM-Base: 95M params (frozen) - feature extraction",
      "Zipformer: 47M params (trainable) - efficient encoding",
      "RoPE positions, SwiGLU FFN, Conv modules",
    ],
  },
  {
    title: "Decoder & Attention",
    subtitle: "Transformer Decoder",
    notes: [
      "Autoregressive token generation",
      "Cross-attention aligns audio frames to text",
      "Supports variable-length sequences",
    ],
  },
  {
    title: "Training Strategy",
    subtitle: "Hybrid Loss & Progress",
    notes: [
      "Hybrid CTC (0.3) + Cross-Entropy (0.7) loss",
      "Stage 1: CTC Pre-training - Complete",
      "Stage 2: Hybrid training - Pending GPU",
      "Target WER: <15% on LibriSpeech",
    ],
  },
  {
    title: "Advanced RAG",
    subtitle: "8-Stage Hybrid Pipeline",
    notes: [
      "Dense (BGE-M3) + Sparse (BM25) hybrid retrieval",
      "RAGAS validation: Faithfulness >0.85, Relevancy >0.90",
      "Uses existing models - not core DL contribution",
    ],
  },
  {
    title: "Avatar & Parallel",
    subtitle: "Simultaneous Execution",
    notes: [
      "ElevenLabs TTS + MuseTalk lip-sync",
      "Avatar speaks while Blender executes in parallel",
      "Parallelism reduces perceived latency",
    ],
  },
  {
    title: "Blender Execution",
    subtitle: "MCP + bpy Integration",
    notes: [
      "Three.js in browser for real-time preview",
      "Blender headless on GPU for actual generation",
      "MCP server for tool orchestration",
      "Exportable .blend files",
    ],
  },
  {
    title: "Demo Result",
    subtitle: "System Observability",
    notes: [
      "Voice → validated instructions → real 3D asset",
      "STT confidence, RAG stages, latency metrics",
      "Full pipeline observability and metrics",
    ],
  },
];

function FullPresentationContent() {
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
        presentationTitle="Deep Learning Final"
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

export default function FullPresentation() {
  return (
    <Suspense fallback={
      <div className="dark min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-slate-400">Loading presentation...</div>
      </div>
    }>
      <FullPresentationContent />
    </Suspense>
  );
}
