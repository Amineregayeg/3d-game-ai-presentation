"use client";

import { Suspense } from "react";
import { TitleSlide } from "@/components/slides/TitleSlide";
import { GoalSlide } from "@/components/slides/GoalSlide";
import { ArchitectureSlide } from "@/components/slides/ArchitectureSlide";
import { STTSlide } from "@/components/slides/STTSlide";
import { RAGSlide } from "@/components/slides/RAGSlide";
import { TTSLipsyncSlide } from "@/components/slides/TTSLipsyncSlide";
import { BlenderMCPSlide } from "@/components/slides/BlenderMCPSlide";
import { UseCasesSlide } from "@/components/slides/UseCasesSlide";
import { ConclusionSlide } from "@/components/slides/ConclusionSlide";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";
import { useSlideNavigation } from "@/hooks/useSlideNavigation";
import { SlideNavigation, PresenterNotes } from "@/components/presentation";

const slides = [
  TitleSlide,
  GoalSlide,
  ArchitectureSlide,
  STTSlide,
  RAGSlide,
  TTSLipsyncSlide,
  BlenderMCPSlide,
  UseCasesSlide,
  ConclusionSlide
];

// Slide metadata for navigation and presenter notes
const slideInfo = [
  {
    title: "3D Game AI Assistant",
    subtitle: "Introduction",
    notes: [
      "Welcome to the 3D Game Generation AI Assistant overview",
      "This system enables intelligent NPCs with natural voice interaction",
      "Four core components: STT, RAG, TTS+LipSync, Blender MCP",
    ],
  },
  {
    title: "Project Goals",
    subtitle: "What we're building",
    notes: [
      "Create believable AI-driven NPCs for games",
      "Real-time voice interaction with natural responses",
      "Procedural animation generation from text",
      "Scalable architecture for multiple concurrent NPCs",
    ],
  },
  {
    title: "System Architecture",
    subtitle: "High-level overview",
    notes: [
      "Voice pipeline: Audio -> STT -> RAG -> LLM -> TTS -> LipSync",
      "Each component is modular and can be upgraded independently",
      "Blender MCP handles 3D asset generation and animation",
      "Central orchestrator manages conversation state",
    ],
  },
  {
    title: "VoxFormer STT",
    subtitle: "Speech-to-Text",
    notes: [
      "Custom Conformer-based architecture for game audio",
      "Optimized for noisy environments (background music, effects)",
      "Three model sizes for different latency/accuracy tradeoffs",
      "See /technical for deep-dive presentation",
    ],
  },
  {
    title: "Advanced RAG",
    subtitle: "Retrieval-Augmented Generation",
    notes: [
      "Hybrid retrieval: dense vectors + sparse BM25",
      "Cross-encoder reranking for precision",
      "Agentic validation ensures answer quality",
      "See /rag for deep-dive presentation",
    ],
  },
  {
    title: "TTS + LipSync",
    subtitle: "Voice synthesis and animation",
    notes: [
      "Neural TTS with emotion and prosody control",
      "Phoneme-based lip sync for realistic mouth movements",
      "Viseme blending for smooth transitions",
      "Real-time processing for conversational latency",
    ],
  },
  {
    title: "Blender MCP",
    subtitle: "3D generation via Model Context Protocol",
    notes: [
      "MCP enables LLM control of Blender operations",
      "Generate 3D models from text descriptions",
      "Animate characters based on dialogue",
      "Procedural environment generation",
    ],
  },
  {
    title: "Use Cases",
    subtitle: "Applications",
    notes: [
      "Quest givers with unique personalities and memories",
      "Shopkeepers that negotiate and remember customers",
      "Companions that react to player decisions",
      "Tutorial NPCs that adapt to player skill level",
    ],
  },
  {
    title: "Conclusion",
    subtitle: "Next steps",
    notes: [
      "Explore the technical presentations for each component",
      "Check /implementation for current progress",
      "Documentation available at /docs",
      "Timeline: 12-16 weeks per component",
    ],
  },
];

function PresentationContent() {
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
        presentationTitle="3D Game AI"
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

      {/* Presentation Dock */}
      <PresentationDock items={dockItems} />
    </div>
  );
}

export default function AcademicPresentation() {
  return (
    <Suspense fallback={
      <div className="dark min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-slate-400">Loading presentation...</div>
      </div>
    }>
      <PresentationContent />
    </Suspense>
  );
}
