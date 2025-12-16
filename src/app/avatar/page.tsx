"use client";

import { Suspense } from "react";
import {
  AvatarTitleSlide,
  AvatarArchitectureSlide,
  ElevenLabsSlide,
  LipSyncSlide,
  VisemeBlendSlide,
  StreamingSlide,
  GameEngineSlide,
  AvatarRoadmapSlide
} from "@/components/avatar-slides";
import Link from "next/link";
import { useSlideNavigation } from "@/hooks/useSlideNavigation";
import { SlideNavigation, PresenterNotes } from "@/components/presentation";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

const slides = [
  AvatarTitleSlide,
  AvatarArchitectureSlide,
  ElevenLabsSlide,
  LipSyncSlide,
  VisemeBlendSlide,
  StreamingSlide,
  GameEngineSlide,
  AvatarRoadmapSlide
];

// Slide metadata for navigation and presenter notes
const slideInfo = [
  {
    title: "TTS + LipSync Introduction",
    subtitle: "Voice synthesis and facial animation",
    notes: [
      "This component brings NPCs to life with realistic voice and lip movements",
      "Combines neural TTS with phoneme-based lip synchronization",
      "Real-time processing for conversational latency requirements",
    ],
  },
  {
    title: "System Architecture",
    subtitle: "End-to-end pipeline",
    notes: [
      "Text input -> TTS Engine -> Audio + Phoneme timing",
      "Phoneme timing -> Viseme mapping -> Blend shapes",
      "Streaming architecture for low-latency playback",
      "Integration points for Unity and Unreal Engine",
    ],
  },
  {
    title: "ElevenLabs Integration",
    subtitle: "Neural TTS engine",
    notes: [
      "ElevenLabs provides high-quality neural voice synthesis",
      "Voice cloning capability for unique NPC voices",
      "Emotion and style control through SSML tags",
      "WebSocket streaming for real-time generation",
    ],
  },
  {
    title: "Lip Sync Pipeline",
    subtitle: "Phoneme to viseme mapping",
    notes: [
      "Phonemes: smallest units of speech sound",
      "Visemes: visual representation of phonemes",
      "CMU phoneme set maps to ~15 core visemes",
      "Timing data extracted from TTS alignment output",
    ],
  },
  {
    title: "Viseme Blending",
    subtitle: "Smooth facial animation",
    notes: [
      "Linear interpolation between viseme poses",
      "Co-articulation: neighboring phonemes influence each other",
      "Blend shape weights for smooth transitions",
      "60fps update rate for smooth animation",
    ],
  },
  {
    title: "Streaming Architecture",
    subtitle: "Real-time processing",
    notes: [
      "Audio chunks processed as they arrive",
      "Ring buffer for smooth playback",
      "Latency target: <200ms from text to speech",
      "Fallback to full synthesis for offline content",
    ],
  },
  {
    title: "Game Engine Integration",
    subtitle: "Unity and Unreal support",
    notes: [
      "Unity: C# SDK with blend shape control",
      "Unreal: Blueprint-compatible plugin",
      "Morph target animation system",
      "Audio source synchronization",
    ],
  },
  {
    title: "Implementation Roadmap",
    subtitle: "8-week development plan",
    notes: [
      "Phase 1: ElevenLabs API integration (Weeks 1-2)",
      "Phase 2: Phoneme extraction pipeline (Weeks 3-4)",
      "Phase 3: Viseme mapping and blending (Weeks 5-6)",
      "Phase 4: Game engine plugins (Weeks 7-8)",
    ],
  },
];

function AvatarPresentationContent() {
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
        presentationTitle="TTS + LipSync Technical"
        accentGradient="from-rose-500 to-pink-500"
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

export default function AvatarPresentation() {
  return (
    <Suspense fallback={
      <div className="dark min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-slate-400">Loading presentation...</div>
      </div>
    }>
      <AvatarPresentationContent />
    </Suspense>
  );
}
