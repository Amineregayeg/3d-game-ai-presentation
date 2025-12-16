"use client";

import { Suspense } from "react";
import {
  MCPTitleSlide,
  MCPArchitectureSlide,
  MCPProtocolSlide,
  MCPToolsSlide,
  AssetSourcesSlide,
  BlenderAddonSlide,
  GameEngineExportSlide,
  MCPRoadmapSlide
} from "@/components/mcp-slides";
import Link from "next/link";
import { useSlideNavigation } from "@/hooks/useSlideNavigation";
import { SlideNavigation, PresenterNotes } from "@/components/presentation";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

const slides = [
  MCPTitleSlide,
  MCPArchitectureSlide,
  MCPProtocolSlide,
  MCPToolsSlide,
  AssetSourcesSlide,
  BlenderAddonSlide,
  GameEngineExportSlide,
  MCPRoadmapSlide
];

// Slide metadata for navigation and presenter notes
const slideInfo = [
  {
    title: "Blender MCP Introduction",
    subtitle: "LLM-driven 3D generation",
    notes: [
      "Model Context Protocol enables LLMs to control external tools",
      "Blender MCP bridges AI and 3D content creation",
      "Use case: procedural asset and animation generation",
    ],
  },
  {
    title: "System Architecture",
    subtitle: "MCP server design",
    notes: [
      "MCP Server runs alongside Blender",
      "JSON-RPC communication between LLM and Blender",
      "Tool definitions expose Blender operations",
      "Stateful session management for complex operations",
    ],
  },
  {
    title: "MCP Protocol",
    subtitle: "Communication layer",
    notes: [
      "JSON-RPC 2.0 based protocol",
      "Tools: functions the LLM can call",
      "Resources: data the LLM can read",
      "Prompts: pre-built templates for common tasks",
    ],
  },
  {
    title: "Available Tools",
    subtitle: "Blender operations",
    notes: [
      "create_mesh: Primitive and custom mesh generation",
      "apply_modifier: Boolean, subdivision, array modifiers",
      "set_material: PBR material assignment",
      "animate_object: Keyframe animation creation",
    ],
  },
  {
    title: "Asset Sources",
    subtitle: "3D content pipelines",
    notes: [
      "Poly Haven: Free CC0 materials and HDRIs",
      "Sketchfab API: Model search and download",
      "Text-to-3D: Integration with generative models",
      "Procedural generation: Geometry nodes",
    ],
  },
  {
    title: "Blender Addon",
    subtitle: "Python integration",
    notes: [
      "Addon registers MCP server as Blender operator",
      "Background thread for non-blocking communication",
      "Event queue for thread-safe Blender operations",
      "Automatic reconnection handling",
    ],
  },
  {
    title: "Game Engine Export",
    subtitle: "Unity and Unreal pipelines",
    notes: [
      "GLTF/GLB export for universal compatibility",
      "FBX export for Unreal Engine",
      "LOD generation for performance",
      "Collision mesh generation",
    ],
  },
  {
    title: "Implementation Roadmap",
    subtitle: "10-week development plan",
    notes: [
      "Phase 1: MCP server skeleton (Weeks 1-2)",
      "Phase 2: Core Blender tools (Weeks 3-5)",
      "Phase 3: Asset source integrations (Weeks 6-7)",
      "Phase 4: Export pipelines (Weeks 8-10)",
    ],
  },
];

function MCPPresentationContent() {
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
        presentationTitle="Blender MCP Technical"
        accentGradient="from-orange-500 to-amber-500"
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

export default function MCPPresentation() {
  return (
    <Suspense fallback={
      <div className="dark min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-slate-400">Loading presentation...</div>
      </div>
    }>
      <MCPPresentationContent />
    </Suspense>
  );
}
