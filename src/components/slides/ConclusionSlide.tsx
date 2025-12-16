"use client";

import Image from "next/image";
import { SlideWrapper } from "./SlideWrapper";

interface ConclusionSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function ConclusionSlide({ slideNumber, totalSlides }: ConclusionSlideProps) {
  const highlights = [
    { icon: "01", text: "Voice-driven 3D asset creation" },
    { icon: "02", text: "Context-aware AI understanding" },
    { icon: "03", text: "Interactive avatar assistant" },
    { icon: "04", text: "Direct Blender integration" }
  ];

  return (
    <SlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="flex flex-col items-center text-center space-y-10">
        {/* Logo */}
        <Image
          src="/medtech-logo.png"
          alt="MedTech Logo"
          width={120}
          height={120}
          className="drop-shadow-2xl"
        />

        {/* Title */}
        <div className="space-y-4">
          <h2 className="text-5xl font-bold text-white">
            Ready to <span className="bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">Transform</span>
          </h2>
          <h3 className="text-4xl font-bold text-white">
            3D Game Development
          </h3>
        </div>

        {/* Highlights */}
        <div className="flex flex-wrap justify-center gap-4 max-w-3xl">
          {highlights.map((item) => (
            <div
              key={item.icon}
              className="flex items-center gap-3 px-6 py-3 bg-slate-800/50 border border-slate-700/50 rounded-full"
            >
              <span className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-full flex items-center justify-center text-xs font-bold text-white">
                {item.icon}
              </span>
              <span className="text-slate-300">{item.text}</span>
            </div>
          ))}
        </div>

        {/* Tech stack summary */}
        <div className="p-8 bg-gradient-to-r from-cyan-500/10 via-purple-500/10 to-pink-500/10 border border-slate-700/50 rounded-3xl max-w-4xl">
          <h4 className="text-lg font-semibold text-white mb-6">Complete Technology Stack</h4>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              "STT (Local/Fine-tuned)",
              "Advanced RAG",
              "LangChain",
              "OpenAI TTS",
              "LipSync Avatar",
              "Blender MCP",
              "Python",
              "TypeScript",
              "Unity/Unreal Ready"
            ].map((tech) => (
              <span
                key={tech}
                className="px-4 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-slate-300 hover:border-cyan-500/50 hover:text-cyan-400 transition-colors"
              >
                {tech}
              </span>
            ))}
          </div>
        </div>

        {/* Call to action */}
        <div className="space-y-4">
          <p className="text-xl text-slate-400 max-w-2xl">
            The future of 3D game asset creation is voice-first, AI-powered, and accessible to everyone.
          </p>
          <div className="flex items-center justify-center gap-2 text-cyan-400">
            <span className="animate-pulse">Thank you</span>
            <span className="text-2xl">|</span>
            <span>Questions?</span>
          </div>
        </div>
      </div>
    </SlideWrapper>
  );
}
