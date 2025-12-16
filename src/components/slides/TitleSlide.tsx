"use client";

import Image from "next/image";
import { SlideWrapper } from "./SlideWrapper";

interface TitleSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function TitleSlide({ slideNumber, totalSlides }: TitleSlideProps) {
  return (
    <SlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="flex flex-col items-center text-center space-y-8">
        {/* Logo */}
        <div className="mb-4">
          <Image
            src="/medtech-logo.png"
            alt="MedTech Logo"
            width={180}
            height={180}
            className="drop-shadow-2xl"
          />
        </div>

        {/* Title */}
        <h1 className="text-6xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">
          3D Game Generation
        </h1>
        <h2 className="text-5xl font-bold text-white">
          AI Assistant
        </h2>

        {/* Subtitle */}
        <p className="text-xl text-slate-400 max-w-2xl mt-6">
          An intelligent voice-controlled system for creating 3D game assets
          using natural language and advanced AI technologies
        </p>

        {/* Tech badges */}
        <div className="flex flex-wrap justify-center gap-3 mt-8">
          {["Speech-to-Text", "Advanced RAG", "Text-to-Speech", "LipSync Avatar", "Blender MCP"].map((tech) => (
            <span
              key={tech}
              className="px-4 py-2 bg-slate-800/50 border border-slate-700/50 rounded-full text-sm text-cyan-400 backdrop-blur-sm"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>
    </SlideWrapper>
  );
}
