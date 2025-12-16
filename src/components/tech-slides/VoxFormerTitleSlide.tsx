"use client";

import Image from "next/image";
import { TechSlideWrapper } from "./TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface VoxFormerTitleSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function VoxFormerTitleSlide({ slideNumber, totalSlides }: VoxFormerTitleSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="flex flex-col items-center justify-center h-full text-center space-y-8">
        {/* Logo */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full blur-2xl opacity-30 scale-150" />
          <Image
            src="/medtech-logo.png"
            alt="MedTech Logo"
            width={100}
            height={100}
            className="relative drop-shadow-2xl"
          />
        </div>

        {/* Title */}
        <div className="space-y-4">
          <div className="flex items-center justify-center gap-3">
            <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10">
              Technical Deep Dive
            </Badge>
            <Badge variant="outline" className="border-purple-500/50 text-purple-400 bg-purple-500/10">
              WavLM + Custom Transformer
            </Badge>
          </div>

          <h1 className="text-6xl md:text-7xl font-bold">
            <span className="bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              VoxFormer
            </span>
          </h1>

          <p className="text-2xl text-slate-400 font-light">
            Elite Speech-to-Text System for Game AI
          </p>

          <p className="text-lg text-slate-500 max-w-2xl mx-auto">
            WavLM backbone + custom Zipformer encoder + Transformer decoder
          </p>
        </div>

        {/* Model specs preview */}
        <div className="flex flex-wrap justify-center gap-6 mt-8">
          {[
            { label: "WavLM", desc: "Feature Extractor" },
            { label: "Zipformer", desc: "6-Block Encoder" },
            { label: "Transformer", desc: "4-Layer Decoder" },
            { label: "Hybrid Loss", desc: "CTC + CE" },
            { label: "Streaming", desc: "<200ms Latency" }
          ].map((item) => (
            <div key={item.label} className="text-center">
              <div className="text-xl font-bold text-white">{item.label}</div>
              <div className="text-xs text-slate-500 uppercase tracking-wider">{item.desc}</div>
            </div>
          ))}
        </div>

        {/* Key metrics */}
        <div className="flex justify-center gap-12 mt-6">
          <div className="text-center">
            <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              &lt;3.5%
            </div>
            <div className="text-sm text-slate-500">WER (LibriSpeech)</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
              $20
            </div>
            <div className="text-sm text-slate-500">Training Cost</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
              7 Days
            </div>
            <div className="text-sm text-slate-500">Development Time</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-500 bg-clip-text text-transparent">
              142M
            </div>
            <div className="text-sm text-slate-500">Parameters</div>
          </div>
        </div>

        {/* Waveform visualization */}
        <div className="w-full max-w-3xl h-20 relative mt-8">
          <svg viewBox="0 0 400 60" className="w-full h-full">
            <defs>
              <linearGradient id="waveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#06b6d4" />
                <stop offset="50%" stopColor="#a855f7" />
                <stop offset="100%" stopColor="#ec4899" />
              </linearGradient>
            </defs>
            {/* Animated waveform bars - using deterministic values to avoid hydration mismatch */}
            {Array.from({ length: 80 }, (_, i) => {
              // Deterministic pseudo-random using sin functions, rounded to avoid float precision issues
              const pseudoRandom = Math.round((Math.abs(Math.sin(i * 12.9898) * 43758.5453) % 1) * 100) / 100;
              const height = Math.round((10 + Math.sin(i * 0.3) * 15 + pseudoRandom * 10) * 100) / 100;
              const opacity = Math.round((0.6 + Math.abs(Math.sin(i * 7.919)) * 0.4) * 100) / 100;
              return (
                <rect
                  key={i}
                  x={i * 5}
                  y={Math.round((30 - height / 2) * 100) / 100}
                  width="3"
                  height={height}
                  fill="url(#waveGradient)"
                  opacity={opacity}
                  rx="1"
                  className="animate-pulse"
                  style={{ animationDelay: `${i * 50}ms` }}
                />
              );
            })}
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs text-slate-500 font-mono bg-slate-900/80 px-3 py-1 rounded">
              Voice Input Stream → WavLM → Zipformer → Text
            </span>
          </div>
        </div>

        {/* Version and date */}
        <div className="text-xs text-slate-600 font-mono">
          Architecture v2.0 | December 2025 | AI-Accelerated Development
        </div>
      </div>
    </TechSlideWrapper>
  );
}
