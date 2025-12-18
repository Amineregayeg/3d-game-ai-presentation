"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Mic, Brain, MessageSquare, Box } from "lucide-react";

interface FullPipelineTitleSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function FullPipelineTitleSlide({ slideNumber, totalSlides }: FullPipelineTitleSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="flex flex-col items-center justify-center h-full text-center space-y-8">
        {/* Icon cluster */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full blur-3xl opacity-20 scale-150" />
          <div className="relative flex items-center gap-4">
            <div className="p-4 bg-cyan-500/10 rounded-2xl border border-cyan-500/30">
              <Mic className="w-10 h-10 text-cyan-400" />
            </div>
            <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <div className="p-4 bg-purple-500/10 rounded-2xl border border-purple-500/30">
              <Brain className="w-10 h-10 text-purple-400" />
            </div>
            <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <div className="p-4 bg-emerald-500/10 rounded-2xl border border-emerald-500/30">
              <MessageSquare className="w-10 h-10 text-emerald-400" />
            </div>
            <svg className="w-8 h-8 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <div className="p-4 bg-orange-500/10 rounded-2xl border border-orange-500/30">
              <Box className="w-10 h-10 text-orange-400" />
            </div>
          </div>
        </div>

        {/* Title */}
        <div className="space-y-4">
          <div className="flex items-center justify-center gap-3 flex-wrap">
            <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10">
              Speech-to-Text
            </Badge>
            <Badge variant="outline" className="border-purple-500/50 text-purple-400 bg-purple-500/10">
              Advanced RAG
            </Badge>
            <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
              Avatar + TTS
            </Badge>
            <Badge variant="outline" className="border-orange-500/50 text-orange-400 bg-orange-500/10">
              Blender MCP
            </Badge>
          </div>

          <h1 className="text-5xl md:text-6xl font-bold">
            <span className="bg-gradient-to-r from-cyan-400 via-purple-500 to-orange-500 bg-clip-text text-transparent">
              Full Pipeline Architecture
            </span>
          </h1>

          <p className="text-2xl text-slate-400 font-light">
            3D Game AI Assistant - End-to-End System
          </p>

          <p className="text-lg text-slate-500 max-w-3xl mx-auto">
            Voice Input → STT → RAG Knowledge → Avatar Response + Blender Execution
          </p>
        </div>

        {/* Pipeline metrics */}
        <div className="flex flex-wrap justify-center gap-8 mt-8">
          {[
            { icon: Mic, label: "STT", desc: "VoxFormer/Whisper", color: "cyan" },
            { icon: Brain, label: "RAG", desc: "8-Stage Pipeline", color: "purple" },
            { icon: MessageSquare, label: "Avatar", desc: "ElevenLabs + Lip-sync", color: "emerald" },
            { icon: Box, label: "Blender", desc: "MCP Execution", color: "orange" },
          ].map((item) => (
            <div key={item.label} className="text-center group">
              <div className={`mx-auto mb-2 p-3 rounded-xl bg-${item.color}-500/10 border border-${item.color}-500/30 transition-all group-hover:scale-110`}>
                <item.icon className={`w-6 h-6 text-${item.color}-400`} />
              </div>
              <div className="text-lg font-bold text-white">{item.label}</div>
              <div className="text-xs text-slate-500 uppercase tracking-wider">{item.desc}</div>
            </div>
          ))}
        </div>

        {/* Key metrics */}
        <div className="flex justify-center gap-12 mt-6">
          <div className="text-center">
            <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              &lt;8s
            </div>
            <div className="text-sm text-slate-500">End-to-End Latency</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
              0.92
            </div>
            <div className="text-sm text-slate-500">RAGAS Score</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-500 bg-clip-text text-transparent">
              Real-time
            </div>
            <div className="text-sm text-slate-500">3D Execution</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent">
              GPU
            </div>
            <div className="text-sm text-slate-500">Accelerated</div>
          </div>
        </div>

        {/* Version */}
        <div className="text-xs text-slate-600 font-mono mt-4">
          Full Demo v1.0 | December 2024 | Voice → Knowledge → Action
        </div>
      </div>
    </TechSlideWrapper>
  );
}
