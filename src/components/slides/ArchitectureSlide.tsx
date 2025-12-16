"use client";

import { SlideWrapper } from "./SlideWrapper";

interface ArchitectureSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function ArchitectureSlide({ slideNumber, totalSlides }: ArchitectureSlideProps) {
  return (
    <SlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h2 className="text-5xl font-bold text-white">
            Technical <span className="bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent">Architecture</span>
          </h2>
          <p className="text-xl text-slate-400">
            End-to-end pipeline for voice-controlled 3D generation
          </p>
        </div>

        {/* Architecture diagram */}
        <div className="relative">
          {/* Flow diagram */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            {/* User Input */}
            <div className="flex flex-col items-center">
              <div className="w-24 h-24 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center shadow-lg shadow-green-500/20">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <span className="mt-2 text-sm text-slate-400 font-medium">User</span>
              <span className="text-xs text-slate-500">Voice Input</span>
            </div>

            <div className="text-cyan-500 text-2xl rotate-90 md:rotate-0">→</div>

            {/* STT Module */}
            <div className="flex flex-col items-center">
              <div className="w-24 h-24 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl flex items-center justify-center shadow-lg shadow-cyan-500/20">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              </div>
              <span className="mt-2 text-sm text-slate-400 font-medium">STT</span>
              <span className="text-xs text-slate-500">Local/Finetuned</span>
            </div>

            <div className="text-cyan-500 text-2xl rotate-90 md:rotate-0">→</div>

            {/* RAG System */}
            <div className="flex flex-col items-center">
              <div className="w-24 h-24 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center shadow-lg shadow-purple-500/20">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <span className="mt-2 text-sm text-slate-400 font-medium">RAG</span>
              <span className="text-xs text-slate-500">Advanced</span>
            </div>

            <div className="text-cyan-500 text-2xl rotate-90 md:rotate-0">→</div>

            {/* TTS + Avatar */}
            <div className="flex flex-col items-center">
              <div className="w-24 h-24 bg-gradient-to-br from-orange-500 to-red-600 rounded-2xl flex items-center justify-center shadow-lg shadow-orange-500/20">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                </svg>
              </div>
              <span className="mt-2 text-sm text-slate-400 font-medium">TTS + Avatar</span>
              <span className="text-xs text-slate-500">LipSync</span>
            </div>

            <div className="text-cyan-500 text-2xl rotate-90 md:rotate-0">→</div>

            {/* Blender MCP */}
            <div className="flex flex-col items-center">
              <div className="w-24 h-24 bg-gradient-to-br from-amber-500 to-yellow-600 rounded-2xl flex items-center justify-center shadow-lg shadow-amber-500/20">
                <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                </svg>
              </div>
              <span className="mt-2 text-sm text-slate-400 font-medium">Blender</span>
              <span className="text-xs text-slate-500">MCP</span>
            </div>
          </div>

          {/* Labels */}
          <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl text-center">
              <div className="text-cyan-400 font-semibold">Speech Input</div>
              <div className="text-xs text-slate-500 mt-1">Natural language commands</div>
            </div>
            <div className="p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl text-center">
              <div className="text-purple-400 font-semibold">AI Processing</div>
              <div className="text-xs text-slate-500 mt-1">Context-aware reasoning</div>
            </div>
            <div className="p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl text-center">
              <div className="text-orange-400 font-semibold">Avatar Response</div>
              <div className="text-xs text-slate-500 mt-1">Visual + audio feedback</div>
            </div>
            <div className="p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl text-center">
              <div className="text-amber-400 font-semibold">3D Output</div>
              <div className="text-xs text-slate-500 mt-1">Generated assets</div>
            </div>
          </div>
        </div>
      </div>
    </SlideWrapper>
  );
}
