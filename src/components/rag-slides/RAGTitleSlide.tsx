"use client";

import Image from "next/image";
import { TechSlideWrapper } from "../tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface RAGTitleSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function RAGTitleSlide({ slideNumber, totalSlides }: RAGTitleSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="flex flex-col items-center justify-center h-full text-center space-y-8">
        {/* Logo */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full blur-2xl opacity-30 scale-150" />
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
            <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
              Technical Deep Dive
            </Badge>
            <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10">
              Production Architecture
            </Badge>
          </div>

          <h1 className="text-6xl md:text-7xl font-bold">
            <span className="bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500 bg-clip-text text-transparent">
              Advanced RAG
            </span>
          </h1>

          <p className="text-2xl text-slate-400 font-light">
            Retrieval-Augmented Generation for 3D Game Development
          </p>
        </div>

        {/* Model specs preview */}
        <div className="flex flex-wrap justify-center gap-6 mt-8">
          {[
            { label: "3,885", desc: "Documents" },
            { label: "MiniLM", desc: "384-dim Embeddings" },
            { label: "RRF", desc: "Hybrid Fusion" },
            { label: "BGE-Reranker", desc: "Cross-Encoder" },
            { label: "GPT-5.1", desc: "Generation" }
          ].map((item) => (
            <div key={item.label} className="text-center">
              <div className="text-xl font-bold text-white">{item.label}</div>
              <div className="text-xs text-slate-500 uppercase tracking-wider">{item.desc}</div>
            </div>
          ))}
        </div>

        {/* Data flow visualization */}
        <div className="w-full max-w-3xl h-20 relative mt-8">
          <svg viewBox="0 0 500 60" className="w-full h-full">
            <defs>
              <linearGradient id="ragGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#10b981" />
                <stop offset="50%" stopColor="#06b6d4" />
                <stop offset="100%" stopColor="#3b82f6" />
              </linearGradient>
              <marker id="ragArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="url(#ragGradient)"/>
              </marker>
            </defs>

            {/* Query box */}
            <rect x="10" y="15" width="70" height="30" rx="6" fill="#1e293b" stroke="#10b981" strokeWidth="1.5"/>
            <text x="45" y="35" textAnchor="middle" fill="#10b981" fontSize="10" fontFamily="monospace">Query</text>

            {/* Arrow 1 */}
            <path d="M85 30 L115 30" stroke="url(#ragGradient)" strokeWidth="2" markerEnd="url(#ragArrow)"/>

            {/* Retrieval box */}
            <rect x="120" y="10" width="80" height="40" rx="6" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="1.5"/>
            <text x="160" y="35" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="bold">Retrieve</text>

            {/* Arrow 2 */}
            <path d="M205 30 L235 30" stroke="url(#ragGradient)" strokeWidth="2" markerEnd="url(#ragArrow)"/>

            {/* Rerank box */}
            <rect x="240" y="10" width="80" height="40" rx="6" fill="#06b6d4" fillOpacity="0.2" stroke="#06b6d4" strokeWidth="1.5"/>
            <text x="280" y="35" textAnchor="middle" fill="#06b6d4" fontSize="10" fontWeight="bold">Rerank</text>

            {/* Arrow 3 */}
            <path d="M325 30 L355 30" stroke="url(#ragGradient)" strokeWidth="2" markerEnd="url(#ragArrow)"/>

            {/* Generate box */}
            <rect x="360" y="10" width="80" height="40" rx="6" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="1.5"/>
            <text x="400" y="35" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold">Generate</text>

            {/* Arrow 4 */}
            <path d="M445 30 L475 30" stroke="url(#ragGradient)" strokeWidth="2" markerEnd="url(#ragArrow)"/>

            {/* Answer indicator */}
            <circle cx="485" cy="30" r="8" fill="url(#ragGradient)"/>
            <text x="485" y="34" textAnchor="middle" fill="white" fontSize="8" fontWeight="bold">A</text>
          </svg>
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <span className="text-xs text-slate-500 font-mono bg-slate-900/80 px-3 py-1 rounded">
              Knowledge-Grounded Response Pipeline
            </span>
          </div>
        </div>

        {/* Version and date */}
        <div className="text-xs text-slate-600 font-mono">
          Production v1.0 | December 2025 | Live at 5.249.161.66:5000
        </div>
      </div>
    </TechSlideWrapper>
  );
}
