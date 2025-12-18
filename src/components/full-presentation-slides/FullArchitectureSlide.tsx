"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface FullArchitectureSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function FullArchitectureSlide({ slideNumber, totalSlides }: FullArchitectureSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="System Architecture">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="text-3xl font-bold text-white mb-2">End-to-End Pipeline Architecture</h2>
          <p className="text-slate-400">Voice → Text → Knowledge → Response → Execution</p>
        </div>

        {/* Architecture SVG Diagram */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 1000 500" className="w-full max-w-5xl h-auto">
            {/* Background grid */}
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" strokeWidth="0.5"/>
              </pattern>

              {/* Gradients */}
              <linearGradient id="cyanGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#a855f7" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="emeraldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#059669" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="orangeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#ea580c" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="pinkGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ec4899" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#db2777" stopOpacity="0.8"/>
              </linearGradient>

              {/* Arrow marker */}
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
              </marker>
            </defs>

            <rect width="100%" height="100%" fill="url(#grid)"/>

            {/* USER INPUT - Left side */}
            <g transform="translate(50, 200)">
              <rect x="0" y="0" width="120" height="100" rx="12" fill="url(#cyanGrad)" fillOpacity="0.15" stroke="#06b6d4" strokeWidth="2"/>
              <text x="60" y="35" textAnchor="middle" fill="#06b6d4" fontSize="14" fontWeight="bold">USER</text>
              <text x="60" y="55" textAnchor="middle" fill="#94a3b8" fontSize="11">Voice Input</text>
              {/* Mic icon */}
              <circle cx="60" cy="75" r="12" fill="#06b6d4" fillOpacity="0.2"/>
              <text x="60" y="80" textAnchor="middle" fill="#06b6d4" fontSize="14">🎤</text>
            </g>

            {/* Arrow to STT */}
            <path d="M 180 250 L 230 250" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)"/>

            {/* STT ENGINE */}
            <g transform="translate(240, 180)">
              <rect x="0" y="0" width="140" height="140" rx="12" fill="url(#cyanGrad)" fillOpacity="0.1" stroke="#06b6d4" strokeWidth="2"/>
              <text x="70" y="30" textAnchor="middle" fill="#06b6d4" fontSize="14" fontWeight="bold">STT ENGINE</text>

              {/* VoxFormer box */}
              <rect x="15" y="45" width="110" height="35" rx="6" fill="#0f172a" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.5"/>
              <text x="70" y="67" textAnchor="middle" fill="#06b6d4" fontSize="10">VoxFormer (Custom)</text>

              {/* Whisper box */}
              <rect x="15" y="90" width="110" height="35" rx="6" fill="#0f172a" stroke="#0ea5e9" strokeWidth="1" strokeOpacity="0.5"/>
              <text x="70" y="112" textAnchor="middle" fill="#0ea5e9" fontSize="10">Whisper (OpenAI)</text>
            </g>

            {/* Arrow to RAG */}
            <path d="M 390 250 L 440 250" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)"/>

            {/* RAG PIPELINE */}
            <g transform="translate(450, 140)">
              <rect x="0" y="0" width="160" height="220" rx="12" fill="url(#purpleGrad)" fillOpacity="0.1" stroke="#a855f7" strokeWidth="2"/>
              <text x="80" y="25" textAnchor="middle" fill="#a855f7" fontSize="14" fontWeight="bold">RAG PIPELINE</text>

              {/* Pipeline stages */}
              {[
                "1. Orchestrator",
                "2. Query Analysis",
                "3. Dense Search",
                "4. Sparse Search",
                "5. RRF Fusion",
                "6. Reranking",
                "7. Generation",
                "8. Validation"
              ].map((stage, i) => (
                <g key={stage} transform={`translate(15, ${40 + i * 22})`}>
                  <rect x="0" y="0" width="130" height="18" rx="3" fill="#0f172a" fillOpacity="0.8"/>
                  <text x="8" y="13" fill="#c4b5fd" fontSize="9">{stage}</text>
                </g>
              ))}
            </g>

            {/* Arrow to Avatar */}
            <path d="M 620 250 L 670 180" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)"/>
            {/* Arrow to Blender */}
            <path d="M 620 250 L 670 320" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)"/>

            {/* AVATAR + TTS */}
            <g transform="translate(680, 100)">
              <rect x="0" y="0" width="140" height="120" rx="12" fill="url(#emeraldGrad)" fillOpacity="0.1" stroke="#10b981" strokeWidth="2"/>
              <text x="70" y="25" textAnchor="middle" fill="#10b981" fontSize="14" fontWeight="bold">AVATAR</text>

              <rect x="15" y="40" width="110" height="30" rx="6" fill="#0f172a" stroke="#10b981" strokeWidth="1" strokeOpacity="0.5"/>
              <text x="70" y="60" textAnchor="middle" fill="#10b981" fontSize="10">ElevenLabs TTS</text>

              <rect x="15" y="78" width="110" height="30" rx="6" fill="#0f172a" stroke="#34d399" strokeWidth="1" strokeOpacity="0.5"/>
              <text x="70" y="98" textAnchor="middle" fill="#34d399" fontSize="10">MuseTalk Lip-sync</text>
            </g>

            {/* BLENDER MCP */}
            <g transform="translate(680, 280)">
              <rect x="0" y="0" width="140" height="120" rx="12" fill="url(#orangeGrad)" fillOpacity="0.1" stroke="#f97316" strokeWidth="2"/>
              <text x="70" y="25" textAnchor="middle" fill="#f97316" fontSize="14" fontWeight="bold">BLENDER</text>

              <rect x="15" y="40" width="110" height="30" rx="6" fill="#0f172a" stroke="#f97316" strokeWidth="1" strokeOpacity="0.5"/>
              <text x="70" y="60" textAnchor="middle" fill="#f97316" fontSize="10">MCP Server (GPU)</text>

              <rect x="15" y="78" width="110" height="30" rx="6" fill="#0f172a" stroke="#fb923c" strokeWidth="1" strokeOpacity="0.5"/>
              <text x="70" y="98" textAnchor="middle" fill="#fb923c" fontSize="10">Three.js Preview</text>
            </g>

            {/* Simultaneous indicator */}
            <g transform="translate(830, 210)">
              <rect x="0" y="0" width="100" height="80" rx="8" fill="url(#pinkGrad)" fillOpacity="0.1" stroke="#ec4899" strokeWidth="2" strokeDasharray="4"/>
              <text x="50" y="25" textAnchor="middle" fill="#ec4899" fontSize="11" fontWeight="bold">PARALLEL</text>
              <text x="50" y="45" textAnchor="middle" fill="#f9a8d4" fontSize="9">Avatar speaks</text>
              <text x="50" y="58" textAnchor="middle" fill="#f9a8d4" fontSize="9">+</text>
              <text x="50" y="71" textAnchor="middle" fill="#f9a8d4" fontSize="9">Blender executes</text>
            </g>

            {/* Arrows to parallel box */}
            <path d="M 820 160 L 840 200" stroke="#64748b" strokeWidth="1" strokeDasharray="4"/>
            <path d="M 820 340 L 840 290" stroke="#64748b" strokeWidth="1" strokeDasharray="4"/>

            {/* OUTPUT arrow */}
            <path d="M 930 250 L 970 250" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)"/>
            <text x="985" y="255" fill="#94a3b8" fontSize="12">OUTPUT</text>

            {/* Labels for sections */}
            <text x="110" y="170" textAnchor="middle" fill="#64748b" fontSize="10">BROWSER</text>
            <text x="310" y="170" textAnchor="middle" fill="#64748b" fontSize="10">GPU SERVER</text>
            <text x="530" y="130" textAnchor="middle" fill="#64748b" fontSize="10">VPS BACKEND</text>
            <text x="750" y="90" textAnchor="middle" fill="#64748b" fontSize="10">GPU + BROWSER</text>
          </svg>
        </div>

        {/* Legend */}
        <div className="flex justify-center gap-6 mt-4">
          {[
            { color: "cyan", label: "Speech Input" },
            { color: "purple", label: "RAG Processing" },
            { color: "emerald", label: "Avatar Response" },
            { color: "orange", label: "3D Execution" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full bg-${item.color}-500`} />
              <span className="text-xs text-slate-400">{item.label}</span>
            </div>
          ))}
          <Badge variant="outline" className="border-pink-500/50 text-pink-400 bg-pink-500/10 text-xs">
            Simultaneous Execution
          </Badge>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
