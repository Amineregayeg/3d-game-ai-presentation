"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface WhySTTSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function WhySTTSlide({ slideNumber, totalSlides }: WhySTTSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Core Contribution">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="text-4xl font-bold text-white mb-2">
            Why <span className="text-cyan-400">Speech-to-Text</span> Is the Core{" "}
            <span className="text-purple-400">Deep Learning</span> Focus
          </h2>
        </div>

        {/* Pipeline with STT Highlighted */}
        <div className="flex-1 flex flex-col items-center justify-center gap-8">
          <svg viewBox="0 0 900 200" className="w-full max-w-4xl h-auto">
            <defs>
              <linearGradient id="sttHighlight" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="1"/>
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="1"/>
              </linearGradient>
              <filter id="highlightGlow">
                <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <marker id="arrow2" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
              </marker>
            </defs>

            {/* Voice Input (dimmed) */}
            <g transform="translate(20, 60)">
              <rect width="120" height="80" rx="10" fill="#1e293b" stroke="#475569" strokeWidth="1.5" opacity="0.6"/>
              <text x="60" y="45" textAnchor="middle" fill="#94a3b8" fontSize="14">Voice</text>
            </g>

            <path d="M 150 100 L 180 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow2)"/>

            {/* STT (HIGHLIGHTED) */}
            <g transform="translate(190, 40)">
              <rect width="180" height="120" rx="12" fill="url(#sttHighlight)" filter="url(#highlightGlow)"/>
              <text x="90" y="40" textAnchor="middle" fill="white" fontSize="18" fontWeight="bold">STT</text>
              <text x="90" y="65" textAnchor="middle" fill="#e0f2fe" fontSize="12">VoxFormer</text>
              <rect x="20" y="80" width="140" height="28" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="90" y="99" textAnchor="middle" fill="#67e8f9" fontSize="10">Custom Built & Trained</text>
            </g>

            <path d="M 380 100 L 420 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow2)"/>

            {/* RAG (dimmed) */}
            <g transform="translate(430, 60)">
              <rect width="120" height="80" rx="10" fill="#1e293b" stroke="#475569" strokeWidth="1.5" opacity="0.6"/>
              <text x="60" y="45" textAnchor="middle" fill="#94a3b8" fontSize="14">RAG</text>
            </g>

            <path d="M 560 100 L 600 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow2)"/>

            {/* Avatar (dimmed) */}
            <g transform="translate(610, 60)">
              <rect width="120" height="80" rx="10" fill="#1e293b" stroke="#475569" strokeWidth="1.5" opacity="0.6"/>
              <text x="60" y="45" textAnchor="middle" fill="#94a3b8" fontSize="14">Avatar</text>
            </g>

            <path d="M 740 100 L 780 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow2)"/>

            {/* Blender (dimmed) */}
            <g transform="translate(790, 60)">
              <rect width="100" height="80" rx="10" fill="#1e293b" stroke="#475569" strokeWidth="1.5" opacity="0.6"/>
              <text x="50" y="45" textAnchor="middle" fill="#94a3b8" fontSize="14">Blender</text>
            </g>

            {/* Error propagation indicator */}
            <g transform="translate(280, 170)">
              <path d="M 0 0 L 500 0" stroke="#ef4444" strokeWidth="1" strokeDasharray="4" opacity="0.6"/>
              <text x="250" y="20" textAnchor="middle" fill="#fca5a5" fontSize="10">Errors propagate downstream →</text>
            </g>
          </svg>

          {/* Key Points */}
          <div className="grid grid-cols-2 gap-6 max-w-4xl w-full">
            <div className="p-5 bg-cyan-500/10 border border-cyan-500/30 rounded-xl">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center">
                  <span className="text-cyan-400 font-bold">1</span>
                </div>
                <h3 className="text-lg font-semibold text-white">First AI Component</h3>
              </div>
              <p className="text-slate-400 text-sm">Entry point of the entire pipeline - accuracy here determines downstream quality</p>
            </div>

            <div className="p-5 bg-purple-500/10 border border-purple-500/30 rounded-xl">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
                  <span className="text-purple-400 font-bold">2</span>
                </div>
                <h3 className="text-lg font-semibold text-white">Error Propagation</h3>
              </div>
              <p className="text-slate-400 text-sm">STT errors cascade through RAG retrieval, response generation, and 3D execution</p>
            </div>

            <div className="p-5 bg-emerald-500/10 border border-emerald-500/30 rounded-xl">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                  <span className="text-emerald-400 font-bold">3</span>
                </div>
                <h3 className="text-lg font-semibold text-white">VoxFormer: Custom Built</h3>
              </div>
              <p className="text-slate-400 text-sm">Novel architecture designed and trained by our team from scratch</p>
            </div>

            <div className="p-5 bg-amber-500/10 border border-amber-500/30 rounded-xl">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
                  <span className="text-amber-400 font-bold">4</span>
                </div>
                <h3 className="text-lg font-semibold text-white">Whisper: Demo Fallback</h3>
              </div>
              <p className="text-slate-400 text-sm">OpenAI Whisper used only as production fallback while training completes</p>
            </div>
          </div>
        </div>

        {/* Badge */}
        <div className="flex justify-center mt-4">
          <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10 px-4 py-2 text-sm">
            VoxFormer = Core Deep Learning Contribution
          </Badge>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
