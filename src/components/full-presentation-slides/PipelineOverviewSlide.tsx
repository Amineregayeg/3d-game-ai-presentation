"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface PipelineOverviewSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function PipelineOverviewSlide({ slideNumber, totalSlides }: PipelineOverviewSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="System Overview">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-8">
          <h2 className="text-4xl font-bold text-white mb-2">
            End-to-End Pipeline: <span className="text-cyan-400">Voice</span> → <span className="text-purple-400">3D</span>
          </h2>
        </div>

        {/* Main Pipeline SVG */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 1000 320" className="w-full max-w-5xl h-auto">
            <defs>
              <linearGradient id="voiceGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="sttGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="ragGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#a855f7" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#9333ea" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="avatarGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#059669" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="blenderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#ea580c" stopOpacity="0.9"/>
              </linearGradient>
              <marker id="pipeArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
              </marker>
              <filter id="boxGlow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* Voice Input */}
            <g transform="translate(20, 110)">
              <rect width="130" height="100" rx="12" fill="url(#voiceGrad)" filter="url(#boxGlow)"/>
              <text x="65" y="35" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">VOICE</text>
              <text x="65" y="55" textAnchor="middle" fill="#e0f2fe" fontSize="11">Input</text>
              <circle cx="65" cy="78" r="16" fill="rgba(255,255,255,0.15)"/>
              <text x="65" y="84" textAnchor="middle" fill="white" fontSize="18">🎤</text>
            </g>

            {/* Arrow */}
            <path d="M 160 160 L 195 160" stroke="#64748b" strokeWidth="3" markerEnd="url(#pipeArrow)"/>

            {/* STT */}
            <g transform="translate(210, 110)">
              <rect width="130" height="100" rx="12" fill="url(#sttGrad)" filter="url(#boxGlow)"/>
              <text x="65" y="35" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">STT</text>
              <text x="65" y="55" textAnchor="middle" fill="#ede9fe" fontSize="11">VoxFormer</text>
              <rect x="20" y="65" width="90" height="24" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="65" y="82" textAnchor="middle" fill="#c4b5fd" fontSize="9">Speech → Text</text>
            </g>

            {/* Arrow */}
            <path d="M 350 160 L 385 160" stroke="#64748b" strokeWidth="3" markerEnd="url(#pipeArrow)"/>

            {/* RAG */}
            <g transform="translate(400, 110)">
              <rect width="130" height="100" rx="12" fill="url(#ragGrad)" filter="url(#boxGlow)"/>
              <text x="65" y="35" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">RAG</text>
              <text x="65" y="55" textAnchor="middle" fill="#f3e8ff" fontSize="11">Retrieval</text>
              <rect x="20" y="65" width="90" height="24" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="65" y="82" textAnchor="middle" fill="#d8b4fe" fontSize="9">Context → Response</text>
            </g>

            {/* Split arrows to Avatar and Blender */}
            <path d="M 540 140 L 590 100" stroke="#64748b" strokeWidth="3" markerEnd="url(#pipeArrow)"/>
            <path d="M 540 180 L 590 220" stroke="#64748b" strokeWidth="3" markerEnd="url(#pipeArrow)"/>

            {/* Avatar */}
            <g transform="translate(600, 50)">
              <rect width="130" height="100" rx="12" fill="url(#avatarGrad)" filter="url(#boxGlow)"/>
              <text x="65" y="35" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">AVATAR</text>
              <text x="65" y="55" textAnchor="middle" fill="#d1fae5" fontSize="11">TTS + Lip-sync</text>
              <rect x="20" y="65" width="90" height="24" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="65" y="82" textAnchor="middle" fill="#6ee7b7" fontSize="9">Audio + Video</text>
            </g>

            {/* Blender */}
            <g transform="translate(600, 170)">
              <rect width="130" height="100" rx="12" fill="url(#blenderGrad)" filter="url(#boxGlow)"/>
              <text x="65" y="35" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">BLENDER</text>
              <text x="65" y="55" textAnchor="middle" fill="#ffedd5" fontSize="11">MCP Execution</text>
              <rect x="20" y="65" width="90" height="24" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="65" y="82" textAnchor="middle" fill="#fdba74" fontSize="9">3D Generation</text>
            </g>

            {/* Parallel execution bracket */}
            <g transform="translate(750, 100)">
              <path d="M 0 0 L 20 0 L 20 120 L 0 120" fill="none" stroke="#ec4899" strokeWidth="2" strokeDasharray="4"/>
              <rect x="30" y="40" width="100" height="40" rx="6" fill="#ec4899" fillOpacity="0.15" stroke="#ec4899" strokeWidth="1"/>
              <text x="80" y="55" textAnchor="middle" fill="#ec4899" fontSize="10" fontWeight="bold">PARALLEL</text>
              <text x="80" y="72" textAnchor="middle" fill="#f9a8d4" fontSize="9">Simultaneous</text>
            </g>

            {/* Output arrows */}
            <path d="M 865 120 L 920 160" stroke="#64748b" strokeWidth="2" strokeDasharray="4"/>
            <path d="M 865 200 L 920 160" stroke="#64748b" strokeWidth="2" strokeDasharray="4"/>

            {/* Output */}
            <g transform="translate(920, 135)">
              <rect width="60" height="50" rx="8" fill="#1e293b" stroke="#10b981" strokeWidth="2"/>
              <text x="30" y="30" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="bold">OUTPUT</text>
            </g>
          </svg>
        </div>

        {/* Badges */}
        <div className="flex justify-center gap-4 mt-6">
          <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10 px-4 py-2">
            <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Parallel Execution
          </Badge>
          <Badge variant="outline" className="border-purple-500/50 text-purple-400 bg-purple-500/10 px-4 py-2">
            <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Real-time Constraints
          </Badge>
          <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10 px-4 py-2">
            <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            GPU Accelerated
          </Badge>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
