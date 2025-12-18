"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";

interface DeploymentArchSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function DeploymentArchSlide({ slideNumber, totalSlides }: DeploymentArchSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Architecture">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className="text-4xl font-bold text-white mb-2">
            Deployment & Component Architecture
          </h2>
        </div>

        {/* Architecture Diagram */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 1000 500" className="w-full max-w-6xl h-auto">
            <defs>
              <linearGradient id="browserGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.2"/>
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.05"/>
              </linearGradient>
              <linearGradient id="vpsGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#a855f7" stopOpacity="0.2"/>
                <stop offset="100%" stopColor="#a855f7" stopOpacity="0.05"/>
              </linearGradient>
              <linearGradient id="gpuGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.2"/>
                <stop offset="100%" stopColor="#f97316" stopOpacity="0.05"/>
              </linearGradient>
            </defs>

            {/* Layer Labels */}
            <text x="50" y="30" fill="#94a3b8" fontSize="12" fontWeight="bold">BROWSER</text>
            <text x="50" y="190" fill="#94a3b8" fontSize="12" fontWeight="bold">VPS BACKEND</text>
            <text x="50" y="360" fill="#94a3b8" fontSize="12" fontWeight="bold">GPU SERVER</text>

            {/* Browser Layer */}
            <g transform="translate(40, 40)">
              <rect width="920" height="130" rx="12" fill="url(#browserGrad)" stroke="#06b6d4" strokeWidth="2"/>

              {/* Mic Input */}
              <g transform="translate(30, 25)">
                <rect width="120" height="80" rx="8" fill="#0f172a" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="60" y="30" textAnchor="middle" fill="#06b6d4" fontSize="11" fontWeight="bold">Microphone</text>
                <text x="60" y="50" textAnchor="middle" fill="#67e8f9" fontSize="9">Voice Input</text>
                <text x="60" y="68" textAnchor="middle" fill="#94a3b8" fontSize="8">WebAudio API</text>
              </g>

              {/* UI */}
              <g transform="translate(180, 25)">
                <rect width="120" height="80" rx="8" fill="#0f172a" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="60" y="30" textAnchor="middle" fill="#06b6d4" fontSize="11" fontWeight="bold">React UI</text>
                <text x="60" y="50" textAnchor="middle" fill="#67e8f9" fontSize="9">Next.js 14</text>
                <text x="60" y="68" textAnchor="middle" fill="#94a3b8" fontSize="8">Tailwind + shadcn</text>
              </g>

              {/* Three.js */}
              <g transform="translate(330, 25)">
                <rect width="120" height="80" rx="8" fill="#0f172a" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="60" y="30" textAnchor="middle" fill="#06b6d4" fontSize="11" fontWeight="bold">Three.js</text>
                <text x="60" y="50" textAnchor="middle" fill="#67e8f9" fontSize="9">3D Preview</text>
                <text x="60" y="68" textAnchor="middle" fill="#94a3b8" fontSize="8">WebGL Renderer</text>
              </g>

              {/* Avatar Player */}
              <g transform="translate(480, 25)">
                <rect width="120" height="80" rx="8" fill="#0f172a" stroke="#10b981" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="60" y="30" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">Avatar</text>
                <text x="60" y="50" textAnchor="middle" fill="#6ee7b7" fontSize="9">Video Player</text>
                <text x="60" y="68" textAnchor="middle" fill="#94a3b8" fontSize="8">Lip-sync Playback</text>
              </g>

              {/* SSE Client */}
              <g transform="translate(630, 25)">
                <rect width="120" height="80" rx="8" fill="#0f172a" stroke="#ec4899" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="60" y="30" textAnchor="middle" fill="#ec4899" fontSize="11" fontWeight="bold">SSE Client</text>
                <text x="60" y="50" textAnchor="middle" fill="#f9a8d4" fontSize="9">Real-time Updates</text>
                <text x="60" y="68" textAnchor="middle" fill="#94a3b8" fontSize="8">EventSource</text>
              </g>

              {/* Waveform */}
              <g transform="translate(780, 25)">
                <rect width="120" height="80" rx="8" fill="#0f172a" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="60" y="30" textAnchor="middle" fill="#06b6d4" fontSize="11" fontWeight="bold">Waveform</text>
                <text x="60" y="50" textAnchor="middle" fill="#67e8f9" fontSize="9">Audio Viz</text>
                <text x="60" y="68" textAnchor="middle" fill="#94a3b8" fontSize="8">Canvas 2D</text>
              </g>
            </g>

            {/* VPS Layer */}
            <g transform="translate(40, 200)">
              <rect width="920" height="130" rx="12" fill="url(#vpsGrad)" stroke="#a855f7" strokeWidth="2"/>

              {/* API Gateway */}
              <g transform="translate(30, 25)">
                <rect width="140" height="80" rx="8" fill="#0f172a" stroke="#a855f7" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="70" y="30" textAnchor="middle" fill="#a855f7" fontSize="11" fontWeight="bold">API Gateway</text>
                <text x="70" y="50" textAnchor="middle" fill="#c4b5fd" fontSize="9">Flask + Gunicorn</text>
                <text x="70" y="68" textAnchor="middle" fill="#94a3b8" fontSize="8">Request Routing</text>
              </g>

              {/* Orchestrator */}
              <g transform="translate(200, 25)">
                <rect width="140" height="80" rx="8" fill="#0f172a" stroke="#a855f7" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="70" y="30" textAnchor="middle" fill="#a855f7" fontSize="11" fontWeight="bold">Orchestrator</text>
                <text x="70" y="50" textAnchor="middle" fill="#c4b5fd" fontSize="9">Pipeline Control</text>
                <text x="70" y="68" textAnchor="middle" fill="#94a3b8" fontSize="8">Async Coordinator</text>
              </g>

              {/* RAG Pipeline */}
              <g transform="translate(370, 25)">
                <rect width="180" height="80" rx="8" fill="#0f172a" stroke="#a855f7" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="90" y="25" textAnchor="middle" fill="#a855f7" fontSize="11" fontWeight="bold">RAG Pipeline</text>
                <text x="90" y="45" textAnchor="middle" fill="#c4b5fd" fontSize="9">BGE-M3 + BM25 + Rerank</text>
                <text x="90" y="62" textAnchor="middle" fill="#94a3b8" fontSize="8">pgvector / LlamaIndex</text>
                <rect x="10" y="68" width="160" height="8" rx="2" fill="#a855f7" fillOpacity="0.3"/>
              </g>

              {/* Avatar TTS */}
              <g transform="translate(580, 25)">
                <rect width="140" height="80" rx="8" fill="#0f172a" stroke="#10b981" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="70" y="30" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">Avatar TTS</text>
                <text x="70" y="50" textAnchor="middle" fill="#6ee7b7" fontSize="9">ElevenLabs API</text>
                <text x="70" y="68" textAnchor="middle" fill="#94a3b8" fontSize="8">Audio Generation</text>
              </g>

              {/* Database */}
              <g transform="translate(750, 25)">
                <rect width="140" height="80" rx="8" fill="#0f172a" stroke="#64748b" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="70" y="30" textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="bold">PostgreSQL</text>
                <text x="70" y="50" textAnchor="middle" fill="#cbd5e1" fontSize="9">+ pgvector</text>
                <text x="70" y="68" textAnchor="middle" fill="#64748b" fontSize="8">Vector Store</text>
              </g>
            </g>

            {/* GPU Layer */}
            <g transform="translate(40, 360)">
              <rect width="920" height="130" rx="12" fill="url(#gpuGrad)" stroke="#f97316" strokeWidth="2"/>

              {/* STT Inference */}
              <g transform="translate(30, 25)">
                <rect width="200" height="80" rx="8" fill="#0f172a" stroke="#8b5cf6" strokeWidth="1.5"/>
                <text x="100" y="25" textAnchor="middle" fill="#8b5cf6" fontSize="11" fontWeight="bold">STT Inference</text>
                <text x="100" y="45" textAnchor="middle" fill="#a78bfa" fontSize="9">VoxFormer / Whisper</text>
                <text x="100" y="62" textAnchor="middle" fill="#94a3b8" fontSize="8">PyTorch + CUDA</text>
                <rect x="10" y="68" width="180" height="8" rx="2" fill="#8b5cf6" fillOpacity="0.3"/>
              </g>

              {/* MuseTalk */}
              <g transform="translate(260, 25)">
                <rect width="160" height="80" rx="8" fill="#0f172a" stroke="#10b981" strokeWidth="1.5"/>
                <text x="80" y="25" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">MuseTalk</text>
                <text x="80" y="45" textAnchor="middle" fill="#6ee7b7" fontSize="9">Lip-sync Generation</text>
                <text x="80" y="62" textAnchor="middle" fill="#94a3b8" fontSize="8">Video Synthesis</text>
                <rect x="10" y="68" width="140" height="8" rx="2" fill="#10b981" fillOpacity="0.3"/>
              </g>

              {/* Blender Headless */}
              <g transform="translate(450, 25)">
                <rect width="200" height="80" rx="8" fill="#0f172a" stroke="#f97316" strokeWidth="1.5"/>
                <text x="100" y="25" textAnchor="middle" fill="#f97316" fontSize="11" fontWeight="bold">Blender Headless</text>
                <text x="100" y="45" textAnchor="middle" fill="#fdba74" fontSize="9">bpy + MCP Server</text>
                <text x="100" y="62" textAnchor="middle" fill="#94a3b8" fontSize="8">3D Generation</text>
                <rect x="10" y="68" width="180" height="8" rx="2" fill="#f97316" fillOpacity="0.3"/>
              </g>

              {/* GPU Info */}
              <g transform="translate(680, 25)">
                <rect width="210" height="80" rx="8" fill="#0f172a" stroke="#f97316" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="105" y="25" textAnchor="middle" fill="#f97316" fontSize="11" fontWeight="bold">RTX 4090</text>
                <text x="105" y="45" textAnchor="middle" fill="#fdba74" fontSize="9">24GB VRAM</text>
                <text x="105" y="62" textAnchor="middle" fill="#94a3b8" fontSize="8">Vast.ai Instance</text>
                <rect x="10" y="68" width="190" height="8" rx="2" fill="#f97316" fillOpacity="0.2"/>
              </g>
            </g>

            {/* Connection arrows */}
            <path d="M 500 170 L 500 200" stroke="#64748b" strokeWidth="2" strokeDasharray="4"/>
            <path d="M 500 330 L 500 360" stroke="#64748b" strokeWidth="2" strokeDasharray="4"/>
          </svg>
        </div>

        {/* Legend */}
        <div className="flex justify-center gap-6 mt-2">
          {[
            { color: "#8b5cf6", label: "STT" },
            { color: "#a855f7", label: "RAG" },
            { color: "#10b981", label: "Avatar" },
            { color: "#f97316", label: "Blender" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
              <span className="text-xs text-slate-400">{item.label}</span>
            </div>
          ))}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
