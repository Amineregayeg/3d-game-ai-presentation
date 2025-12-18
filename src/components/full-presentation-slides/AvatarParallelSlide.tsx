"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface AvatarParallelSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function AvatarParallelSlide({ slideNumber, totalSlides }: AvatarParallelSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Avatar & Execution">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className="text-4xl font-bold text-white mb-2">
            Avatar Response in <span className="text-emerald-400">Parallel</span> with <span className="text-orange-400">3D Execution</span>
          </h2>
        </div>

        {/* Parallel Timeline Visualization */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 900 350" className="w-full max-w-5xl h-auto">
            <defs>
              <linearGradient id="avatarTimeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#059669" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="blenderTimeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#ea580c" stopOpacity="0.9"/>
              </linearGradient>
              <marker id="timeArrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#64748b"/>
              </marker>
            </defs>

            {/* Timeline axis */}
            <line x1="100" y1="280" x2="800" y2="280" stroke="#475569" strokeWidth="2"/>
            <text x="850" y="285" fill="#64748b" fontSize="12">time</text>

            {/* Time markers */}
            {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((t) => (
              <g key={t} transform={`translate(${100 + t * 80}, 280)`}>
                <line x1="0" y1="0" x2="0" y2="8" stroke="#475569" strokeWidth="1"/>
                <text x="0" y="22" textAnchor="middle" fill="#64748b" fontSize="10">{t}s</text>
              </g>
            ))}

            {/* RAG Response arrives - vertical line */}
            <line x1="100" y1="60" x2="100" y2="270" stroke="#a855f7" strokeWidth="2" strokeDasharray="4"/>
            <text x="100" y="50" textAnchor="middle" fill="#a855f7" fontSize="11" fontWeight="bold">RAG Response</text>

            {/* Avatar Track */}
            <g transform="translate(0, 100)">
              <text x="50" y="35" textAnchor="middle" fill="#10b981" fontSize="12" fontWeight="bold">Avatar</text>

              {/* TTS Generation */}
              <rect x="100" y="10" width="160" height="50" rx="8" fill="url(#avatarTimeGrad)"/>
              <text x="180" y="30" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">ElevenLabs TTS</text>
              <text x="180" y="48" textAnchor="middle" fill="#d1fae5" fontSize="9">~2s generation</text>

              {/* Lip-sync Generation */}
              <rect x="260" y="10" width="200" height="50" rx="8" fill="#10b981" fillOpacity="0.7"/>
              <text x="360" y="30" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">MuseTalk Lip-sync</text>
              <text x="360" y="48" textAnchor="middle" fill="#d1fae5" fontSize="9">~2.5s rendering</text>

              {/* Playback */}
              <rect x="460" y="10" width="240" height="50" rx="8" fill="#10b981" fillOpacity="0.5"/>
              <text x="580" y="30" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Avatar Playback</text>
              <text x="580" y="48" textAnchor="middle" fill="#d1fae5" fontSize="9">Speaking to user</text>
            </g>

            {/* Blender Track */}
            <g transform="translate(0, 180)">
              <text x="50" y="35" textAnchor="middle" fill="#f97316" fontSize="12" fontWeight="bold">Blender</text>

              {/* Code Generation */}
              <rect x="100" y="10" width="120" height="50" rx="8" fill="url(#blenderTimeGrad)"/>
              <text x="160" y="30" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Code Gen</text>
              <text x="160" y="48" textAnchor="middle" fill="#ffedd5" fontSize="9">bpy script</text>

              {/* MCP Execution */}
              <rect x="220" y="10" width="180" height="50" rx="8" fill="#f97316" fillOpacity="0.7"/>
              <text x="310" y="30" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">MCP Execution</text>
              <text x="310" y="48" textAnchor="middle" fill="#ffedd5" fontSize="9">Headless Blender</text>

              {/* Preview Update */}
              <rect x="400" y="10" width="140" height="50" rx="8" fill="#f97316" fillOpacity="0.5"/>
              <text x="470" y="30" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Three.js Update</text>
              <text x="470" y="48" textAnchor="middle" fill="#ffedd5" fontSize="9">Live preview</text>
            </g>

            {/* Parallel bracket */}
            <g transform="translate(720, 100)">
              <path d="M 0 0 L 20 0 L 20 130 L 0 130" fill="none" stroke="#ec4899" strokeWidth="2"/>
              <rect x="30" y="45" width="120" height="40" rx="6" fill="#ec4899" fillOpacity="0.15" stroke="#ec4899" strokeWidth="1"/>
              <text x="90" y="62" textAnchor="middle" fill="#ec4899" fontSize="10" fontWeight="bold">PARALLEL</text>
              <text x="90" y="78" textAnchor="middle" fill="#f9a8d4" fontSize="9">Same time window</text>
            </g>

            {/* Result ready marker */}
            <g transform="translate(540, 280)">
              <circle cx="0" cy="0" r="8" fill="#10b981"/>
              <text x="0" y="40" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">Result Ready</text>
              <text x="0" y="55" textAnchor="middle" fill="#64748b" fontSize="9">~5.5s total</text>
            </g>

            {/* Latency annotation */}
            <g transform="translate(600, 60)">
              <rect x="0" y="0" width="200" height="50" rx="8" fill="#ec4899" fillOpacity="0.1" stroke="#ec4899" strokeWidth="1" strokeDasharray="4"/>
              <text x="100" y="22" textAnchor="middle" fill="#ec4899" fontSize="10" fontWeight="bold">Parallelism reduces</text>
              <text x="100" y="40" textAnchor="middle" fill="#ec4899" fontSize="10" fontWeight="bold">perceived latency</text>
            </g>
          </svg>
        </div>

        {/* Technology Stack */}
        <div className="grid grid-cols-3 gap-4 max-w-3xl mx-auto mt-2">
          <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-center">
            <div className="text-emerald-400 font-semibold mb-1">ElevenLabs TTS</div>
            <div className="text-slate-400 text-xs">29+ premium voices</div>
          </div>
          <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-center">
            <div className="text-emerald-400 font-semibold mb-1">MuseTalk</div>
            <div className="text-slate-400 text-xs">Real-time lip-sync</div>
          </div>
          <div className="p-3 bg-orange-500/10 border border-orange-500/30 rounded-lg text-center">
            <div className="text-orange-400 font-semibold mb-1">Blender MCP</div>
            <div className="text-slate-400 text-xs">GPU headless execution</div>
          </div>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
