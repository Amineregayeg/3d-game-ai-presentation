"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";

interface BlenderMCPSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function BlenderMCPSlide({ slideNumber, totalSlides }: BlenderMCPSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Blender Execution">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className="text-4xl font-bold text-white mb-2">
            Blender Execution Layer <span className="text-slate-500">(MCP + bpy)</span>
          </h2>
        </div>

        {/* Hybrid Architecture Diagram */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 900 360" className="w-full max-w-5xl h-auto">
            <defs>
              <linearGradient id="browserBoxGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.2"/>
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.05"/>
              </linearGradient>
              <linearGradient id="gpuBoxGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.2"/>
                <stop offset="100%" stopColor="#f97316" stopOpacity="0.05"/>
              </linearGradient>
              <linearGradient id="mcpGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#ea580c" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="threeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.9"/>
              </linearGradient>
              <marker id="mcpArrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#64748b"/>
              </marker>
            </defs>

            {/* Browser Section */}
            <g transform="translate(50, 30)">
              <rect width="350" height="300" rx="12" fill="url(#browserBoxGrad)" stroke="#06b6d4" strokeWidth="2"/>
              <text x="175" y="30" textAnchor="middle" fill="#06b6d4" fontSize="14" fontWeight="bold">BROWSER</text>

              {/* Three.js Preview */}
              <g transform="translate(30, 50)">
                <rect width="290" height="120" rx="10" fill="url(#threeGrad)"/>
                <text x="145" y="30" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">Three.js Viewport</text>
                <text x="145" y="50" textAnchor="middle" fill="#e0f2fe" fontSize="11">Real-time 3D Preview</text>

                {/* 3D scene representation */}
                <g transform="translate(70, 60)">
                  <polygon points="75,5 145,45 75,85 5,45" fill="#0f172a" stroke="#67e8f9" strokeWidth="1"/>
                  <polygon points="75,5 75,45 5,45 5,5" fill="#0f172a" stroke="#67e8f9" strokeWidth="1" transform="translate(0,0)"/>
                  <circle cx="75" cy="45" r="15" fill="#06b6d4" fillOpacity="0.5"/>
                </g>
              </g>

              {/* WebGL info */}
              <g transform="translate(30, 185)">
                <rect width="140" height="50" rx="6" fill="#0f172a" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="70" y="22" textAnchor="middle" fill="#67e8f9" fontSize="10" fontWeight="bold">WebGL Renderer</text>
                <text x="70" y="38" textAnchor="middle" fill="#94a3b8" fontSize="9">GPU-accelerated</text>
              </g>

              {/* No installation */}
              <g transform="translate(180, 185)">
                <rect width="140" height="50" rx="6" fill="#0f172a" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.5"/>
                <text x="70" y="22" textAnchor="middle" fill="#67e8f9" fontSize="10" fontWeight="bold">No Installation</text>
                <text x="70" y="38" textAnchor="middle" fill="#94a3b8" fontSize="9">Browser-only</text>
              </g>

              {/* SSE Updates */}
              <g transform="translate(30, 250)">
                <rect width="290" height="35" rx="6" fill="#ec4899" fillOpacity="0.1" stroke="#ec4899" strokeWidth="1"/>
                <text x="145" y="22" textAnchor="middle" fill="#ec4899" fontSize="10">← SSE Scene Updates (WebSocket)</text>
              </g>
            </g>

            {/* Connection arrows */}
            <g transform="translate(400, 130)">
              <path d="M 0 50 L 50 50" stroke="#64748b" strokeWidth="2" markerEnd="url(#mcpArrow)"/>
              <path d="M 50 80 L 0 80" stroke="#ec4899" strokeWidth="2" strokeDasharray="4" markerEnd="url(#mcpArrow)"/>
              <text x="25" y="40" textAnchor="middle" fill="#64748b" fontSize="9">Commands</text>
              <text x="25" y="100" textAnchor="middle" fill="#ec4899" fontSize="9">Updates</text>
            </g>

            {/* GPU Server Section */}
            <g transform="translate(500, 30)">
              <rect width="350" height="300" rx="12" fill="url(#gpuBoxGrad)" stroke="#f97316" strokeWidth="2"/>
              <text x="175" y="30" textAnchor="middle" fill="#f97316" fontSize="14" fontWeight="bold">GPU SERVER</text>

              {/* MCP Server */}
              <g transform="translate(30, 50)">
                <rect width="290" height="80" rx="10" fill="url(#mcpGrad)"/>
                <text x="145" y="30" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">MCP Server</text>
                <text x="145" y="50" textAnchor="middle" fill="#ffedd5" fontSize="11">Model Context Protocol</text>
                <text x="145" y="68" textAnchor="middle" fill="#fdba74" fontSize="9">Tool orchestration layer</text>
              </g>

              {/* Blender Headless */}
              <g transform="translate(30, 145)">
                <rect width="140" height="90" rx="8" fill="#0f172a" stroke="#f97316" strokeWidth="1.5"/>
                <text x="70" y="25" textAnchor="middle" fill="#f97316" fontSize="11" fontWeight="bold">Blender</text>
                <text x="70" y="42" textAnchor="middle" fill="#f97316" fontSize="11" fontWeight="bold">Headless</text>
                <rect x="15" y="52" width="110" height="25" rx="4" fill="#f97316" fillOpacity="0.2"/>
                <text x="70" y="70" textAnchor="middle" fill="#fdba74" fontSize="9">bpy Python API</text>
              </g>

              {/* Export */}
              <g transform="translate(180, 145)">
                <rect width="140" height="90" rx="8" fill="#0f172a" stroke="#10b981" strokeWidth="1.5"/>
                <text x="70" y="25" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">.blend Export</text>
                <rect x="15" y="42" width="110" height="22" rx="4" fill="#10b981" fillOpacity="0.2"/>
                <text x="70" y="58" textAnchor="middle" fill="#6ee7b7" fontSize="9">Downloadable</text>
                <rect x="15" y="68" width="110" height="18" rx="4" fill="#10b981" fillOpacity="0.1"/>
                <text x="70" y="81" textAnchor="middle" fill="#94a3b8" fontSize="8">GLTF / FBX</text>
              </g>

              {/* GPU info */}
              <g transform="translate(30, 250)">
                <rect width="290" height="35" rx="6" fill="#f97316" fillOpacity="0.1" stroke="#f97316" strokeWidth="1"/>
                <text x="145" y="22" textAnchor="middle" fill="#f97316" fontSize="10">RTX 4090 • CUDA Rendering • 24GB VRAM</text>
              </g>
            </g>
          </svg>
        </div>

        {/* Key Points */}
        <div className="grid grid-cols-3 gap-4 max-w-3xl mx-auto mt-2">
          <div className="p-3 bg-orange-500/10 border border-orange-500/30 rounded-lg text-center">
            <div className="text-orange-400 font-semibold">Deterministic execution</div>
          </div>
          <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-center">
            <div className="text-emerald-400 font-semibold">Exportable .blend files</div>
          </div>
          <div className="p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-center">
            <div className="text-cyan-400 font-semibold">Browser-based preview</div>
          </div>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
