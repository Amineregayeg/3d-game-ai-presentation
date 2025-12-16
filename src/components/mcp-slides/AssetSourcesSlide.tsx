"use client";

import { MCPSlideWrapper } from "./MCPSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface AssetSourcesSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function AssetSourcesSlide({ slideNumber, totalSlides }: AssetSourcesSlideProps) {
  const assetSources = [
    {
      priority: 1,
      name: "Sketchfab",
      description: "3D Model Marketplace",
      bestFor: "Specific named assets (sword, chair, car)",
      quality: "Professional artist-created",
      latency: "10-30s",
      cost: "Free tier + paid",
      color: "blue",
      icon: "M21 12a9 9 0 11-18 0 9 9 0 0118 0z M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
    },
    {
      priority: 2,
      name: "Poly Haven",
      description: "HDRI, Textures, Models",
      bestFor: "Environment assets, textures, HDRIs",
      quality: "Photogrammetry-scanned",
      latency: "5-20s",
      cost: "Free",
      color: "emerald",
      icon: "M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
    },
    {
      priority: 3,
      name: "Hyper3D Rodin",
      description: "AI 3D Generation",
      bestFor: "Custom assets not found elsewhere",
      quality: "AI-generated (variable)",
      latency: "30-120s",
      cost: "Credits-based",
      color: "violet",
      icon: "M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
    },
    {
      priority: 4,
      name: "Hunyuan3D",
      description: "Tencent AI Generation",
      bestFor: "Custom unique assets",
      quality: "AI-generated (variable)",
      latency: "30-120s",
      cost: "Pay-per-use",
      color: "pink",
      icon: "M13 10V3L4 14h7v7l9-11h-7z"
    },
    {
      priority: 5,
      name: "Python Scripting",
      description: "Procedural Generation",
      bestFor: "Procedural/parametric geometry",
      quality: "Dependent on script",
      latency: "<5s",
      cost: "Free",
      color: "amber",
      icon: "M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
    }
  ];

  const colorMap: Record<string, { bg: string; border: string; text: string; gradient: string }> = {
    blue: { bg: "bg-blue-500/10", border: "border-blue-500/40", text: "text-blue-400", gradient: "from-blue-500 to-cyan-500" },
    emerald: { bg: "bg-emerald-500/10", border: "border-emerald-500/40", text: "text-emerald-400", gradient: "from-emerald-500 to-teal-500" },
    violet: { bg: "bg-violet-500/10", border: "border-violet-500/40", text: "text-violet-400", gradient: "from-violet-500 to-purple-500" },
    pink: { bg: "bg-pink-500/10", border: "border-pink-500/40", text: "text-pink-400", gradient: "from-pink-500 to-rose-500" },
    amber: { bg: "bg-amber-500/10", border: "border-amber-500/40", text: "text-amber-400", gradient: "from-amber-500 to-orange-500" }
  };

  return (
    <MCPSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Asset Sources">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              External Asset <span className="text-orange-400">Sources</span>
            </h2>
            <p className="text-slate-400">Multi-source asset acquisition with priority fallback</p>
          </div>
          <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/50">
            5 Sources
          </Badge>
        </div>

        {/* Priority Flow */}
        <div className="relative mb-4 py-4">
          <div className="absolute top-1/2 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-violet-500 to-amber-500 rounded-full transform -translate-y-1/2" />
          <div className="flex justify-between relative">
            {assetSources.map((source, i) => (
              <div key={source.name} className="flex flex-col items-center">
                <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${colorMap[source.color].gradient} flex items-center justify-center text-white font-bold text-sm z-10 shadow-lg`}>
                  {i + 1}
                </div>
                <div className={`mt-2 text-xs ${colorMap[source.color].text} font-semibold`}>{source.name}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Source Cards */}
        <div className="grid grid-cols-5 gap-3 flex-1">
          {assetSources.map((source) => (
            <Card key={source.name} className={`${colorMap[source.color].bg} ${colorMap[source.color].border} border`}>
              <CardContent className="p-3">
                <div className="flex items-center gap-2 mb-2">
                  <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${colorMap[source.color].gradient} bg-opacity-20 flex items-center justify-center`}>
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={source.icon} />
                    </svg>
                  </div>
                  <div>
                    <div className={`text-sm font-bold ${colorMap[source.color].text}`}>{source.name}</div>
                    <div className="text-xs text-slate-500">{source.description}</div>
                  </div>
                </div>

                <div className="space-y-2 mt-3">
                  <div className="bg-slate-900/50 rounded p-2">
                    <div className="text-xs text-slate-500 mb-0.5">Best For</div>
                    <div className="text-xs text-slate-300">{source.bestFor}</div>
                  </div>

                  <div className="grid grid-cols-2 gap-1">
                    <div className="bg-slate-900/50 rounded p-1.5 text-center">
                      <div className="text-xs text-slate-500">Latency</div>
                      <div className={`text-xs font-bold ${colorMap[source.color].text}`}>{source.latency}</div>
                    </div>
                    <div className="bg-slate-900/50 rounded p-1.5 text-center">
                      <div className="text-xs text-slate-500">Cost</div>
                      <div className="text-xs text-slate-300">{source.cost}</div>
                    </div>
                  </div>

                  <div className="flex items-center gap-1 mt-2">
                    <span className="text-xs text-slate-500">Quality:</span>
                    <span className="text-xs text-slate-400">{source.quality}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Fallback Strategy Diagram */}
        <div className="mt-4 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="flex items-center gap-2 mb-3">
            <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span className="text-sm font-semibold text-white">Asset Acquisition Strategy</span>
          </div>

          <svg viewBox="0 0 1000 100" className="w-full h-auto">
            <defs>
              <marker id="arrowAsset" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#64748b" />
              </marker>
            </defs>

            {/* User Request */}
            <rect x="20" y="30" width="120" height="40" rx="6" fill="#1e293b" stroke="#f97316" strokeWidth="1.5"/>
            <text x="80" y="55" textAnchor="middle" fill="#f97316" fontSize="10">User Request</text>

            <path d="M145,50 L175,50" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAsset)"/>

            {/* Try Sketchfab */}
            <rect x="180" y="30" width="100" height="40" rx="6" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="1"/>
            <text x="230" y="50" textAnchor="middle" fill="#3b82f6" fontSize="9" fontWeight="bold">Sketchfab</text>
            <text x="230" y="62" textAnchor="middle" fill="#93c5fd" fontSize="8">Search</text>

            <path d="M285,50 L315,50" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAsset)"/>
            <text x="300" y="42" textAnchor="middle" fill="#ef4444" fontSize="7">404?</text>

            {/* Try Poly Haven */}
            <rect x="320" y="30" width="100" height="40" rx="6" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="1"/>
            <text x="370" y="50" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold">Poly Haven</text>
            <text x="370" y="62" textAnchor="middle" fill="#6ee7b7" fontSize="8">Search</text>

            <path d="M425,50 L455,50" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAsset)"/>
            <text x="440" y="42" textAnchor="middle" fill="#ef4444" fontSize="7">404?</text>

            {/* Try Hyper3D */}
            <rect x="460" y="30" width="100" height="40" rx="6" fill="#8b5cf6" fillOpacity="0.2" stroke="#8b5cf6" strokeWidth="1"/>
            <text x="510" y="50" textAnchor="middle" fill="#8b5cf6" fontSize="9" fontWeight="bold">Hyper3D</text>
            <text x="510" y="62" textAnchor="middle" fill="#c4b5fd" fontSize="8">Generate</text>

            <path d="M565,50 L595,50" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAsset)"/>
            <text x="580" y="42" textAnchor="middle" fill="#ef4444" fontSize="7">Fail?</text>

            {/* Try Hunyuan */}
            <rect x="600" y="30" width="100" height="40" rx="6" fill="#ec4899" fillOpacity="0.2" stroke="#ec4899" strokeWidth="1"/>
            <text x="650" y="50" textAnchor="middle" fill="#ec4899" fontSize="9" fontWeight="bold">Hunyuan3D</text>
            <text x="650" y="62" textAnchor="middle" fill="#f9a8d4" fontSize="8">Generate</text>

            <path d="M705,50 L735,50" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAsset)"/>
            <text x="720" y="42" textAnchor="middle" fill="#ef4444" fontSize="7">Fail?</text>

            {/* Python Fallback */}
            <rect x="740" y="30" width="100" height="40" rx="6" fill="#f59e0b" fillOpacity="0.2" stroke="#f59e0b" strokeWidth="1"/>
            <text x="790" y="50" textAnchor="middle" fill="#f59e0b" fontSize="9" fontWeight="bold">Python</text>
            <text x="790" y="62" textAnchor="middle" fill="#fcd34d" fontSize="8">Procedural</text>

            <path d="M845,50 L875,50" stroke="#22c55e" strokeWidth="1.5" markerEnd="url(#arrowAsset)"/>

            {/* Success */}
            <rect x="880" y="30" width="100" height="40" rx="6" fill="#22c55e" fillOpacity="0.2" stroke="#22c55e" strokeWidth="1.5"/>
            <text x="930" y="55" textAnchor="middle" fill="#22c55e" fontSize="10" fontWeight="bold">Asset Ready</text>
          </svg>
        </div>
      </div>
    </MCPSlideWrapper>
  );
}
