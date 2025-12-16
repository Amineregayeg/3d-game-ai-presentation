"use client";

import { MCPSlideWrapper } from "./MCPSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface MCPRoadmapSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function MCPRoadmapSlide({ slideNumber, totalSlides }: MCPRoadmapSlideProps) {
  const phases = [
    {
      phase: "Phase 1",
      title: "Basic MCP Setup",
      color: "orange",
      items: [
        "Install Blender 3.6+",
        "Install BlenderMCP addon",
        "Configure MCP server",
        "Test socket connection"
      ],
      milestone: "AI controls Blender"
    },
    {
      phase: "Phase 2",
      title: "Asset Integration",
      color: "amber",
      items: [
        "Configure Sketchfab API",
        "Setup Poly Haven",
        "Configure Hyper3D Rodin",
        "Implement caching"
      ],
      milestone: "External assets available"
    },
    {
      phase: "Phase 3",
      title: "Game Engine Bridge",
      color: "yellow",
      items: [
        "FBX export for Unity",
        "FBX export for UE5",
        "GLTF export pipeline",
        "Material conversion"
      ],
      milestone: "Assets in game engine"
    },
    {
      phase: "Phase 4",
      title: "Production Hardening",
      color: "lime",
      items: [
        "Code validation/sandbox",
        "Error logging setup",
        "Load testing",
        "Security audit"
      ],
      milestone: "Production ready"
    }
  ];

  const colorMap: Record<string, { bg: string; border: string; text: string; gradient: string }> = {
    orange: { bg: "bg-orange-500/20", border: "border-orange-500/50", text: "text-orange-400", gradient: "from-orange-500 to-amber-500" },
    amber: { bg: "bg-amber-500/20", border: "border-amber-500/50", text: "text-amber-400", gradient: "from-amber-500 to-yellow-500" },
    yellow: { bg: "bg-yellow-500/20", border: "border-yellow-500/50", text: "text-yellow-400", gradient: "from-yellow-500 to-lime-500" },
    lime: { bg: "bg-lime-500/20", border: "border-lime-500/50", text: "text-lime-400", gradient: "from-lime-500 to-green-500" }
  };

  return (
    <MCPSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Roadmap">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Implementation <span className="text-orange-400">Roadmap</span>
            </h2>
            <p className="text-slate-400">4-phase production deployment strategy</p>
          </div>
          <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
            Production Ready
          </Badge>
        </div>

        {/* Timeline */}
        <div className="relative mb-6">
          <div className="absolute top-6 left-0 right-0 h-1 bg-gradient-to-r from-orange-500 via-amber-500 to-lime-500 rounded-full" />
          <div className="flex justify-between relative px-8">
            {phases.map((p, i) => (
              <div key={p.phase} className="flex flex-col items-center">
                <div className={`w-12 h-12 rounded-full bg-gradient-to-br ${colorMap[p.color].gradient} flex items-center justify-center text-white font-bold text-sm z-10 shadow-lg`}>
                  {i + 1}
                </div>
                <div className={`mt-3 text-xs ${colorMap[p.color].text} font-semibold`}>{p.phase}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Phase Cards */}
        <div className="grid grid-cols-4 gap-4 flex-1">
          {phases.map((p) => (
            <Card key={p.phase} className={`${colorMap[p.color].bg} ${colorMap[p.color].border} border`}>
              <CardContent className="p-4">
                <h3 className={`text-sm font-bold ${colorMap[p.color].text} mb-3`}>{p.title}</h3>
                <div className="space-y-2 mb-4">
                  {p.items.map((item) => (
                    <div key={item} className="flex items-start gap-2">
                      <div className={`w-1.5 h-1.5 rounded-full ${colorMap[p.color].text.replace('text', 'bg')} mt-1.5 flex-shrink-0`} />
                      <span className="text-xs text-slate-400">{item}</span>
                    </div>
                  ))}
                </div>
                <div className={`text-xs ${colorMap[p.color].text} font-semibold border-t border-slate-700/50 pt-3 flex items-center gap-1`}>
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  {p.milestone}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Performance Targets & Config */}
        <div className="grid grid-cols-5 gap-4 mt-4">
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-3">
              <div className="text-xs text-slate-500 mb-1">Socket Response</div>
              <div className="text-xl font-bold text-orange-400">&lt;100ms</div>
              <div className="text-xs text-slate-600">Local connection</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-3">
              <div className="text-xs text-slate-500 mb-1">Scene Query</div>
              <div className="text-xl font-bold text-amber-400">&lt;50ms</div>
              <div className="text-xs text-slate-600">Cached metadata</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-3">
              <div className="text-xs text-slate-500 mb-1">Asset Import</div>
              <div className="text-xl font-bold text-yellow-400">&lt;30s</div>
              <div className="text-xs text-slate-600">Size dependent</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-3">
              <div className="text-xs text-slate-500 mb-1">AI Generation</div>
              <div className="text-xl font-bold text-lime-400">30-120s</div>
              <div className="text-xs text-slate-600">External API</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-3">
              <div className="text-xs text-slate-500 mb-1">Game Export</div>
              <div className="text-xl font-bold text-green-400">&lt;10s</div>
              <div className="text-xs text-slate-600">FBX/GLTF</div>
            </CardContent>
          </Card>
        </div>

        {/* Final Summary */}
        <div className="mt-4 p-4 bg-gradient-to-r from-orange-500/10 via-amber-500/10 to-lime-500/10 rounded-xl border border-orange-500/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-sm text-white font-semibold">24 MCP Tools</span>
              </div>
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-sm text-white font-semibold">5 Asset Sources</span>
              </div>
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-sm text-white font-semibold">Unity + UE5 Export</span>
              </div>
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-sm text-white font-semibold">Blender 3.6+</span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-slate-500">TCP Socket</div>
              <div className="text-lg font-bold bg-gradient-to-r from-orange-400 to-amber-400 bg-clip-text text-transparent">
                localhost:9876 | JSON-RPC 2.0
              </div>
            </div>
          </div>
        </div>
      </div>
    </MCPSlideWrapper>
  );
}
