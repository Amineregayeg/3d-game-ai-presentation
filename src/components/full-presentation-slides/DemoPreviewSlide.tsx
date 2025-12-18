"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Mic, Settings, MessageSquare, Box, CheckCircle, ChevronRight } from "lucide-react";

interface DemoPreviewSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function DemoPreviewSlide({ slideNumber, totalSlides }: DemoPreviewSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Interface Preview">
      <div className="space-y-4">
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className="text-2xl font-bold text-white mb-2">Expected Demo Interface</h2>
          <p className="text-slate-400 text-sm">Full pipeline visualization in action</p>
        </div>

        {/* Interface mockup */}
        <Card className="bg-slate-900/80 border-slate-700/50 overflow-hidden">
          <CardContent className="p-0">
            {/* Header bar */}
            <div className="flex items-center justify-between px-4 py-2 bg-slate-800/80 border-b border-slate-700/50">
              <div className="flex items-center gap-2">
                <ChevronRight className="w-4 h-4 text-slate-500" />
                <span className="text-sm text-white font-medium">3D Game AI Assistant - Full Pipeline Demo</span>
              </div>
              <Settings className="w-4 h-4 text-slate-400" />
            </div>

            {/* Main content */}
            <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Left column - Avatar + Viewport */}
              <div className="space-y-3">
                {/* Avatar section */}
                <div className="p-3 bg-slate-800/50 rounded-lg border border-emerald-500/30">
                  <div className="flex items-center gap-2 mb-2">
                    <MessageSquare className="w-3 h-3 text-emerald-400" />
                    <span className="text-xs text-emerald-400">AVATAR</span>
                    <Badge variant="outline" className="text-[10px] border-emerald-500/30 text-emerald-400 bg-emerald-500/10 py-0">
                      Speaking...
                    </Badge>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-16 h-16 bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 rounded-lg flex items-center justify-center border border-emerald-500/30">
                      <div className="text-2xl">👤</div>
                    </div>
                    <div className="flex-1">
                      <div className="h-2 bg-emerald-500/30 rounded-full mb-1.5">
                        <div className="h-full bg-emerald-500 rounded-full w-3/4 animate-pulse" />
                      </div>
                      <div className="text-xs text-slate-400">Voice: Rachel | Duration: 8s</div>
                    </div>
                  </div>
                </div>

                {/* 3D Viewport */}
                <div className="p-3 bg-slate-800/50 rounded-lg border border-orange-500/30">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Box className="w-3 h-3 text-orange-400" />
                      <span className="text-xs text-orange-400">THREE.JS VIEWPORT</span>
                    </div>
                    <Badge variant="outline" className="text-[10px] border-emerald-500/30 text-emerald-400 bg-emerald-500/10 py-0">
                      🟢 Connected
                    </Badge>
                  </div>
                  <div className="h-32 bg-slate-900 rounded border border-slate-700 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-3xl mb-1">🌲</div>
                      <div className="text-xs text-slate-500">Low-poly tree</div>
                      <div className="text-[10px] text-slate-600 mt-1">Objects: 4 | Vertices: 156</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right column - Input + Pipeline + Response */}
              <div className="space-y-3">
                {/* Voice input */}
                <div className="p-3 bg-slate-800/50 rounded-lg border border-cyan-500/30">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Mic className="w-3 h-3 text-cyan-400" />
                      <span className="text-xs text-cyan-400">VOICE INPUT</span>
                    </div>
                    <Badge variant="outline" className="text-[10px] border-purple-500/30 text-purple-400 bg-purple-500/10 py-0">
                      Whisper
                    </Badge>
                  </div>
                  <div className="p-2 bg-slate-900 rounded border border-slate-700">
                    <div className="text-sm text-white mb-1">&quot;Create a low-poly tree with green leaves&quot;</div>
                    <div className="flex gap-3 text-[10px] text-slate-500">
                      <span>Confidence: <span className="text-cyan-400">94.2%</span></span>
                      <span>RTF: <span className="text-cyan-400">0.32</span></span>
                      <span>Words: <span className="text-cyan-400">8</span></span>
                    </div>
                  </div>
                </div>

                {/* RAG Pipeline mini */}
                <div className="p-3 bg-slate-800/50 rounded-lg border border-purple-500/30">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs text-purple-400">RAG PIPELINE</span>
                    <Badge variant="outline" className="text-[10px] border-emerald-500/30 text-emerald-400 bg-emerald-500/10 py-0">
                      Complete
                    </Badge>
                  </div>
                  <div className="grid grid-cols-4 gap-1">
                    {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => (
                      <div key={n} className="flex items-center gap-1 p-1 bg-slate-900 rounded text-[9px]">
                        <CheckCircle className="w-2 h-2 text-emerald-400" />
                        <span className="text-slate-400">Stage {n}</span>
                      </div>
                    ))}
                  </div>
                  <div className="flex gap-2 mt-2 text-[10px]">
                    <span className="text-slate-500">Docs: <span className="text-purple-400">5</span></span>
                    <span className="text-slate-500">RAGAS: <span className="text-emerald-400">0.92</span></span>
                    <span className="text-slate-500">Latency: <span className="text-cyan-400">1.87s</span></span>
                  </div>
                </div>

                {/* Response */}
                <div className="p-3 bg-slate-800/50 rounded-lg border border-slate-600/50">
                  <div className="text-xs text-slate-400 mb-2">RESPONSE</div>
                  <div className="text-xs text-slate-300 leading-relaxed">
                    &quot;I&apos;ll create a low-poly tree for you! I&apos;m using a cylinder for the trunk with a brown material...&quot;
                  </div>
                  <div className="flex gap-1 mt-2">
                    <Badge variant="outline" className="text-[9px] border-slate-600 text-slate-400 py-0">[1] mesh_primitives.md</Badge>
                    <Badge variant="outline" className="text-[9px] border-slate-600 text-slate-400 py-0">[2] low_poly.md</Badge>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom metrics bar */}
            <div className="px-4 py-2 bg-slate-800/50 border-t border-slate-700/50 flex items-center justify-center gap-6 text-[10px]">
              <span className="text-slate-400">STT: <span className="text-cyan-400">94.2%</span></span>
              <span className="text-slate-400">RAG: <span className="text-purple-400">0.92</span></span>
              <span className="text-slate-400">Total: <span className="text-emerald-400">3.2s</span></span>
              <span className="text-slate-400">GPU: <span className="text-orange-400">78%</span></span>
              <span className="text-slate-400">Blender: <span className="text-emerald-400">✓</span></span>
            </div>
          </CardContent>
        </Card>

        {/* Key features */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { label: "Auto-greeting", desc: "Avatar welcomes user on load", color: "emerald" },
            { label: "Dual STT", desc: "VoxFormer or Whisper toggle", color: "cyan" },
            { label: "Live Pipeline", desc: "8-stage RAG visualization", color: "purple" },
            { label: "Parallel Exec", desc: "Avatar + Blender simultaneous", color: "orange" },
          ].map((feat) => (
            <Card key={feat.label} className={`bg-slate-900/50 border-${feat.color}-500/30 backdrop-blur-sm`}>
              <CardContent className="p-3 text-center">
                <div className={`text-sm font-semibold text-${feat.color}-400`}>{feat.label}</div>
                <div className="text-xs text-slate-500">{feat.desc}</div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
