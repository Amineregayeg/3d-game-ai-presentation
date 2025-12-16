"use client";

import { AvatarSlideWrapper } from "./AvatarSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface AvatarRoadmapSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function AvatarRoadmapSlide({ slideNumber, totalSlides }: AvatarRoadmapSlideProps) {
  const phases = [
    {
      phase: "Phase 1",
      title: "Core TTS Integration",
      color: "rose",
      items: [
        "ElevenLabs API setup",
        "WebSocket streaming",
        "Voice selection",
        "Basic audio playback"
      ],
      milestone: "First voice output"
    },
    {
      phase: "Phase 2",
      title: "Lip-Sync Integration",
      color: "pink",
      items: [
        "Wav2Lip deployment",
        "MFCC extraction",
        "Viseme mapping",
        "Animation generation"
      ],
      milestone: "Synced mouth movement"
    },
    {
      phase: "Phase 3",
      title: "Game Engine Bridge",
      color: "fuchsia",
      items: [
        "Unity C# manager",
        "UE5 C++ module",
        "MetaHuman integration",
        "Blend shape mapping"
      ],
      milestone: "In-game animation"
    },
    {
      phase: "Phase 4",
      title: "Optimization",
      color: "violet",
      items: [
        "Dialogue caching",
        "Latency profiling",
        "Parallel processing",
        "Buffer tuning"
      ],
      milestone: "<200ms E2E latency"
    },
    {
      phase: "Phase 5",
      title: "Production Hardening",
      color: "purple",
      items: [
        "Fallback chains",
        "Error monitoring",
        "Load testing",
        "Security audit"
      ],
      milestone: "Production ready"
    }
  ];

  const colorMap: Record<string, { bg: string; border: string; text: string; gradient: string }> = {
    rose: { bg: "bg-rose-500/20", border: "border-rose-500/50", text: "text-rose-400", gradient: "from-rose-500 to-pink-500" },
    pink: { bg: "bg-pink-500/20", border: "border-pink-500/50", text: "text-pink-400", gradient: "from-pink-500 to-fuchsia-500" },
    fuchsia: { bg: "bg-fuchsia-500/20", border: "border-fuchsia-500/50", text: "text-fuchsia-400", gradient: "from-fuchsia-500 to-violet-500" },
    violet: { bg: "bg-violet-500/20", border: "border-violet-500/50", text: "text-violet-400", gradient: "from-violet-500 to-purple-500" },
    purple: { bg: "bg-purple-500/20", border: "border-purple-500/50", text: "text-purple-400", gradient: "from-purple-500 to-indigo-500" }
  };

  return (
    <AvatarSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Implementation Roadmap">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Implementation <span className="text-rose-400">Roadmap</span>
            </h2>
            <p className="text-slate-400">5-phase production deployment strategy</p>
          </div>
          <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
            Production Ready
          </Badge>
        </div>

        {/* Timeline */}
        <div className="relative mb-6">
          <div className="absolute top-6 left-0 right-0 h-1 bg-gradient-to-r from-rose-500 via-fuchsia-500 to-purple-500 rounded-full" />
          <div className="flex justify-between relative">
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
        <div className="grid grid-cols-5 gap-3 flex-1">
          {phases.map((p) => (
            <Card key={p.phase} className={`${colorMap[p.color].bg} ${colorMap[p.color].border} border`}>
              <CardContent className="p-3">
                <h3 className={`text-sm font-bold ${colorMap[p.color].text} mb-2`}>{p.title}</h3>
                <div className="space-y-1.5 mb-3">
                  {p.items.map((item) => (
                    <div key={item} className="flex items-start gap-1.5">
                      <div className={`w-1 h-1 rounded-full ${colorMap[p.color].text.replace('text', 'bg')} mt-1.5 flex-shrink-0`} />
                      <span className="text-xs text-slate-400">{item}</span>
                    </div>
                  ))}
                </div>
                <div className={`text-xs ${colorMap[p.color].text} font-semibold border-t border-slate-700/50 pt-2 flex items-center gap-1`}>
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  {p.milestone}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Cost & Metrics Summary */}
        <div className="grid grid-cols-4 gap-4 mt-4">
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-4">
              <div className="text-xs text-slate-500 mb-1">Small Game (10K DAU)</div>
              <div className="text-xl font-bold text-rose-400">$180-300/mo</div>
              <div className="text-xs text-slate-500">Infrastructure</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-4">
              <div className="text-xs text-slate-500 mb-1">Medium Game (100K DAU)</div>
              <div className="text-xl font-bold text-pink-400">$600-800/mo</div>
              <div className="text-xs text-slate-500">Infrastructure</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-4">
              <div className="text-xs text-slate-500 mb-1">Cache Hit Rate</div>
              <div className="text-xl font-bold text-emerald-400">80-90%</div>
              <div className="text-xs text-slate-500">Cost Reduction</div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-4">
              <div className="text-xs text-slate-500 mb-1">Target SLA</div>
              <div className="text-xl font-bold text-violet-400">99.9%</div>
              <div className="text-xs text-slate-500">Uptime</div>
            </CardContent>
          </Card>
        </div>

        {/* Final Summary */}
        <div className="mt-4 p-4 bg-gradient-to-r from-rose-500/10 via-fuchsia-500/10 to-violet-500/10 rounded-xl border border-rose-500/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-sm text-white font-semibold">ElevenLabs Flash 2.5</span>
              </div>
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-sm text-white font-semibold">Wav2Lip + SadTalker</span>
              </div>
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-sm text-white font-semibold">Unity + UE5 Ready</span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-slate-500">Target Achieved</div>
              <div className="text-lg font-bold bg-gradient-to-r from-rose-400 to-fuchsia-400 bg-clip-text text-transparent">
                75ms TTFB | 4.14 MOS | 95%+ Lip-Sync
              </div>
            </div>
          </div>
        </div>
      </div>
    </AvatarSlideWrapper>
  );
}
