"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface DataFlowSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function DataFlowSlide({ slideNumber, totalSlides }: DataFlowSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Data Flow">
      <div className="space-y-4">
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className="text-3xl font-bold text-white mb-2">Request/Response Flow</h2>
          <p className="text-slate-400">SSE streaming for real-time progress updates</p>
        </div>

        {/* Sequence diagram */}
        <div className="flex justify-center">
          <svg viewBox="0 0 900 450" className="w-full max-w-4xl h-auto">
            {/* Background */}
            <defs>
              <pattern id="flow-grid" width="30" height="30" patternUnits="userSpaceOnUse">
                <path d="M 30 0 L 0 0 0 30" fill="none" stroke="#1e293b" strokeWidth="0.5"/>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#flow-grid)"/>

            {/* Column headers */}
            <g transform="translate(0, 20)">
              {/* User */}
              <rect x="50" y="0" width="80" height="30" rx="4" fill="#06b6d4" fillOpacity="0.2" stroke="#06b6d4"/>
              <text x="90" y="20" textAnchor="middle" fill="#06b6d4" fontSize="11" fontWeight="bold">User</text>
              <line x1="90" y1="35" x2="90" y2="420" stroke="#06b6d4" strokeWidth="1" strokeDasharray="4"/>

              {/* Frontend */}
              <rect x="180" y="0" width="80" height="30" rx="4" fill="#a855f7" fillOpacity="0.2" stroke="#a855f7"/>
              <text x="220" y="20" textAnchor="middle" fill="#a855f7" fontSize="11" fontWeight="bold">Frontend</text>
              <line x1="220" y1="35" x2="220" y2="420" stroke="#a855f7" strokeWidth="1" strokeDasharray="4"/>

              {/* API */}
              <rect x="340" y="0" width="80" height="30" rx="4" fill="#f59e0b" fillOpacity="0.2" stroke="#f59e0b"/>
              <text x="380" y="20" textAnchor="middle" fill="#f59e0b" fontSize="11" fontWeight="bold">API</text>
              <line x1="380" y1="35" x2="380" y2="420" stroke="#f59e0b" strokeWidth="1" strokeDasharray="4"/>

              {/* STT */}
              <rect x="480" y="0" width="80" height="30" rx="4" fill="#10b981" fillOpacity="0.2" stroke="#10b981"/>
              <text x="520" y="20" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">STT</text>
              <line x1="520" y1="35" x2="520" y2="420" stroke="#10b981" strokeWidth="1" strokeDasharray="4"/>

              {/* RAG */}
              <rect x="600" y="0" width="80" height="30" rx="4" fill="#ec4899" fillOpacity="0.2" stroke="#ec4899"/>
              <text x="640" y="20" textAnchor="middle" fill="#ec4899" fontSize="11" fontWeight="bold">RAG</text>
              <line x1="640" y1="35" x2="640" y2="420" stroke="#ec4899" strokeWidth="1" strokeDasharray="4"/>

              {/* Avatar */}
              <rect x="720" y="0" width="80" height="30" rx="4" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6"/>
              <text x="760" y="20" textAnchor="middle" fill="#3b82f6" fontSize="11" fontWeight="bold">Avatar</text>
              <line x1="760" y1="35" x2="760" y2="420" stroke="#3b82f6" strokeWidth="1" strokeDasharray="4"/>

              {/* Blender */}
              <rect x="820" y="0" width="60" height="30" rx="4" fill="#f97316" fillOpacity="0.2" stroke="#f97316"/>
              <text x="850" y="20" textAnchor="middle" fill="#f97316" fontSize="10" fontWeight="bold">Blender</text>
              <line x1="850" y1="35" x2="850" y2="420" stroke="#f97316" strokeWidth="1" strokeDasharray="4"/>
            </g>

            {/* Messages */}
            <g transform="translate(0, 70)">
              {/* 1. Speak */}
              <g transform="translate(0, 0)">
                <line x1="90" y1="10" x2="220" y2="10" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrow)"/>
                <text x="155" y="6" textAnchor="middle" fill="#94a3b8" fontSize="9">Speak (audio blob)</text>
              </g>

              {/* 2. POST /pipeline */}
              <g transform="translate(0, 35)">
                <line x1="220" y1="10" x2="380" y2="10" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrow)"/>
                <text x="300" y="6" textAnchor="middle" fill="#94a3b8" fontSize="9">POST /api/full-demo/pipeline</text>
              </g>

              {/* 3. SSE: stt_start */}
              <g transform="translate(0, 60)">
                <line x1="380" y1="10" x2="220" y2="10" stroke="#10b981" strokeWidth="1" strokeDasharray="3" markerEnd="url(#arrow)"/>
                <text x="300" y="6" textAnchor="middle" fill="#10b981" fontSize="9">SSE: stt_start</text>
              </g>

              {/* 4. Transcribe */}
              <g transform="translate(0, 85)">
                <line x1="380" y1="10" x2="520" y2="10" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrow)"/>
                <text x="450" y="6" textAnchor="middle" fill="#94a3b8" fontSize="9">transcribe(audio)</text>
              </g>

              {/* 5. Return text */}
              <g transform="translate(0, 110)">
                <line x1="520" y1="10" x2="380" y2="10" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrow)"/>
                <text x="450" y="6" textAnchor="middle" fill="#94a3b8" fontSize="9">text + confidence</text>
              </g>

              {/* 6. SSE: stt_complete */}
              <g transform="translate(0, 135)">
                <line x1="380" y1="10" x2="220" y2="10" stroke="#10b981" strokeWidth="1" strokeDasharray="3" markerEnd="url(#arrow)"/>
                <text x="300" y="6" textAnchor="middle" fill="#10b981" fontSize="9">SSE: stt_complete</text>
              </g>

              {/* 7. RAG query */}
              <g transform="translate(0, 160)">
                <line x1="380" y1="10" x2="640" y2="10" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrow)"/>
                <text x="510" y="6" textAnchor="middle" fill="#94a3b8" fontSize="9">query(text)</text>
              </g>

              {/* 8. RAG stages (SSE) */}
              <g transform="translate(0, 185)">
                <line x1="380" y1="10" x2="220" y2="10" stroke="#ec4899" strokeWidth="1" strokeDasharray="3" markerEnd="url(#arrow)"/>
                <text x="300" y="6" textAnchor="middle" fill="#ec4899" fontSize="9">SSE: rag_stages (1-8)</text>
              </g>

              {/* 9. RAG response */}
              <g transform="translate(0, 210)">
                <line x1="640" y1="10" x2="380" y2="10" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrow)"/>
                <text x="510" y="6" textAnchor="middle" fill="#94a3b8" fontSize="9">answer + citations</text>
              </g>

              {/* 10. Avatar speak */}
              <g transform="translate(0, 235)">
                <line x1="380" y1="10" x2="760" y2="10" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrow)"/>
                <text x="570" y="6" textAnchor="middle" fill="#94a3b8" fontSize="9">speak(answer)</text>
              </g>

              {/* 11. Avatar ready */}
              <g transform="translate(0, 260)">
                <line x1="760" y1="10" x2="380" y2="10" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrow)"/>
                <text x="570" y="6" textAnchor="middle" fill="#94a3b8" fontSize="9">video_url + audio_url</text>
              </g>

              {/* 12. Parallel execution box */}
              <rect x="360" y="275" width="510" height="45" rx="4" fill="none" stroke="#f97316" strokeWidth="1" strokeDasharray="4"/>
              <text x="615" y="288" textAnchor="middle" fill="#f97316" fontSize="9">PARALLEL EXECUTION</text>

              {/* 13. Blender execute */}
              <g transform="translate(0, 295)">
                <line x1="380" y1="10" x2="850" y2="10" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrow)"/>
                <text x="615" y="6" textAnchor="middle" fill="#94a3b8" fontSize="9">execute_commands()</text>
              </g>

              {/* 14. SSE updates to frontend */}
              <g transform="translate(0, 330)">
                <line x1="380" y1="10" x2="220" y2="10" stroke="#3b82f6" strokeWidth="1" strokeDasharray="3" markerEnd="url(#arrow)"/>
                <text x="300" y="6" textAnchor="middle" fill="#3b82f6" fontSize="9">SSE: avatar + blender updates</text>
              </g>

              {/* 15. Display to user */}
              <g transform="translate(0, 355)">
                <line x1="220" y1="10" x2="90" y2="10" stroke="#64748b" strokeWidth="1" markerEnd="url(#arrow)"/>
                <text x="155" y="6" textAnchor="middle" fill="#94a3b8" fontSize="9">Show video + 3D result</text>
              </g>
            </g>

            {/* Arrow marker */}
            <defs>
              <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
              </marker>
            </defs>
          </svg>
        </div>

        {/* Latency breakdown */}
        <Card className="bg-slate-900/30 border-slate-700/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="h-1 w-8 bg-gradient-to-r from-cyan-500 to-orange-500 rounded-full" />
              <span className="text-sm font-mono text-cyan-400">LATENCY BREAKDOWN</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-center text-sm">
              {[
                { stage: "STT", time: "~2s", color: "cyan" },
                { stage: "RAG", time: "~1.8s", color: "purple" },
                { stage: "Avatar Gen", time: "~3s", color: "blue" },
                { stage: "Blender", time: "~1.5s", color: "orange" },
                { stage: "Total", time: "<8s", color: "emerald" },
              ].map((item) => (
                <div key={item.stage} className="p-2 bg-slate-800/50 rounded">
                  <div className={`text-${item.color}-400 font-mono text-lg`}>{item.time}</div>
                  <div className="text-slate-500 text-xs">{item.stage}</div>
                </div>
              ))}
            </div>
            <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
              <Badge variant="outline" className="border-emerald-500/30 text-emerald-400">Note</Badge>
              Avatar + Blender execute in parallel, reducing total time
            </div>
          </CardContent>
        </Card>
      </div>
    </TechSlideWrapper>
  );
}
