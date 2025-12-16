"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface StreamingInferenceSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function StreamingInferenceSlide({ slideNumber, totalSlides }: StreamingInferenceSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Real-Time Processing">
      <div className="flex flex-col h-full">
        <h2 className="text-4xl font-bold text-white mb-2">
          Streaming <span className="text-cyan-400">Inference</span>
        </h2>
        <p className="text-slate-400 mb-6">Sub-200ms latency for real-time game voice commands</p>

        <div className="flex-1 grid grid-cols-3 gap-6">
          {/* Streaming Pipeline Visualization */}
          <Card className="col-span-2 bg-slate-800/30 border-slate-700/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Badge className="bg-cyan-500/20 text-cyan-400">Pipeline</Badge>
                Chunked Audio Processing
              </CardTitle>
            </CardHeader>
            <CardContent>
              <svg viewBox="0 0 600 350" className="w-full h-full">
                <defs>
                  <linearGradient id="streamGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#06b6d4"/>
                    <stop offset="50%" stopColor="#a855f7"/>
                    <stop offset="100%" stopColor="#ec4899"/>
                  </linearGradient>
                </defs>

                {/* Audio Stream */}
                <g transform="translate(20, 30)">
                  <text x="0" y="0" fill="#94a3b8" fontSize="11" fontWeight="bold">Continuous Audio Stream</text>

                  {/* Waveform representation */}
                  <rect x="0" y="15" width="560" height="40" rx="4" fill="#1e293b" stroke="#334155" strokeWidth="1"/>

                  {/* Chunks */}
                  {[0, 1, 2, 3, 4, 5, 6].map((i) => (
                    <g key={i}>
                      <rect
                        x={i * 80}
                        y="17"
                        width="78"
                        height="36"
                        rx="2"
                        fill={i === 3 ? "#06b6d4" : "#334155"}
                        opacity={i === 3 ? 0.4 : 0.3}
                        stroke={i === 3 ? "#06b6d4" : "none"}
                        strokeWidth="2"
                      />
                      <text x={i * 80 + 39} y="40" textAnchor="middle" fill={i === 3 ? "#67e8f9" : "#64748b"} fontSize="8">
                        {i === 3 ? "Current" : `Chunk ${i + 1}`}
                      </text>
                    </g>
                  ))}

                  {/* Context window */}
                  <path d="M80 60 L80 70 L320 70 L320 60" stroke="#a855f7" strokeWidth="1.5" fill="none"/>
                  <text x="200" y="85" textAnchor="middle" fill="#a855f7" fontSize="9">Left Context (0.5-1.0s)</text>
                </g>

                {/* Processing Pipeline */}
                <g transform="translate(20, 120)">
                  <text x="0" y="0" fill="#94a3b8" fontSize="11" fontWeight="bold">Processing Pipeline</text>

                  {/* Stage boxes */}
                  <g transform="translate(0, 20)">
                    {/* Audio Capture */}
                    <rect x="0" y="0" width="100" height="50" rx="6" fill="#1e293b" stroke="#06b6d4" strokeWidth="2"/>
                    <text x="50" y="22" textAnchor="middle" fill="#06b6d4" fontSize="10">Audio</text>
                    <text x="50" y="36" textAnchor="middle" fill="#06b6d4" fontSize="10">Capture</text>
                    <text x="50" y="60" textAnchor="middle" fill="#67e8f9" fontSize="9">20ms</text>

                    <path d="M105 25 L135 25" stroke="#64748b" strokeWidth="2" markerEnd="url(#streamArrow)"/>

                    {/* WavLM */}
                    <rect x="140" y="0" width="100" height="50" rx="6" fill="#a855f7" opacity="0.3" stroke="#a855f7" strokeWidth="2"/>
                    <text x="190" y="22" textAnchor="middle" fill="#c4b5fd" fontSize="10">WavLM</text>
                    <text x="190" y="36" textAnchor="middle" fill="#c4b5fd" fontSize="10">Features</text>
                    <text x="190" y="60" textAnchor="middle" fill="#e9d5ff" fontSize="9">40ms</text>

                    <path d="M245 25 L275 25" stroke="#64748b" strokeWidth="2" markerEnd="url(#streamArrow)"/>

                    {/* Encoder */}
                    <rect x="280" y="0" width="100" height="50" rx="6" fill="#06b6d4" opacity="0.3" stroke="#06b6d4" strokeWidth="2"/>
                    <text x="330" y="22" textAnchor="middle" fill="#a5f3fc" fontSize="10">Zipformer</text>
                    <text x="330" y="36" textAnchor="middle" fill="#a5f3fc" fontSize="10">Encoder</text>
                    <text x="330" y="60" textAnchor="middle" fill="#67e8f9" fontSize="9">60ms</text>

                    <path d="M385 25 L415 25" stroke="#64748b" strokeWidth="2" markerEnd="url(#streamArrow)"/>

                    {/* Decoder */}
                    <rect x="420" y="0" width="100" height="50" rx="6" fill="#ec4899" opacity="0.3" stroke="#ec4899" strokeWidth="2"/>
                    <text x="470" y="22" textAnchor="middle" fill="#f9a8d4" fontSize="10">Decoder</text>
                    <text x="470" y="36" textAnchor="middle" fill="#f9a8d4" fontSize="10">+ KV-Cache</text>
                    <text x="470" y="60" textAnchor="middle" fill="#fbcfe8" fontSize="9">40ms</text>
                  </g>

                  {/* Total latency bar */}
                  <g transform="translate(0, 100)">
                    <text x="0" y="0" fill="#64748b" fontSize="9">Total Latency</text>
                    <rect x="0" y="8" width="520" height="20" rx="4" fill="#1e293b"/>
                    <rect x="0" y="8" width={520 * 0.2 / 0.2} height="20" rx="4" fill="url(#streamGrad)" opacity="0.6"/>
                    <text x="530" y="22" fill="#10b981" fontSize="11" fontWeight="bold">~160ms</text>
                  </g>
                </g>

                {/* KV-Cache Visualization */}
                <g transform="translate(20, 270)">
                  <text x="0" y="0" fill="#94a3b8" fontSize="11" fontWeight="bold">Decoder KV-Cache</text>

                  <g transform="translate(0, 15)">
                    {[0, 1, 2, 3, 4].map((i) => (
                      <g key={i} transform={`translate(${i * 110}, 0)`}>
                        <rect width="100" height="45" rx="4" fill={i < 3 ? "#ec4899" : "#334155"} opacity={i < 3 ? 0.2 : 0.3}/>
                        <text x="50" y="18" textAnchor="middle" fill={i < 3 ? "#f9a8d4" : "#64748b"} fontSize="9">
                          {i < 3 ? `Token ${i + 1}` : "..."}
                        </text>
                        <text x="50" y="35" textAnchor="middle" fill={i < 3 ? "#f472b6" : "#475569"} fontSize="8">
                          {i < 3 ? "K,V cached" : ""}
                        </text>
                      </g>
                    ))}
                    <text x="560" y="28" fill="#10b981" fontSize="9">Reused!</text>
                  </g>
                </g>

                <defs>
                  <marker id="streamArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
                  </marker>
                </defs>
              </svg>
            </CardContent>
          </Card>

          {/* Right Panel - Specs and Config */}
          <div className="space-y-4">
            <Card className="bg-slate-800/30 border-slate-700/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-slate-400">Latency Breakdown</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {[
                  { label: "Audio Capture + VAD", value: "20ms", pct: 12, color: "bg-slate-400" },
                  { label: "WavLM Features", value: "40ms", pct: 25, color: "bg-purple-400" },
                  { label: "Zipformer Encoder", value: "60ms", pct: 38, color: "bg-cyan-400" },
                  { label: "Decoder Generation", value: "40ms", pct: 25, color: "bg-pink-400" },
                ].map((item) => (
                  <div key={item.label} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400">{item.label}</span>
                      <span className="text-white font-mono">{item.value}</span>
                    </div>
                    <div className="h-2 bg-slate-900/50 rounded-full overflow-hidden">
                      <div className={`h-full ${item.color} rounded-full`} style={{ width: `${item.pct}%` }}/>
                    </div>
                  </div>
                ))}
                <div className="pt-2 border-t border-slate-700/50">
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-400">Total</span>
                    <span className="text-lg font-bold text-emerald-400">~160ms</span>
                  </div>
                  <div className="text-xs text-slate-500">Within 200ms budget</div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800/30 border-slate-700/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-slate-400">Streaming Config</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {[
                  { label: "Chunk Size", value: "160-200ms" },
                  { label: "Left Context", value: "0.5-1.0s" },
                  { label: "Right Context", value: "0ms (causal)" },
                  { label: "KV-Cache", value: "Enabled" },
                  { label: "Decoding", value: "Greedy" },
                ].map((item) => (
                  <div key={item.label} className="flex justify-between items-center p-2 bg-slate-900/30 rounded">
                    <span className="text-xs text-slate-400">{item.label}</span>
                    <span className="text-xs font-mono text-cyan-400">{item.value}</span>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card className="bg-slate-800/30 border-slate-700/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-slate-400">Decoding Modes</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="p-2 bg-cyan-500/10 border border-cyan-500/30 rounded">
                  <div className="text-xs text-cyan-400 font-semibold">Greedy (Real-time)</div>
                  <div className="text-xs text-slate-500">Fastest, used during streaming</div>
                </div>
                <div className="p-2 bg-purple-500/10 border border-purple-500/30 rounded">
                  <div className="text-xs text-purple-400 font-semibold">Beam Search (size=4)</div>
                  <div className="text-xs text-slate-500">Better WER, after utterance ends</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
