"use client";

import { AvatarSlideWrapper } from "./AvatarSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface StreamingSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function StreamingSlide({ slideNumber, totalSlides }: StreamingSlideProps) {
  return (
    <AvatarSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Real-Time Streaming">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              WebSocket <span className="text-rose-400">Streaming</span> Architecture
            </h2>
            <p className="text-slate-400">Chunked text streaming with parallel audio processing</p>
          </div>
          <div className="flex gap-2">
            <Badge variant="outline" className="border-rose-500/50 text-rose-400 bg-rose-500/10">
              75ms TTFB
            </Badge>
            <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
              Parallel Processing
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6 flex-1">
          {/* WebSocket vs REST Comparison */}
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-4">
              <h3 className="text-lg font-semibold text-white mb-4">WebSocket vs REST Trade-offs</h3>
              <div className="relative h-48">
                <svg viewBox="0 0 400 180" className="w-full h-full">
                  {/* WebSocket Box */}
                  <rect x="10" y="10" width="180" height="160" rx="8" fill="#f43f5e" fillOpacity="0.1" stroke="#f43f5e" strokeWidth="1.5"/>
                  <text x="100" y="35" textAnchor="middle" fill="#f43f5e" fontSize="12" fontWeight="bold">WebSocket</text>

                  {/* WebSocket Stats */}
                  <g transform="translate(25, 50)">
                    <rect width="150" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                    <text x="10" y="17" fill="#fda4af" fontSize="9">TTFB</text>
                    <text x="140" y="17" textAnchor="end" fill="white" fontSize="10" fontWeight="bold">75ms</text>
                  </g>
                  <g transform="translate(25, 80)">
                    <rect width="150" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                    <text x="10" y="17" fill="#fda4af" fontSize="9">Connection</text>
                    <text x="140" y="17" textAnchor="end" fill="white" fontSize="10" fontWeight="bold">Persistent</text>
                  </g>
                  <g transform="translate(25, 110)">
                    <rect width="150" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                    <text x="10" y="17" fill="#fda4af" fontSize="9">Streaming</text>
                    <text x="140" y="17" textAnchor="end" fill="white" fontSize="10" fontWeight="bold">Native</text>
                  </g>
                  <g transform="translate(25, 140)">
                    <rect width="150" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                    <text x="10" y="17" fill="#fda4af" fontSize="9">Overhead</text>
                    <text x="140" y="17" textAnchor="end" fill="#10b981" fontSize="10" fontWeight="bold">Low</text>
                  </g>

                  {/* REST Box */}
                  <rect x="210" y="10" width="180" height="160" rx="8" fill="#64748b" fillOpacity="0.1" stroke="#64748b" strokeWidth="1.5"/>
                  <text x="300" y="35" textAnchor="middle" fill="#94a3b8" fontSize="12" fontWeight="bold">REST API</text>

                  {/* REST Stats */}
                  <g transform="translate(225, 50)">
                    <rect width="150" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                    <text x="10" y="17" fill="#94a3b8" fontSize="9">TTFB</text>
                    <text x="140" y="17" textAnchor="end" fill="#fbbf24" fontSize="10" fontWeight="bold">150-300ms</text>
                  </g>
                  <g transform="translate(225, 80)">
                    <rect width="150" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                    <text x="10" y="17" fill="#94a3b8" fontSize="9">Connection</text>
                    <text x="140" y="17" textAnchor="end" fill="white" fontSize="10" fontWeight="bold">Per-request</text>
                  </g>
                  <g transform="translate(225, 110)">
                    <rect width="150" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                    <text x="10" y="17" fill="#94a3b8" fontSize="9">Streaming</text>
                    <text x="140" y="17" textAnchor="end" fill="white" fontSize="10" fontWeight="bold">Polling</text>
                  </g>
                  <g transform="translate(225, 140)">
                    <rect width="150" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                    <text x="10" y="17" fill="#94a3b8" fontSize="9">Overhead</text>
                    <text x="140" y="17" textAnchor="end" fill="#fbbf24" fontSize="10" fontWeight="bold">Higher</text>
                  </g>
                </svg>
              </div>
              <div className="mt-3 p-2 bg-rose-500/10 border border-rose-500/30 rounded-lg">
                <span className="text-xs text-rose-400">Recommendation: WebSocket for &lt;100ms latency requirement; REST for fallback/batch</span>
              </div>
            </CardContent>
          </Card>

          {/* Chunked Streaming Diagram */}
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-4">
              <h3 className="text-lg font-semibold text-white mb-4">Chunked Text Streaming</h3>
              <div className="relative h-56">
                <svg viewBox="0 0 400 200" className="w-full h-full">
                  <defs>
                    <linearGradient id="chunkGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#f43f5e"/>
                      <stop offset="100%" stopColor="#ec4899"/>
                    </linearGradient>
                  </defs>

                  {/* Long Text Input */}
                  <rect x="10" y="10" width="380" height="30" rx="6" fill="#1e293b" stroke="#f43f5e" strokeWidth="1"/>
                  <text x="200" y="29" textAnchor="middle" fill="#fda4af" fontSize="9">&quot;Hello there, adventurer! Welcome to the kingdom. How may I assist you?&quot;</text>

                  {/* Split Arrow */}
                  <path d="M200 45 L200 60" stroke="#64748b" strokeWidth="1.5"/>
                  <path d="M200 60 L100 75" stroke="#64748b" strokeWidth="1"/>
                  <path d="M200 60 L200 75" stroke="#64748b" strokeWidth="1"/>
                  <path d="M200 60 L300 75" stroke="#64748b" strokeWidth="1"/>

                  {/* Chunks */}
                  <rect x="30" y="80" width="100" height="25" rx="4" fill="#f43f5e" fillOpacity="0.3" stroke="#f43f5e" strokeWidth="1"/>
                  <text x="80" y="96" textAnchor="middle" fill="white" fontSize="8">Chunk 1</text>

                  <rect x="150" y="80" width="100" height="25" rx="4" fill="#ec4899" fillOpacity="0.3" stroke="#ec4899" strokeWidth="1"/>
                  <text x="200" y="96" textAnchor="middle" fill="white" fontSize="8">Chunk 2</text>

                  <rect x="270" y="80" width="100" height="25" rx="4" fill="#d946ef" fillOpacity="0.3" stroke="#d946ef" strokeWidth="1"/>
                  <text x="320" y="96" textAnchor="middle" fill="white" fontSize="8">Chunk 3</text>

                  {/* Parallel Processing */}
                  <path d="M80 110 L80 125" stroke="#f43f5e" strokeWidth="1"/>
                  <path d="M200 110 L200 125" stroke="#ec4899" strokeWidth="1"/>
                  <path d="M320 110 L320 125" stroke="#d946ef" strokeWidth="1"/>

                  {/* WebSocket Connections */}
                  <rect x="30" y="130" width="100" height="20" rx="3" fill="#f43f5e" fillOpacity="0.2"/>
                  <text x="80" y="143" textAnchor="middle" fill="#fda4af" fontSize="7">WS Connection 1</text>

                  <rect x="150" y="130" width="100" height="20" rx="3" fill="#ec4899" fillOpacity="0.2"/>
                  <text x="200" y="143" textAnchor="middle" fill="#f9a8d4" fontSize="7">WS Connection 2</text>

                  <rect x="270" y="130" width="100" height="20" rx="3" fill="#d946ef" fillOpacity="0.2"/>
                  <text x="320" y="143" textAnchor="middle" fill="#e9d5ff" fontSize="7">WS Connection 3</text>

                  {/* Timeline */}
                  <line x1="30" y1="175" x2="370" y2="175" stroke="#334155" strokeWidth="1"/>
                  <text x="80" y="188" textAnchor="middle" fill="#f43f5e" fontSize="7">75ms</text>
                  <text x="200" y="188" textAnchor="middle" fill="#ec4899" fontSize="7">200ms</text>
                  <text x="320" y="188" textAnchor="middle" fill="#d946ef" fontSize="7">400ms</text>

                  {/* Playback indicator */}
                  <rect x="80" y="160" width="250" height="8" rx="2" fill="url(#chunkGrad)" opacity="0.4"/>
                  <text x="205" y="168" textAnchor="middle" fill="white" fontSize="6">Seamless Audio Playback</text>
                </svg>
              </div>
              <div className="text-xs text-slate-500 text-center mt-2">
                Parallel chunk processing: User hears speech at 75ms, full dialogue by 400ms
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Buffer Management & Latency */}
        <div className="grid grid-cols-3 gap-4 mt-4">
          {/* Circular Buffer */}
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-4">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-pink-500" />
                Circular Buffer (512KB)
              </h3>
              <div className="relative h-20">
                <svg viewBox="0 0 200 70" className="w-full h-full">
                  {/* Buffer ring */}
                  <circle cx="100" cy="35" r="30" fill="none" stroke="#334155" strokeWidth="8"/>
                  <circle cx="100" cy="35" r="30" fill="none" stroke="#ec4899" strokeWidth="8"
                    strokeDasharray="120 69" strokeDashoffset="-20"/>

                  {/* Pointers */}
                  <circle cx="70" cy="35" r="4" fill="#10b981"/>
                  <text x="70" y="55" textAnchor="middle" fill="#10b981" fontSize="6">Read</text>

                  <circle cx="130" cy="35" r="4" fill="#f43f5e"/>
                  <text x="130" y="55" textAnchor="middle" fill="#f43f5e" fontSize="6">Write</text>

                  {/* Fill level */}
                  <text x="100" y="38" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">65%</text>
                </svg>
              </div>
            </CardContent>
          </Card>

          {/* Latency Budget */}
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-4">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-fuchsia-500" />
                Latency Budget
              </h3>
              <div className="space-y-2">
                {[
                  { step: "SSML Preprocessing", time: "5ms", color: "bg-rose-500" },
                  { step: "ElevenLabs TTFB", time: "75ms", color: "bg-pink-500" },
                  { step: "Audio Buffering", time: "50ms", color: "bg-fuchsia-500" },
                  { step: "Lip-Sync (parallel)", time: "100ms", color: "bg-violet-500" }
                ].map((item, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${item.color}`} />
                      <span className="text-xs text-slate-400">{item.step}</span>
                    </div>
                    <span className="text-xs font-mono text-white">{item.time}</span>
                  </div>
                ))}
                <div className="border-t border-slate-700 pt-2 flex items-center justify-between">
                  <span className="text-xs text-slate-300 font-semibold">Total E2E</span>
                  <span className="text-sm font-mono text-rose-400 font-bold">&lt;200ms</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Fallback Strategy */}
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-4">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-orange-500" />
                Fallback Chain
              </h3>
              <div className="space-y-2">
                {[
                  { level: "Primary", method: "WebSocket", timeout: "5s", color: "text-rose-400" },
                  { level: "Secondary", method: "REST API", timeout: "10s", color: "text-amber-400" },
                  { level: "Tertiary", method: "Local Cache", timeout: "-", color: "text-emerald-400" },
                  { level: "Fallback", method: "Silent Mode", timeout: "-", color: "text-slate-400" }
                ].map((item, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <span className={item.color}>{item.level}</span>
                    <span className="text-slate-400">{item.method}</span>
                    <span className="text-slate-500 font-mono">{item.timeout}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 text-xs text-slate-500">
                Exponential backoff: 1s → 2s → 4s (max 3 retries)
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </AvatarSlideWrapper>
  );
}
