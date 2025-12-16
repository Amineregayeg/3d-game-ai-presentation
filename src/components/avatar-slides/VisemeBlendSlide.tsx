"use client";

import { AvatarSlideWrapper } from "./AvatarSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface VisemeBlendSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function VisemeBlendSlide({ slideNumber, totalSlides }: VisemeBlendSlideProps) {
  const visemes = [
    { id: "A", phonemes: "/ɑ/, /ɔ/", example: "father", shape: "Wide open" },
    { id: "B/M/P", phonemes: "/b/, /m/, /p/", example: "bat, mat", shape: "Closed lips" },
    { id: "E", phonemes: "/e/, /i/", example: "see, pet", shape: "Smile" },
    { id: "F/V", phonemes: "/f/, /v/", example: "fit, van", shape: "Lip-teeth" },
    { id: "O", phonemes: "/o/, /ʊ/", example: "go, put", shape: "Rounded" },
    { id: "U", phonemes: "/u/", example: "blue", shape: "Rounded tense" }
  ];

  const blendShapes = [
    { name: "BS_Mouth_A", desc: "Open 'ah'", value: 0.8 },
    { name: "BS_Mouth_B", desc: "Closed 'oh'", value: 0.6 },
    { name: "BS_Mouth_C", desc: "Smile 'ee'", value: 0.7 },
    { name: "BS_Mouth_D", desc: "Rounded 'oo'", value: 0.5 },
    { name: "BS_Mouth_Wide", desc: "Smile width", value: 0.4 },
    { name: "BS_Jaw_Forward", desc: "Jaw protrusion", value: 0.3 }
  ];

  return (
    <AvatarSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Avatar Animation">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Viseme <span className="text-rose-400">→</span> Blend Shape <span className="text-pink-400">Mapping</span>
            </h2>
            <p className="text-slate-400">Phoneme-to-mouth-shape conversion for realistic avatar animation</p>
          </div>
          <Badge variant="outline" className="border-orange-500/50 text-orange-400 bg-orange-500/10">
            22-Viseme Standard
          </Badge>
        </div>

        <div className="grid grid-cols-3 gap-4 flex-1">
          {/* Viseme Table */}
          <Card className="bg-slate-800/50 border-slate-700/50 col-span-1">
            <CardContent className="p-4">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-rose-500" />
                Viseme Set (Core 6)
              </h3>
              <div className="space-y-2">
                {visemes.map((v) => (
                  <div key={v.id} className="bg-slate-900/50 rounded-lg p-2 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-rose-500 to-pink-500 flex items-center justify-center text-white text-sm font-bold">
                        {v.id.length > 3 ? v.id.substring(0, 3) : v.id}
                      </div>
                      <div>
                        <div className="text-xs font-mono text-rose-400">{v.phonemes}</div>
                        <div className="text-xs text-slate-500">{v.shape}</div>
                      </div>
                    </div>
                    <div className="text-xs text-slate-400 italic">&quot;{v.example}&quot;</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Mapping Diagram */}
          <Card className="bg-slate-800/50 border-slate-700/50 col-span-1">
            <CardContent className="p-4">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-pink-500" />
                Audio → Animation Pipeline
              </h3>
              <div className="relative h-full">
                <svg viewBox="0 0 250 280" className="w-full h-64">
                  <defs>
                    <linearGradient id="mapGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#f43f5e"/>
                      <stop offset="50%" stopColor="#ec4899"/>
                      <stop offset="100%" stopColor="#d946ef"/>
                    </linearGradient>
                  </defs>

                  {/* Audio Input */}
                  <rect x="80" y="10" width="90" height="35" rx="6" fill="#1e293b" stroke="#f43f5e" strokeWidth="1.5"/>
                  <text x="125" y="32" textAnchor="middle" fill="#f43f5e" fontSize="10">Audio Stream</text>

                  {/* Arrow */}
                  <path d="M125 50 L125 65" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

                  {/* Phoneme Extraction */}
                  <rect x="60" y="70" width="130" height="40" rx="6" fill="#f43f5e" fillOpacity="0.2" stroke="#f43f5e" strokeWidth="1"/>
                  <text x="125" y="88" textAnchor="middle" fill="#f43f5e" fontSize="9" fontWeight="bold">Phoneme Extraction</text>
                  <text x="125" y="102" textAnchor="middle" fill="#fda4af" fontSize="8">Forced Alignment</text>

                  {/* Arrow */}
                  <path d="M125 115 L125 130" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

                  {/* Viseme Mapping */}
                  <rect x="60" y="135" width="130" height="40" rx="6" fill="#ec4899" fillOpacity="0.2" stroke="#ec4899" strokeWidth="1"/>
                  <text x="125" y="153" textAnchor="middle" fill="#ec4899" fontSize="9" fontWeight="bold">Viseme Lookup</text>
                  <text x="125" y="167" textAnchor="middle" fill="#f9a8d4" fontSize="8">22-Viseme Mapping</text>

                  {/* Arrow */}
                  <path d="M125 180 L125 195" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

                  {/* Blend Shape Values */}
                  <rect x="50" y="200" width="150" height="40" rx="6" fill="#d946ef" fillOpacity="0.2" stroke="#d946ef" strokeWidth="1"/>
                  <text x="125" y="218" textAnchor="middle" fill="#d946ef" fontSize="9" fontWeight="bold">Blend Shape Values</text>
                  <text x="125" y="232" textAnchor="middle" fill="#e9d5ff" fontSize="8">Interpolated (0-1)</text>

                  {/* Arrow */}
                  <path d="M125 245 L125 260" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

                  {/* Avatar */}
                  <rect x="70" y="265" width="110" height="30" rx="6" fill="url(#mapGrad)" fillOpacity="0.4" stroke="#d946ef" strokeWidth="1"/>
                  <text x="125" y="284" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Avatar Mesh</text>

                  {/* Timeline indicator */}
                  <g transform="translate(210, 70)">
                    <text x="0" y="8" fill="#64748b" fontSize="7">t=0ms</text>
                    <text x="0" y="73" fill="#64748b" fontSize="7">t=50ms</text>
                    <text x="0" y="138" fill="#64748b" fontSize="7">t=100ms</text>
                    <text x="0" y="203" fill="#64748b" fontSize="7">t=150ms</text>
                  </g>
                </svg>
              </div>
            </CardContent>
          </Card>

          {/* Blend Shapes */}
          <Card className="bg-slate-800/50 border-slate-700/50 col-span-1">
            <CardContent className="p-4">
              <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-fuchsia-500" />
                Blend Shape Controls
              </h3>
              <div className="space-y-3">
                {blendShapes.map((bs) => (
                  <div key={bs.name} className="bg-slate-900/50 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <code className="text-xs font-mono text-fuchsia-400">{bs.name}</code>
                      <span className="text-xs text-slate-500">{bs.desc}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-fuchsia-500 to-pink-500 rounded-full transition-all"
                          style={{ width: `${bs.value * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-fuchsia-400 font-mono w-8">{bs.value.toFixed(1)}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* MetaHuman note */}
              <div className="mt-4 p-3 bg-violet-500/10 border border-violet-500/30 rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <svg className="w-4 h-4 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-xs text-violet-400 font-semibold">UE5 MetaHuman</span>
                </div>
                <span className="text-xs text-slate-400">Audio-driven lip-sync via Animation Blueprint with automatic viseme detection</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Bottom: Animation Timeline Preview */}
        <div className="mt-4 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="flex items-center gap-4 mb-3">
            <span className="text-sm text-rose-400 font-semibold">Animation Timeline</span>
            <span className="text-xs text-slate-500">30 FPS interpolation</span>
          </div>
          <div className="relative h-16">
            <svg viewBox="0 0 800 50" className="w-full h-full">
              {/* Timeline base */}
              <line x1="40" y1="40" x2="760" y2="40" stroke="#334155" strokeWidth="2"/>

              {/* Time markers */}
              {[0, 100, 200, 300, 400, 500, 600].map((ms, i) => (
                <g key={ms} transform={`translate(${40 + i * 120}, 0)`}>
                  <line x1="0" y1="35" x2="0" y2="45" stroke="#64748b" strokeWidth="1"/>
                  <text x="0" y="50" textAnchor="middle" fill="#64748b" fontSize="8">{ms}ms</text>
                </g>
              ))}

              {/* Viseme sequence */}
              {[
                { x: 60, w: 80, label: "A", color: "#f43f5e" },
                { x: 160, w: 60, label: "E", color: "#ec4899" },
                { x: 240, w: 100, label: "O", color: "#d946ef" },
                { x: 360, w: 80, label: "B/M", color: "#a78bfa" },
                { x: 460, w: 120, label: "A", color: "#f43f5e" },
                { x: 600, w: 80, label: "Neutral", color: "#64748b" }
              ].map((v, i) => (
                <g key={i}>
                  <rect
                    x={v.x}
                    y="10"
                    width={v.w}
                    height="20"
                    rx="4"
                    fill={v.color}
                    opacity="0.4"
                    stroke={v.color}
                    strokeWidth="1"
                  />
                  <text
                    x={v.x + v.w / 2}
                    y="24"
                    textAnchor="middle"
                    fill="white"
                    fontSize="9"
                    fontWeight="bold"
                  >
                    {v.label}
                  </text>
                </g>
              ))}

              {/* Playhead */}
              <line x1="300" y1="5" x2="300" y2="45" stroke="#10b981" strokeWidth="2"/>
              <polygon points="295,5 305,5 300,0" fill="#10b981"/>
            </svg>
          </div>
        </div>
      </div>
    </AvatarSlideWrapper>
  );
}
