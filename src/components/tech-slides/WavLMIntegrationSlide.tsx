"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface WavLMIntegrationSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function WavLMIntegrationSlide({ slideNumber, totalSlides }: WavLMIntegrationSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Feature Extraction">
      <div className="flex flex-col h-full">
        <h2 className="text-4xl font-bold text-white mb-2">
          <span className="text-purple-400">WavLM</span> Integration
        </h2>
        <p className="text-slate-400 mb-6">Pretrained audio backbone with learnable layer combination</p>

        <div className="flex-1 grid grid-cols-3 gap-6">
          {/* WavLM Architecture */}
          <Card className="col-span-2 bg-slate-800/30 border-slate-700/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Badge className="bg-purple-500/20 text-purple-400">WavLM-Base</Badge>
                Frozen Feature Extractor
              </CardTitle>
            </CardHeader>
            <CardContent>
              <svg viewBox="0 0 600 380" className="w-full h-full">
                <defs>
                  <linearGradient id="wavlmGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#a855f7" stopOpacity="0.8"/>
                    <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.8"/>
                  </linearGradient>
                  <linearGradient id="layerGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8"/>
                    <stop offset="100%" stopColor="#0891b2" stopOpacity="0.8"/>
                  </linearGradient>
                  <linearGradient id="adapterGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#10b981" stopOpacity="0.8"/>
                    <stop offset="100%" stopColor="#059669" stopOpacity="0.8"/>
                  </linearGradient>
                </defs>

                {/* Raw Audio Input */}
                <g transform="translate(20, 160)">
                  <rect width="80" height="60" rx="6" fill="#1e293b" stroke="#334155" strokeWidth="2"/>
                  <text x="40" y="28" textAnchor="middle" fill="#94a3b8" fontSize="10">Raw Audio</text>
                  <text x="40" y="42" textAnchor="middle" fill="#64748b" fontSize="8">16kHz</text>
                  {/* Waveform */}
                  <path d="M15 52 Q25 45 35 52 Q45 59 55 52 Q65 45 70 52" stroke="#06b6d4" fill="none" strokeWidth="1.5"/>
                </g>

                {/* Arrow */}
                <path d="M105 190 L135 190" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowWavlm)"/>

                {/* WavLM Block */}
                <g transform="translate(140, 40)">
                  <rect width="180" height="300" rx="10" fill="url(#wavlmGrad)" opacity="0.3"/>
                  <rect width="180" height="300" rx="10" fill="none" stroke="#a855f7" strokeWidth="2"/>
                  <text x="90" y="25" textAnchor="middle" fill="#c4b5fd" fontSize="12" fontWeight="bold">WavLM-Base</text>
                  <text x="90" y="42" textAnchor="middle" fill="#a78bfa" fontSize="9">95M params (frozen)</text>

                  {/* CNN Feature Extractor */}
                  <rect x="15" y="55" width="150" height="35" rx="4" fill="#1e293b"/>
                  <text x="90" y="77" textAnchor="middle" fill="#e9d5ff" fontSize="9">CNN Feature Extractor</text>

                  {/* Transformer Layers */}
                  {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((i) => (
                    <g key={i} transform={`translate(15, ${100 + i * 16})`}>
                      <rect width="150" height="14" rx="2" fill={i >= 9 ? "#7c3aed" : "#374151"} opacity={i >= 9 ? 0.6 : 0.5}/>
                      <text x="75" y="10" textAnchor="middle" fill={i >= 9 ? "#e9d5ff" : "#94a3b8"} fontSize="7">
                        Layer {i + 1} {i >= 9 ? "(unfrozen Stage 2)" : ""}
                      </text>
                    </g>
                  ))}
                </g>

                {/* Output arrows from each layer */}
                {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((i) => (
                  <path key={i} d={`M320 ${107 + i * 16} L340 ${107 + i * 16}`} stroke="#a855f7" strokeWidth="1" opacity={0.3 + i * 0.05}/>
                ))}

                {/* Weighted Layer Sum */}
                <g transform="translate(345, 80)">
                  <rect width="120" height="220" rx="8" fill="url(#layerGrad)"/>
                  <text x="60" y="22" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Weighted</text>
                  <text x="60" y="35" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Layer Sum</text>
                  <text x="60" y="52" textAnchor="middle" fill="#a5f3fc" fontSize="8">Learnable weights</text>

                  {/* Weight visualization */}
                  {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((i) => {
                    const weight = 0.05 + (i / 11) * 0.12;
                    return (
                      <g key={i} transform={`translate(10, ${65 + i * 12})`}>
                        <rect width={weight * 600} height="8" rx="1" fill="#0f172a" opacity="0.5"/>
                        <rect width={weight * 600} height="8" rx="1" fill="#06b6d4" opacity={0.4 + weight * 2}/>
                        <text x="105" y="7" textAnchor="end" fill="#67e8f9" fontSize="6">w{i + 1}</text>
                      </g>
                    );
                  })}

                  <text x="60" y="215" textAnchor="middle" fill="#a5f3fc" fontSize="8">softmax(weights)</text>
                </g>

                {/* Arrow to adapter */}
                <path d="M470 190 L500 190" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowWavlm)"/>

                {/* Adapter Module */}
                <g transform="translate(505, 120)">
                  <rect width="80" height="140" rx="6" fill="url(#adapterGrad)"/>
                  <text x="40" y="20" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Adapter</text>
                  <text x="40" y="35" textAnchor="middle" fill="#a7f3d0" fontSize="8">768 → 512</text>

                  <rect x="8" y="50" width="64" height="18" rx="2" fill="#0f172a" opacity="0.5"/>
                  <text x="40" y="63" textAnchor="middle" fill="#a7f3d0" fontSize="7">LayerNorm</text>

                  <rect x="8" y="72" width="64" height="18" rx="2" fill="#0f172a" opacity="0.5"/>
                  <text x="40" y="85" textAnchor="middle" fill="#a7f3d0" fontSize="7">Linear 768→512</text>

                  <rect x="8" y="94" width="64" height="18" rx="2" fill="#0f172a" opacity="0.5"/>
                  <text x="40" y="107" textAnchor="middle" fill="#a7f3d0" fontSize="7">GELU + Drop</text>

                  <rect x="8" y="116" width="64" height="18" rx="2" fill="#0f172a" opacity="0.5"/>
                  <text x="40" y="129" textAnchor="middle" fill="#a7f3d0" fontSize="7">Linear 512→512</text>
                </g>

                {/* Output label */}
                <text x="545" y="280" textAnchor="middle" fill="#10b981" fontSize="9">To Encoder</text>
                <text x="545" y="295" textAnchor="middle" fill="#64748b" fontSize="8">512-dim @ 50fps</text>

                <defs>
                  <marker id="arrowWavlm" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
                  </marker>
                </defs>
              </svg>
            </CardContent>
          </Card>

          {/* Key Features */}
          <div className="space-y-4">
            <Card className="bg-slate-800/30 border-slate-700/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-slate-400">WavLM Specs</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {[
                  { label: "Pretrained Hours", value: "94,000h", color: "text-purple-400" },
                  { label: "Parameters", value: "95M", color: "text-cyan-400" },
                  { label: "Output Dim", value: "768", color: "text-emerald-400" },
                  { label: "Layers", value: "12", color: "text-pink-400" },
                  { label: "Output FPS", value: "50", color: "text-amber-400" },
                ].map((item) => (
                  <div key={item.label} className="flex justify-between items-center p-2 bg-slate-900/30 rounded">
                    <span className="text-xs text-slate-400">{item.label}</span>
                    <span className={`text-sm font-mono font-bold ${item.color}`}>{item.value}</span>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card className="bg-slate-800/30 border-slate-700/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-slate-400">Training Stages</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-cyan-400">Stage 1: Frozen</span>
                    <span className="text-slate-500">30h GPU</span>
                  </div>
                  <Progress value={66} className="h-1.5" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-purple-400">Stage 2: Top 3 Unfrozen</span>
                    <span className="text-slate-500">5h GPU</span>
                  </div>
                  <Progress value={11} className="h-1.5" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-pink-400">Stage 3: Gaming</span>
                    <span className="text-slate-500">10h GPU</span>
                  </div>
                  <Progress value={22} className="h-1.5" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800/30 border-slate-700/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-slate-400">Why Weighted Sum?</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-xs text-slate-400 space-y-2">
                  <p>Different layers capture different features:</p>
                  <div className="space-y-1 ml-2">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-purple-400"/>
                      <span>Lower: acoustic</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-cyan-400"/>
                      <span>Middle: phonetic</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-emerald-400"/>
                      <span>Upper: semantic</span>
                    </div>
                  </div>
                  <p className="text-emerald-400 font-medium mt-2">+0.2-0.5% WER improvement</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
