"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface TrainingStrategySlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function TrainingStrategySlide({ slideNumber, totalSlides }: TrainingStrategySlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Training Configuration">
      <div className="flex flex-col h-full">
        <h2 className="text-4xl font-bold text-white mb-2">
          Training <span className="text-pink-400">Strategy</span>
        </h2>
        <p className="text-slate-400 mb-6">Hybrid CTC + Cross-Entropy loss with advanced optimization</p>

        <div className="flex-1 grid grid-cols-3 gap-6">
          {/* Loss Function */}
          <Card className="bg-slate-800/30 border-slate-700/50">
            <CardHeader>
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Badge className="bg-pink-500/20 text-pink-400">Loss Function</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Loss visualization */}
              <div className="p-4 bg-slate-900/50 rounded-lg">
                <div className="text-center mb-4">
                  <span className="text-2xl font-mono text-white">L = </span>
                  <span className="text-2xl font-mono text-cyan-400">λ<sub>ce</sub></span>
                  <span className="text-2xl font-mono text-white">L<sub>CE</sub> + </span>
                  <span className="text-2xl font-mono text-purple-400">λ<sub>ctc</sub></span>
                  <span className="text-2xl font-mono text-white">L<sub>CTC</sub></span>
                </div>

                <div className="flex gap-4 justify-center">
                  <div className="text-center">
                    <div className="w-16 h-16 rounded-full bg-cyan-500/20 border-2 border-cyan-500 flex items-center justify-center">
                      <span className="text-xl font-bold text-cyan-400">0.7</span>
                    </div>
                    <span className="text-xs text-slate-500 mt-1">CE Weight</span>
                  </div>
                  <div className="text-center">
                    <div className="w-16 h-16 rounded-full bg-purple-500/20 border-2 border-purple-500 flex items-center justify-center">
                      <span className="text-xl font-bold text-purple-400">0.3</span>
                    </div>
                    <span className="text-xs text-slate-500 mt-1">CTC Weight</span>
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <div className="p-2 bg-cyan-500/10 border border-cyan-500/30 rounded">
                  <div className="text-xs text-cyan-400 font-semibold">Cross-Entropy (Decoder)</div>
                  <div className="text-xs text-slate-400">Token-level supervision</div>
                </div>
                <div className="p-2 bg-purple-500/10 border border-purple-500/30 rounded">
                  <div className="text-xs text-purple-400 font-semibold">CTC (Encoder)</div>
                  <div className="text-xs text-slate-400">Alignment-free, auxiliary</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Optimization */}
          <Card className="bg-slate-800/30 border-slate-700/50">
            <CardHeader>
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Badge className="bg-emerald-500/20 text-emerald-400">Optimization</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-2">
                {[
                  { label: "Optimizer", value: "AdamW", color: "emerald" },
                  { label: "Learning Rate", value: "1e-4", color: "cyan" },
                  { label: "Weight Decay", value: "0.01", color: "purple" },
                  { label: "Betas", value: "(0.9, 0.98)", color: "pink" },
                  { label: "Gradient Clip", value: "1.0", color: "amber" }
                ].map((item) => (
                  <div key={item.label} className="flex justify-between items-center p-2 bg-slate-900/30 rounded">
                    <span className="text-xs text-slate-400">{item.label}</span>
                    <span className={`text-sm font-mono font-bold text-${item.color}-400`}>{item.value}</span>
                  </div>
                ))}
              </div>

              {/* LR Schedule */}
              <div className="mt-4">
                <div className="text-xs text-slate-500 mb-2">Learning Rate Schedule</div>
                <svg viewBox="0 0 200 80" className="w-full">
                  <defs>
                    <linearGradient id="lrGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#06b6d4"/>
                      <stop offset="100%" stopColor="#8b5cf6"/>
                    </linearGradient>
                  </defs>

                  {/* Axes */}
                  <line x1="20" y1="60" x2="180" y2="60" stroke="#475569" strokeWidth="1"/>
                  <line x1="20" y1="60" x2="20" y2="10" stroke="#475569" strokeWidth="1"/>

                  {/* Warmup + Cosine decay curve */}
                  <path
                    d="M20 60 L50 15 Q100 15 180 55"
                    fill="none"
                    stroke="url(#lrGrad)"
                    strokeWidth="2"
                  />

                  {/* Labels */}
                  <text x="35" y="75" fill="#94a3b8" fontSize="8">Warmup</text>
                  <text x="110" y="75" fill="#94a3b8" fontSize="8">Cosine Decay</text>
                  <text x="5" y="15" fill="#06b6d4" fontSize="8">LR</text>
                </svg>
                <div className="flex justify-between text-xs text-slate-500">
                  <span>10K warmup</span>
                  <span>500K total steps</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 3-Stage Training */}
          <Card className="bg-slate-800/30 border-slate-700/50">
            <CardHeader>
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Badge className="bg-amber-500/20 text-amber-400">3-Stage Training</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Stage 1 */}
              <div className="p-2 bg-cyan-500/10 border border-cyan-500/30 rounded">
                <div className="flex justify-between items-center">
                  <div className="text-xs text-cyan-400 font-semibold">Stage 1: LibriSpeech</div>
                  <span className="text-xs text-slate-500">30h GPU</span>
                </div>
                <div className="text-xs text-slate-400 mt-1">WavLM frozen, train enc+dec</div>
                <Progress value={60} className="h-1 mt-1" />
              </div>

              {/* Stage 2 */}
              <div className="p-2 bg-purple-500/10 border border-purple-500/30 rounded">
                <div className="flex justify-between items-center">
                  <div className="text-xs text-purple-400 font-semibold">Stage 2: Fine-tune WavLM</div>
                  <span className="text-xs text-slate-500">5h GPU</span>
                </div>
                <div className="text-xs text-slate-400 mt-1">Unfreeze top 3 layers, 10x lower LR</div>
                <Progress value={10} className="h-1 mt-1" />
              </div>

              {/* Stage 3 */}
              <div className="p-2 bg-pink-500/10 border border-pink-500/30 rounded">
                <div className="flex justify-between items-center">
                  <div className="text-xs text-pink-400 font-semibold">Stage 3: Gaming Domain</div>
                  <span className="text-xs text-slate-500">10h GPU</span>
                </div>
                <div className="text-xs text-slate-400 mt-1">Fine-tune on gaming vocab</div>
                <Progress value={20} className="h-1 mt-1" />
              </div>

              {/* Total cost */}
              <div className="mt-3 p-2 bg-emerald-500/10 border border-emerald-500/30 rounded text-center">
                <div className="text-lg font-bold text-emerald-400">$20 Total</div>
                <div className="text-xs text-slate-500">45h A100 @ $0.40/hr</div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Bottom: Training Config Summary */}
        <div className="mt-6 grid grid-cols-5 gap-4">
          {[
            { label: "Batch Size", value: "32 × 4 GPU", icon: "B" },
            { label: "Grad Accum", value: "4 steps", icon: "G" },
            { label: "Mixed Precision", value: "FP16", icon: "F" },
            { label: "Label Smooth", value: "0.1", icon: "L" },
            { label: "Total Steps", value: "500K", icon: "S" }
          ].map((item) => (
            <div key={item.label} className="p-3 bg-slate-800/50 border border-slate-700/50 rounded-lg text-center">
              <div className="w-8 h-8 mx-auto mb-2 rounded-full bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center">
                <span className="text-cyan-400 text-sm font-bold">{item.icon}</span>
              </div>
              <div className="text-sm font-bold text-white">{item.value}</div>
              <div className="text-xs text-slate-500">{item.label}</div>
            </div>
          ))}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
