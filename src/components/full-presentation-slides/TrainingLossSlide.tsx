"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface TrainingLossSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function TrainingLossSlide({ slideNumber, totalSlides }: TrainingLossSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Training Strategy">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className="text-4xl font-bold text-white mb-2">
            VoxFormer Training: <span className="text-amber-400">Loss Design</span> & <span className="text-cyan-400">Current Status</span>
          </h2>
        </div>

        <div className="flex-1 grid grid-cols-2 gap-6">
          {/* Left: Hybrid Loss Diagram */}
          <div className="flex flex-col">
            <h3 className="text-lg font-semibold text-white mb-3 text-center">Hybrid Loss Design</h3>
            <div className="flex-1 bg-slate-800/30 rounded-xl p-4 border border-slate-700/50">
              <svg viewBox="0 0 400 280" className="w-full h-auto">
                <defs>
                  <linearGradient id="ctcLossGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.9"/>
                    <stop offset="100%" stopColor="#d97706" stopOpacity="0.9"/>
                  </linearGradient>
                  <linearGradient id="ceLossGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#ec4899" stopOpacity="0.9"/>
                    <stop offset="100%" stopColor="#db2777" stopOpacity="0.9"/>
                  </linearGradient>
                </defs>

                {/* Formula */}
                <text x="200" y="30" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold">
                  L = λ<tspan fill="#f59e0b">ctc</tspan> · L<tspan fill="#f59e0b">CTC</tspan> + λ<tspan fill="#ec4899">ce</tspan> · L<tspan fill="#ec4899">CE</tspan>
                </text>

                {/* Encoder output */}
                <rect x="150" y="50" width="100" height="30" rx="6" fill="#1e293b" stroke="#8b5cf6" strokeWidth="1.5"/>
                <text x="200" y="70" textAnchor="middle" fill="#8b5cf6" fontSize="10">Encoder Output</text>

                {/* Split */}
                <path d="M 150 80 L 80 110" stroke="#64748b" strokeWidth="1.5"/>
                <path d="M 250 80 L 320 110" stroke="#64748b" strokeWidth="1.5"/>

                {/* CTC Branch */}
                <g transform="translate(20, 110)">
                  <rect width="120" height="100" rx="8" fill="url(#ctcLossGrad)"/>
                  <text x="60" y="25" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">CTC Head</text>
                  <rect x="10" y="35" width="100" height="22" rx="4" fill="rgba(0,0,0,0.3)"/>
                  <text x="60" y="51" textAnchor="middle" fill="#fcd34d" fontSize="9">Alignment-free</text>
                  <rect x="10" y="62" width="100" height="22" rx="4" fill="rgba(0,0,0,0.3)"/>
                  <text x="60" y="78" textAnchor="middle" fill="#fcd34d" fontSize="9">Fast convergence</text>
                </g>

                {/* CE Branch */}
                <g transform="translate(260, 110)">
                  <rect width="120" height="100" rx="8" fill="url(#ceLossGrad)"/>
                  <text x="60" y="25" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">CE Decoder</text>
                  <rect x="10" y="35" width="100" height="22" rx="4" fill="rgba(0,0,0,0.3)"/>
                  <text x="60" y="51" textAnchor="middle" fill="#fbcfe8" fontSize="9">Better final WER</text>
                  <rect x="10" y="62" width="100" height="22" rx="4" fill="rgba(0,0,0,0.3)"/>
                  <text x="60" y="78" textAnchor="middle" fill="#fbcfe8" fontSize="9">Token accuracy</text>
                </g>

                {/* Combine */}
                <path d="M 80 215 L 150 245" stroke="#f59e0b" strokeWidth="1.5"/>
                <path d="M 320 215 L 250 245" stroke="#ec4899" strokeWidth="1.5"/>

                {/* Total Loss */}
                <rect x="140" y="240" width="120" height="30" rx="6" fill="#1e293b" stroke="#10b981" strokeWidth="1.5"/>
                <text x="200" y="260" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">Total Loss</text>

                {/* Weights */}
                <text x="80" y="235" textAnchor="middle" fill="#f59e0b" fontSize="10">λ=0.3</text>
                <text x="320" y="235" textAnchor="middle" fill="#ec4899" fontSize="10">λ=0.7</text>
              </svg>
            </div>
          </div>

          {/* Right: Training Progress */}
          <div className="flex flex-col">
            <h3 className="text-lg font-semibold text-white mb-3 text-center">Training Progress</h3>
            <div className="flex-1 bg-slate-800/30 rounded-xl p-4 border border-slate-700/50">
              {/* Stage Progress */}
              <div className="space-y-4">
                {/* Stage 1 */}
                <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-emerald-400 font-medium">Stage 1: CTC Pre-training</span>
                    <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
                      Complete
                    </Badge>
                  </div>
                  <div className="w-full h-2 bg-slate-700 rounded-full">
                    <div className="w-full h-full bg-emerald-500 rounded-full" />
                  </div>
                  <div className="text-xs text-slate-500 mt-1">10 epochs on LibriSpeech clean-100</div>
                </div>

                {/* Stage 2 */}
                <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-amber-400 font-medium">Stage 2: Hybrid CTC+CE</span>
                    <Badge variant="outline" className="border-amber-500/50 text-amber-400 bg-amber-500/10">
                      Pending GPU
                    </Badge>
                  </div>
                  <div className="w-full h-2 bg-slate-700 rounded-full">
                    <div className="w-[15%] h-full bg-amber-500 rounded-full" />
                  </div>
                  <div className="text-xs text-slate-500 mt-1">~15% complete, waiting for GPU time</div>
                </div>

                {/* Stage 3 */}
                <div className="p-3 bg-slate-500/10 border border-slate-500/30 rounded-lg opacity-60">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-slate-400 font-medium">Stage 3: Full Fine-tuning</span>
                    <Badge variant="outline" className="border-slate-500/50 text-slate-400 bg-slate-500/10">
                      Planned
                    </Badge>
                  </div>
                  <div className="w-full h-2 bg-slate-700 rounded-full" />
                  <div className="text-xs text-slate-500 mt-1">LibriSpeech full 960h</div>
                </div>
              </div>

              {/* Target metrics */}
              <div className="mt-4 p-3 bg-slate-900/50 rounded-lg">
                <div className="text-xs text-slate-500 mb-2">TARGET METRICS</div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center">
                    <div className="text-xl font-bold text-cyan-400">&lt;15%</div>
                    <div className="text-xs text-slate-500">Target WER</div>
                  </div>
                  <div className="text-center">
                    <div className="text-xl font-bold text-purple-400">142M</div>
                    <div className="text-xs text-slate-500">Parameters</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Key Points */}
        <div className="grid grid-cols-4 gap-4 mt-4">
          {[
            { text: "Hybrid CTC + CE loss", color: "amber" },
            { text: "Stage-wise training", color: "purple" },
            { text: "Compute constraints", color: "slate" },
            { text: "WER evaluation pending", color: "cyan" }
          ].map((point) => (
            <div key={point.text} className={`p-3 bg-${point.color}-500/10 border border-${point.color}-500/30 rounded-lg text-center`}>
              <span className="text-slate-300 text-sm">{point.text}</span>
            </div>
          ))}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
