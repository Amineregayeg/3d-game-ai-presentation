"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface DSPPipelineSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function DSPPipelineSlide({ slideNumber, totalSlides }: DSPPipelineSlideProps) {
  const stages = [
    {
      num: "01",
      title: "Signal Conditioning",
      color: "cyan",
      items: ["DC Offset Removal", "Pre-Emphasis Filter", "Sample Rate Conv."]
    },
    {
      num: "02",
      title: "Voice Activity Detection",
      color: "emerald",
      items: ["Energy-Based VAD", "Spectral Entropy", "Neural VAD (DNN)"]
    },
    {
      num: "03",
      title: "Noise Estimation",
      color: "purple",
      items: ["MCRA Algorithm", "Adaptive Tracking", "SNR Estimation"]
    },
    {
      num: "04",
      title: "Noise Reduction",
      color: "pink",
      items: ["Spectral Subtraction", "Wiener Filter", "MMSE-STSA"]
    },
    {
      num: "05",
      title: "Echo Cancellation",
      color: "amber",
      items: ["Adaptive Filter", "NLMS/RLS", "Double-Talk Det."]
    },
    {
      num: "06",
      title: "Voice Isolation",
      color: "rose",
      items: ["Deep Attractor Net", "Source Separation", "Mask Estimation"]
    }
  ];

  const colorMap: Record<string, { bg: string; border: string; text: string; glow: string }> = {
    cyan: { bg: "bg-cyan-500/20", border: "border-cyan-500/50", text: "text-cyan-400", glow: "shadow-cyan-500/20" },
    emerald: { bg: "bg-emerald-500/20", border: "border-emerald-500/50", text: "text-emerald-400", glow: "shadow-emerald-500/20" },
    purple: { bg: "bg-purple-500/20", border: "border-purple-500/50", text: "text-purple-400", glow: "shadow-purple-500/20" },
    pink: { bg: "bg-pink-500/20", border: "border-pink-500/50", text: "text-pink-400", glow: "shadow-pink-500/20" },
    amber: { bg: "bg-amber-500/20", border: "border-amber-500/50", text: "text-amber-400", glow: "shadow-amber-500/20" },
    rose: { bg: "bg-rose-500/20", border: "border-rose-500/50", text: "text-rose-400", glow: "shadow-rose-500/20" }
  };

  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Digital Signal Processing">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Voice Isolation <span className="text-cyan-400">Pipeline</span>
            </h2>
            <p className="text-slate-400">6-stage DSP pipeline for robust voice extraction</p>
          </div>
          <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
            All Low-Level Implementation
          </Badge>
        </div>

        {/* Pipeline Visualization */}
        <div className="flex-1 relative">
          {/* Input/Output Labels */}
          <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4">
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-slate-800 border-2 border-slate-600 flex items-center justify-center">
                <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              </div>
              <span className="text-xs text-slate-500 mt-2">Noisy Input</span>
            </div>
          </div>

          <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4">
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-cyan-500 to-purple-500 flex items-center justify-center shadow-lg shadow-cyan-500/30">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
              </div>
              <span className="text-xs text-cyan-400 mt-2">Clean Voice</span>
            </div>
          </div>

          {/* Pipeline Stages */}
          <div className="grid grid-cols-6 gap-3 px-20 h-full items-center">
            {stages.map((stage, idx) => (
              <div key={stage.num} className="relative">
                {/* Connection Arrow */}
                {idx < stages.length - 1 && (
                  <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 z-10">
                    <svg className="w-6 h-6 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                )}

                {/* Stage Card */}
                <div className={`p-4 rounded-xl ${colorMap[stage.color].bg} border ${colorMap[stage.color].border} shadow-lg ${colorMap[stage.color].glow} h-full`}>
                  <div className={`text-3xl font-bold ${colorMap[stage.color].text} opacity-30 mb-1`}>
                    {stage.num}
                  </div>
                  <h3 className="text-sm font-semibold text-white mb-3 leading-tight">
                    {stage.title}
                  </h3>
                  <div className="space-y-1.5">
                    {stage.items.map((item) => (
                      <div key={item} className="text-xs text-slate-400 flex items-center gap-1.5">
                        <div className={`w-1 h-1 rounded-full ${colorMap[stage.color].text.replace('text-', 'bg-')}`} />
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom: Key Algorithms */}
        <div className="mt-6 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div>
                <span className="text-xs text-slate-500 uppercase tracking-wider">Core DSP</span>
                <div className="text-sm text-slate-300">Custom FFT (Cooley-Tukey) | FIR/IIR Filters | Adaptive NLMS</div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-cyan-400">{"<"}10ms</div>
                <div className="text-xs text-slate-500">Latency</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">{">"}-20dB</div>
                <div className="text-xs text-slate-500">Noise Reduction</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
