"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface ImplementationRoadmapSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function ImplementationRoadmapSlide({ slideNumber, totalSlides }: ImplementationRoadmapSlideProps) {
  const phases = [
    {
      phase: "1",
      title: "Core Architecture",
      weeks: "Day 1",
      status: "current",
      tasks: ["WeightedLayerSum module", "WavLMAdapter (768→512)", "ConformerBlock implementation", "ZipformerEncoder (6 blocks)"],
      color: "cyan"
    },
    {
      phase: "2",
      title: "Training Infra",
      weeks: "Day 2",
      status: "upcoming",
      tasks: ["HybridCTCAttentionLoss", "ASRDataset for LibriSpeech", "BPE tokenizer (2K vocab)", "Trainer with checkpointing"],
      color: "emerald"
    },
    {
      phase: "3",
      title: "Local Validation",
      weeks: "Day 3",
      status: "upcoming",
      tasks: ["Train on 100h subset", "WER evaluation pipeline", "Debug gradient issues", "Prepare A100 configs"],
      color: "purple"
    },
    {
      phase: "4",
      title: "Stage 1 (A100)",
      weeks: "Day 4",
      status: "upcoming",
      tasks: ["Full LibriSpeech 960h", "WavLM frozen training", "30 GPU hours ($12)", "Target: WER < 5%"],
      color: "pink"
    },
    {
      phase: "5",
      title: "Stage 2 & 3",
      weeks: "Day 5",
      status: "upcoming",
      tasks: ["Unfreeze WavLM top 3", "Gaming domain fine-tune", "15 GPU hours ($6)", "Target: WER < 3.5%"],
      color: "amber"
    },
    {
      phase: "6",
      title: "Optimize & Release",
      weeks: "Day 6-7",
      status: "upcoming",
      tasks: ["Streaming inference", "KV-cache decoder", "ONNX + INT8 export", "gRPC server + clients"],
      color: "rose"
    }
  ];

  const colorMap: Record<string, { bg: string; border: string; text: string; badge: string }> = {
    cyan: { bg: "bg-cyan-500/10", border: "border-cyan-500/50", text: "text-cyan-400", badge: "bg-cyan-500/20" },
    emerald: { bg: "bg-emerald-500/10", border: "border-emerald-500/50", text: "text-emerald-400", badge: "bg-emerald-500/20" },
    purple: { bg: "bg-purple-500/10", border: "border-purple-500/50", text: "text-purple-400", badge: "bg-purple-500/20" },
    pink: { bg: "bg-pink-500/10", border: "border-pink-500/50", text: "text-pink-400", badge: "bg-pink-500/20" },
    amber: { bg: "bg-amber-500/10", border: "border-amber-500/50", text: "text-amber-400", badge: "bg-amber-500/20" },
    rose: { bg: "bg-rose-500/10", border: "border-rose-500/50", text: "text-rose-400", badge: "bg-rose-500/20" }
  };

  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Implementation Plan">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Implementation <span className="text-cyan-400">Roadmap</span>
            </h2>
            <p className="text-slate-400">7-day AI-accelerated development plan for VoxFormer STT</p>
          </div>
          <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10 px-4 py-2">
            $20 Budget | 50 GPU Hours
          </Badge>
        </div>

        {/* Timeline */}
        <div className="flex-1">
          {/* Timeline line */}
          <div className="relative">
            <div className="absolute top-12 left-0 right-0 h-1 bg-slate-800 rounded-full">
              <div className="absolute inset-y-0 left-0 w-1/6 bg-gradient-to-r from-cyan-500 to-cyan-400 rounded-full" />
            </div>

            {/* Phase cards */}
            <div className="grid grid-cols-6 gap-4 pt-20">
              {phases.map((phase) => (
                <div key={phase.phase} className="relative">
                  {/* Timeline dot */}
                  <div className={`absolute -top-10 left-1/2 -translate-x-1/2 w-6 h-6 rounded-full ${colorMap[phase.color].bg} border-2 ${colorMap[phase.color].border} flex items-center justify-center z-10`}>
                    <span className={`text-xs font-bold ${colorMap[phase.color].text}`}>{phase.phase}</span>
                  </div>

                  {/* Phase card */}
                  <Card className={`${colorMap[phase.color].bg} ${colorMap[phase.color].border} border h-full ${phase.status === 'current' ? 'ring-2 ring-cyan-500/50' : ''}`}>
                    <CardContent className="p-3">
                      <div className="flex items-center justify-between mb-2">
                        <Badge className={`${colorMap[phase.color].badge} ${colorMap[phase.color].text} text-xs`}>
                          W{phase.weeks}
                        </Badge>
                        {phase.status === 'current' && (
                          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                        )}
                      </div>
                      <h3 className={`text-sm font-semibold ${colorMap[phase.color].text} mb-2`}>
                        {phase.title}
                      </h3>
                      <ul className="space-y-1">
                        {phase.tasks.map((task, idx) => (
                          <li key={idx} className="text-xs text-slate-400 flex items-start gap-1.5">
                            <span className={`w-1 h-1 rounded-full mt-1.5 ${colorMap[phase.color].text.replace('text-', 'bg-')}`} />
                            {task}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom: Target Metrics */}
        <div className="mt-8 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="flex items-center justify-between">
            <div className="text-sm text-slate-400">
              <span className="font-semibold text-white">Target Performance</span> (LibriSpeech test-clean)
            </div>
            <div className="flex gap-8">
              <div className="text-center">
                <div className="text-2xl font-bold text-cyan-400">{"<"}3.5%</div>
                <div className="text-xs text-slate-500">WER</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400">{"<"}200ms</div>
                <div className="text-xs text-slate-500">Latency</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-emerald-400">142M</div>
                <div className="text-xs text-slate-500">Parameters</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-pink-400">$20</div>
                <div className="text-xs text-slate-500">Training Cost</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
