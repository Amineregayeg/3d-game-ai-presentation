"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Cpu,
  Database,
  Zap,
  TrendingDown,
  Server,
  CheckCircle2,
  PlayCircle,
} from "lucide-react";

interface TrainingProgressSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export default function TrainingProgressSlide({
  slideNumber,
  totalSlides,
}: TrainingProgressSlideProps) {
  // Stage 1 results (December 9, 2025) - WavLM frozen, 20 epochs
  const stage1Epochs = [
    { epoch: 0, totalLoss: 7.13, ctcLoss: 2.85, ceLoss: 4.28, lr: "1.0e-04" },
    { epoch: 5, totalLoss: 1.04, ctcLoss: 0.42, ceLoss: 0.62, lr: "8.0e-05" },
    { epoch: 10, totalLoss: 0.90, ctcLoss: 0.36, ceLoss: 0.54, lr: "6.0e-05" },
    { epoch: 19, totalLoss: 1.01, ctcLoss: 0.40, ceLoss: 0.60, lr: "2.0e-05" },
  ];

  // Stage 2 results (December 11, 2025) - WavLM top 3 layers unfrozen, 5 epochs
  const stage2Epochs = [
    { epoch: 0, totalLoss: 3.92, ctcLoss: 1.80, ceLoss: 5.35, lr: "8.5e-06" },
    { epoch: 1, totalLoss: 0, ctcLoss: 0, ceLoss: 0, lr: "1.0e-05", inProgress: true },
  ];

  return (
    <TechSlideWrapper
      title="VoxFormer Training Progress"
      slideNumber={slideNumber}
      totalSlides={totalSlides}
    >
      <div className="grid grid-cols-12 gap-6 h-full">
        {/* Left Column - Infrastructure */}
        <div className="col-span-4 space-y-4">
          <Card className="bg-slate-800/60 border-cyan-500/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2 text-cyan-400">
                <Server className="h-5 w-5" />
                GPU Infrastructure
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">GPU</span>
                <span className="text-white font-mono">RTX 4090 24GB</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">CUDA</span>
                <span className="text-white font-mono">12.6</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Provider</span>
                <span className="text-white font-mono">Vast.ai</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Cost</span>
                <span className="text-emerald-400 font-mono">$0.40/hr</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/60 border-purple-500/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2 text-purple-400">
                <Database className="h-5 w-5" />
                Training Data
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Dataset</span>
                <span className="text-white font-mono text-sm">LibriSpeech</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Subset</span>
                <span className="text-white font-mono text-sm">train-clean-100</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Samples</span>
                <span className="text-white font-mono">28,539</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Duration</span>
                <span className="text-white font-mono">~100 hours</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Speakers</span>
                <span className="text-white font-mono">251</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Sample Rate</span>
                <span className="text-white font-mono">16 kHz</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Language</span>
                <span className="text-white font-mono">English</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Source</span>
                <span className="text-cyan-400 font-mono text-sm">OpenSLR</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/60 border-emerald-500/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2 text-emerald-400">
                <Cpu className="h-5 w-5" />
                Model Config
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Total Params</span>
                <span className="text-white font-mono">204.8M</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Stage 1 Trainable</span>
                <span className="text-cyan-400 font-mono">110.4M</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Stage 2 Trainable</span>
                <span className="text-purple-400 font-mono">131.6M</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Batch Size</span>
                <span className="text-white font-mono">8 × 4 = 32</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Precision</span>
                <span className="text-white font-mono">FP16</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Training Progress */}
        <div className="col-span-8 space-y-3">
          {/* Stage 1 Progress Bar */}
          <Card className="bg-slate-800/60 border-emerald-500/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center justify-between">
                <span className="flex items-center gap-2 text-emerald-400">
                  <CheckCircle2 className="h-4 w-4" />
                  Stage 1: WavLM Frozen
                </span>
                <span className="text-emerald-400 font-mono text-sm">Complete (20/20)</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <Progress value={100} className="h-2 mb-1" />
              <div className="flex justify-between text-xs text-slate-400">
                <span>110.4M trainable • LR: 1e-4</span>
                <span>Final loss: 1.01</span>
              </div>
            </CardContent>
          </Card>

          {/* Stage 2 Progress Bar */}
          <Card className="bg-slate-800/60 border-purple-500/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center justify-between">
                <span className="flex items-center gap-2 text-purple-400">
                  <PlayCircle className="h-4 w-4 animate-pulse" />
                  Stage 2: WavLM Top 3 Unfrozen
                </span>
                <span className="text-purple-400 font-mono text-sm">In Progress (1/5)</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <Progress value={20} className="h-2 mb-1" />
              <div className="flex justify-between text-xs text-slate-400">
                <span>131.6M trainable • LR: 1e-5 (10x lower)</span>
                <span>Current loss: 3.92</span>
              </div>
            </CardContent>
          </Card>

          {/* Loss Chart - Two Stage Tables */}
          <Card className="bg-slate-800/60 border-cyan-500/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2 text-cyan-400">
                <TrendingDown className="h-5 w-5" />
                Loss Convergence
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                {/* Stage 1 Table */}
                <div>
                  <div className="text-xs text-emerald-400 mb-2 font-medium">Stage 1 (WavLM Frozen)</div>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="text-left py-1 text-slate-400">Ep</th>
                        <th className="text-right py-1 text-slate-400">Loss</th>
                        <th className="text-right py-1 text-slate-400">CTC</th>
                        <th className="text-right py-1 text-slate-400">CE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stage1Epochs.map((e, i) => (
                        <tr key={i} className="border-b border-slate-800">
                          <td className="py-1 text-white font-mono">{e.epoch}</td>
                          <td className="py-1 text-right font-mono text-cyan-400">{e.totalLoss.toFixed(2)}</td>
                          <td className="py-1 text-right font-mono text-purple-400">{e.ctcLoss.toFixed(2)}</td>
                          <td className="py-1 text-right font-mono text-emerald-400">{e.ceLoss.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {/* Stage 2 Table */}
                <div>
                  <div className="text-xs text-purple-400 mb-2 font-medium">Stage 2 (WavLM Top 3 Unfrozen)</div>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-slate-700">
                        <th className="text-left py-1 text-slate-400">Ep</th>
                        <th className="text-right py-1 text-slate-400">Loss</th>
                        <th className="text-right py-1 text-slate-400">CTC</th>
                        <th className="text-right py-1 text-slate-400">CE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stage2Epochs.map((e, i) => (
                        <tr key={i} className={`border-b border-slate-800 ${e.inProgress ? 'animate-pulse' : ''}`}>
                          <td className="py-1 text-white font-mono">
                            {e.epoch}
                            {e.inProgress && <span className="text-purple-400 ml-1">⏳</span>}
                          </td>
                          <td className="py-1 text-right font-mono text-cyan-400">
                            {e.inProgress ? '...' : e.totalLoss.toFixed(2)}
                          </td>
                          <td className="py-1 text-right font-mono text-purple-400">
                            {e.inProgress ? '...' : e.ctcLoss.toFixed(2)}
                          </td>
                          <td className="py-1 text-right font-mono text-emerald-400">
                            {e.inProgress ? '...' : e.ceLoss.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Visual Loss Bars - Combined Training Journey */}
              <div className="mt-4 space-y-1">
                <div className="text-xs text-slate-500 mb-2">Training Loss Journey</div>
                <div className="flex items-center gap-2">
                  <span className="w-16 text-xs text-emerald-400">S1:E0</span>
                  <div className="flex-1 h-3 bg-slate-700 rounded overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-red-500 to-orange-500" style={{ width: "100%" }} />
                  </div>
                  <span className="w-10 text-xs text-slate-400 text-right">7.13</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-16 text-xs text-emerald-400">S1:E5</span>
                  <div className="flex-1 h-3 bg-slate-700 rounded overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-yellow-500 to-lime-500" style={{ width: "15%" }} />
                  </div>
                  <span className="w-10 text-xs text-slate-400 text-right">1.04</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-16 text-xs text-emerald-400">S1:E19</span>
                  <div className="flex-1 h-3 bg-slate-700 rounded overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500" style={{ width: "14%" }} />
                  </div>
                  <span className="w-10 text-xs text-slate-400 text-right">1.01</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-16 text-xs text-purple-400">S2:E0</span>
                  <div className="flex-1 h-3 bg-slate-700 rounded overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-purple-500 to-pink-500" style={{ width: "55%" }} />
                  </div>
                  <span className="w-10 text-xs text-slate-400 text-right">3.92</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-16 text-xs text-purple-400 animate-pulse">S2:E1</span>
                  <div className="flex-1 h-3 bg-slate-700 rounded overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-purple-500/50 to-pink-500/50 animate-pulse" style={{ width: "40%" }} />
                  </div>
                  <span className="w-10 text-xs text-purple-400 text-right animate-pulse">...</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Key Achievements */}
          <Card className="bg-slate-800/60 border-emerald-500/30">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2 text-emerald-400">
                <Zap className="h-5 w-5" />
                Training Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-4 gap-3">
                <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                  <div className="text-2xl font-bold text-cyan-400">86%</div>
                  <div className="text-xs text-slate-400 mt-1">
                    S1 Loss drop<br />7.13→1.01
                  </div>
                </div>
                <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-400">~10h</div>
                  <div className="text-xs text-slate-400 mt-1">
                    Total time<br />S1+S2
                  </div>
                </div>
                <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                  <div className="text-2xl font-bold text-emerald-400">$4.00</div>
                  <div className="text-xs text-slate-400 mt-1">
                    Total cost<br />$0.40/hr
                  </div>
                </div>
                <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                  <div className="text-2xl font-bold text-yellow-400">84%</div>
                  <div className="text-xs text-slate-400 mt-1">
                    complete<br />21/25 epochs
                  </div>
                </div>
              </div>

              <div className="mt-3 space-y-1">
                <div className="flex items-center gap-2 text-sm text-emerald-400">
                  <CheckCircle2 className="h-4 w-4" />
                  <span>Stage 1 complete (20/20) - WavLM frozen, 110.4M params</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-purple-400">
                  <PlayCircle className="h-4 w-4 animate-pulse" />
                  <span>Stage 2 in progress (1/5) - WavLM top 3 unfrozen, 131.6M params</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
