"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Mic, Zap, Brain, CheckCircle } from "lucide-react";

interface STTComponentSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function STTComponentSlide({ slideNumber, totalSlides }: STTComponentSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Speech-to-Text Engine">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
        {/* VoxFormer */}
        <Card className="bg-slate-900/50 border-cyan-500/30 backdrop-blur-sm">
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
                  <Brain className="w-5 h-5 text-cyan-400" />
                </div>
                <div>
                  <CardTitle className="text-xl text-white">VoxFormer</CardTitle>
                  <p className="text-sm text-cyan-400">Custom Trained Model</p>
                </div>
              </div>
              <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10">
                In Training
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Architecture */}
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xs text-slate-500 mb-2">ARCHITECTURE</div>
              <div className="flex items-center gap-2 text-sm flex-wrap">
                <Badge variant="outline" className="border-slate-600 text-slate-300">WavLM-Base</Badge>
                <span className="text-slate-500">→</span>
                <Badge variant="outline" className="border-slate-600 text-slate-300">Zipformer</Badge>
                <span className="text-slate-500">→</span>
                <Badge variant="outline" className="border-slate-600 text-slate-300">Transformer</Badge>
              </div>
            </div>

            {/* Specs */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-500">PARAMETERS</div>
                <div className="text-lg font-bold text-cyan-400">142M</div>
                <div className="text-xs text-slate-500">47M trainable</div>
              </div>
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-500">TARGET WER</div>
                <div className="text-lg font-bold text-cyan-400">&lt;15%</div>
                <div className="text-xs text-slate-500">LibriSpeech</div>
              </div>
            </div>

            {/* Training Progress */}
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-slate-500">TRAINING PROGRESS</span>
                <span className="text-xs text-cyan-400">Stage 4: Hybrid CTC-Attention</span>
              </div>
              <Progress value={15} className="h-2" />
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>Epoch 1.5/10</span>
                <span>15% complete</span>
              </div>
            </div>

            {/* Features */}
            <div className="space-y-2">
              {[
                { label: "Custom domain training", done: false },
                { label: "Hybrid CTC + Cross-Entropy loss", done: true },
                { label: "Streaming inference ready", done: false },
              ].map((feat) => (
                <div key={feat.label} className="flex items-center gap-2 text-sm">
                  <div className={`w-4 h-4 rounded-full ${feat.done ? 'bg-cyan-500' : 'border border-slate-600'} flex items-center justify-center`}>
                    {feat.done && <CheckCircle className="w-3 h-3 text-white" />}
                  </div>
                  <span className={feat.done ? 'text-slate-300' : 'text-slate-500'}>{feat.label}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Whisper */}
        <Card className="bg-slate-900/50 border-purple-500/30 backdrop-blur-sm">
          <CardHeader className="pb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-500/10 rounded-lg border border-purple-500/30">
                  <Mic className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <CardTitle className="text-xl text-white">Whisper</CardTitle>
                  <p className="text-sm text-purple-400">OpenAI Pre-trained</p>
                </div>
              </div>
              <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
                Recommended
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Architecture */}
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xs text-slate-500 mb-2">MODEL</div>
              <div className="flex items-center gap-2 text-sm flex-wrap">
                <Badge variant="outline" className="border-purple-500/30 text-purple-300">whisper-large-v3</Badge>
                <span className="text-slate-500">via</span>
                <Badge variant="outline" className="border-slate-600 text-slate-300">openai-whisper</Badge>
              </div>
            </div>

            {/* Specs */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-500">PARAMETERS</div>
                <div className="text-lg font-bold text-purple-400">1.5B</div>
                <div className="text-xs text-slate-500">large-v3</div>
              </div>
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-500">WER</div>
                <div className="text-lg font-bold text-purple-400">&lt;5%</div>
                <div className="text-xs text-slate-500">LibriSpeech</div>
              </div>
            </div>

            {/* Features */}
            <div className="p-3 bg-slate-800/50 rounded-lg">
              <div className="text-xs text-slate-500 mb-2">ADVANTAGES</div>
              <div className="space-y-2">
                {[
                  "State-of-the-art accuracy",
                  "99+ language support",
                  "Timestamp & word confidence",
                  "Production-ready immediately",
                ].map((adv) => (
                  <div key={adv} className="flex items-center gap-2 text-sm text-slate-300">
                    <Zap className="w-3 h-3 text-purple-400" />
                    {adv}
                  </div>
                ))}
              </div>
            </div>

            {/* Priority indicator */}
            <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <span className="text-sm text-emerald-400 font-medium">Selected for Demo</span>
              </div>
              <p className="text-xs text-slate-400 mt-1">Accuracy is priority - Whisper recommended until VoxFormer training completes</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bottom comparison */}
      <Card className="mt-4 bg-slate-900/30 border-slate-700/50">
        <CardContent className="p-4">
          <div className="grid grid-cols-4 gap-4 text-center text-sm">
            <div>
              <div className="text-slate-500 mb-1">Output Format</div>
              <div className="text-white">Text + Word Timestamps + Confidence Scores</div>
            </div>
            <div>
              <div className="text-slate-500 mb-1">Processing</div>
              <div className="text-white">GPU Server (VPS → GPU)</div>
            </div>
            <div>
              <div className="text-slate-500 mb-1">Latency</div>
              <div className="text-white">&lt;2s for 10s audio</div>
            </div>
            <div>
              <div className="text-slate-500 mb-1">Toggle</div>
              <div className="text-white">User selectable in settings</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </TechSlideWrapper>
  );
}
