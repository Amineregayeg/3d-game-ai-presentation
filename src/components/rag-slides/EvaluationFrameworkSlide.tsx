"use client";

import { TechSlideWrapper } from "../tech-slides/TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface EvaluationFrameworkSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function EvaluationFrameworkSlide({ slideNumber, totalSlides }: EvaluationFrameworkSlideProps) {
  const ragasMetrics = [
    { name: "Faithfulness", target: 0.85, current: 0.88, desc: "Answer grounded in context", color: "emerald" },
    { name: "Answer Relevancy", target: 0.80, current: 0.84, desc: "Answer addresses query", color: "cyan" },
    { name: "Context Precision", target: 0.75, current: 0.82, desc: "Retrieved docs are relevant", color: "blue" },
    { name: "Context Recall", target: 0.70, current: 0.76, desc: "All relevant info retrieved", color: "purple" }
  ];

  const gameDevMetrics = [
    { name: "Code Correctness", target: 0.95, current: 0.97, weight: "20%", color: "amber" },
    { name: "API Existence", target: 0.90, current: 0.94, weight: "20%", color: "rose" },
    { name: "Completeness", target: 0.80, current: 0.85, weight: "15%", color: "indigo" },
    { name: "Version Compat", target: 0.85, current: 0.91, weight: "10%", color: "teal" }
  ];

  const colorMap: Record<string, string> = {
    emerald: "bg-emerald-500",
    cyan: "bg-cyan-500",
    blue: "bg-blue-500",
    purple: "bg-purple-500",
    amber: "bg-amber-500",
    rose: "bg-rose-500",
    indigo: "bg-indigo-500",
    teal: "bg-teal-500"
  };

  const textColorMap: Record<string, string> = {
    emerald: "text-emerald-400",
    cyan: "text-cyan-400",
    blue: "text-blue-400",
    purple: "text-purple-400",
    amber: "text-amber-400",
    rose: "text-rose-400",
    indigo: "text-indigo-400",
    teal: "text-teal-400"
  };

  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Evaluation Framework">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              RAGAS <span className="text-emerald-400">Evaluation</span> Framework
            </h2>
            <p className="text-slate-400">Continuous quality assurance with domain-specific metrics</p>
          </div>
          <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
            Reference-Free Evaluation
          </Badge>
        </div>

        <div className="grid grid-cols-2 gap-6 flex-1">
          {/* Left: RAGAS Metrics */}
          <div className="space-y-3">
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold text-white">RAGAS Metrics</h3>
                  <Badge variant="outline" className="border-slate-600 text-slate-400 text-xs">
                    Standard RAG
                  </Badge>
                </div>
                <div className="space-y-4">
                  {ragasMetrics.map((metric) => (
                    <div key={metric.name} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className={`text-sm font-medium ${textColorMap[metric.color]}`}>{metric.name}</span>
                          <span className="text-xs text-slate-500 ml-2">({metric.desc})</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-slate-500">Target: {metric.target}</span>
                          <span className={`text-sm font-bold ${textColorMap[metric.color]}`}>{metric.current}</span>
                        </div>
                      </div>
                      <div className="relative">
                        <Progress value={metric.current * 100} className="h-2 bg-slate-700" />
                        <div
                          className="absolute top-0 h-2 w-0.5 bg-white/50"
                          style={{ left: `${metric.target * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Composite Score */}
            <Card className="bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 border-emerald-500/30">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-sm text-slate-400">Composite RAGAS Score</h4>
                    <p className="text-xs text-slate-500">Weighted average of all metrics</p>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-emerald-400">0.82</div>
                    <div className="text-xs text-emerald-500">Target: 0.80</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right: Game Dev Specific Metrics */}
          <div className="space-y-3">
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold text-white">Game Dev Metrics</h3>
                  <Badge variant="outline" className="border-amber-500/50 text-amber-400 text-xs">
                    Domain-Specific
                  </Badge>
                </div>
                <div className="space-y-4">
                  {gameDevMetrics.map((metric) => (
                    <div key={metric.name} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className={`text-sm font-medium ${textColorMap[metric.color]}`}>{metric.name}</span>
                          <span className="text-xs text-slate-600 bg-slate-700/50 px-1.5 py-0.5 rounded">{metric.weight}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-slate-500">Target: {metric.target}</span>
                          <span className={`text-sm font-bold ${textColorMap[metric.color]}`}>{metric.current}</span>
                        </div>
                      </div>
                      <div className="relative">
                        <Progress value={metric.current * 100} className="h-2 bg-slate-700" />
                        <div
                          className="absolute top-0 h-2 w-0.5 bg-white/50"
                          style={{ left: `${metric.target * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Evaluation Pipeline */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h4 className="text-sm font-semibold text-white mb-3">Continuous Evaluation</h4>
                <div className="bg-slate-900/50 rounded p-3 font-mono text-xs space-y-1">
                  <div className="text-purple-400">def continuous_eval_loop():</div>
                  <div className="text-slate-400 pl-4">queries = fetch_queries(last_24h)</div>
                  <div className="text-slate-400 pl-4"><span className="text-cyan-400">for</span> q <span className="text-cyan-400">in</span> queries:</div>
                  <div className="text-slate-400 pl-8">ragas = compute_ragas(q)</div>
                  <div className="text-slate-400 pl-8">game_metrics = GameDevMetrics(q)</div>
                  <div className="text-slate-400 pl-8"><span className="text-cyan-400">if</span> score &lt; threshold:</div>
                  <div className="text-rose-400 pl-12">alert_degradation(q)</div>
                  <div className="text-slate-400 pl-4">generate_daily_report()</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Bottom: Key Thresholds */}
        <div className="mt-3 grid grid-cols-5 gap-3">
          {[
            { label: "Hallucination", value: "<5%", status: "pass", color: "emerald" },
            { label: "MRR@10", value: ">0.80", status: "pass", color: "cyan" },
            { label: "P95 Latency", value: "<200ms", status: "pass", color: "blue" },
            { label: "Code Valid", value: ">95%", status: "pass", color: "amber" },
            { label: "Daily Evals", value: "1000+", status: "running", color: "purple" }
          ].map((item) => (
            <Card key={item.label} className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-3 text-center">
                <div className={`text-lg font-bold ${textColorMap[item.color]}`}>{item.value}</div>
                <div className="text-xs text-slate-500">{item.label}</div>
                <div className={`text-xs mt-1 ${item.status === 'pass' ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {item.status === 'pass' ? '✓ Pass' : '● Running'}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
