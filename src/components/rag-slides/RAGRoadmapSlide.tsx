"use client";

import { TechSlideWrapper } from "../tech-slides/TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface RAGRoadmapSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function RAGRoadmapSlide({ slideNumber, totalSlides }: RAGRoadmapSlideProps) {
  const phases = [
    {
      num: "01",
      title: "Foundation",
      weeks: "1-4",
      color: "emerald",
      progress: 100,
      items: [
        { task: "PostgreSQL + pgvector setup", done: true },
        { task: "Document schema design", done: true },
        { task: "MiniLM embedding deployment", done: true },
        { task: "Chunking pipeline (3,885 docs)", done: true },
        { task: "HNSW index optimization", done: true }
      ]
    },
    {
      num: "02",
      title: "Retrieval",
      weeks: "5-8",
      color: "cyan",
      progress: 100,
      items: [
        { task: "Dense retrieval (384-dim)", done: true },
        { task: "BM25 sparse retrieval", done: true },
        { task: "RRF fusion algorithm", done: true },
        { task: "Metadata filtering system", done: true },
        { task: "Blender Manual ingestion", done: true }
      ]
    },
    {
      num: "03",
      title: "Quality",
      weeks: "9-12",
      color: "amber",
      progress: 100,
      items: [
        { task: "BGE-reranker cross-encoder", done: true },
        { task: "Context assembly module", done: true },
        { task: "Agentic validation loop", done: true },
        { task: "GPT-5.1 generation", done: true },
        { task: "Citation system", done: true }
      ]
    },
    {
      num: "04",
      title: "Production",
      weeks: "13-16",
      color: "purple",
      progress: 100,
      items: [
        { task: "Flask API endpoints", done: true },
        { task: "Session memory", done: true },
        { task: "Frontend integration", done: true },
        { task: "VPS deployment", done: true },
        { task: "Live at 5.249.161.66", done: true }
      ]
    }
  ];

  const colorMap: Record<string, { bg: string; border: string; text: string; progress: string }> = {
    emerald: { bg: "bg-emerald-500/20", border: "border-emerald-500/50", text: "text-emerald-400", progress: "bg-emerald-500" },
    cyan: { bg: "bg-cyan-500/20", border: "border-cyan-500/50", text: "text-cyan-400", progress: "bg-cyan-500" },
    amber: { bg: "bg-amber-500/20", border: "border-amber-500/50", text: "text-amber-400", progress: "bg-amber-500" },
    purple: { bg: "bg-purple-500/20", border: "border-purple-500/50", text: "text-purple-400", progress: "bg-purple-500" }
  };

  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Implementation Roadmap">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Implementation <span className="text-purple-400">Roadmap</span>
            </h2>
            <p className="text-slate-400">16-week development plan with phased delivery</p>
          </div>
          <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
            Complete - Live in Production
          </Badge>
        </div>

        {/* Timeline */}
        <div className="relative mb-4">
          <div className="absolute top-4 left-0 right-0 h-1 bg-slate-700 rounded-full">
            <div className="h-full bg-gradient-to-r from-emerald-500 via-cyan-500 via-amber-500 to-purple-500 rounded-full" style={{ width: '100%' }}></div>
          </div>
          <div className="flex justify-between relative">
            {phases.map((phase, idx) => (
              <div key={phase.num} className="flex flex-col items-center" style={{ width: '25%' }}>
                <div className={`w-8 h-8 rounded-full ${phase.progress === 100 ? colorMap[phase.color].progress : phase.progress > 0 ? 'bg-slate-600' : 'bg-slate-700'} border-2 ${colorMap[phase.color].border} flex items-center justify-center z-10`}>
                  {phase.progress === 100 ? (
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <span className={`text-xs font-bold ${colorMap[phase.color].text}`}>{phase.num}</span>
                  )}
                </div>
                <span className={`text-xs mt-2 ${colorMap[phase.color].text} font-semibold`}>{phase.title}</span>
                <span className="text-xs text-slate-500">Weeks {phase.weeks}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Phase Cards */}
        <div className="grid grid-cols-4 gap-3 flex-1">
          {phases.map((phase) => (
            <Card key={phase.num} className={`${colorMap[phase.color].bg} ${colorMap[phase.color].border} border`}>
              <CardContent className="p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-2xl font-bold ${colorMap[phase.color].text} opacity-50`}>{phase.num}</span>
                  <span className={`text-xs ${colorMap[phase.color].text}`}>{phase.progress}%</span>
                </div>
                <h3 className="text-sm font-semibold text-white mb-1">{phase.title}</h3>
                <Progress value={phase.progress} className="h-1 mb-3 bg-slate-700" />
                <div className="space-y-1.5">
                  {phase.items.map((item) => (
                    <div key={item.task} className="flex items-center gap-2 text-xs">
                      {item.done ? (
                        <svg className="w-3 h-3 text-emerald-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <div className="w-3 h-3 rounded-full border border-slate-500 flex-shrink-0" />
                      )}
                      <span className={item.done ? 'text-slate-400 line-through' : 'text-slate-300'}>{item.task}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Tech Stack & Resources */}
        <div className="mt-3 grid grid-cols-2 gap-4">
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-3">
              <h4 className="text-sm font-semibold text-white mb-2">Production Tech Stack</h4>
              <div className="flex flex-wrap gap-2">
                {[
                  { name: "PostgreSQL + pgvector", color: "blue" },
                  { name: "MiniLM-L6 (384d)", color: "cyan" },
                  { name: "BGE-reranker-v2-m3", color: "emerald" },
                  { name: "GPT-5.1", color: "amber" },
                  { name: "Flask API", color: "purple" },
                  { name: "3,885 docs", color: "rose" }
                ].map((tech) => (
                  <Badge key={tech.name} variant="outline" className={`border-${tech.color}-500/50 text-${tech.color}-400 bg-${tech.color}-500/10 text-xs`}>
                    {tech.name}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-3">
              <h4 className="text-sm font-semibold text-white mb-2">Live Performance</h4>
              <div className="grid grid-cols-4 gap-2 text-center">
                {[
                  { value: "~2s", label: "Retrieval" },
                  { value: "~46s", label: "Reranking" },
                  { value: "~80s", label: "Total" },
                  { value: "3,885", label: "Documents" }
                ].map((target) => (
                  <div key={target.label}>
                    <div className="text-sm font-bold text-cyan-400">{target.value}</div>
                    <div className="text-xs text-slate-500">{target.label}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
