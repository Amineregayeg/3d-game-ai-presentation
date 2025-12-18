"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Workflow, Search, GitMerge, ArrowDownUp,
  Sparkles, ShieldCheck, Database, FileText
} from "lucide-react";

interface RAGPipelineSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function RAGPipelineSlide({ slideNumber, totalSlides }: RAGPipelineSlideProps) {
  const stages = [
    {
      num: 1,
      name: "Orchestrator",
      icon: Workflow,
      time: "~10ms",
      description: "Routes query through pipeline",
      color: "cyan",
    },
    {
      num: 2,
      name: "Query Analysis",
      icon: Search,
      time: "~90ms",
      description: "Intent detection & expansion",
      color: "cyan",
    },
    {
      num: 3,
      name: "Dense Search",
      icon: Database,
      time: "~150ms",
      description: "BGE-M3 vector similarity",
      color: "purple",
    },
    {
      num: 4,
      name: "Sparse Search",
      icon: FileText,
      time: "~50ms",
      description: "BM25 keyword matching",
      color: "purple",
    },
    {
      num: 5,
      name: "RRF Fusion",
      icon: GitMerge,
      time: "~5ms",
      description: "Reciprocal Rank Fusion",
      color: "amber",
    },
    {
      num: 6,
      name: "Reranking",
      icon: ArrowDownUp,
      time: "~230ms",
      description: "Cross-encoder reranking",
      color: "amber",
    },
    {
      num: 7,
      name: "Generation",
      icon: Sparkles,
      time: "~1.2s",
      description: "LLM response generation",
      color: "emerald",
    },
    {
      num: 8,
      name: "Validation",
      icon: ShieldCheck,
      time: "~90ms",
      description: "RAGAS quality check",
      color: "emerald",
    },
  ];

  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="RAG Pipeline">
      <div className="space-y-4">
        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="text-3xl font-bold text-white mb-2">8-Stage Advanced RAG Pipeline</h2>
          <p className="text-slate-400">Hybrid retrieval with agentic validation</p>
        </div>

        {/* Pipeline visualization */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2">
          {stages.map((stage, idx) => {
            const Icon = stage.icon;
            return (
              <Card
                key={stage.num}
                className={`bg-slate-900/50 border-${stage.color}-500/30 backdrop-blur-sm relative group hover:border-${stage.color}-500/60 transition-all`}
              >
                <CardContent className="p-3 text-center">
                  {/* Stage number */}
                  <Badge
                    variant="outline"
                    className={`border-${stage.color}-500/50 text-${stage.color}-400 bg-${stage.color}-500/10 text-xs mb-2`}
                  >
                    {stage.num}
                  </Badge>

                  {/* Icon */}
                  <div className={`mx-auto p-2 rounded-lg bg-${stage.color}-500/10 border border-${stage.color}-500/30 w-fit mb-2`}>
                    <Icon className={`w-4 h-4 text-${stage.color}-400`} />
                  </div>

                  {/* Name */}
                  <h3 className="text-xs font-semibold text-white mb-1 truncate">{stage.name}</h3>

                  {/* Time */}
                  <div className={`text-xs text-${stage.color}-400 font-mono`}>{stage.time}</div>

                  {/* Description - tooltip on hover */}
                  <div className="hidden group-hover:block absolute -bottom-8 left-1/2 -translate-x-1/2 bg-slate-800 px-2 py-1 rounded text-xs text-slate-300 whitespace-nowrap z-10 border border-slate-700">
                    {stage.description}
                  </div>

                  {/* Arrow to next */}
                  {idx < stages.length - 1 && (
                    <div className="absolute -right-1.5 top-1/2 -translate-y-1/2 text-slate-600 z-10 hidden lg:block">
                      <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/>
                      </svg>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Details grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          {/* Hybrid Retrieval */}
          <Card className="bg-slate-900/50 border-purple-500/30 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <GitMerge className="w-4 h-4 text-purple-400" />
                <h3 className="text-sm font-semibold text-white">Hybrid Retrieval</h3>
              </div>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-slate-400">Dense (BGE-M3)</span>
                  <span className="text-purple-400">4096-dim vectors</span>
                </div>
                <div className="flex justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-slate-400">Sparse (BM25)</span>
                  <span className="text-purple-400">Keyword matching</span>
                </div>
                <div className="flex justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-slate-400">Fusion</span>
                  <span className="text-purple-400">RRF (k=60)</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Evaluation */}
          <Card className="bg-slate-900/50 border-emerald-500/30 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <h3 className="text-sm font-semibold text-white">RAGAS Evaluation</h3>
              </div>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-slate-400">Faithfulness</span>
                  <span className="text-emerald-400">&gt;0.85</span>
                </div>
                <div className="flex justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-slate-400">Relevancy</span>
                  <span className="text-emerald-400">&gt;0.90</span>
                </div>
                <div className="flex justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-slate-400">Context Precision</span>
                  <span className="text-emerald-400">&gt;0.88</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Performance */}
          <Card className="bg-slate-900/50 border-cyan-500/30 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-semibold text-white">Performance</h3>
              </div>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-slate-400">Total Latency</span>
                  <span className="text-cyan-400">~1.8s avg</span>
                </div>
                <div className="flex justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-slate-400">Documents</span>
                  <span className="text-cyan-400">Top 5 retrieved</span>
                </div>
                <div className="flex justify-between p-2 bg-slate-800/50 rounded">
                  <span className="text-slate-400">Citations</span>
                  <span className="text-cyan-400">Auto-generated</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Example query */}
        <Card className="bg-slate-900/30 border-slate-700/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-1 w-8 bg-gradient-to-r from-purple-500 to-emerald-500 rounded-full" />
              <span className="text-sm font-mono text-purple-400">EXAMPLE QUERY</span>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <span className="text-slate-400">Input:</span>
              <Badge variant="outline" className="border-slate-600 text-slate-300 bg-slate-800/50">
                &quot;How do I create a metallic material in Blender?&quot;
              </Badge>
              <span className="text-slate-500">→</span>
              <span className="text-slate-400">Retrieved:</span>
              <Badge variant="outline" className="border-purple-500/30 text-purple-300 bg-purple-500/10">
                materials/principled_bsdf.md
              </Badge>
              <Badge variant="outline" className="border-purple-500/30 text-purple-300 bg-purple-500/10">
                tutorials/pbr_materials.md
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </TechSlideWrapper>
  );
}
