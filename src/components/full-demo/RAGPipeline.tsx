"use client";

import { motion } from "framer-motion";
import {
  Brain,
  Search,
  Database,
  Hash,
  GitBranch,
  BarChart3,
  Layers,
  Sparkles,
  CheckCircle2,
  Loader2,
  Check,
  FileText,
  BookOpen,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";

// Types
export interface StageResult {
  stage: string;
  status: string;
  duration_ms: number;
  results_count?: number;
  details?: Record<string, unknown>;
}

export interface Document {
  id: string;
  content: string;
  title?: string;
  source: string;
  version: string;
  category: string;
  dense_score: number;
  sparse_score: number;
  rrf_score: number;
  rerank_score: number;
}

export interface QueryAnalysis {
  intent: string;
  entities: string[];
  confidence: number;
  is_multi_step: boolean;
}

export interface RAGMetrics {
  faithfulness: number;
  relevancy: number;
  completeness: number;
  composite_score: number;
}

export interface Citation {
  index: number;
  doc_id: string;
  text: string;
}

// Pipeline Stages Configuration
export const PIPELINE_STAGES = [
  { id: "orchestration", label: "Orchestrator", icon: Brain, color: "emerald", description: "Plans query execution strategy", model: "GPT-4" },
  { id: "query_analysis", label: "Query Analysis", icon: Search, color: "cyan", description: "Extract intent and entities", model: "GPT-4" },
  { id: "retrieval_dense", label: "Dense Search", icon: Database, color: "purple", description: "Semantic vector similarity", model: "BGE-M3" },
  { id: "retrieval_sparse", label: "Sparse Search", icon: Hash, color: "purple", description: "BM25 keyword retrieval", model: "BM25" },
  { id: "rrf_fusion", label: "RRF Fusion", icon: GitBranch, color: "amber", description: "Reciprocal Rank Fusion", model: "Algorithm" },
  { id: "reranking", label: "Reranking", icon: BarChart3, color: "orange", description: "Cross-encoder precision", model: "BGE-Reranker" },
  { id: "context_assembly", label: "Context", icon: Layers, color: "blue", description: "Assemble context window", model: "GPT-4" },
  { id: "generation", label: "Generation", icon: Sparkles, color: "emerald", description: "Generate answer", model: "GPT-4" },
  { id: "validation", label: "Validation", icon: CheckCircle2, color: "green", description: "RAGAS quality check", model: "GPT-4" },
];

// Metric Gauge Component
export function MetricGauge({ label, value, color }: { label: string; value: number; color: string }) {
  const percentage = Math.round(value * 100);
  const circumference = 2 * Math.PI * 36;
  const strokeDashoffset = circumference - (value * circumference);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-20 h-20">
        <svg className="w-20 h-20 transform -rotate-90">
          <circle
            cx="40"
            cy="40"
            r="36"
            stroke="currentColor"
            strokeWidth="6"
            fill="none"
            className="text-slate-700"
          />
          <motion.circle
            cx="40"
            cy="40"
            r="36"
            stroke={color}
            strokeWidth="6"
            fill="none"
            strokeLinecap="round"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1, ease: "easeOut" }}
            style={{ strokeDasharray: circumference }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-lg font-bold text-white">{percentage}%</span>
        </div>
      </div>
      <span className="mt-1 text-xs text-slate-400">{label}</span>
    </div>
  );
}

// Pipeline Stage Component
function PipelineStage({
  stageConfig,
  stageData,
  isActive,
  isComplete,
}: {
  stageConfig: typeof PIPELINE_STAGES[0];
  stageData?: StageResult;
  isActive: boolean;
  isComplete: boolean;
}) {
  const Icon = stageConfig.icon;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <motion.div
            className={`flex items-center gap-2 p-2 rounded-lg transition-all cursor-help ${
              isComplete
                ? "bg-emerald-500/20 border border-emerald-500/50"
                : isActive
                ? "bg-cyan-500/20 border border-cyan-500/50"
                : "bg-slate-800/50 border border-slate-700/50"
            }`}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div
              className={`p-1.5 rounded ${
                isComplete ? "bg-emerald-500/30" : isActive ? "bg-cyan-500/30" : "bg-slate-700/50"
              }`}
            >
              {isActive && !isComplete ? (
                <Loader2 className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
              ) : (
                <Icon className={`w-3.5 h-3.5 ${isComplete ? "text-emerald-400" : "text-slate-400"}`} />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-xs font-medium text-white truncate">{stageConfig.label}</div>
              {stageData && (
                <div className="text-[10px] text-slate-500">{stageData.duration_ms}ms</div>
              )}
            </div>
            {isComplete && <Check className="w-3.5 h-3.5 text-emerald-400" />}
          </motion.div>
        </TooltipTrigger>
        <TooltipContent side="right" className="max-w-xs">
          <div className="space-y-1">
            <p className="font-medium">{stageConfig.label}</p>
            <p className="text-xs text-muted-foreground">{stageConfig.description}</p>
            <p className="text-xs text-cyan-400">Model: {stageConfig.model}</p>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Document Card Component
export function DocumentCard({
  doc,
  index,
  isExpanded,
  onToggle,
}: {
  doc: Document;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const scoreColor =
    doc.rerank_score >= 0.9
      ? "text-emerald-400"
      : doc.rerank_score >= 0.7
      ? "text-cyan-400"
      : doc.rerank_score >= 0.5
      ? "text-amber-400"
      : "text-red-400";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="bg-slate-800/50 rounded-lg border border-slate-700/50 overflow-hidden"
    >
      <div
        className="flex items-center gap-3 p-3 cursor-pointer hover:bg-slate-700/30 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center justify-center w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-400 font-bold text-xs">
          [{index + 1}]
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-medium text-white text-sm truncate">{doc.title || "Document"}</div>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Badge variant="outline" className="text-[10px] px-1 py-0 border-slate-600">
              {doc.source}
            </Badge>
            <span>v{doc.version}</span>
          </div>
        </div>
        <div className="text-right">
          <div className={`text-sm font-bold ${scoreColor}`}>
            {Math.round(doc.rerank_score * 100)}%
          </div>
        </div>
      </div>
      {isExpanded && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          className="border-t border-slate-700/50 p-3"
        >
          <div className="grid grid-cols-4 gap-2 text-xs mb-2">
            <div className="bg-slate-900/50 p-1.5 rounded text-center">
              <div className="text-slate-500">Dense</div>
              <div className="text-purple-400">{(doc.dense_score * 100).toFixed(0)}%</div>
            </div>
            <div className="bg-slate-900/50 p-1.5 rounded text-center">
              <div className="text-slate-500">Sparse</div>
              <div className="text-cyan-400">{(doc.sparse_score * 100).toFixed(0)}%</div>
            </div>
            <div className="bg-slate-900/50 p-1.5 rounded text-center">
              <div className="text-slate-500">RRF</div>
              <div className="text-amber-400">{(doc.rrf_score * 100).toFixed(0)}%</div>
            </div>
            <div className="bg-slate-900/50 p-1.5 rounded text-center">
              <div className="text-slate-500">Final</div>
              <div className="text-emerald-400">{(doc.rerank_score * 100).toFixed(0)}%</div>
            </div>
          </div>
          <div className="bg-slate-900/50 p-2 rounded text-xs text-slate-300 max-h-32 overflow-auto">
            {doc.content}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

// Citation Preview Component
export function CitationPreview({
  citation,
  documents,
}: {
  citation: Citation;
  documents: Document[];
}) {
  const doc = documents.find((d) => d.id === citation.doc_id);
  if (!doc) return <sup className="text-cyan-400">[{citation.index}]</sup>;

  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <sup className="text-cyan-400 cursor-pointer hover:text-cyan-300 font-bold">
          [{citation.index}]
        </sup>
      </HoverCardTrigger>
      <HoverCardContent className="w-72 bg-slate-900 border-slate-700">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-cyan-400" />
            <span className="font-medium text-white text-sm">{doc.title || "Document"}</span>
          </div>
          <div className="flex gap-1">
            <Badge variant="outline" className="text-[10px] border-slate-600">
              {doc.source}
            </Badge>
            <Badge variant="outline" className="text-[10px] border-emerald-600 text-emerald-400">
              {Math.round(doc.rerank_score * 100)}%
            </Badge>
          </div>
          <p className="text-xs text-slate-400 line-clamp-2">{citation.text}</p>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

// Main RAG Pipeline Component
interface RAGPipelineProps {
  isActive: boolean;
  currentStage: string;
  stages: StageResult[];
  analysis?: QueryAnalysis;
  documents?: Document[];
  answer?: string;
  citations?: Citation[];
  metrics?: RAGMetrics;
  isComplete: boolean;
}

export function RAGPipeline({
  isActive,
  currentStage,
  stages,
  analysis,
  documents = [],
  answer,
  citations = [],
  metrics,
  isComplete,
}: RAGPipelineProps) {
  const completedStages = new Set(stages.map((s) => s.stage));
  const activeStageIndex = PIPELINE_STAGES.findIndex((s) => s.id === currentStage);

  return (
    <div className="space-y-4">
      {/* Pipeline Stages */}
      <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-white flex items-center gap-2">
            <Brain className="w-4 h-4 text-cyan-400" />
            RAG Pipeline
            {isActive && !isComplete && (
              <Badge variant="outline" className="ml-auto text-[10px] border-cyan-500/50 text-cyan-400 animate-pulse">
                Processing
              </Badge>
            )}
            {isComplete && (
              <Badge variant="outline" className="ml-auto text-[10px] border-emerald-500/50 text-emerald-400">
                Complete
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-2">
            {PIPELINE_STAGES.map((stageConfig, index) => {
              const stageData = stages.find((s) => s.stage === stageConfig.id);
              const isStageActive = isActive && index === activeStageIndex;
              const isStageComplete = completedStages.has(stageConfig.id) || (isComplete && index <= activeStageIndex);

              return (
                <PipelineStage
                  key={stageConfig.id}
                  stageConfig={stageConfig}
                  stageData={stageData}
                  isActive={isStageActive}
                  isComplete={isStageComplete}
                />
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Query Analysis */}
      {analysis && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-white flex items-center gap-2">
                <Search className="w-4 h-4 text-purple-400" />
                Query Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="bg-slate-900/50 p-2 rounded">
                  <div className="text-slate-500">Intent</div>
                  <div className="text-cyan-400 font-medium">{analysis.intent}</div>
                </div>
                <div className="bg-slate-900/50 p-2 rounded">
                  <div className="text-slate-500">Confidence</div>
                  <div className="text-emerald-400 font-medium">
                    {Math.round(analysis.confidence * 100)}%
                  </div>
                </div>
                <div className="bg-slate-900/50 p-2 rounded">
                  <div className="text-slate-500">Multi-step</div>
                  <div className="text-white font-medium">
                    {analysis.is_multi_step ? "Yes" : "No"}
                  </div>
                </div>
              </div>
              {analysis.entities.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {analysis.entities.map((entity, i) => (
                    <Badge key={i} variant="outline" className="text-[10px] border-purple-500/50 text-purple-400">
                      {entity}
                    </Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* RAGAS Metrics */}
      {metrics && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-white flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-amber-400" />
                RAGAS Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-around">
                <MetricGauge label="Faithfulness" value={metrics.faithfulness} color="#10b981" />
                <MetricGauge label="Relevancy" value={metrics.relevancy} color="#06b6d4" />
                <MetricGauge label="Completeness" value={metrics.completeness} color="#8b5cf6" />
                <MetricGauge label="Composite" value={metrics.composite_score} color="#f59e0b" />
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
