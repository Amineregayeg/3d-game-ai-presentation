"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  Send,
  Loader2,
  Database,
  Brain,
  CheckCircle2,
  AlertCircle,
  Copy,
  RefreshCw,
  Settings2,
  Sparkles,
  ChevronDown,
  Hash,
  Layers,
  BarChart3,
  Activity,
  GitBranch,
  MessageSquare,
  Cpu,
  Check,
  Zap,
  FileText,
  Code,
  BookOpen,
  Workflow,
  Info,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";
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
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Kbd } from "@/components/ui/kbd";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupTextarea,
  InputGroupText,
} from "@/components/ui/input-group";
import { Separator } from "@/components/ui/separator";

// =============================================================================
// Types
// =============================================================================

interface QueryAnalysis {
  intent: string;
  entities: string[];
  variations: string[];
  detected_filters: Record<string, string>;
  is_multi_step: boolean;
  sub_queries: string[];
  confidence: number;
}

interface Document {
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

interface ValidationResult {
  faithfulness: number;
  relevancy: number;
  completeness: number;
  code_validity?: number;
  composite_score: number;
  issues: string[];
  passed: boolean;
  attempt: number;
}

interface StageResult {
  stage: string;
  status: string;
  duration_ms: number;
  results_count?: number;
  details?: Record<string, unknown>;
}

interface RAGResponse {
  request_id: string;
  session_id: string;
  query: string;
  analysis: QueryAnalysis;
  documents: Document[];
  answer: string;
  citations: Array<{ index: number; doc_id: string; text: string }>;
  validation: ValidationResult;
  metrics: Record<string, number>;
  latency: Record<string, number>;
  stages: StageResult[];
  status: string;
  warning?: string;
  demo_mode: boolean;
}

interface RAGSettings {
  retrieval_mode: "hybrid" | "dense" | "sparse";
  top_k: number;
  enable_reranking: boolean;
  show_debug: boolean;
}

// =============================================================================
// Sample Queries with Categories
// =============================================================================

const SAMPLE_QUERIES = [
  { label: "Select All Faces", query: "How do I select all faces in Blender?", category: "mesh", icon: Layers },
  { label: "Python API", query: "What is bpy.ops.mesh.select_all?", category: "scripting", icon: Code },
  { label: "Procedural Material", query: "How to create a procedural material with nodes?", category: "materials", icon: Sparkles },
  { label: "Animation Keyframes", query: "Add keyframes for object animation using Python", category: "animation", icon: Workflow },
  { label: "Multi-step Workflow", query: "Select all faces, extrude them, and apply smooth shading", category: "workflow", icon: GitBranch },
  { label: "UV Unwrapping", query: "How to unwrap UV coordinates automatically?", category: "uv", icon: FileText },
];

// =============================================================================
// Pipeline Stages Configuration with Tooltips
// =============================================================================

const PIPELINE_STAGES = [
  { id: "orchestration", label: "Orchestrator", icon: Brain, color: "emerald", description: "GPT-5.1 plans query execution strategy", model: "gpt-5.1" },
  { id: "query_analysis", label: "Query Analysis", icon: Search, color: "cyan", description: "Extract intent, entities, generate variations", model: "gpt-5-mini" },
  { id: "retrieval_dense", label: "Dense Search", icon: Database, color: "purple", description: "BGE-M3 semantic vector similarity search", model: "bge-m3" },
  { id: "retrieval_sparse", label: "Sparse Search", icon: Hash, color: "purple", description: "BM25 keyword-based retrieval", model: "bm25" },
  { id: "rrf_fusion", label: "RRF Fusion", icon: GitBranch, color: "amber", description: "Reciprocal Rank Fusion (k=60)", model: "algorithm" },
  { id: "reranking", label: "Reranking", icon: BarChart3, color: "orange", description: "Cross-encoder reranking for precision", model: "bge-reranker" },
  { id: "context_assembly", label: "Context Assembly", icon: Layers, color: "blue", description: "Assemble optimal context window", model: "gpt-5-nano" },
  { id: "generation", label: "Generation", icon: Sparkles, color: "emerald", description: "Generate answer with citations", model: "gpt-5.1" },
  { id: "validation", label: "Validation", icon: CheckCircle2, color: "green", description: "RAGAS quality metrics validation", model: "gpt-5-mini" },
];

// =============================================================================
// API Functions
// =============================================================================

// Use empty string for relative URLs (through nginx proxy), or explicit URL for direct access
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? '';

async function queryRAG(query: string, sessionId: string | null, settings: RAGSettings): Promise<RAGResponse> {
  const response = await fetch(`${API_BASE}/api/rag/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      session_id: sessionId,
      settings: {
        retrieval_mode: settings.retrieval_mode,
        top_k: settings.top_k,
        enable_reranking: settings.enable_reranking,
      },
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Query failed");
  }

  return response.json();
}

async function getRAGStatus(): Promise<{ status: string; demo_mode: boolean; components: Record<string, boolean> }> {
  const response = await fetch(`${API_BASE}/api/rag/status`);
  return response.json();
}

// =============================================================================
// Components
// =============================================================================

function MetricGauge({ label, value, color }: { label: string; value: number; color: string }) {
  const percentage = Math.round(value * 100);
  const circumference = 2 * Math.PI * 40;
  const strokeDashoffset = circumference - (value * circumference);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="w-24 h-24 transform -rotate-90">
          <circle
            cx="48"
            cy="48"
            r="40"
            stroke="currentColor"
            strokeWidth="8"
            fill="none"
            className="text-slate-700"
          />
          <motion.circle
            cx="48"
            cy="48"
            r="40"
            stroke={color}
            strokeWidth="8"
            fill="none"
            strokeLinecap="round"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1, ease: "easeOut" }}
            style={{ strokeDasharray: circumference }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xl font-bold text-white">{percentage}%</span>
        </div>
      </div>
      <span className="mt-2 text-xs text-slate-400">{label}</span>
    </div>
  );
}

function PipelineStage({
  stage,
  stageConfig,
  isActive,
  isComplete,
}: {
  stage?: StageResult;
  stageConfig: typeof PIPELINE_STAGES[0];
  isActive: boolean;
  isComplete: boolean;
}) {
  const Icon = stageConfig.icon;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <motion.div
          className={`flex items-center gap-2 p-2 rounded-lg transition-all cursor-help ${
            isComplete
              ? "bg-emerald-500/20 border border-emerald-500/50"
              : isActive
              ? "bg-cyan-500/20 border border-cyan-500/50 animate-pulse"
              : "bg-slate-800/50 border border-slate-700/50"
          }`}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div
            className={`p-1.5 rounded ${
              isComplete ? "bg-emerald-500/30" : isActive ? "bg-cyan-500/30" : "bg-slate-700/50"
            }`}
          >
            {isActive && !isComplete ? (
              <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />
            ) : (
              <Icon className={`w-4 h-4 ${isComplete ? "text-emerald-400" : "text-slate-400"}`} />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-xs font-medium text-white truncate">{stageConfig.label}</div>
            {stage && (
              <div className="text-[10px] text-slate-500">
                {stage.duration_ms}ms
                {stage.results_count !== undefined && ` • ${stage.results_count} results`}
              </div>
            )}
          </div>
          {isComplete && <Check className="w-4 h-4 text-emerald-400" />}
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
  );
}

function DocumentCard({
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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="bg-slate-800/50 rounded-lg border border-slate-700/50 overflow-hidden"
    >
      <div
        className="flex items-center gap-3 p-3 cursor-pointer hover:bg-slate-700/30 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-cyan-500/20 text-cyan-400 font-bold text-sm">
          [{index + 1}]
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-medium text-white truncate">{doc.title || "Document"}</div>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-slate-600">
              {doc.source}
            </Badge>
            <span>v{doc.version}</span>
            <span>•</span>
            <span>{doc.category}</span>
          </div>
        </div>
        <div className="text-right">
          <div className={`text-lg font-bold ${scoreColor}`}>
            {Math.round(doc.rerank_score * 100)}%
          </div>
          <div className="text-[10px] text-slate-500">relevance</div>
        </div>
        <ChevronDown
          className={`w-4 h-4 text-slate-400 transition-transform ${isExpanded ? "rotate-180" : ""}`}
        />
      </div>
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-slate-700/50"
          >
            <div className="p-3 space-y-3">
              <div className="grid grid-cols-4 gap-2 text-xs">
                <div className="bg-slate-900/50 p-2 rounded">
                  <div className="text-slate-500">Dense</div>
                  <div className="text-purple-400 font-medium">{(doc.dense_score * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-slate-900/50 p-2 rounded">
                  <div className="text-slate-500">Sparse</div>
                  <div className="text-cyan-400 font-medium">{(doc.sparse_score * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-slate-900/50 p-2 rounded">
                  <div className="text-slate-500">RRF</div>
                  <div className="text-amber-400 font-medium">{(doc.rrf_score * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-slate-900/50 p-2 rounded">
                  <div className="text-slate-500">Rerank</div>
                  <div className="text-emerald-400 font-medium">{(doc.rerank_score * 100).toFixed(1)}%</div>
                </div>
              </div>
              <div className="bg-slate-900/50 p-3 rounded text-sm text-slate-300 font-mono whitespace-pre-wrap max-h-48 overflow-auto">
                {doc.content}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// Citation HoverCard component
function CitationPreview({ citation, documents }: { citation: { index: number; doc_id: string; text: string }; documents: Document[] }) {
  const doc = documents.find(d => d.id === citation.doc_id);
  if (!doc) return null;

  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <sup className="text-cyan-400 cursor-pointer hover:text-cyan-300 font-bold">[{citation.index}]</sup>
      </HoverCardTrigger>
      <HoverCardContent className="w-80 bg-slate-900 border-slate-700">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-cyan-400" />
            <span className="font-medium text-white text-sm">{doc.title || "Document"}</span>
          </div>
          <div className="flex gap-2">
            <Badge variant="outline" className="text-[10px] border-slate-600">{doc.source}</Badge>
            <Badge variant="outline" className="text-[10px] border-slate-600">v{doc.version}</Badge>
            <Badge variant="outline" className="text-[10px] border-emerald-600 text-emerald-400">
              {Math.round(doc.rerank_score * 100)}% relevant
            </Badge>
          </div>
          <p className="text-xs text-slate-400 line-clamp-3">{citation.text}</p>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

// =============================================================================
// Main Page Component
// =============================================================================

export default function RAGDemoPage() {
  // State
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<RAGResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [systemStatus, setSystemStatus] = useState<{
    status: string;
    demo_mode: boolean;
    components: Record<string, boolean>;
  } | null>(null);

  // Settings
  const [settings, setSettings] = useState<RAGSettings>({
    retrieval_mode: "hybrid",
    top_k: 10,
    enable_reranking: true,
    show_debug: false,
  });

  // Pipeline animation state
  const [activeStageIndex, setActiveStageIndex] = useState(-1);
  const [completedStages, setCompletedStages] = useState<Set<string>>(new Set());

  // Check system status on mount
  useEffect(() => {
    getRAGStatus()
      .then(setSystemStatus)
      .catch(() => setSystemStatus({ status: "unknown", demo_mode: true, components: {} }));
  }, []);

  // Keyboard shortcut for command palette
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setShowCommandPalette(prev => !prev);
      }
      if (e.key === "Escape") {
        setShowCommandPalette(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Pipeline animation
  useEffect(() => {
    if (isLoading && !response) {
      let stageIndex = 0;
      const interval = setInterval(() => {
        if (stageIndex < PIPELINE_STAGES.length) {
          setActiveStageIndex(stageIndex);
          if (stageIndex > 0) {
            setCompletedStages((prev) => new Set([...prev, PIPELINE_STAGES[stageIndex - 1].id]));
          }
          stageIndex++;
        } else {
          clearInterval(interval);
        }
      }, 200);
      return () => clearInterval(interval);
    }
  }, [isLoading, response]);

  // Handle query submission
  const handleSubmit = useCallback(async () => {
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setResponse(null);
    setActiveStageIndex(0);
    setCompletedStages(new Set());
    setShowCommandPalette(false);

    try {
      const result = await queryRAG(query, sessionId, settings);
      setResponse(result);
      setSessionId(result.session_id);

      // Mark all stages complete
      setCompletedStages(new Set(PIPELINE_STAGES.map((s) => s.id)));
      setActiveStageIndex(-1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  }, [query, sessionId, settings, isLoading]);

  // Copy answer to clipboard
  const handleCopy = async () => {
    if (response?.answer) {
      await navigator.clipboard.writeText(response.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Reset conversation
  const handleReset = () => {
    setQuery("");
    setResponse(null);
    setError(null);
    setSessionId(null);
    setCompletedStages(new Set());
    setActiveStageIndex(-1);
  };

  // Render answer with hoverable citations
  const renderAnswerWithCitations = (answer: string, citations: RAGResponse["citations"], documents: Document[]) => {
    // Parse markdown-like formatting
    let html = answer
      .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre class="bg-slate-800 p-3 rounded-lg overflow-x-auto my-2"><code>$2</code></pre>')
      .replace(/`([^`]+)`/g, '<code class="bg-slate-800 px-1.5 py-0.5 rounded text-cyan-400 text-sm">$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong class="text-white font-semibold">$1</strong>');

    // Split by citation markers and rebuild with components
    const parts = html.split(/(\[\d+\])/g);

    return (
      <div className="text-slate-300 leading-relaxed">
        {parts.map((part, i) => {
          const citationMatch = part.match(/\[(\d+)\]/);
          if (citationMatch) {
            const citationIndex = parseInt(citationMatch[1]);
            const citation = citations.find(c => c.index === citationIndex);
            if (citation) {
              return <CitationPreview key={i} citation={citation} documents={documents} />;
            }
          }
          return <span key={i} dangerouslySetInnerHTML={{ __html: part }} />;
        })}
      </div>
    );
  };

  return (
    <TooltipProvider>
      <div className="min-h-screen w-full flex flex-col relative overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
        {/* Animated grid background - matching /technical */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#0ea5e920_1px,transparent_1px),linear-gradient(to_bottom,#0ea5e920_1px,transparent_1px)] bg-[size:3rem_3rem]" />

        {/* Gradient orbs - matching /technical */}
        <div className="absolute top-0 left-0 w-[600px] h-[600px] bg-cyan-500/10 rounded-full blur-[120px] -translate-x-1/2 -translate-y-1/2" />
        <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-purple-500/10 rounded-full blur-[120px] translate-x-1/2 translate-y-1/2" />
        <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] bg-emerald-500/5 rounded-full blur-[100px] -translate-x-1/2 -translate-y-1/2" />

        {/* Circuit pattern overlay */}
        <div className="absolute inset-0 opacity-5">
          <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="circuit" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
                <path d="M10 10h80v80H10z" fill="none" stroke="currentColor" strokeWidth="0.5"/>
                <circle cx="10" cy="10" r="2" fill="currentColor"/>
                <circle cx="90" cy="10" r="2" fill="currentColor"/>
                <circle cx="10" cy="90" r="2" fill="currentColor"/>
                <circle cx="90" cy="90" r="2" fill="currentColor"/>
                <path d="M50 10v30M10 50h30M50 90v-30M90 50h-30" stroke="currentColor" strokeWidth="0.5"/>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#circuit)" className="text-cyan-400"/>
          </svg>
        </div>

        {/* Command Palette Modal */}
        <AnimatePresence>
          {showCommandPalette && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh] bg-black/50 backdrop-blur-sm"
              onClick={() => setShowCommandPalette(false)}
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.95, y: -20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: -20 }}
                onClick={(e) => e.stopPropagation()}
                className="w-full max-w-lg"
              >
                <Command className="rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">
                  <CommandInput placeholder="Search queries or type a question..." className="border-b border-slate-700" />
                  <CommandList className="max-h-80">
                    <CommandEmpty>No results found. Press Enter to search.</CommandEmpty>
                    <CommandGroup heading="Mesh Operations">
                      {SAMPLE_QUERIES.filter(q => q.category === "mesh" || q.category === "workflow").map((sq) => (
                        <CommandItem
                          key={sq.label}
                          onSelect={() => {
                            setQuery(sq.query);
                            setShowCommandPalette(false);
                          }}
                          className="cursor-pointer"
                        >
                          <sq.icon className="mr-2 h-4 w-4 text-cyan-400" />
                          <span>{sq.label}</span>
                          <span className="ml-auto text-xs text-slate-500">{sq.category}</span>
                        </CommandItem>
                      ))}
                    </CommandGroup>
                    <CommandSeparator />
                    <CommandGroup heading="Scripting & Materials">
                      {SAMPLE_QUERIES.filter(q => q.category === "scripting" || q.category === "materials" || q.category === "animation" || q.category === "uv").map((sq) => (
                        <CommandItem
                          key={sq.label}
                          onSelect={() => {
                            setQuery(sq.query);
                            setShowCommandPalette(false);
                          }}
                          className="cursor-pointer"
                        >
                          <sq.icon className="mr-2 h-4 w-4 text-purple-400" />
                          <span>{sq.label}</span>
                          <span className="ml-auto text-xs text-slate-500">{sq.category}</span>
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="relative z-10 container mx-auto px-4 py-6 max-w-7xl flex-1">
          {/* Title Bar - matching /technical style */}
          <div className="mb-6">
            <div className="flex items-center gap-4 mb-4">
              <div className="h-1 w-12 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full" />
              <span className="text-sm font-mono text-cyan-400 uppercase tracking-wider">Agentic RAG System</span>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                  <Brain className="w-8 h-8 text-cyan-400" />
                  RAG Demo
                </h1>
                <p className="text-slate-400 text-sm mt-1">
                  7-Layer Fully Agentic Retrieval-Augmented Generation • GPT-5.1 + BGE-M3
                </p>
              </div>
              <div className="flex items-center gap-3">
                {systemStatus && (
                  <Badge
                    variant="outline"
                    className={`${
                      systemStatus.demo_mode
                        ? "border-amber-500/50 text-amber-400"
                        : "border-emerald-500/50 text-emerald-400"
                    }`}
                  >
                    <Zap className="w-3 h-3 mr-1" />
                    {systemStatus.demo_mode ? "Demo Mode" : "Live"}
                  </Badge>
                )}
                <Badge
                  variant="outline"
                  className={`${
                    systemStatus?.status === "healthy"
                      ? "border-emerald-500/50 text-emerald-400"
                      : "border-slate-500/50 text-slate-400"
                  }`}
                >
                  <Activity className="w-3 h-3 mr-1" />
                  {systemStatus?.status || "Checking..."}
                </Badge>
              </div>
            </div>
          </div>

          {/* Main Grid Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column - Input & Settings */}
            <div className="lg:col-span-4 space-y-4">
              {/* Query Input with InputGroup */}
              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base text-white flex items-center gap-2">
                    <MessageSquare className="w-4 h-4 text-cyan-400" />
                    Query
                    <Button
                      variant="ghost"
                      size="sm"
                      className="ml-auto h-6 px-2 text-xs text-slate-400 hover:text-white"
                      onClick={() => setShowCommandPalette(true)}
                    >
                      <Search className="w-3 h-3 mr-1" />
                      <Kbd className="ml-1">⌘K</Kbd>
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <InputGroup className="bg-slate-800/50 border-slate-700/50">
                    <InputGroupTextarea
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSubmit();
                        }
                      }}
                      placeholder="Ask about Blender, 3D modeling, Python scripting..."
                      className="min-h-[80px] bg-transparent border-0 text-white placeholder-slate-500 focus-visible:ring-0"
                      disabled={isLoading}
                    />
                    <InputGroupAddon align="block-end" className="border-t border-slate-700/50">
                      <InputGroupText className="text-xs text-slate-500">
                        {query.length}/2000
                      </InputGroupText>
                      <Separator orientation="vertical" className="h-4 mx-2" />
                      <InputGroupText className="text-xs text-slate-500">
                        <Kbd>Enter</Kbd> to send
                      </InputGroupText>
                      <InputGroupButton
                        variant="default"
                        size="sm"
                        className="ml-auto bg-cyan-600 hover:bg-cyan-500"
                        onClick={handleSubmit}
                        disabled={!query.trim() || isLoading}
                      >
                        {isLoading ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <>
                            <Send className="w-4 h-4" />
                            <span className="ml-1">Send</span>
                          </>
                        )}
                      </InputGroupButton>
                    </InputGroupAddon>
                  </InputGroup>

                  {/* Quick Query Buttons */}
                  <div className="flex flex-wrap gap-1.5">
                    {SAMPLE_QUERIES.slice(0, 4).map((sq) => (
                      <Tooltip key={sq.label}>
                        <TooltipTrigger asChild>
                          <Button
                            variant="outline"
                            size="sm"
                            className="text-xs h-7 border-slate-700 text-slate-400 hover:text-white hover:border-cyan-500/50"
                            onClick={() => setQuery(sq.query)}
                            disabled={isLoading}
                          >
                            <sq.icon className="w-3 h-3 mr-1" />
                            {sq.label}
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs">{sq.query}</p>
                        </TooltipContent>
                      </Tooltip>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Settings Panel with Accordion */}
              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base text-white flex items-center gap-2">
                    <Settings2 className="w-4 h-4 text-purple-400" />
                    Settings
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Retrieval Mode with ToggleGroup */}
                  <div className="space-y-2">
                    <label className="text-xs text-slate-400 flex items-center gap-1">
                      Retrieval Mode
                      <Tooltip>
                        <TooltipTrigger>
                          <Info className="w-3 h-3 text-slate-500" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p className="text-xs max-w-xs">
                            <strong>Hybrid:</strong> Best of both worlds (recommended)<br/>
                            <strong>Dense:</strong> Semantic similarity only<br/>
                            <strong>Sparse:</strong> Keyword matching only
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </label>
                    <ToggleGroup
                      type="single"
                      value={settings.retrieval_mode}
                      onValueChange={(v) => v && setSettings({ ...settings, retrieval_mode: v as RAGSettings["retrieval_mode"] })}
                      className="w-full justify-start bg-slate-800/50 p-1 rounded-lg"
                    >
                      <ToggleGroupItem value="hybrid" className="flex-1 text-xs data-[state=on]:bg-cyan-500/20 data-[state=on]:text-cyan-400">
                        Hybrid
                      </ToggleGroupItem>
                      <ToggleGroupItem value="dense" className="flex-1 text-xs data-[state=on]:bg-purple-500/20 data-[state=on]:text-purple-400">
                        Dense
                      </ToggleGroupItem>
                      <ToggleGroupItem value="sparse" className="flex-1 text-xs data-[state=on]:bg-amber-500/20 data-[state=on]:text-amber-400">
                        Sparse
                      </ToggleGroupItem>
                    </ToggleGroup>
                  </div>

                  {/* Top-K Slider */}
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <label className="text-xs text-slate-400">Top-K Results</label>
                      <span className="text-xs text-cyan-400 font-mono">{settings.top_k}</span>
                    </div>
                    <Slider
                      value={[settings.top_k]}
                      onValueChange={([v]) => setSettings({ ...settings, top_k: v })}
                      min={1}
                      max={20}
                      step={1}
                      className="w-full"
                    />
                  </div>

                  {/* Advanced Settings Accordion */}
                  <Accordion type="single" collapsible className="w-full">
                    <AccordionItem value="advanced" className="border-slate-700/50">
                      <AccordionTrigger className="text-xs text-slate-400 hover:text-white py-2">
                        Advanced Settings
                      </AccordionTrigger>
                      <AccordionContent className="space-y-3 pt-2">
                        <div className="flex items-center justify-between">
                          <label className="text-xs text-slate-400">Enable Reranking</label>
                          <Switch
                            checked={settings.enable_reranking}
                            onCheckedChange={(v) => setSettings({ ...settings, enable_reranking: v })}
                          />
                        </div>
                        <div className="flex items-center justify-between">
                          <label className="text-xs text-slate-400">Show Debug Info</label>
                          <Switch
                            checked={settings.show_debug}
                            onCheckedChange={(v) => setSettings({ ...settings, show_debug: v })}
                          />
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>

                  {/* Reset Button */}
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full border-slate-700 text-slate-400 hover:text-white hover:border-red-500/50"
                    onClick={handleReset}
                  >
                    <RefreshCw className="w-3 h-3 mr-2" />
                    Reset Conversation
                  </Button>
                </CardContent>
              </Card>

              {/* Pipeline Visualization with Tooltips */}
              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-base text-white flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-emerald-400" />
                    Pipeline Stages
                    <Badge variant="outline" className="ml-auto text-[10px] border-slate-600">
                      9 stages
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {PIPELINE_STAGES.map((stageConfig, index) => {
                      const stageData = response?.stages.find((s) => s.stage === stageConfig.id);
                      return (
                        <PipelineStage
                          key={stageConfig.id}
                          stageConfig={stageConfig}
                          stage={stageData}
                          isActive={activeStageIndex === index}
                          isComplete={completedStages.has(stageConfig.id)}
                        />
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Right Column - Results */}
            <div className="lg:col-span-8 space-y-4">
              {/* Query Analysis */}
              {response?.analysis && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                  <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base text-white flex items-center gap-2">
                        <Search className="w-4 h-4 text-cyan-400" />
                        Query Analysis
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="bg-slate-800/50 p-3 rounded-lg">
                          <div className="text-xs text-slate-500 mb-1">Intent</div>
                          <div className="text-sm font-medium text-cyan-400">{response.analysis.intent}</div>
                        </div>
                        <div className="bg-slate-800/50 p-3 rounded-lg">
                          <div className="text-xs text-slate-500 mb-1">Confidence</div>
                          <div className="text-sm font-medium text-emerald-400">
                            {Math.round(response.analysis.confidence * 100)}%
                          </div>
                        </div>
                        <div className="bg-slate-800/50 p-3 rounded-lg">
                          <div className="text-xs text-slate-500 mb-1">Multi-step</div>
                          <div className="text-sm font-medium text-white">
                            {response.analysis.is_multi_step ? "Yes" : "No"}
                          </div>
                        </div>
                        <div className="bg-slate-800/50 p-3 rounded-lg">
                          <div className="text-xs text-slate-500 mb-1">Entities</div>
                          <div className="text-sm font-medium text-purple-400">
                            {response.analysis.entities.length}
                          </div>
                        </div>
                      </div>
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {response.analysis.entities.map((entity, i) => (
                          <Badge key={i} variant="outline" className="text-xs border-purple-500/50 text-purple-400">
                            {entity}
                          </Badge>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {/* Retrieved Documents */}
              {response?.documents && response.documents.length > 0 && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
                  <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base text-white flex items-center gap-2">
                        <Database className="w-4 h-4 text-purple-400" />
                        Retrieved Documents
                        <Badge variant="outline" className="ml-auto text-xs border-slate-600">
                          {response.documents.length} docs
                        </Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 max-h-80 overflow-y-auto">
                      {response.documents.map((doc, index) => (
                        <DocumentCard
                          key={doc.id}
                          doc={doc}
                          index={index}
                          isExpanded={expandedDoc === doc.id}
                          onToggle={() => setExpandedDoc(expandedDoc === doc.id ? null : doc.id)}
                        />
                      ))}
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {/* Generated Answer with HoverCard Citations */}
              {response?.answer && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                  <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-base text-white flex items-center gap-2">
                          <Sparkles className="w-4 h-4 text-emerald-400" />
                          Generated Answer
                        </CardTitle>
                        <div className="flex items-center gap-2">
                          {response.validation.passed ? (
                            <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/50">
                              <CheckCircle2 className="w-3 h-3 mr-1" />
                              Validated
                            </Badge>
                          ) : (
                            <Badge className="bg-amber-500/20 text-amber-400 border-amber-500/50">
                              <AlertCircle className="w-3 h-3 mr-1" />
                              Warning
                            </Badge>
                          )}
                          <Button variant="ghost" size="sm" onClick={handleCopy}>
                            {copied ? (
                              <Check className="w-4 h-4 text-emerald-400" />
                            ) : (
                              <Copy className="w-4 h-4 text-slate-400" />
                            )}
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="prose prose-invert prose-sm max-w-none">
                        {renderAnswerWithCitations(response.answer, response.citations, response.documents)}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {/* Metrics Dashboard */}
              {response?.metrics && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
                  <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base text-white flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-amber-400" />
                        Quality Metrics (RAGAS)
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex justify-around items-center">
                        <MetricGauge label="Faithfulness" value={response.metrics.faithfulness} color="#10b981" />
                        <MetricGauge label="Relevancy" value={response.metrics.relevancy} color="#06b6d4" />
                        <MetricGauge label="Completeness" value={response.metrics.completeness} color="#8b5cf6" />
                        <MetricGauge
                          label="Composite"
                          value={response.metrics.composite_score}
                          color="#f59e0b"
                        />
                      </div>
                      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2 text-center">
                        <div className="bg-slate-800/50 p-2 rounded">
                          <div className="text-lg font-bold text-white">{response.latency.total}ms</div>
                          <div className="text-[10px] text-slate-500">Total Latency</div>
                        </div>
                        <div className="bg-slate-800/50 p-2 rounded">
                          <div className="text-lg font-bold text-purple-400">{response.latency.retrieval_dense}ms</div>
                          <div className="text-[10px] text-slate-500">Dense Search</div>
                        </div>
                        <div className="bg-slate-800/50 p-2 rounded">
                          <div className="text-lg font-bold text-cyan-400">{response.latency.generation}ms</div>
                          <div className="text-[10px] text-slate-500">Generation</div>
                        </div>
                        <div className="bg-slate-800/50 p-2 rounded">
                          <div className="text-lg font-bold text-emerald-400">{response.validation.attempt}</div>
                          <div className="text-[10px] text-slate-500">Attempts</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {/* Error State */}
              {error && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                  <Card className="bg-red-900/20 border-red-700/50">
                    <CardContent className="flex items-center gap-3 py-4">
                      <AlertCircle className="w-5 h-5 text-red-400" />
                      <div className="text-red-400">{error}</div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}

              {/* Loading State */}
              {isLoading && !response && (
                <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                  <CardContent className="py-8">
                    <div className="flex flex-col items-center justify-center gap-4">
                      <div className="relative">
                        <div className="w-16 h-16 border-4 border-cyan-500/30 rounded-full animate-pulse" />
                        <Loader2 className="w-8 h-8 text-cyan-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
                      </div>
                      <div className="text-slate-400 text-sm">Processing query through 7-layer pipeline...</div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Empty State */}
              {!isLoading && !response && !error && (
                <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                  <CardContent className="py-12">
                    <div className="flex flex-col items-center justify-center gap-4 text-center">
                      <div className="p-4 bg-cyan-500/10 rounded-full">
                        <Brain className="w-10 h-10 text-cyan-400" />
                      </div>
                      <div>
                        <h3 className="text-lg font-medium text-white mb-1">Ready to Query</h3>
                        <p className="text-slate-400 text-sm max-w-md">
                          Ask anything about Blender, 3D modeling, Python scripting, or the Blender Python API.
                          The agentic RAG system will retrieve relevant documentation and generate accurate answers.
                        </p>
                      </div>
                      <div className="flex gap-2 flex-wrap justify-center">
                        <Button
                          variant="outline"
                          size="sm"
                          className="border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10"
                          onClick={() => setShowCommandPalette(true)}
                        >
                          <Search className="w-3 h-3 mr-1" />
                          Browse Queries
                          <Kbd className="ml-2">⌘K</Kbd>
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="border-purple-500/50 text-purple-400 hover:bg-purple-500/10"
                          onClick={() => setQuery(SAMPLE_QUERIES[0].query)}
                        >
                          Try: &quot;{SAMPLE_QUERIES[0].label}&quot;
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>

        {/* Presentation Dock */}
        <PresentationDock items={dockItems} />
      </div>
    </TooltipProvider>
  );
}
