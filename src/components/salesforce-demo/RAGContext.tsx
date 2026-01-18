"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BookOpen,
  FileText,
  ChevronDown,
  ExternalLink,
  Sparkles,
  GraduationCap,
  Code,
  Shield,
  BarChart3,
  Loader2,
  Check,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  SalesforceDocument,
  RAGStageResult,
  SalesforceRAGMetrics,
  SalesforceQueryAnalysis,
  SALESFORCE_PIPELINE_STAGES,
} from "./types";

interface RAGContextProps {
  isActive: boolean;
  currentStage: string;
  stages: RAGStageResult[];
  analysis?: SalesforceQueryAnalysis;
  documents: SalesforceDocument[];
  metrics?: SalesforceRAGMetrics;
  isComplete: boolean;
}

// Helper to get source icon
function getSourceIcon(source: SalesforceDocument["source"]) {
  switch (source) {
    case "help_docs":
      return <FileText className="w-3 h-3" />;
    case "trailhead":
      return <GraduationCap className="w-3 h-3" />;
    case "apex_guide":
      return <Code className="w-3 h-3" />;
    case "admin_guide":
      return <Shield className="w-3 h-3" />;
    case "community":
      return <BookOpen className="w-3 h-3" />;
  }
}

// Helper to get source color
function getSourceColor(source: SalesforceDocument["source"]) {
  switch (source) {
    case "help_docs":
      return "border-[#0176D3]/50 text-[#0176D3]";
    case "trailhead":
      return "border-purple-500/50 text-purple-400";
    case "apex_guide":
      return "border-orange-500/50 text-orange-400";
    case "admin_guide":
      return "border-emerald-500/50 text-emerald-400";
    case "community":
      return "border-cyan-500/50 text-cyan-400";
  }
}

// Metric Gauge Component
function MetricGauge({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const percentage = Math.round(value * 100);
  const circumference = 2 * Math.PI * 28;
  const strokeDashoffset = circumference - value * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-16 h-16">
        <svg className="w-16 h-16 transform -rotate-90">
          <circle
            cx="32"
            cy="32"
            r="28"
            stroke="currentColor"
            strokeWidth="4"
            fill="none"
            className="text-slate-700"
          />
          <motion.circle
            cx="32"
            cy="32"
            r="28"
            stroke={color}
            strokeWidth="4"
            fill="none"
            strokeLinecap="round"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1, ease: "easeOut" }}
            style={{ strokeDasharray: circumference }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold text-white">{percentage}%</span>
        </div>
      </div>
      <span className="mt-1 text-[10px] text-slate-400">{label}</span>
    </div>
  );
}

export function RAGContext({
  isActive,
  currentStage,
  stages,
  analysis,
  documents,
  metrics,
  isComplete,
}: RAGContextProps) {
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  const completedStages = new Set(stages.map((s) => s.stage));

  return (
    <div className="space-y-4">
      {/* Pipeline Stages */}
      <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base text-white flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-[#0176D3]" />
            Knowledge Retrieval
            {isActive && !isComplete && (
              <Badge
                variant="outline"
                className="ml-auto text-[10px] border-[#0176D3]/50 text-[#0176D3] animate-pulse"
              >
                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                Processing
              </Badge>
            )}
            {isComplete && (
              <Badge
                variant="outline"
                className="ml-auto text-[10px] border-emerald-500/50 text-emerald-400"
              >
                <Check className="w-3 h-3 mr-1" />
                Complete
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-1.5">
            {SALESFORCE_PIPELINE_STAGES.map((stageConfig, index) => {
              const stageData = stages.find((s) => s.stage === stageConfig.id);
              const isStageActive =
                isActive && stageConfig.id === currentStage;
              const isStageComplete =
                completedStages.has(stageConfig.id) ||
                (isComplete &&
                  SALESFORCE_PIPELINE_STAGES.findIndex(
                    (s) => s.id === currentStage
                  ) >= index);

              return (
                <motion.div
                  key={stageConfig.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.03 }}
                  className={`flex items-center gap-1.5 p-1.5 rounded text-[10px] ${
                    isStageComplete
                      ? "bg-emerald-500/20 border border-emerald-500/30"
                      : isStageActive
                      ? "bg-[#0176D3]/20 border border-[#0176D3]/30"
                      : "bg-slate-800/50 border border-slate-700/30"
                  }`}
                >
                  {isStageActive && !isStageComplete ? (
                    <Loader2 className="w-3 h-3 text-[#0176D3] animate-spin flex-shrink-0" />
                  ) : isStageComplete ? (
                    <Check className="w-3 h-3 text-emerald-400 flex-shrink-0" />
                  ) : (
                    <div className="w-3 h-3 rounded-full bg-slate-600 flex-shrink-0" />
                  )}
                  <span
                    className={`truncate ${
                      isStageComplete
                        ? "text-emerald-400"
                        : isStageActive
                        ? "text-[#0176D3]"
                        : "text-slate-400"
                    }`}
                  >
                    {stageConfig.label}
                  </span>
                  {stageData?.duration_ms && (
                    <span className="text-slate-500 ml-auto">
                      {stageData.duration_ms}ms
                    </span>
                  )}
                </motion.div>
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
                <BarChart3 className="w-4 h-4 text-purple-400" />
                Query Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="bg-slate-900/50 p-2 rounded">
                  <div className="text-slate-500">Intent</div>
                  <div className="text-[#0176D3] font-medium truncate">
                    {analysis.intent}
                  </div>
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
              {analysis.suggested_objects && analysis.suggested_objects.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {analysis.suggested_objects.map((obj, i) => (
                    <Badge
                      key={i}
                      variant="outline"
                      className="text-[10px] border-[#0176D3]/50 text-[#0176D3]"
                    >
                      {obj}
                    </Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Retrieved Documents */}
      {documents.length > 0 && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-white flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-cyan-400" />
                Knowledge Sources
                <Badge
                  variant="outline"
                  className="ml-auto text-[10px] border-slate-600"
                >
                  {documents.length} docs
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {documents.slice(0, 5).map((doc, index) => (
                <motion.div
                  key={doc.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-slate-900/50 rounded-lg border border-slate-700/50 overflow-hidden"
                >
                  <div
                    className="flex items-center gap-2 p-2 cursor-pointer hover:bg-slate-800/50"
                    onClick={() =>
                      setExpandedDoc(expandedDoc === doc.id ? null : doc.id)
                    }
                  >
                    <div className="flex items-center justify-center w-5 h-5 rounded bg-[#0176D3]/20 text-[#0176D3] font-bold text-[10px]">
                      {index + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-white text-xs font-medium truncate">
                        {doc.title}
                      </div>
                      <div className="flex items-center gap-1 mt-0.5">
                        <Badge
                          variant="outline"
                          className={`text-[9px] px-1 py-0 ${getSourceColor(
                            doc.source
                          )}`}
                        >
                          {getSourceIcon(doc.source)}
                          <span className="ml-1">{doc.source.replace("_", " ")}</span>
                        </Badge>
                      </div>
                    </div>
                    <div className="text-right">
                      <div
                        className={`text-xs font-bold ${
                          doc.rerank_score >= 0.9
                            ? "text-emerald-400"
                            : doc.rerank_score >= 0.7
                            ? "text-cyan-400"
                            : "text-amber-400"
                        }`}
                      >
                        {Math.round(doc.rerank_score * 100)}%
                      </div>
                    </div>
                    <ChevronDown
                      className={`w-3 h-3 text-slate-400 transition-transform ${
                        expandedDoc === doc.id ? "rotate-180" : ""
                      }`}
                    />
                  </div>
                  <AnimatePresence>
                    {expandedDoc === doc.id && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="border-t border-slate-700/50"
                      >
                        <div className="p-2">
                          <p className="text-[10px] text-slate-400 line-clamp-3">
                            {doc.content}
                          </p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
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
                Quality Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex justify-around">
                <MetricGauge
                  label="Faithfulness"
                  value={metrics.faithfulness}
                  color="#10b981"
                />
                <MetricGauge
                  label="Relevancy"
                  value={metrics.relevancy}
                  color="#0176D3"
                />
                <MetricGauge
                  label="Complete"
                  value={metrics.completeness}
                  color="#8b5cf6"
                />
                <MetricGauge
                  label="Overall"
                  value={metrics.composite_score}
                  color="#f59e0b"
                />
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
