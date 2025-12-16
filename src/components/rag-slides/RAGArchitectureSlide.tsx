"use client";

import { TechSlideWrapper } from "../tech-slides/TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface RAGArchitectureSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function RAGArchitectureSlide({ slideNumber, totalSlides }: RAGArchitectureSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="System Architecture">
      <div className="flex flex-col h-full">
        <h2 className="text-4xl font-bold text-white mb-2">
          RAG <span className="text-emerald-400">Pipeline Architecture</span>
        </h2>
        <p className="text-slate-400 mb-4">Hybrid retrieval with agentic validation for grounded responses</p>

        {/* Main Architecture Diagram */}
        <div className="flex-1 relative">
          <svg viewBox="0 0 1000 420" className="w-full h-full">
            <defs>
              <linearGradient id="gradEmerald" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#059669" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradCyan" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradBlue" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#2563eb" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradPurple" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#a855f7" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradAmber" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#d97706" stopOpacity="0.8"/>
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
              </marker>
            </defs>

            {/* User Query Input */}
            <g transform="translate(20, 160)">
              <rect width="80" height="80" rx="8" fill="#1e293b" stroke="#334155" strokeWidth="2"/>
              <text x="40" y="30" textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="monospace">User</text>
              <text x="40" y="45" textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="monospace">Query</text>
              <circle cx="40" cy="62" r="10" fill="#10b981" fillOpacity="0.3" stroke="#10b981"/>
              <text x="40" y="66" textAnchor="middle" fill="#10b981" fontSize="10">?</text>
            </g>

            {/* Arrow 1 */}
            <path d="M105 200 L140 200" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>

            {/* Query Transformer */}
            <g transform="translate(145, 145)">
              <rect width="110" height="110" rx="8" fill="url(#gradEmerald)" filter="url(#glow)"/>
              <text x="55" y="25" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Query</text>
              <text x="55" y="40" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Transformer</text>
              <rect x="10" y="50" width="90" height="20" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="55" y="64" textAnchor="middle" fill="#a7f3d0" fontSize="8">Intent Extract</text>
              <rect x="10" y="73" width="90" height="20" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="55" y="87" textAnchor="middle" fill="#a7f3d0" fontSize="8">Multi-hop Decomp</text>
            </g>

            {/* Arrow to Dense */}
            <path d="M260 180 L295 140" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>
            {/* Arrow to Sparse */}
            <path d="M260 220 L295 260" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>

            {/* Dense Search (Vector) */}
            <g transform="translate(300, 80)">
              <rect width="130" height="100" rx="8" fill="url(#gradCyan)" filter="url(#glow)"/>
              <text x="65" y="22" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Dense Search</text>
              <text x="65" y="38" textAnchor="middle" fill="#a5f3fc" fontSize="9">(Vector Similarity)</text>
              <rect x="10" y="48" width="110" height="18" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="65" y="60" textAnchor="middle" fill="#a5f3fc" fontSize="8">MiniLM-L6 (384 dims)</text>
              <rect x="10" y="70" width="110" height="18" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="65" y="82" textAnchor="middle" fill="#a5f3fc" fontSize="8">HNSW Index</text>
            </g>

            {/* Sparse Search (BM25) */}
            <g transform="translate(300, 220)">
              <rect width="130" height="100" rx="8" fill="url(#gradBlue)" filter="url(#glow)"/>
              <text x="65" y="22" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Sparse Search</text>
              <text x="65" y="38" textAnchor="middle" fill="#bfdbfe" fontSize="9">(Keyword Matching)</text>
              <rect x="10" y="48" width="110" height="18" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="65" y="60" textAnchor="middle" fill="#bfdbfe" fontSize="8">BM25 Algorithm</text>
              <rect x="10" y="70" width="110" height="18" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="65" y="82" textAnchor="middle" fill="#bfdbfe" fontSize="8">GIN Index (PostgreSQL)</text>
            </g>

            {/* Arrows to RRF */}
            <path d="M435 130 L475 175" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>
            <path d="M435 270 L475 225" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>

            {/* RRF Fusion */}
            <g transform="translate(480, 160)">
              <rect width="100" height="80" rx="8" fill="url(#gradPurple)" filter="url(#glow)"/>
              <text x="50" y="25" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">RRF Fusion</text>
              <text x="50" y="45" textAnchor="middle" fill="#e9d5ff" fontSize="9">Merge Results</text>
              <text x="50" y="62" textAnchor="middle" fill="#c4b5fd" fontSize="8">1/(k + rank)</text>
            </g>

            {/* Arrow to Reranker */}
            <path d="M585 200 L620 200" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>

            {/* Cross-Encoder Reranker */}
            <g transform="translate(625, 145)">
              <rect width="120" height="110" rx="8" fill="url(#gradAmber)" filter="url(#glow)"/>
              <text x="60" y="25" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Reranker</text>
              <text x="60" y="40" textAnchor="middle" fill="#fef3c7" fontSize="9">(Cross-Encoder)</text>
              <rect x="10" y="50" width="100" height="18" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="60" y="62" textAnchor="middle" fill="#fef3c7" fontSize="8">BGE-reranker-v2-m3</text>
              <rect x="10" y="72" width="100" height="18" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="60" y="84" textAnchor="middle" fill="#fef3c7" fontSize="8">Top-10 Context</text>
            </g>

            {/* Arrow to LLM */}
            <path d="M750 200 L785 200" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>

            {/* LLM Generation */}
            <g transform="translate(790, 130)">
              <rect width="100" height="140" rx="8" fill="#1e293b" stroke="#ec4899" strokeWidth="2"/>
              <text x="50" y="25" textAnchor="middle" fill="#ec4899" fontSize="11" fontWeight="bold">GPT-5.1</text>
              <text x="50" y="40" textAnchor="middle" fill="#64748b" fontSize="9">Generation</text>
              <rect x="10" y="50" width="80" height="18" rx="3" fill="#ec4899" fillOpacity="0.2"/>
              <text x="50" y="62" textAnchor="middle" fill="#fce7f3" fontSize="8">Context + Query</text>
              <rect x="10" y="72" width="80" height="18" rx="3" fill="#ec4899" fillOpacity="0.2"/>
              <text x="50" y="84" textAnchor="middle" fill="#fce7f3" fontSize="8">Grounded Answer</text>
              <rect x="10" y="94" width="80" height="18" rx="3" fill="#ec4899" fillOpacity="0.2"/>
              <text x="50" y="106" textAnchor="middle" fill="#fce7f3" fontSize="8">Source Citations</text>
            </g>

            {/* Output */}
            <g transform="translate(900, 175)">
              <rect width="70" height="50" rx="8" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="2"/>
              <text x="35" y="30" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="bold">Answer</text>
            </g>
            <path d="M890 200 L898 200" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>

            {/* Database (bottom) */}
            <g transform="translate(340, 350)">
              <rect width="160" height="50" rx="6" fill="#374151" stroke="#4b5563" strokeWidth="1"/>
              <text x="80" y="20" textAnchor="middle" fill="#9ca3af" fontSize="10" fontWeight="bold">PostgreSQL + pgvector</text>
              <text x="80" y="38" textAnchor="middle" fill="#6b7280" fontSize="9">Documents + Embeddings + Metadata</text>
            </g>
            {/* Connection lines to DB */}
            <path d="M365 180 L365 348" stroke="#64748b" strokeWidth="1" strokeDasharray="4"/>
            <path d="M365 320 L365 348" stroke="#64748b" strokeWidth="1" strokeDasharray="4"/>

            {/* Validation Loop (top) */}
            <g transform="translate(625, 30)">
              <rect width="160" height="45" rx="6" fill="#374151" stroke="#f59e0b" strokeWidth="1" strokeDasharray="4"/>
              <text x="80" y="18" textAnchor="middle" fill="#f59e0b" fontSize="9" fontWeight="bold">Agentic Validation</text>
              <text x="80" y="34" textAnchor="middle" fill="#6b7280" fontSize="8">Code Check | API Verify | Hallucination Detect</text>
            </g>
            {/* Validation loop arrow */}
            <path d="M705 75 L705 143" stroke="#f59e0b" strokeWidth="1" strokeDasharray="4" markerEnd="url(#arrow)"/>
          </svg>
        </div>

        {/* Performance Metrics */}
        <div className="flex justify-center gap-4 mt-2">
          {[
            { metric: "<200ms", label: "Retrieval", color: "text-emerald-400" },
            { metric: ">0.85", label: "Precision", color: "text-cyan-400" },
            { metric: "<5%", label: "Hallucination", color: "text-amber-400" },
            { metric: ">0.80", label: "MRR@10", color: "text-purple-400" }
          ].map((item) => (
            <Card key={item.label} className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-3 text-center">
                <div className={`text-lg font-bold ${item.color}`}>{item.metric}</div>
                <div className="text-xs text-slate-500">{item.label}</div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
