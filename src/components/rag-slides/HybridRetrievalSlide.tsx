"use client";

import { TechSlideWrapper } from "../tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface HybridRetrievalSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function HybridRetrievalSlide({ slideNumber, totalSlides }: HybridRetrievalSlideProps) {
  const stages = [
    {
      num: "01",
      title: "Dense Search",
      subtitle: "Semantic Understanding",
      color: "cyan",
      items: ["MiniLM-L6-v2", "384 Dimensions", "Cosine Similarity", "HNSW Index"]
    },
    {
      num: "02",
      title: "Sparse Search",
      subtitle: "Keyword Matching",
      color: "blue",
      items: ["BM25 Algorithm", "TF-IDF Scoring", "Exact Terms", "GIN Index"]
    },
    {
      num: "03",
      title: "RRF Fusion",
      subtitle: "Result Merging",
      color: "purple",
      items: ["Rank Combination", "No Tuning Needed", "Statistical Optimal", "k = 60"]
    },
    {
      num: "04",
      title: "Metadata Filter",
      subtitle: "Domain Filtering",
      color: "amber",
      items: ["Version Match", "Category Filter", "Code vs Docs", "Priority Boost"]
    }
  ];

  const colorMap: Record<string, { bg: string; border: string; text: string; glow: string }> = {
    cyan: { bg: "bg-cyan-500/20", border: "border-cyan-500/50", text: "text-cyan-400", glow: "shadow-cyan-500/20" },
    blue: { bg: "bg-blue-500/20", border: "border-blue-500/50", text: "text-blue-400", glow: "shadow-blue-500/20" },
    purple: { bg: "bg-purple-500/20", border: "border-purple-500/50", text: "text-purple-400", glow: "shadow-purple-500/20" },
    amber: { bg: "bg-amber-500/20", border: "border-amber-500/50", text: "text-amber-400", glow: "shadow-amber-500/20" }
  };

  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Hybrid Retrieval">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Hybrid <span className="text-cyan-400">Retrieval</span> Pipeline
            </h2>
            <p className="text-slate-400">Dense semantic + Sparse lexical search with RRF fusion</p>
          </div>
          <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10">
            Best of Both Worlds
          </Badge>
        </div>

        {/* Main Diagram */}
        <div className="flex-1 relative mb-4">
          <svg viewBox="0 0 900 280" className="w-full h-full">
            <defs>
              <linearGradient id="cyanGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#06b6d4"/>
                <stop offset="100%" stopColor="#0891b2"/>
              </linearGradient>
              <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#3b82f6"/>
                <stop offset="100%" stopColor="#2563eb"/>
              </linearGradient>
              <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#a855f7"/>
                <stop offset="100%" stopColor="#7c3aed"/>
              </linearGradient>
              <filter id="glowSmall">
                <feGaussianBlur stdDeviation="1.5" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <marker id="arrowSmall" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#64748b"/>
              </marker>
            </defs>

            {/* Query Input */}
            <g transform="translate(20, 110)">
              <rect width="100" height="60" rx="8" fill="#1e293b" stroke="#10b981" strokeWidth="2"/>
              <text x="50" y="28" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">User Query</text>
              <text x="50" y="45" textAnchor="middle" fill="#64748b" fontSize="9">&quot;select all faces&quot;</text>
            </g>

            {/* Split arrows */}
            <path d="M125 125 L170 70" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowSmall)"/>
            <path d="M125 155 L170 210" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowSmall)"/>

            {/* Dense Search Branch */}
            <g transform="translate(175, 30)">
              <rect width="160" height="100" rx="10" fill="url(#cyanGrad)" fillOpacity="0.15" stroke="#06b6d4" strokeWidth="2" filter="url(#glowSmall)"/>
              <text x="80" y="25" textAnchor="middle" fill="#06b6d4" fontSize="12" fontWeight="bold">Dense Search</text>

              {/* Embedding visualization */}
              <g transform="translate(15, 40)">
                <rect width="130" height="50" rx="6" fill="#0f172a" opacity="0.6"/>
                <text x="65" y="18" textAnchor="middle" fill="#a5f3fc" fontSize="9">Query Embedding</text>
                {/* Vector dots */}
                {[0,1,2,3,4,5,6,7].map((i) => (
                  <circle key={i} cx={20 + i * 14} cy="35" r="4" fill="#06b6d4" opacity={0.3 + (i % 3) * 0.2}/>
                ))}
                <text x="65" y="47" textAnchor="middle" fill="#64748b" fontSize="7">384 dims</text>
              </g>
            </g>

            {/* Results box - Dense */}
            <g transform="translate(355, 45)">
              <rect width="80" height="70" rx="6" fill="#1e293b" stroke="#06b6d4" strokeWidth="1"/>
              <text x="40" y="18" textAnchor="middle" fill="#06b6d4" fontSize="9" fontWeight="bold">Top 100</text>
              {[0,1,2,3].map((i) => (
                <g key={i} transform={`translate(8, ${25 + i * 12})`}>
                  <rect width="64" height="10" rx="2" fill="#06b6d4" opacity={0.4 - i * 0.08}/>
                  <text x="58" y="8" textAnchor="end" fill="#a5f3fc" fontSize="7">{(0.95 - i * 0.05).toFixed(2)}</text>
                </g>
              ))}
            </g>

            {/* Sparse Search Branch */}
            <g transform="translate(175, 150)">
              <rect width="160" height="100" rx="10" fill="url(#blueGrad)" fillOpacity="0.15" stroke="#3b82f6" strokeWidth="2" filter="url(#glowSmall)"/>
              <text x="80" y="25" textAnchor="middle" fill="#3b82f6" fontSize="12" fontWeight="bold">Sparse Search</text>

              {/* BM25 visualization */}
              <g transform="translate(15, 40)">
                <rect width="130" height="50" rx="6" fill="#0f172a" opacity="0.6"/>
                <text x="65" y="18" textAnchor="middle" fill="#bfdbfe" fontSize="9">BM25 Tokens</text>
                {/* Keyword boxes */}
                <rect x="8" y="25" width="35" height="15" rx="3" fill="#3b82f6" opacity="0.5"/>
                <text x="25" y="36" textAnchor="middle" fill="white" fontSize="7">select</text>
                <rect x="48" y="25" width="25" height="15" rx="3" fill="#3b82f6" opacity="0.5"/>
                <text x="60" y="36" textAnchor="middle" fill="white" fontSize="7">all</text>
                <rect x="78" y="25" width="35" height="15" rx="3" fill="#3b82f6" opacity="0.5"/>
                <text x="95" y="36" textAnchor="middle" fill="white" fontSize="7">faces</text>
              </g>
            </g>

            {/* Results box - Sparse */}
            <g transform="translate(355, 165)">
              <rect width="80" height="70" rx="6" fill="#1e293b" stroke="#3b82f6" strokeWidth="1"/>
              <text x="40" y="18" textAnchor="middle" fill="#3b82f6" fontSize="9" fontWeight="bold">Top 100</text>
              {[0,1,2,3].map((i) => (
                <g key={i} transform={`translate(8, ${25 + i * 12})`}>
                  <rect width="64" height="10" rx="2" fill="#3b82f6" opacity={0.4 - i * 0.08}/>
                  <text x="58" y="8" textAnchor="end" fill="#bfdbfe" fontSize="7">{(12.5 - i * 1.5).toFixed(1)}</text>
                </g>
              ))}
            </g>

            {/* Arrows to RRF */}
            <path d="M440 80 L485 125" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowSmall)"/>
            <path d="M440 200 L485 155" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowSmall)"/>

            {/* RRF Fusion */}
            <g transform="translate(490, 95)">
              <rect width="140" height="90" rx="10" fill="url(#purpleGrad)" fillOpacity="0.15" stroke="#a855f7" strokeWidth="2" filter="url(#glowSmall)"/>
              <text x="70" y="22" textAnchor="middle" fill="#a855f7" fontSize="12" fontWeight="bold">RRF Fusion</text>

              {/* Formula */}
              <g transform="translate(10, 32)">
                <rect width="120" height="45" rx="6" fill="#0f172a" opacity="0.6"/>
                <text x="60" y="18" textAnchor="middle" fill="#e9d5ff" fontSize="9" fontFamily="monospace">score(d) = Σ</text>
                <text x="60" y="34" textAnchor="middle" fill="#c4b5fd" fontSize="10" fontFamily="monospace">1 / (k + rank)</text>
              </g>
            </g>

            {/* Arrow to output */}
            <path d="M635 140 L680 140" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowSmall)"/>

            {/* Fused Output */}
            <g transform="translate(685, 80)">
              <rect width="100" height="120" rx="8" fill="#1e293b" stroke="#10b981" strokeWidth="2"/>
              <text x="50" y="22" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">Fused Top-50</text>
              {[0,1,2,3,4].map((i) => (
                <g key={i} transform={`translate(10, ${32 + i * 16})`}>
                  <rect width="80" height="12" rx="3" fill="#10b981" opacity={0.5 - i * 0.08}/>
                  <text x="75" y="10" textAnchor="end" fill="#a7f3d0" fontSize="7">{(0.032 - i * 0.003).toFixed(3)}</text>
                </g>
              ))}
            </g>

            {/* Arrow to reranker */}
            <path d="M790 140 L825 140" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowSmall)"/>

            {/* To Reranker indicator */}
            <g transform="translate(830, 115)">
              <rect width="60" height="50" rx="6" fill="#f59e0b" fillOpacity="0.2" stroke="#f59e0b" strokeWidth="1.5"/>
              <text x="30" y="22" textAnchor="middle" fill="#f59e0b" fontSize="9" fontWeight="bold">To</text>
              <text x="30" y="38" textAnchor="middle" fill="#fbbf24" fontSize="9">Reranker</text>
            </g>
          </svg>
        </div>

        {/* Pipeline Stages Cards */}
        <div className="grid grid-cols-4 gap-3">
          {stages.map((stage) => (
            <div key={stage.num} className={`p-3 rounded-xl ${colorMap[stage.color].bg} border ${colorMap[stage.color].border} shadow-lg ${colorMap[stage.color].glow}`}>
              <div className={`text-2xl font-bold ${colorMap[stage.color].text} opacity-30 mb-1`}>
                {stage.num}
              </div>
              <h3 className="text-sm font-semibold text-white leading-tight">{stage.title}</h3>
              <p className="text-xs text-slate-500 mb-2">{stage.subtitle}</p>
              <div className="space-y-1">
                {stage.items.map((item) => (
                  <div key={item} className="text-xs text-slate-400 flex items-center gap-1.5">
                    <div className={`w-1 h-1 rounded-full ${colorMap[stage.color].text.replace('text-', 'bg-')}`} />
                    {item}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Bottom: Why Hybrid */}
        <div className="mt-3 p-3 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div>
                <span className="text-xs text-slate-500 uppercase tracking-wider">Why Hybrid?</span>
                <div className="text-sm text-slate-300">Dense: &quot;rotate objects&quot; | Sparse: &quot;bpy.ops.transform.rotate()&quot;</div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-center">
                <div className="text-xl font-bold text-cyan-400">+15%</div>
                <div className="text-xs text-slate-500">Recall Boost</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-purple-400">0 tuning</div>
                <div className="text-xs text-slate-500">RRF Params</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
