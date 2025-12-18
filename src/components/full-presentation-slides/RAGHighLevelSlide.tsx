"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface RAGHighLevelSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function RAGHighLevelSlide({ slideNumber, totalSlides }: RAGHighLevelSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="RAG Pipeline">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className="text-4xl font-bold text-white mb-2">
            RAG: Grounding Responses in <span className="text-purple-400">Blender Documentation</span>
          </h2>
        </div>

        {/* 8-Stage Pipeline Visualization */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 1000 320" className="w-full max-w-5xl h-auto">
            <defs>
              <linearGradient id="ragStage1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="ragStage2" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="ragStage3" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#d97706" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="ragStage4" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#059669" stopOpacity="0.9"/>
              </linearGradient>
              <marker id="ragArrow" markerWidth="6" markerHeight="5" refX="5" refY="2.5" orient="auto">
                <polygon points="0 0, 6 2.5, 0 5" fill="#64748b"/>
              </marker>
            </defs>

            {/* Stage boxes - Row 1 */}
            {[
              { num: 1, name: "Orchestrator", color: "url(#ragStage1)", x: 30 },
              { num: 2, name: "Query Analysis", color: "url(#ragStage1)", x: 150 },
              { num: 3, name: "Dense Search", color: "url(#ragStage2)", x: 270 },
              { num: 4, name: "Sparse Search", color: "url(#ragStage2)", x: 390 }
            ].map((stage) => (
              <g key={stage.num} transform={`translate(${stage.x}, 50)`}>
                <rect width="110" height="100" rx="10" fill={stage.color}/>
                <circle cx="55" cy="25" r="14" fill="rgba(255,255,255,0.2)"/>
                <text x="55" y="30" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">{stage.num}</text>
                <text x="55" y="60" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">{stage.name.split(" ")[0]}</text>
                {stage.name.split(" ")[1] && (
                  <text x="55" y="78" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="10">{stage.name.split(" ")[1]}</text>
                )}
              </g>
            ))}

            {/* Arrows Row 1 */}
            <path d="M 145 100 L 150 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#ragArrow)"/>
            <path d="M 265 100 L 270 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#ragArrow)"/>
            <path d="M 385 100 L 390 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#ragArrow)"/>

            {/* Curve down */}
            <path d="M 500 100 Q 550 100 550 160" stroke="#64748b" strokeWidth="2" fill="none"/>
            <path d="M 550 160 Q 550 220 500 220" stroke="#64748b" strokeWidth="2" fill="none" markerEnd="url(#ragArrow)"/>

            {/* Stage boxes - Row 2 */}
            {[
              { num: 5, name: "RRF Fusion", color: "url(#ragStage3)", x: 390 },
              { num: 6, name: "Reranking", color: "url(#ragStage3)", x: 270 },
              { num: 7, name: "Generation", color: "url(#ragStage4)", x: 150 },
              { num: 8, name: "Validation", color: "url(#ragStage4)", x: 30 }
            ].map((stage) => (
              <g key={stage.num} transform={`translate(${stage.x}, 170)`}>
                <rect width="110" height="100" rx="10" fill={stage.color}/>
                <circle cx="55" cy="25" r="14" fill="rgba(255,255,255,0.2)"/>
                <text x="55" y="30" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">{stage.num}</text>
                <text x="55" y="60" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">{stage.name.split(" ")[0]}</text>
                {stage.name.split(" ")[1] && (
                  <text x="55" y="78" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="10">{stage.name.split(" ")[1]}</text>
                )}
              </g>
            ))}

            {/* Arrows Row 2 (reversed) */}
            <path d="M 385 220 L 380 220" stroke="#64748b" strokeWidth="2" markerEnd="url(#ragArrow)"/>
            <path d="M 265 220 L 260 220" stroke="#64748b" strokeWidth="2" markerEnd="url(#ragArrow)"/>
            <path d="M 145 220 L 140 220" stroke="#64748b" strokeWidth="2" markerEnd="url(#ragArrow)"/>

            {/* Right side: Hybrid Retrieval detail */}
            <g transform="translate(580, 50)">
              <rect width="380" height="220" rx="12" fill="#1e293b" stroke="#a855f7" strokeWidth="2"/>
              <text x="190" y="30" textAnchor="middle" fill="#a855f7" fontSize="14" fontWeight="bold">Hybrid Retrieval</text>

              {/* Dense */}
              <g transform="translate(20, 50)">
                <rect width="160" height="70" rx="8" fill="#8b5cf6" fillOpacity="0.2" stroke="#8b5cf6" strokeWidth="1"/>
                <text x="80" y="25" textAnchor="middle" fill="#a78bfa" fontSize="11" fontWeight="bold">Dense Search</text>
                <text x="80" y="45" textAnchor="middle" fill="#94a3b8" fontSize="9">BGE-M3 embeddings</text>
                <text x="80" y="60" textAnchor="middle" fill="#64748b" fontSize="9">4096-dim vectors</text>
              </g>

              {/* Sparse */}
              <g transform="translate(200, 50)">
                <rect width="160" height="70" rx="8" fill="#8b5cf6" fillOpacity="0.2" stroke="#8b5cf6" strokeWidth="1"/>
                <text x="80" y="25" textAnchor="middle" fill="#a78bfa" fontSize="11" fontWeight="bold">Sparse Search</text>
                <text x="80" y="45" textAnchor="middle" fill="#94a3b8" fontSize="9">BM25 keywords</text>
                <text x="80" y="60" textAnchor="middle" fill="#64748b" fontSize="9">TF-IDF scoring</text>
              </g>

              {/* Validation */}
              <g transform="translate(20, 140)">
                <rect width="340" height="60" rx="8" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="1"/>
                <text x="170" y="25" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">RAGAS Validation Layer</text>
                <text x="170" y="45" textAnchor="middle" fill="#94a3b8" fontSize="9">Faithfulness &gt;0.85 | Relevancy &gt;0.90 | Context Precision &gt;0.88</text>
              </g>
            </g>
          </svg>
        </div>

        {/* Key Points */}
        <div className="flex justify-center gap-4 mt-4">
          <Badge variant="outline" className="border-purple-500/50 text-purple-400 bg-purple-500/10 px-4 py-2">
            Prevents hallucination
          </Badge>
          <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10 px-4 py-2">
            Uses existing models
          </Badge>
          <Badge variant="outline" className="border-slate-500/50 text-slate-400 bg-slate-500/10 px-4 py-2">
            Not core DL contribution
          </Badge>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
