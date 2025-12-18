"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface DemoResultSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function DemoResultSlide({ slideNumber, totalSlides }: DemoResultSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="System Observability">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className="text-4xl font-bold text-white mb-2">
            End-to-End Result & <span className="text-cyan-400">System Observability</span>
          </h2>
        </div>

        {/* Request/Response Flow */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 900 350" className="w-full max-w-5xl h-auto">
            <defs>
              <linearGradient id="requestGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="responseGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#f97316" stopOpacity="0.9"/>
              </linearGradient>
              <marker id="flowArrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#64748b"/>
              </marker>
            </defs>

            {/* Input Section */}
            <g transform="translate(30, 50)">
              <rect width="200" height="100" rx="12" fill="#1e293b" stroke="#06b6d4" strokeWidth="2"/>
              <text x="100" y="30" textAnchor="middle" fill="#06b6d4" fontSize="14" fontWeight="bold">Voice Input</text>
              <text x="100" y="55" textAnchor="middle" fill="#94a3b8" fontSize="11">&quot;Create a low-poly tree&quot;</text>
              <rect x="20" y="70" width="160" height="20" rx="4" fill="#06b6d4" fillOpacity="0.2"/>
              <text x="100" y="84" textAnchor="middle" fill="#67e8f9" fontSize="9">Waveform captured</text>
            </g>

            {/* Arrow to STT */}
            <path d="M 240 100 L 280 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#flowArrow)"/>

            {/* STT Processing */}
            <g transform="translate(290, 40)">
              <rect width="160" height="120" rx="10" fill="url(#requestGrad)"/>
              <text x="80" y="30" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">STT</text>
              <rect x="15" y="45" width="130" height="25" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="80" y="62" textAnchor="middle" fill="#e0f2fe" fontSize="9">Confidence: 94%</text>
              <rect x="15" y="78" width="130" height="25" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="80" y="95" textAnchor="middle" fill="#e0f2fe" fontSize="9">Latency: 1.8s</text>
            </g>

            {/* Arrow to RAG */}
            <path d="M 460 100 L 500 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#flowArrow)"/>

            {/* RAG Processing */}
            <g transform="translate(510, 30)">
              <rect width="180" height="140" rx="10" fill="#a855f7" fillOpacity="0.8"/>
              <text x="90" y="25" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">RAG Pipeline</text>

              {/* Stage indicators */}
              {["Query Analysis", "Retrieval", "Reranking", "Generation", "Validation"].map((stage, i) => (
                <g key={stage} transform={`translate(15, ${38 + i * 20})`}>
                  <circle cx="8" cy="8" r="6" fill="#10b981"/>
                  <text x="8" y="11" textAnchor="middle" fill="white" fontSize="7">✓</text>
                  <text x="22" y="12" fill="#e9d5ff" fontSize="9">{stage}</text>
                </g>
              ))}
            </g>

            {/* Arrow to Output */}
            <path d="M 700 100 L 740 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#flowArrow)"/>

            {/* Output Section */}
            <g transform="translate(750, 40)">
              <rect width="130" height="120" rx="10" fill="url(#responseGrad)"/>
              <text x="65" y="30" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">Output</text>
              <rect x="15" y="45" width="100" height="60" rx="6" fill="rgba(0,0,0,0.3)"/>
              {/* 3D preview icon */}
              <g transform="translate(35, 55)">
                <polygon points="30,5 55,20 30,35 5,20" fill="#10b981" fillOpacity="0.5" stroke="#10b981" strokeWidth="1"/>
              </g>
              <text x="65" y="110" textAnchor="middle" fill="#d1fae5" fontSize="9">3D Asset Ready</text>
            </g>

            {/* Metrics Panel */}
            <g transform="translate(30, 190)">
              <rect width="850" height="130" rx="12" fill="#0f172a" stroke="#475569" strokeWidth="1"/>
              <text x="425" y="25" textAnchor="middle" fill="#94a3b8" fontSize="12" fontWeight="bold">SYSTEM METRICS</text>

              {/* Metric boxes */}
              <g transform="translate(20, 40)">
                {/* STT Confidence */}
                <g transform="translate(0, 0)">
                  <rect width="150" height="70" rx="8" fill="#8b5cf6" fillOpacity="0.1" stroke="#8b5cf6" strokeWidth="1"/>
                  <text x="75" y="22" textAnchor="middle" fill="#a78bfa" fontSize="10">STT Confidence</text>
                  <text x="75" y="50" textAnchor="middle" fill="#8b5cf6" fontSize="22" fontWeight="bold">94%</text>
                </g>

                {/* RAG Latency */}
                <g transform="translate(170, 0)">
                  <rect width="150" height="70" rx="8" fill="#a855f7" fillOpacity="0.1" stroke="#a855f7" strokeWidth="1"/>
                  <text x="75" y="22" textAnchor="middle" fill="#c4b5fd" fontSize="10">RAG Latency</text>
                  <text x="75" y="50" textAnchor="middle" fill="#a855f7" fontSize="22" fontWeight="bold">1.8s</text>
                </g>

                {/* Relevancy Score */}
                <g transform="translate(340, 0)">
                  <rect width="150" height="70" rx="8" fill="#10b981" fillOpacity="0.1" stroke="#10b981" strokeWidth="1"/>
                  <text x="75" y="22" textAnchor="middle" fill="#6ee7b7" fontSize="10">Relevancy</text>
                  <text x="75" y="50" textAnchor="middle" fill="#10b981" fontSize="22" fontWeight="bold">0.92</text>
                </g>

                {/* Total Pipeline */}
                <g transform="translate(510, 0)">
                  <rect width="150" height="70" rx="8" fill="#f97316" fillOpacity="0.1" stroke="#f97316" strokeWidth="1"/>
                  <text x="75" y="22" textAnchor="middle" fill="#fdba74" fontSize="10">Total Pipeline</text>
                  <text x="75" y="50" textAnchor="middle" fill="#f97316" fontSize="22" fontWeight="bold">5.5s</text>
                </g>

                {/* GPU Usage */}
                <g transform="translate(680, 0)">
                  <rect width="150" height="70" rx="8" fill="#06b6d4" fillOpacity="0.1" stroke="#06b6d4" strokeWidth="1"/>
                  <text x="75" y="22" textAnchor="middle" fill="#67e8f9" fontSize="10">GPU Usage</text>
                  <text x="75" y="50" textAnchor="middle" fill="#06b6d4" fontSize="22" fontWeight="bold">78%</text>
                </g>
              </g>
            </g>
          </svg>
        </div>

        {/* Summary badge */}
        <div className="flex justify-center mt-2">
          <Badge variant="outline" className="border-emerald-500/50 text-emerald-400 bg-emerald-500/10 px-6 py-2 text-sm">
            Voice → validated instructions → real 3D asset
          </Badge>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
