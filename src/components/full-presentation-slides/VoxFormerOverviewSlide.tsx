"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface VoxFormerOverviewSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function VoxFormerOverviewSlide({ slideNumber, totalSlides }: VoxFormerOverviewSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="VoxFormer Architecture">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="text-4xl font-bold text-white mb-2">
            VoxFormer Overview <span className="text-slate-500">(Seq2Seq ASR)</span>
          </h2>
        </div>

        {/* Architecture Diagram */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 1000 350" className="w-full max-w-5xl h-auto">
            <defs>
              <linearGradient id="wavlmGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="zipGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="decoderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#ea580c" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="outputGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#059669" stopOpacity="0.9"/>
              </linearGradient>
              <marker id="archArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
              </marker>
              <filter id="archGlow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* Raw Audio Input */}
            <g transform="translate(30, 120)">
              <rect width="120" height="110" rx="12" fill="#1e293b" stroke="#64748b" strokeWidth="2"/>
              <text x="60" y="35" textAnchor="middle" fill="#94a3b8" fontSize="14" fontWeight="bold">Raw Audio</text>
              {/* Waveform visualization */}
              <g transform="translate(15, 50)">
                {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                  <rect
                    key={i}
                    x={i * 10}
                    y={25 - Math.sin(i * 0.8) * 20}
                    width="6"
                    height={Math.abs(Math.sin(i * 0.8) * 40) + 5}
                    rx="2"
                    fill="#64748b"
                  />
                ))}
              </g>
              <text x="60" y="100" textAnchor="middle" fill="#64748b" fontSize="10">16kHz waveform</text>
            </g>

            {/* Arrow */}
            <path d="M 160 175 L 200 175" stroke="#64748b" strokeWidth="3" markerEnd="url(#archArrow)"/>

            {/* WavLM Feature Extractor */}
            <g transform="translate(210, 100)">
              <rect width="160" height="150" rx="12" fill="url(#wavlmGrad)" filter="url(#archGlow)"/>
              <text x="80" y="35" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold">WavLM</text>
              <text x="80" y="55" textAnchor="middle" fill="#e0f2fe" fontSize="11">Feature Extractor</text>

              <rect x="15" y="70" width="130" height="30" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="80" y="90" textAnchor="middle" fill="#67e8f9" fontSize="10">Pre-trained (Frozen)</text>

              <rect x="15" y="108" width="130" height="30" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="80" y="128" textAnchor="middle" fill="#67e8f9" fontSize="10">768-dim features</text>
            </g>

            {/* Arrow */}
            <path d="M 380 175 L 420 175" stroke="#64748b" strokeWidth="3" markerEnd="url(#archArrow)"/>

            {/* Zipformer Encoder */}
            <g transform="translate(430, 80)">
              <rect width="180" height="190" rx="12" fill="url(#zipGrad)" filter="url(#archGlow)"/>
              <text x="90" y="35" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold">Zipformer</text>
              <text x="90" y="55" textAnchor="middle" fill="#ede9fe" fontSize="11">Encoder</text>

              {/* Encoder blocks */}
              {["Self-Attention", "RoPE Positions", "SwiGLU FFN", "Conv Module"].map((block, i) => (
                <g key={block} transform={`translate(15, ${70 + i * 28})`}>
                  <rect width="150" height="24" rx="4" fill="rgba(0,0,0,0.3)"/>
                  <text x="75" y="16" textAnchor="middle" fill="#c4b5fd" fontSize="9">{block}</text>
                </g>
              ))}
            </g>

            {/* Arrow */}
            <path d="M 620 175 L 660 175" stroke="#64748b" strokeWidth="3" markerEnd="url(#archArrow)"/>

            {/* Transformer Decoder */}
            <g transform="translate(670, 100)">
              <rect width="160" height="150" rx="12" fill="url(#decoderGrad)" filter="url(#archGlow)"/>
              <text x="80" y="35" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold">Transformer</text>
              <text x="80" y="55" textAnchor="middle" fill="#ffedd5" fontSize="11">Decoder</text>

              <rect x="15" y="70" width="130" height="30" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="80" y="90" textAnchor="middle" fill="#fdba74" fontSize="10">Cross-Attention</text>

              <rect x="15" y="108" width="130" height="30" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="80" y="128" textAnchor="middle" fill="#fdba74" fontSize="10">Autoregressive</text>
            </g>

            {/* Arrow */}
            <path d="M 840 175 L 880 175" stroke="#64748b" strokeWidth="3" markerEnd="url(#archArrow)"/>

            {/* Text Output */}
            <g transform="translate(890, 120)">
              <rect width="100" height="110" rx="12" fill="url(#outputGrad)" filter="url(#archGlow)"/>
              <text x="50" y="40" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">Text</text>
              <text x="50" y="60" textAnchor="middle" fill="#d1fae5" fontSize="11">Output</text>

              <rect x="10" y="75" width="80" height="25" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="50" y="92" textAnchor="middle" fill="#6ee7b7" fontSize="9">BPE Tokens</text>
            </g>

            {/* Labels below */}
            <text x="90" y="260" textAnchor="middle" fill="#64748b" fontSize="10">Input</text>
            <text x="290" y="280" textAnchor="middle" fill="#06b6d4" fontSize="10">95M params (frozen)</text>
            <text x="520" y="300" textAnchor="middle" fill="#8b5cf6" fontSize="10">47M params (trainable)</text>
            <text x="750" y="280" textAnchor="middle" fill="#f97316" fontSize="10">Autoregressive</text>
            <text x="940" y="260" textAnchor="middle" fill="#10b981" fontSize="10">Output</text>
          </svg>
        </div>

        {/* Key Points */}
        <div className="flex justify-center gap-4 mt-4">
          <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10 px-4 py-2">
            Transformer-based ASR
          </Badge>
          <Badge variant="outline" className="border-purple-500/50 text-purple-400 bg-purple-500/10 px-4 py-2">
            Designed for Low Latency
          </Badge>
          <Badge variant="outline" className="border-amber-500/50 text-amber-400 bg-amber-500/10 px-4 py-2">
            Hybrid CTC + Cross-Entropy
          </Badge>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
