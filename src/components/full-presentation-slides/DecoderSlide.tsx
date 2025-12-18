"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";

interface DecoderSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function DecoderSlide({ slideNumber, totalSlides }: DecoderSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Decoder & Attention">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className="text-4xl font-bold text-white mb-2">
            Decoder & <span className="text-purple-400">Attention</span> Mechanism
          </h2>
        </div>

        {/* Decoder Architecture */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 900 400" className="w-full max-w-5xl h-auto">
            <defs>
              <linearGradient id="encOutGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.1"/>
              </linearGradient>
              <linearGradient id="decGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#ea580c" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="crossAttnGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#ec4899" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#db2777" stopOpacity="0.9"/>
              </linearGradient>
              <marker id="decArrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#64748b"/>
              </marker>
              <marker id="attnArrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#ec4899"/>
              </marker>
            </defs>

            {/* Encoder Output (left side) */}
            <g transform="translate(50, 80)">
              <rect width="180" height="240" rx="12" fill="url(#encOutGrad)" stroke="#8b5cf6" strokeWidth="2" strokeDasharray="4"/>
              <text x="90" y="35" textAnchor="middle" fill="#8b5cf6" fontSize="14" fontWeight="bold">Encoder Output</text>
              <text x="90" y="55" textAnchor="middle" fill="#a78bfa" fontSize="11">[B, T, 512]</text>

              {/* Audio frames representation */}
              <g transform="translate(20, 80)">
                {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
                  <rect
                    key={i}
                    x={0}
                    y={i * 18}
                    width="140"
                    height="14"
                    rx="2"
                    fill="#8b5cf6"
                    fillOpacity={0.3 - i * 0.02}
                  />
                ))}
              </g>
              <text x="90" y="230" textAnchor="middle" fill="#64748b" fontSize="10">Audio frames (T)</text>
            </g>

            {/* Cross-Attention visualization */}
            <g transform="translate(250, 120)">
              {/* Attention lines */}
              {[0, 1, 2, 3, 4].map((i) => (
                <path
                  key={i}
                  d={`M 0 ${30 + i * 40} Q 80 ${30 + i * 40 + (i - 2) * 30} 160 ${100 + i * 15}`}
                  stroke="#ec4899"
                  strokeWidth="1.5"
                  strokeOpacity={0.4 + i * 0.1}
                  fill="none"
                  markerEnd="url(#attnArrow)"
                />
              ))}

              {/* Attention label */}
              <rect x="50" y="70" width="60" height="60" rx="30" fill="url(#crossAttnGrad)" />
              <text x="80" y="95" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Cross</text>
              <text x="80" y="110" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Attn</text>
            </g>

            {/* Decoder Block */}
            <g transform="translate(430, 50)">
              <rect width="220" height="300" rx="12" fill="url(#decGrad)"/>
              <text x="110" y="35" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold">Transformer Decoder</text>
              <text x="110" y="55" textAnchor="middle" fill="#ffedd5" fontSize="11">4 Layers</text>

              {/* Masked Self-Attention */}
              <g transform="translate(15, 70)">
                <rect width="190" height="45" rx="6" fill="rgba(0,0,0,0.3)"/>
                <text x="95" y="20" textAnchor="middle" fill="#fdba74" fontSize="11" fontWeight="bold">Masked Self-Attention</text>
                <text x="95" y="36" textAnchor="middle" fill="#fbbf24" fontSize="9">Causal mask (no future)</text>
              </g>

              {/* Cross-Attention */}
              <g transform="translate(15, 125)">
                <rect width="190" height="45" rx="6" fill="url(#crossAttnGrad)"/>
                <text x="95" y="20" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Cross-Attention</text>
                <text x="95" y="36" textAnchor="middle" fill="#fce7f3" fontSize="9">Attends to encoder output</text>
              </g>

              {/* FFN */}
              <g transform="translate(15, 180)">
                <rect width="190" height="45" rx="6" fill="rgba(0,0,0,0.3)"/>
                <text x="95" y="20" textAnchor="middle" fill="#fdba74" fontSize="11" fontWeight="bold">Feed-Forward</text>
                <text x="95" y="36" textAnchor="middle" fill="#fbbf24" fontSize="9">SwiGLU (2048-dim)</text>
              </g>

              {/* Output projection */}
              <g transform="translate(15, 235)">
                <rect width="190" height="45" rx="6" fill="rgba(0,0,0,0.3)"/>
                <text x="95" y="20" textAnchor="middle" fill="#fdba74" fontSize="11" fontWeight="bold">Output Projection</text>
                <text x="95" y="36" textAnchor="middle" fill="#fbbf24" fontSize="9">Linear → Vocab (2000)</text>
              </g>
            </g>

            <path d="M 660 200 L 690 200" stroke="#64748b" strokeWidth="2" markerEnd="url(#decArrow)"/>

            {/* Text Output */}
            <g transform="translate(700, 110)">
              <rect width="160" height="180" rx="12" fill="#1e293b" stroke="#10b981" strokeWidth="2"/>
              <text x="80" y="35" textAnchor="middle" fill="#10b981" fontSize="14" fontWeight="bold">Text Output</text>
              <text x="80" y="55" textAnchor="middle" fill="#6ee7b7" fontSize="11">BPE Tokens</text>

              {/* Token sequence */}
              <g transform="translate(15, 70)">
                {["<sos>", "create", "a", "cube", "<eos>"].map((tok, i) => (
                  <g key={tok} transform={`translate(0, ${i * 22})`}>
                    <rect width="130" height="18" rx="3" fill="#10b981" fillOpacity={0.2}/>
                    <text x="65" y="13" textAnchor="middle" fill="#6ee7b7" fontSize="10">{tok}</text>
                  </g>
                ))}
              </g>
            </g>

            {/* Autoregressive arrow */}
            <g transform="translate(700, 300)">
              <path d="M 80 0 L 80 30 L 0 30 L 0 -170 L -40 -170" stroke="#64748b" strokeWidth="1.5" fill="none" strokeDasharray="4"/>
              <text x="40" y="50" textAnchor="middle" fill="#64748b" fontSize="9">Autoregressive</text>
            </g>
          </svg>
        </div>

        {/* Key Points */}
        <div className="grid grid-cols-3 gap-6 max-w-3xl mx-auto mt-2">
          {[
            { text: "Autoregressive token generation", color: "orange" },
            { text: "Attention aligns audio → text", color: "pink" },
            { text: "Supports variable-length sequences", color: "emerald" }
          ].map((point) => (
            <div key={point.text} className={`p-3 bg-${point.color}-500/10 border border-${point.color}-500/30 rounded-lg text-center`}>
              <span className="text-slate-300 text-sm">{point.text}</span>
            </div>
          ))}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
