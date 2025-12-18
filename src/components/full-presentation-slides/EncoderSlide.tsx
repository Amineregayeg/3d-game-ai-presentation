"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";

interface EncoderSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function EncoderSlide({ slideNumber, totalSlides }: EncoderSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Encoder Design">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-4">
          <h2 className="text-4xl font-bold text-white mb-2">
            Encoder: <span className="text-cyan-400">WavLM</span> + <span className="text-purple-400">Zipformer</span> + <span className="text-amber-400">RoPE</span>
          </h2>
        </div>

        {/* Encoder Architecture Diagram */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 900 420" className="w-full max-w-5xl h-auto">
            <defs>
              <linearGradient id="wavlmEncGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="zipEncGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="ropeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#d97706" stopOpacity="0.9"/>
              </linearGradient>
              <marker id="encArrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#64748b"/>
              </marker>
            </defs>

            {/* Audio Input */}
            <g transform="translate(50, 180)">
              <rect width="100" height="60" rx="8" fill="#1e293b" stroke="#64748b" strokeWidth="1.5"/>
              <text x="50" y="25" textAnchor="middle" fill="#94a3b8" fontSize="12" fontWeight="bold">Audio</text>
              <text x="50" y="45" textAnchor="middle" fill="#64748b" fontSize="10">16kHz</text>
            </g>

            <path d="M 160 210 L 190 210" stroke="#64748b" strokeWidth="2" markerEnd="url(#encArrow)"/>

            {/* WavLM Block */}
            <g transform="translate(200, 80)">
              <rect width="180" height="260" rx="12" fill="url(#wavlmEncGrad)"/>
              <text x="90" y="35" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold">WavLM-Base</text>
              <text x="90" y="55" textAnchor="middle" fill="#e0f2fe" fontSize="11">Feature Extractor</text>

              {/* Internal blocks */}
              <rect x="15" y="70" width="150" height="35" rx="6" fill="rgba(0,0,0,0.3)"/>
              <text x="90" y="92" textAnchor="middle" fill="#67e8f9" fontSize="10">CNN Feature Encoder</text>

              <rect x="15" y="115" width="150" height="35" rx="6" fill="rgba(0,0,0,0.3)"/>
              <text x="90" y="137" textAnchor="middle" fill="#67e8f9" fontSize="10">12 Transformer Layers</text>

              <rect x="15" y="160" width="150" height="35" rx="6" fill="rgba(0,0,0,0.3)"/>
              <text x="90" y="182" textAnchor="middle" fill="#67e8f9" fontSize="10">768-dim Output</text>

              {/* Frozen indicator */}
              <rect x="15" y="210" width="150" height="35" rx="6" fill="#0f172a" stroke="#64748b" strokeWidth="1" strokeDasharray="3"/>
              <text x="90" y="232" textAnchor="middle" fill="#94a3b8" fontSize="10">🔒 Frozen (95M)</text>
            </g>

            <path d="M 390 210 L 420 210" stroke="#64748b" strokeWidth="2" markerEnd="url(#encArrow)"/>

            {/* Zipformer Block */}
            <g transform="translate(430, 50)">
              <rect width="220" height="320" rx="12" fill="url(#zipEncGrad)"/>
              <text x="110" y="35" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold">Zipformer</text>
              <text x="110" y="55" textAnchor="middle" fill="#ede9fe" fontSize="11">Encoder (6 Layers)</text>

              {/* Multi-Head Self-Attention */}
              <g transform="translate(15, 70)">
                <rect width="190" height="50" rx="6" fill="rgba(0,0,0,0.3)"/>
                <text x="95" y="20" textAnchor="middle" fill="#c4b5fd" fontSize="11" fontWeight="bold">Multi-Head Self-Attention</text>
                <text x="95" y="38" textAnchor="middle" fill="#a78bfa" fontSize="9">8 heads, 512-dim</text>
              </g>

              {/* RoPE */}
              <g transform="translate(15, 130)">
                <rect width="190" height="40" rx="6" fill="url(#ropeGrad)"/>
                <text x="95" y="25" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">RoPE Positions</text>
              </g>

              {/* Conv Module */}
              <g transform="translate(15, 180)">
                <rect width="190" height="40" rx="6" fill="rgba(0,0,0,0.3)"/>
                <text x="95" y="25" textAnchor="middle" fill="#c4b5fd" fontSize="11">Conv Module (k=31)</text>
              </g>

              {/* SwiGLU FFN */}
              <g transform="translate(15, 230)">
                <rect width="190" height="40" rx="6" fill="rgba(0,0,0,0.3)"/>
                <text x="95" y="25" textAnchor="middle" fill="#c4b5fd" fontSize="11">SwiGLU FFN (2048-dim)</text>
              </g>

              {/* Trainable indicator */}
              <rect x="15" y="280" width="190" height="28" rx="6" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="1"/>
              <text x="110" y="299" textAnchor="middle" fill="#10b981" fontSize="10">✓ Trainable (47M)</text>
            </g>

            <path d="M 660 210 L 690 210" stroke="#64748b" strokeWidth="2" markerEnd="url(#encArrow)"/>

            {/* Output */}
            <g transform="translate(700, 160)">
              <rect width="150" height="100" rx="12" fill="#1e293b" stroke="#10b981" strokeWidth="2"/>
              <text x="75" y="35" textAnchor="middle" fill="#10b981" fontSize="14" fontWeight="bold">Encoder</text>
              <text x="75" y="55" textAnchor="middle" fill="#10b981" fontSize="14" fontWeight="bold">Output</text>
              <text x="75" y="80" textAnchor="middle" fill="#6ee7b7" fontSize="11">[B, T/4, 512]</text>
            </g>

            {/* Key callouts */}
            <g transform="translate(700, 300)">
              <rect x="0" y="0" width="150" height="80" rx="8" fill="#0f172a" stroke="#64748b" strokeWidth="1" strokeDasharray="3"/>
              <text x="75" y="25" textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="bold">Key Features</text>
              <text x="75" y="45" textAnchor="middle" fill="#67e8f9" fontSize="9">• Efficient long-seq</text>
              <text x="75" y="60" textAnchor="middle" fill="#c4b5fd" fontSize="9">• Better temporal</text>
              <text x="75" y="75" textAnchor="middle" fill="#f59e0b" fontSize="9">• 4x downsampling</text>
            </g>
          </svg>
        </div>

        {/* Key Points */}
        <div className="flex justify-center gap-6 mt-2">
          {[
            { text: "Efficient long-sequence modeling", color: "cyan" },
            { text: "Better temporal representation", color: "purple" },
            { text: "Accuracy vs efficiency trade-off", color: "amber" }
          ].map((point) => (
            <div key={point.text} className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full bg-${point.color}-500`} />
              <span className="text-slate-400 text-sm">{point.text}</span>
            </div>
          ))}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
