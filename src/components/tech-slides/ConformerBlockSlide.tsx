"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface ConformerBlockSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function ConformerBlockSlide({ slideNumber, totalSlides }: ConformerBlockSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Conformer Architecture">
      <div className="flex flex-col h-full">
        <h2 className="text-4xl font-bold text-white mb-2">
          Conformer <span className="text-emerald-400">Encoder Block</span>
        </h2>
        <p className="text-slate-400 mb-6">Combining convolution and self-attention for speech recognition</p>

        <div className="flex-1 grid grid-cols-3 gap-6">
          {/* Conformer Block Diagram */}
          <Card className="col-span-2 bg-slate-800/30 border-slate-700/50">
            <CardContent className="p-6">
              <svg viewBox="0 0 600 450" className="w-full h-full">
                <defs>
                  <linearGradient id="ffnGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#f97316" stopOpacity="0.8"/>
                    <stop offset="100%" stopColor="#ea580c" stopOpacity="0.8"/>
                  </linearGradient>
                  <linearGradient id="attnGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.8"/>
                    <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.8"/>
                  </linearGradient>
                  <linearGradient id="convGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8"/>
                    <stop offset="100%" stopColor="#0891b2" stopOpacity="0.8"/>
                  </linearGradient>
                  <linearGradient id="normGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#10b981" stopOpacity="0.8"/>
                    <stop offset="100%" stopColor="#059669" stopOpacity="0.8"/>
                  </linearGradient>
                </defs>

                {/* Input */}
                <rect x="250" y="10" width="100" height="35" rx="6" fill="#1e293b" stroke="#334155" strokeWidth="2"/>
                <text x="300" y="33" textAnchor="middle" fill="#94a3b8" fontSize="12">Input X</text>

                {/* Arrow down */}
                <path d="M300 45 L300 65" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>

                {/* FFN 1/2 */}
                <g transform="translate(150, 70)">
                  <rect width="300" height="50" rx="8" fill="url(#ffnGrad)"/>
                  <text x="150" y="30" textAnchor="middle" fill="white" fontSize="13" fontWeight="bold">Feed-Forward Module (1/2)</text>
                  <text x="150" y="45" textAnchor="middle" fill="#fed7aa" fontSize="9">SwiGLU + Residual × 0.5</text>

                  {/* Residual connection */}
                  <path d="M-30 25 L-10 25" stroke="#f97316" strokeWidth="2" strokeDasharray="4"/>
                  <circle cx="-30" cy="25" r="4" fill="#f97316"/>
                </g>

                {/* Arrow */}
                <path d="M300 125 L300 145" stroke="#64748b" strokeWidth="2"/>

                {/* Multi-Head Attention */}
                <g transform="translate(150, 150)">
                  <rect width="300" height="60" rx="8" fill="url(#attnGrad)"/>
                  <text x="150" y="25" textAnchor="middle" fill="white" fontSize="13" fontWeight="bold">Multi-Head Self-Attention</text>
                  <text x="150" y="42" textAnchor="middle" fill="#ddd6fe" fontSize="9">8 heads × 64 dim + RoPE</text>
                  <text x="150" y="55" textAnchor="middle" fill="#c4b5fd" fontSize="8">+ Dropout + Residual</text>

                  {/* Residual */}
                  <path d="M-30 30 L-10 30" stroke="#8b5cf6" strokeWidth="2" strokeDasharray="4"/>
                  <circle cx="-30" cy="30" r="4" fill="#8b5cf6"/>
                </g>

                {/* Arrow */}
                <path d="M300 215 L300 235" stroke="#64748b" strokeWidth="2"/>

                {/* Convolution Module */}
                <g transform="translate(150, 240)">
                  <rect width="300" height="80" rx="8" fill="url(#convGrad)"/>
                  <text x="150" y="22" textAnchor="middle" fill="white" fontSize="13" fontWeight="bold">Convolution Module</text>

                  {/* Sub-components */}
                  <g transform="translate(15, 32)">
                    <rect width="55" height="20" rx="3" fill="#0f172a" opacity="0.5"/>
                    <text x="27" y="14" textAnchor="middle" fill="#a5f3fc" fontSize="7">Pointwise</text>
                  </g>
                  <g transform="translate(75, 32)">
                    <rect width="40" height="20" rx="3" fill="#0f172a" opacity="0.5"/>
                    <text x="20" y="14" textAnchor="middle" fill="#a5f3fc" fontSize="7">GLU</text>
                  </g>
                  <g transform="translate(120, 32)">
                    <rect width="60" height="20" rx="3" fill="#0f172a" opacity="0.5"/>
                    <text x="30" y="14" textAnchor="middle" fill="#a5f3fc" fontSize="7">Depthwise</text>
                  </g>
                  <g transform="translate(185, 32)">
                    <rect width="45" height="20" rx="3" fill="#0f172a" opacity="0.5"/>
                    <text x="22" y="14" textAnchor="middle" fill="#a5f3fc" fontSize="7">BN+SiLU</text>
                  </g>
                  <g transform="translate(235, 32)">
                    <rect width="50" height="20" rx="3" fill="#0f172a" opacity="0.5"/>
                    <text x="25" y="14" textAnchor="middle" fill="#a5f3fc" fontSize="7">Pointwise</text>
                  </g>

                  <text x="150" y="70" textAnchor="middle" fill="#67e8f9" fontSize="9">Kernel Size: 31 | Groups: d_model</text>

                  {/* Residual */}
                  <path d="M-30 40 L-10 40" stroke="#06b6d4" strokeWidth="2" strokeDasharray="4"/>
                  <circle cx="-30" cy="40" r="4" fill="#06b6d4"/>
                </g>

                {/* Arrow */}
                <path d="M300 325 L300 345" stroke="#64748b" strokeWidth="2"/>

                {/* FFN 2/2 */}
                <g transform="translate(150, 350)">
                  <rect width="300" height="50" rx="8" fill="url(#ffnGrad)"/>
                  <text x="150" y="30" textAnchor="middle" fill="white" fontSize="13" fontWeight="bold">Feed-Forward Module (1/2)</text>
                  <text x="150" y="45" textAnchor="middle" fill="#fed7aa" fontSize="9">SwiGLU + Residual × 0.5</text>

                  {/* Residual */}
                  <path d="M-30 25 L-10 25" stroke="#f97316" strokeWidth="2" strokeDasharray="4"/>
                  <circle cx="-30" cy="25" r="4" fill="#f97316"/>
                </g>

                {/* Arrow */}
                <path d="M300 405 L300 420" stroke="#64748b" strokeWidth="2"/>

                {/* Layer Norm */}
                <rect x="200" y="425" width="200" height="25" rx="6" fill="url(#normGrad)"/>
                <text x="300" y="442" textAnchor="middle" fill="white" fontSize="11">RMS Layer Norm</text>

                {/* Residual connection line (full) */}
                <path d="M100 27 L100 437 L195 437" stroke="#475569" strokeWidth="1.5" strokeDasharray="4" fill="none"/>
                <circle cx="100" cy="27" r="4" fill="#475569"/>

                <defs>
                  <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
                  </marker>
                </defs>
              </svg>
            </CardContent>
          </Card>

          {/* Key Features */}
          <div className="space-y-4">
            <Card className="bg-slate-800/30 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <Badge className="bg-purple-500/20 text-purple-400">Key Insight</Badge>
                </h3>
                <p className="text-sm text-slate-400">
                  Conformer combines the <span className="text-purple-400">global context</span> of self-attention
                  with <span className="text-cyan-400">local patterns</span> from convolution, achieving
                  state-of-the-art speech recognition.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-slate-800/30 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-sm font-semibold text-slate-400 mb-3">Module Parameters</h3>
                <div className="space-y-2">
                  {[
                    { label: "Model Dim (d)", value: "512", color: "text-cyan-400" },
                    { label: "FFN Dim", value: "2048", color: "text-orange-400" },
                    { label: "Attention Heads", value: "8", color: "text-purple-400" },
                    { label: "Conv Kernel", value: "31", color: "text-cyan-400" },
                    { label: "Dropout", value: "0.1", color: "text-slate-400" }
                  ].map((item) => (
                    <div key={item.label} className="flex justify-between items-center p-2 bg-slate-900/30 rounded">
                      <span className="text-xs text-slate-400">{item.label}</span>
                      <span className={`text-sm font-mono font-bold ${item.color}`}>{item.value}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800/30 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-sm font-semibold text-slate-400 mb-3">Macaron-Style FFN</h3>
                <div className="text-xs text-slate-500 space-y-1">
                  <div>• Two half-strength FFN modules</div>
                  <div>• Better gradient flow</div>
                  <div>• Residual weight: 0.5×</div>
                  <div>• SwiGLU activation</div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
