"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface ArchitectureOverviewSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function ArchitectureOverviewSlide({ slideNumber, totalSlides }: ArchitectureOverviewSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="System Architecture">
      <div className="flex flex-col h-full">
        <h2 className="text-4xl font-bold text-white mb-2">
          VoxFormer <span className="text-cyan-400">Architecture</span>
        </h2>
        <p className="text-slate-400 mb-6">WavLM backbone + custom Zipformer encoder + Transformer decoder</p>

        {/* Main Architecture Diagram */}
        <div className="flex-1 relative">
          <svg viewBox="0 0 1000 500" className="w-full h-full">
            <defs>
              <linearGradient id="gradCyan" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradPurple" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#a855f7" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradEmerald" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#059669" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradPink" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ec4899" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#db2777" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradAmber" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#d97706" stopOpacity="0.8"/>
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
              </marker>
            </defs>

            {/* Audio Input */}
            <g transform="translate(20, 200)">
              <rect width="70" height="100" rx="8" fill="#1e293b" stroke="#334155" strokeWidth="2"/>
              <text x="35" y="40" textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="monospace">Raw</text>
              <text x="35" y="55" textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="monospace">Audio</text>
              <path d="M15 70 Q25 60 35 70 Q45 80 55 70" stroke="#06b6d4" fill="none" strokeWidth="2"/>
              <path d="M15 80 Q25 70 35 80 Q45 90 55 80" stroke="#06b6d4" fill="none" strokeWidth="2"/>
            </g>

            {/* Arrow 1 */}
            <path d="M95 250 L125 250" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)"/>

            {/* WavLM Feature Extractor */}
            <g transform="translate(130, 160)">
              <rect width="140" height="180" rx="8" fill="url(#gradPurple)" filter="url(#glow)"/>
              <text x="70" y="25" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">WavLM-Base</text>
              <Badge className="absolute">
                <text x="70" y="42" textAnchor="middle" fill="#e9d5ff" fontSize="8">95M params (frozen)</text>
              </Badge>
              <rect x="10" y="55" width="120" height="28" rx="4" fill="#0f172a" opacity="0.5"/>
              <text x="70" y="73" textAnchor="middle" fill="#e9d5ff" fontSize="9">12 Transformer Layers</text>
              <rect x="10" y="88" width="120" height="28" rx="4" fill="#0f172a" opacity="0.5"/>
              <text x="70" y="106" textAnchor="middle" fill="#e9d5ff" fontSize="9">Weighted Layer Sum</text>
              <rect x="10" y="121" width="120" height="28" rx="4" fill="#0f172a" opacity="0.5"/>
              <text x="70" y="139" textAnchor="middle" fill="#e9d5ff" fontSize="9">768-dim @ 50fps</text>
              <text x="70" y="165" textAnchor="middle" fill="#c4b5fd" fontSize="8">Pretrained on 94K hours</text>
            </g>

            {/* Arrow 2 */}
            <path d="M275 250 L305 250" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)"/>

            {/* Adapter Module */}
            <g transform="translate(310, 200)">
              <rect width="90" height="100" rx="8" fill="url(#gradEmerald)" filter="url(#glow)"/>
              <text x="45" y="25" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Adapter</text>
              <text x="45" y="42" textAnchor="middle" fill="#a7f3d0" fontSize="8">768 → 512</text>
              <rect x="8" y="52" width="74" height="18" rx="3" fill="#0f172a" opacity="0.5"/>
              <text x="45" y="65" textAnchor="middle" fill="#a7f3d0" fontSize="7">LayerNorm</text>
              <rect x="8" y="72" width="74" height="18" rx="3" fill="#0f172a" opacity="0.5"/>
              <text x="45" y="85" textAnchor="middle" fill="#a7f3d0" fontSize="7">Linear + GELU</text>
            </g>

            {/* Arrow 3 */}
            <path d="M405 250 L435 250" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)"/>

            {/* Zipformer Encoder Stack */}
            <g transform="translate(440, 120)">
              <rect width="180" height="260" rx="12" fill="#1e293b" stroke="#06b6d4" strokeWidth="2"/>
              <text x="90" y="25" textAnchor="middle" fill="#06b6d4" fontSize="12" fontWeight="bold">ZIPFORMER ENCODER</text>
              <text x="90" y="42" textAnchor="middle" fill="#64748b" fontSize="9">6 Blocks | 25M params</text>

              {/* Conformer Block */}
              <g transform="translate(15, 55)">
                <rect width="150" height="190" rx="8" fill="url(#gradCyan)" filter="url(#glow)"/>
                <rect x="10" y="12" width="130" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                <text x="75" y="29" textAnchor="middle" fill="#a5f3fc" fontSize="9">FFN (1/2 residual)</text>
                <rect x="10" y="42" width="130" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                <text x="75" y="59" textAnchor="middle" fill="#a5f3fc" fontSize="9">Multi-Head Attn (8 heads)</text>
                <rect x="10" y="72" width="130" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                <text x="75" y="89" textAnchor="middle" fill="#a5f3fc" fontSize="9">Depthwise Conv (k=31)</text>
                <rect x="10" y="102" width="130" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                <text x="75" y="119" textAnchor="middle" fill="#a5f3fc" fontSize="9">FFN (1/2 residual)</text>
                <rect x="10" y="132" width="130" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                <text x="75" y="149" textAnchor="middle" fill="#a5f3fc" fontSize="9">LayerNorm</text>
                <text x="75" y="175" textAnchor="middle" fill="#67e8f9" fontSize="8">U-Net: 50→25→12.5 fps</text>
              </g>
            </g>

            {/* CTC Branch */}
            <g transform="translate(470, 400)">
              <rect width="120" height="50" rx="8" fill="#374151" stroke="#f59e0b" strokeWidth="1"/>
              <text x="60" y="22" textAnchor="middle" fill="#fbbf24" fontSize="10" fontWeight="bold">CTC Head</text>
              <text x="60" y="38" textAnchor="middle" fill="#9ca3af" fontSize="8">0.3 weight (aux)</text>
            </g>
            <path d="M530 380 L530 398" stroke="#f59e0b" strokeWidth="2" strokeDasharray="4" markerEnd="url(#arrowhead)"/>

            {/* Arrow 4 */}
            <path d="M625 250 L655 250" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)"/>

            {/* Transformer Decoder */}
            <g transform="translate(660, 140)">
              <rect width="160" height="220" rx="12" fill="#1e293b" stroke="#ec4899" strokeWidth="2"/>
              <text x="80" y="25" textAnchor="middle" fill="#ec4899" fontSize="12" fontWeight="bold">TRANSFORMER DECODER</text>
              <text x="80" y="42" textAnchor="middle" fill="#64748b" fontSize="9">4 Layers | 20M params</text>

              <g transform="translate(15, 55)">
                <rect width="130" height="150" rx="8" fill="url(#gradPink)" filter="url(#glow)"/>
                <rect x="10" y="12" width="110" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                <text x="65" y="29" textAnchor="middle" fill="#fce7f3" fontSize="8">Masked Self-Attention</text>
                <rect x="10" y="42" width="110" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                <text x="65" y="59" textAnchor="middle" fill="#fce7f3" fontSize="8">Cross-Attention</text>
                <rect x="10" y="72" width="110" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                <text x="65" y="89" textAnchor="middle" fill="#fce7f3" fontSize="8">Feed-Forward (2048)</text>
                <rect x="10" y="102" width="110" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
                <text x="65" y="119" textAnchor="middle" fill="#fce7f3" fontSize="8">LayerNorm + KV-Cache</text>
              </g>
              <text x="80" y="215" textAnchor="middle" fill="#f9a8d4" fontSize="8">BPE vocab: 2000 tokens</text>
            </g>

            {/* Arrow 5 */}
            <path d="M825 250 L855 250" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)"/>

            {/* Output */}
            <g transform="translate(860, 190)">
              <rect width="90" height="120" rx="8" fill="#1e293b" stroke="#10b981" strokeWidth="2"/>
              <text x="45" y="30" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">Output</text>
              <text x="45" y="50" textAnchor="middle" fill="#94a3b8" fontSize="9">CE Loss</text>
              <text x="45" y="65" textAnchor="middle" fill="#94a3b8" fontSize="8">0.7 weight</text>
              <text x="45" y="90" textAnchor="middle" fill="#10b981" fontSize="24">T</text>
              <text x="45" y="110" textAnchor="middle" fill="#6ee7b7" fontSize="8">Text Output</text>
            </g>

            {/* Token embedding input to decoder */}
            <g transform="translate(700, 385)">
              <rect width="80" height="40" rx="4" fill="#374151"/>
              <text x="40" y="18" textAnchor="middle" fill="#9ca3af" fontSize="8">Token Embed</text>
              <text x="40" y="32" textAnchor="middle" fill="#64748b" fontSize="7">+ Pos Encoding</text>
            </g>
            <path d="M740 375 L740 362" stroke="#64748b" strokeWidth="1" strokeDasharray="3"/>
          </svg>
        </div>

        {/* Model Specs */}
        <div className="flex justify-center gap-4 mt-4">
          {[
            { name: "WavLM", params: "95M", desc: "Frozen backbone", color: "purple" },
            { name: "Adapter", params: "2M", desc: "768→512 projection", color: "emerald" },
            { name: "Zipformer", params: "25M", desc: "6 Conformer blocks", color: "cyan", active: true },
            { name: "Decoder", params: "20M", desc: "4 Transformer layers", color: "pink" }
          ].map((config) => (
            <Card key={config.name} className={`bg-slate-800/50 border-slate-700/50 ${config.active ? 'ring-2 ring-cyan-500/50' : ''}`}>
              <CardContent className="p-3 text-center">
                <div className={`text-sm font-bold text-${config.color}-400`}>{config.name}</div>
                <div className="text-xs text-white">{config.params}</div>
                <div className="text-xs text-slate-500">{config.desc}</div>
              </CardContent>
            </Card>
          ))}
          <Card className="bg-slate-800/50 border-slate-700/50 border-amber-500/30">
            <CardContent className="p-3 text-center">
              <div className="text-sm font-bold text-amber-400">Total</div>
              <div className="text-xs text-white">142M</div>
              <div className="text-xs text-slate-500">47M trainable</div>
            </CardContent>
          </Card>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
