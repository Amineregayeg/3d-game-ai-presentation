"use client";

import { TechSlideWrapper } from "../tech-slides/TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface RerankingSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function RerankingSlide({ slideNumber, totalSlides }: RerankingSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Reranking & Context Assembly">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Cross-Encoder <span className="text-amber-400">Reranking</span>
            </h2>
            <p className="text-slate-400">BGE cross-encoder for precise relevance scoring and context assembly</p>
          </div>
          <Badge variant="outline" className="border-amber-500/50 text-amber-400 bg-amber-500/10">
            BGE-reranker-v2-m3
          </Badge>
        </div>

        <div className="grid grid-cols-2 gap-6 flex-1">
          {/* Left: Cross-Encoder vs Bi-Encoder */}
          <div className="space-y-4">
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-3">Bi-Encoder vs Cross-Encoder</h3>
                <svg viewBox="0 0 380 220" className="w-full">
                  <defs>
                    <linearGradient id="amberGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#f59e0b"/>
                      <stop offset="100%" stopColor="#d97706"/>
                    </linearGradient>
                  </defs>

                  {/* Bi-Encoder (Fast, less accurate) */}
                  <g transform="translate(10, 10)">
                    <text x="85" y="15" textAnchor="middle" fill="#64748b" fontSize="11" fontWeight="bold">Bi-Encoder (Dense)</text>

                    {/* Query path */}
                    <rect x="10" y="30" width="60" height="30" rx="4" fill="#1e293b" stroke="#06b6d4" strokeWidth="1"/>
                    <text x="40" y="49" textAnchor="middle" fill="#06b6d4" fontSize="9">Query</text>

                    <rect x="10" y="70" width="60" height="35" rx="4" fill="#06b6d4" fillOpacity="0.2" stroke="#06b6d4"/>
                    <text x="40" y="92" textAnchor="middle" fill="#06b6d4" fontSize="8">Encoder</text>

                    <rect x="10" y="115" width="60" height="25" rx="4" fill="#1e293b" stroke="#06b6d4" strokeWidth="1"/>
                    <text x="40" y="131" textAnchor="middle" fill="#67e8f9" fontSize="8">Q_vec</text>

                    {/* Document path */}
                    <rect x="100" y="30" width="60" height="30" rx="4" fill="#1e293b" stroke="#3b82f6" strokeWidth="1"/>
                    <text x="130" y="49" textAnchor="middle" fill="#3b82f6" fontSize="9">Doc</text>

                    <rect x="100" y="70" width="60" height="35" rx="4" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6"/>
                    <text x="130" y="92" textAnchor="middle" fill="#3b82f6" fontSize="8">Encoder</text>

                    <rect x="100" y="115" width="60" height="25" rx="4" fill="#1e293b" stroke="#3b82f6" strokeWidth="1"/>
                    <text x="130" y="131" textAnchor="middle" fill="#93c5fd" fontSize="8">D_vec</text>

                    {/* Cosine similarity */}
                    <path d="M70 128 L100 128" stroke="#64748b" strokeWidth="1" strokeDasharray="3"/>
                    <circle cx="85" cy="155" r="15" fill="#475569"/>
                    <text x="85" y="159" textAnchor="middle" fill="white" fontSize="8">cos</text>

                    {/* Lines to cosine */}
                    <path d="M40 140 L75 155" stroke="#64748b" strokeWidth="1"/>
                    <path d="M130 140 L95 155" stroke="#64748b" strokeWidth="1"/>

                    {/* Speed badge */}
                    <g transform="translate(55, 175)">
                      <rect width="60" height="20" rx="10" fill="#10b981" fillOpacity="0.2" stroke="#10b981"/>
                      <text x="30" y="14" textAnchor="middle" fill="#10b981" fontSize="8">Fast (5ms)</text>
                    </g>
                  </g>

                  {/* Cross-Encoder (Slower, more accurate) */}
                  <g transform="translate(200, 10)">
                    <text x="85" y="15" textAnchor="middle" fill="#f59e0b" fontSize="11" fontWeight="bold">Cross-Encoder</text>

                    {/* Combined input */}
                    <rect x="10" y="30" width="150" height="35" rx="4" fill="#1e293b" stroke="#f59e0b" strokeWidth="1.5"/>
                    <text x="85" y="45" textAnchor="middle" fill="#fbbf24" fontSize="8">[CLS] Query [SEP] Document</text>
                    <text x="85" y="58" textAnchor="middle" fill="#64748b" fontSize="7">Concatenated Input</text>

                    {/* Transformer */}
                    <rect x="30" y="75" width="110" height="50" rx="6" fill="url(#amberGrad)" fillOpacity="0.3" stroke="#f59e0b" strokeWidth="2"/>
                    <text x="85" y="97" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="bold">Transformer</text>
                    <text x="85" y="112" textAnchor="middle" fill="#fef3c7" fontSize="8">Full Attention</text>

                    {/* Direct score */}
                    <rect x="45" y="135" width="80" height="30" rx="4" fill="#1e293b" stroke="#10b981" strokeWidth="1.5"/>
                    <text x="85" y="154" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="bold">0.92</text>

                    {/* Accuracy badge */}
                    <g transform="translate(55, 175)">
                      <rect width="60" height="20" rx="10" fill="#f59e0b" fillOpacity="0.2" stroke="#f59e0b"/>
                      <text x="30" y="14" textAnchor="middle" fill="#f59e0b" fontSize="8">Accurate</text>
                    </g>
                  </g>

                  {/* VS divider */}
                  <line x1="190" y1="30" x2="190" y2="190" stroke="#334155" strokeWidth="1" strokeDasharray="4"/>
                  <circle cx="190" cy="110" r="12" fill="#1e293b" stroke="#334155"/>
                  <text x="190" y="114" textAnchor="middle" fill="#64748b" fontSize="9">VS</text>
                </svg>
              </CardContent>
            </Card>

            {/* Reranking Process */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-sm font-semibold text-white mb-3">Reranking Pipeline</h3>
                <div className="space-y-2">
                  {[
                    { step: "1", text: "Receive 50 candidates from RRF fusion", color: "purple" },
                    { step: "2", text: "Create [query, doc] pairs for each candidate", color: "cyan" },
                    { step: "3", text: "Batch process through MiniLM (64 pairs/batch)", color: "amber" },
                    { step: "4", text: "Sort by relevance score (0-1)", color: "emerald" },
                    { step: "5", text: "Return top-10 for context assembly", color: "blue" }
                  ].map((item) => (
                    <div key={item.step} className="flex items-center gap-3">
                      <div className={`w-6 h-6 rounded-full bg-${item.color}-500/20 border border-${item.color}-500/50 flex items-center justify-center text-${item.color}-400 text-xs font-bold`}>
                        {item.step}
                      </div>
                      <span className="text-sm text-slate-300">{item.text}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right: Context Assembly */}
          <div className="space-y-4">
            <Card className="bg-slate-800/50 border-slate-700/50 flex-1">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-3">Context Assembly</h3>
                <div className="bg-slate-900/50 rounded-lg p-3 font-mono text-xs space-y-2 max-h-[220px] overflow-y-auto">
                  <div className="text-emerald-400">=== RETRIEVED CONTEXT ===</div>
                  <div className="text-slate-500">Query: &quot;select all faces in Blender&quot;</div>
                  <div className="text-slate-600">Sources: 10 documents</div>
                  <div className="text-slate-600">========================================</div>

                  <div className="border-l-2 border-cyan-500 pl-2 mt-2">
                    <div className="text-cyan-400">[Source 1 | Relevance: 95%]</div>
                    <div className="text-slate-300 text-xs">bpy.ops.mesh.select_all(action=&apos;SELECT&apos;)</div>
                    <div className="text-slate-400 text-xs">Selects all mesh elements in edit mode...</div>
                  </div>

                  <div className="border-l-2 border-blue-500 pl-2">
                    <div className="text-blue-400">[Source 2 | Relevance: 89%]</div>
                    <div className="text-slate-300 text-xs">To select faces, enter Edit Mode (Tab)...</div>
                  </div>

                  <div className="border-l-2 border-purple-500 pl-2">
                    <div className="text-purple-400">[Source 3 | Relevance: 82%]</div>
                    <div className="text-slate-300 text-xs">Face selection mode: Press &apos;3&apos; in edit mode...</div>
                  </div>

                  <div className="text-slate-600">... 7 more sources</div>
                  <div className="text-slate-600">========================================</div>
                  <div className="text-amber-400 text-xs">Respond based only on the above context.</div>
                </div>
              </CardContent>
            </Card>

            {/* Token Budget */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-sm font-semibold text-white mb-2">Token Budget Management</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-400">Max Context</span>
                    <span className="text-sm font-mono text-cyan-400">2,000 tokens</span>
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-2">
                    <div className="bg-gradient-to-r from-cyan-500 to-emerald-500 h-2 rounded-full" style={{ width: '75%' }}></div>
                  </div>
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>Per doc: 400 max</span>
                    <span>Used: 1,500</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Performance Metrics */}
            <div className="grid grid-cols-3 gap-2">
              {[
                { value: "15ms", label: "Per Pair", color: "text-amber-400" },
                { value: "~750ms", label: "50 Docs", color: "text-cyan-400" },
                { value: "0.89", label: "Accuracy", color: "text-emerald-400" }
              ].map((metric) => (
                <Card key={metric.label} className="bg-slate-800/50 border-slate-700/50">
                  <CardContent className="p-2 text-center">
                    <div className={`text-lg font-bold ${metric.color}`}>{metric.value}</div>
                    <div className="text-xs text-slate-500">{metric.label}</div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom: BGE Reranker Specs */}
        <div className="mt-3 p-3 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div>
                <span className="text-xs text-slate-500 uppercase tracking-wider">Model</span>
                <div className="text-sm text-slate-300 font-mono">BAAI/bge-reranker-v2-m3</div>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <div className="text-center">
                <div className="text-sm font-bold text-amber-400">568M params</div>
                <div className="text-xs text-slate-500">Model Size</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold text-cyan-400">Multi-lingual</div>
                <div className="text-xs text-slate-500">100+ Languages</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-bold text-emerald-400">~46s/batch</div>
                <div className="text-xs text-slate-500">CPU Latency</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
