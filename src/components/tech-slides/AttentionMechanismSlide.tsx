"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface AttentionMechanismSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function AttentionMechanismSlide({ slideNumber, totalSlides }: AttentionMechanismSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Core Components">
      <div className="flex flex-col h-full">
        <h2 className="text-4xl font-bold text-white mb-2">
          Multi-Head <span className="text-purple-400">Attention</span> + <span className="text-cyan-400">RoPE</span>
        </h2>
        <p className="text-slate-400 mb-6">Custom attention implementation with Rotary Position Embeddings</p>

        <Tabs defaultValue="attention" className="flex-1">
          <TabsList className="bg-slate-800/50 border border-slate-700/50">
            <TabsTrigger value="attention" className="data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400">
              Self-Attention
            </TabsTrigger>
            <TabsTrigger value="rope" className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400">
              RoPE Embeddings
            </TabsTrigger>
            <TabsTrigger value="flash" className="data-[state=active]:bg-emerald-500/20 data-[state=active]:text-emerald-400">
              Flash Attention
            </TabsTrigger>
          </TabsList>

          <TabsContent value="attention" className="flex-1 mt-4">
            <div className="grid grid-cols-2 gap-6 h-full">
              {/* Attention Diagram */}
              <Card className="bg-slate-800/30 border-slate-700/50">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg text-white flex items-center gap-2">
                    <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/50">Architecture</Badge>
                    Scaled Dot-Product Attention
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <svg viewBox="0 0 400 300" className="w-full h-64">
                    <defs>
                      <linearGradient id="qGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#f472b6"/>
                        <stop offset="100%" stopColor="#ec4899"/>
                      </linearGradient>
                      <linearGradient id="kGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#a78bfa"/>
                        <stop offset="100%" stopColor="#8b5cf6"/>
                      </linearGradient>
                      <linearGradient id="vGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#34d399"/>
                        <stop offset="100%" stopColor="#10b981"/>
                      </linearGradient>
                    </defs>

                    {/* Input X */}
                    <rect x="170" y="10" width="60" height="30" rx="4" fill="#334155"/>
                    <text x="200" y="30" textAnchor="middle" fill="#94a3b8" fontSize="12">X</text>

                    {/* Q, K, V Projections */}
                    <rect x="50" y="70" width="60" height="30" rx="4" fill="url(#qGrad)"/>
                    <text x="80" y="90" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Q</text>

                    <rect x="170" y="70" width="60" height="30" rx="4" fill="url(#kGrad)"/>
                    <text x="200" y="90" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">K</text>

                    <rect x="290" y="70" width="60" height="30" rx="4" fill="url(#vGrad)"/>
                    <text x="320" y="90" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">V</text>

                    {/* Arrows from X */}
                    <path d="M200 40 L80 70" stroke="#64748b" strokeWidth="1.5"/>
                    <path d="M200 40 L200 70" stroke="#64748b" strokeWidth="1.5"/>
                    <path d="M200 40 L320 70" stroke="#64748b" strokeWidth="1.5"/>

                    {/* MatMul Q*K^T */}
                    <circle cx="140" cy="140" r="20" fill="#1e293b" stroke="#8b5cf6" strokeWidth="2"/>
                    <text x="140" y="145" textAnchor="middle" fill="#c4b5fd" fontSize="10">QK^T</text>

                    <path d="M80 100 L130 125" stroke="#f472b6" strokeWidth="1.5"/>
                    <path d="M200 100 L150 125" stroke="#8b5cf6" strokeWidth="1.5"/>

                    {/* Scale */}
                    <rect x="110" y="175" width="60" height="25" rx="4" fill="#374151"/>
                    <text x="140" y="192" textAnchor="middle" fill="#94a3b8" fontSize="9">/sqrt(d_k)</text>
                    <path d="M140 160 L140 175" stroke="#64748b" strokeWidth="1.5"/>

                    {/* Softmax */}
                    <rect x="110" y="215" width="60" height="25" rx="4" fill="#7c3aed"/>
                    <text x="140" y="232" textAnchor="middle" fill="white" fontSize="9">Softmax</text>
                    <path d="M140 200 L140 215" stroke="#64748b" strokeWidth="1.5"/>

                    {/* MatMul with V */}
                    <circle cx="220" cy="228" r="20" fill="#1e293b" stroke="#10b981" strokeWidth="2"/>
                    <text x="220" y="233" textAnchor="middle" fill="#6ee7b7" fontSize="10">@ V</text>

                    <path d="M170 228 L200 228" stroke="#64748b" strokeWidth="1.5"/>
                    <path d="M320 100 L320 200 L240 228" stroke="#10b981" strokeWidth="1.5"/>

                    {/* Output */}
                    <rect x="260" y="260" width="80" height="30" rx="4" fill="#0f766e"/>
                    <text x="300" y="280" textAnchor="middle" fill="white" fontSize="11">Output</text>
                    <path d="M220 248 L300 260" stroke="#64748b" strokeWidth="1.5"/>
                  </svg>
                </CardContent>
              </Card>

              {/* Formula and Code */}
              <div className="space-y-4">
                <Card className="bg-slate-800/30 border-slate-700/50">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm text-slate-400">Mathematical Formulation</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm">
                      <div className="text-purple-400">Attention(Q, K, V) =</div>
                      <div className="text-cyan-400 ml-4">softmax(QK<sup>T</sup> / sqrt(d_k)) V</div>
                      <div className="mt-4 text-slate-500 text-xs">
                        <div>Q = X @ W_q {"  // Query projection"}</div>
                        <div>K = X @ W_k {"  // Key projection"}</div>
                        <div>V = X @ W_v {"  // Value projection"}</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-slate-800/30 border-slate-700/50">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm text-slate-400">Multi-Head Configuration</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                        <div className="text-2xl font-bold text-purple-400">8</div>
                        <div className="text-xs text-slate-500">Attention Heads</div>
                      </div>
                      <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                        <div className="text-2xl font-bold text-cyan-400">64</div>
                        <div className="text-xs text-slate-500">Head Dimension</div>
                      </div>
                      <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                        <div className="text-2xl font-bold text-emerald-400">512</div>
                        <div className="text-xs text-slate-500">Model Dimension</div>
                      </div>
                      <div className="text-center p-3 bg-slate-900/50 rounded-lg">
                        <div className="text-2xl font-bold text-pink-400">0</div>
                        <div className="text-xs text-slate-500">Bias Terms</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="rope" className="flex-1 mt-4">
            <div className="grid grid-cols-2 gap-6">
              <Card className="bg-slate-800/30 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-lg text-white">Rotary Position Embedding</CardTitle>
                </CardHeader>
                <CardContent>
                  <svg viewBox="0 0 350 250" className="w-full">
                    {/* Rotation visualization */}
                    <g transform="translate(175, 125)">
                      {/* Unit circle */}
                      <circle cx="0" cy="0" r="80" fill="none" stroke="#334155" strokeWidth="1"/>

                      {/* Angle arcs */}
                      {[0, 1, 2, 3].map((i) => {
                        const angle = (i * Math.PI) / 6;
                        const x = 80 * Math.cos(angle);
                        const y = -80 * Math.sin(angle);
                        return (
                          <g key={i}>
                            <line x1="0" y1="0" x2={x} y2={y} stroke="#06b6d4" strokeWidth="2" opacity={0.3 + i * 0.2}/>
                            <circle cx={x} cy={y} r="6" fill="#06b6d4" opacity={0.5 + i * 0.15}/>
                            <text x={x * 1.15} y={y * 1.15} textAnchor="middle" fill="#67e8f9" fontSize="10">
                              pos={i}
                            </text>
                          </g>
                        );
                      })}

                      {/* Rotation arrow */}
                      <path d="M30 -10 A35 35 0 0 1 10 -30" fill="none" stroke="#a855f7" strokeWidth="2" markerEnd="url(#arrowPurple)"/>
                      <text x="45" y="-25" fill="#c084fc" fontSize="9">mθ</text>
                    </g>

                    <defs>
                      <marker id="arrowPurple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#a855f7"/>
                      </marker>
                    </defs>

                    {/* Labels */}
                    <text x="175" y="230" textAnchor="middle" fill="#94a3b8" fontSize="11">
                      Position encoded via rotation in 2D subspaces
                    </text>
                  </svg>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/30 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-lg text-white">RoPE Formula</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm">
                    <div className="text-cyan-400">R(θ, m) = </div>
                    <div className="text-slate-300 ml-4">
                      [cos(mθ)  -sin(mθ)]
                    </div>
                    <div className="text-slate-300 ml-4">
                      [sin(mθ)   cos(mθ)]
                    </div>
                    <div className="mt-3 text-slate-500 text-xs">
                      θ_i = 10000^(-2i/d)
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-2 bg-slate-900/30 rounded">
                      <span className="text-sm text-slate-400">Relative Position Aware</span>
                      <Badge className="bg-emerald-500/20 text-emerald-400">Yes</Badge>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-slate-900/30 rounded">
                      <span className="text-sm text-slate-400">Extrapolation</span>
                      <Badge className="bg-emerald-500/20 text-emerald-400">Strong</Badge>
                    </div>
                    <div className="flex items-center justify-between p-2 bg-slate-900/30 rounded">
                      <span className="text-sm text-slate-400">Max Sequence</span>
                      <Badge className="bg-cyan-500/20 text-cyan-400">8192</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="flash" className="flex-1 mt-4">
            <div className="grid grid-cols-3 gap-6">
              <Card className="bg-slate-800/30 border-slate-700/50 col-span-2">
                <CardHeader>
                  <CardTitle className="text-lg text-white">Flash Attention - Memory Efficient</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                      <div className="text-red-400 font-semibold mb-2">Standard Attention</div>
                      <div className="text-3xl font-bold text-red-400">O(N<sup>2</sup>)</div>
                      <div className="text-sm text-slate-400 mt-1">Memory Complexity</div>
                      <div className="text-xs text-slate-500 mt-2">Stores full N×N attention matrix</div>
                    </div>
                    <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                      <div className="text-emerald-400 font-semibold mb-2">Flash Attention</div>
                      <div className="text-3xl font-bold text-emerald-400">O(N)</div>
                      <div className="text-sm text-slate-400 mt-1">Memory Complexity</div>
                      <div className="text-xs text-slate-500 mt-2">Tiled computation, no full matrix</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/30 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-sm text-slate-400">Benefits</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center">
                      <span className="text-emerald-400">4x</span>
                    </div>
                    <span className="text-sm text-slate-300">Longer sequences</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-cyan-500/20 flex items-center justify-center">
                      <span className="text-cyan-400">2x</span>
                    </div>
                    <span className="text-sm text-slate-300">Faster training</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center">
                      <span className="text-purple-400">IO</span>
                    </div>
                    <span className="text-sm text-slate-300">Aware algorithm</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </TechSlideWrapper>
  );
}
