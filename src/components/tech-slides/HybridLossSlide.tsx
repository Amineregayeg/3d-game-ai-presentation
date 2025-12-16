"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface HybridLossSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function HybridLossSlide({ slideNumber, totalSlides }: HybridLossSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Loss Function">
      <div className="flex flex-col h-full">
        <h2 className="text-4xl font-bold text-white mb-2">
          Hybrid <span className="text-amber-400">CTC</span> + <span className="text-pink-400">Attention</span> Loss
        </h2>
        <p className="text-slate-400 mb-6">Combining alignment-free CTC with high-accuracy cross-entropy</p>

        <Tabs defaultValue="overview" className="flex-1">
          <TabsList className="bg-slate-800/50 border border-slate-700/50">
            <TabsTrigger value="overview" className="data-[state=active]:bg-amber-500/20 data-[state=active]:text-amber-400">
              Loss Overview
            </TabsTrigger>
            <TabsTrigger value="ctc" className="data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400">
              CTC Branch
            </TabsTrigger>
            <TabsTrigger value="attention" className="data-[state=active]:bg-pink-500/20 data-[state=active]:text-pink-400">
              Attention Branch
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="flex-1 mt-4">
            <div className="grid grid-cols-2 gap-6 h-full">
              {/* Loss Formula Visualization */}
              <Card className="bg-slate-800/30 border-slate-700/50">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg text-white">Combined Loss Function</CardTitle>
                </CardHeader>
                <CardContent>
                  <svg viewBox="0 0 450 320" className="w-full h-full">
                    <defs>
                      <linearGradient id="ctcGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.8"/>
                        <stop offset="100%" stopColor="#d97706" stopOpacity="0.8"/>
                      </linearGradient>
                      <linearGradient id="ceGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#ec4899" stopOpacity="0.8"/>
                        <stop offset="100%" stopColor="#db2777" stopOpacity="0.8"/>
                      </linearGradient>
                    </defs>

                    {/* Main formula */}
                    <text x="225" y="40" textAnchor="middle" fill="white" fontSize="20" fontWeight="bold">
                      L = λ<tspan fill="#f59e0b">ctc</tspan> · L<tspan fill="#f59e0b">CTC</tspan> + λ<tspan fill="#ec4899">ce</tspan> · L<tspan fill="#ec4899">CE</tspan>
                    </text>

                    {/* Encoder output */}
                    <rect x="175" y="70" width="100" height="35" rx="6" fill="#1e293b" stroke="#06b6d4" strokeWidth="2"/>
                    <text x="225" y="93" textAnchor="middle" fill="#06b6d4" fontSize="11">Encoder Output</text>

                    {/* Split arrows */}
                    <path d="M175 105 L100 140" stroke="#64748b" strokeWidth="2"/>
                    <path d="M275 105 L350 140" stroke="#64748b" strokeWidth="2"/>

                    {/* CTC Branch */}
                    <g transform="translate(30, 145)">
                      <rect width="140" height="100" rx="8" fill="url(#ctcGrad)"/>
                      <text x="70" y="25" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">CTC Head</text>
                      <rect x="15" y="35" width="110" height="20" rx="3" fill="#0f172a" opacity="0.5"/>
                      <text x="70" y="49" textAnchor="middle" fill="#fcd34d" fontSize="9">Linear → Softmax</text>
                      <rect x="15" y="60" width="110" height="20" rx="3" fill="#0f172a" opacity="0.5"/>
                      <text x="70" y="74" textAnchor="middle" fill="#fcd34d" fontSize="9">CTC Loss (blank=0)</text>
                      <text x="70" y="95" textAnchor="middle" fill="#fef3c7" fontSize="10">Weight: 0.3</text>
                    </g>

                    {/* Attention Branch */}
                    <g transform="translate(280, 145)">
                      <rect width="140" height="100" rx="8" fill="url(#ceGrad)"/>
                      <text x="70" y="25" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">Decoder</text>
                      <rect x="15" y="35" width="110" height="20" rx="3" fill="#0f172a" opacity="0.5"/>
                      <text x="70" y="49" textAnchor="middle" fill="#fbcfe8" fontSize="9">Cross-Attention</text>
                      <rect x="15" y="60" width="110" height="20" rx="3" fill="#0f172a" opacity="0.5"/>
                      <text x="70" y="74" textAnchor="middle" fill="#fbcfe8" fontSize="9">CE Loss (smooth=0.1)</text>
                      <text x="70" y="95" textAnchor="middle" fill="#fce7f3" fontSize="10">Weight: 0.7</text>
                    </g>

                    {/* Combine arrows */}
                    <path d="M100 250 L180 280" stroke="#f59e0b" strokeWidth="2"/>
                    <path d="M350 250 L270 280" stroke="#ec4899" strokeWidth="2"/>

                    {/* Combined Loss */}
                    <rect x="160" y="275" width="130" height="35" rx="6" fill="#1e293b" stroke="#10b981" strokeWidth="2"/>
                    <text x="225" y="298" textAnchor="middle" fill="#10b981" fontSize="12" fontWeight="bold">Total Loss</text>

                  </svg>
                </CardContent>
              </Card>

              {/* Benefits and Config */}
              <div className="space-y-4">
                <Card className="bg-slate-800/30 border-slate-700/50">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm text-slate-400">Why Hybrid Loss?</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                      <div className="text-amber-400 font-semibold text-sm mb-1">CTC Benefits</div>
                      <ul className="text-xs text-slate-400 space-y-1">
                        <li>• No forced alignment needed</li>
                        <li>• Fast convergence early training</li>
                        <li>• Handles variable-length I/O</li>
                        <li>• Provides monotonic alignment</li>
                      </ul>
                    </div>
                    <div className="p-3 bg-pink-500/10 border border-pink-500/30 rounded-lg">
                      <div className="text-pink-400 font-semibold text-sm mb-1">Attention Benefits</div>
                      <ul className="text-xs text-slate-400 space-y-1">
                        <li>• Better final WER</li>
                        <li>• Handles non-monotonic alignment</li>
                        <li>• Token-level accuracy</li>
                        <li>• Stronger language modeling</li>
                      </ul>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-slate-800/30 border-slate-700/50">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm text-slate-400">Warmup Schedule</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <svg viewBox="0 0 250 100" className="w-full">
                      {/* Axes */}
                      <line x1="30" y1="80" x2="230" y2="80" stroke="#475569" strokeWidth="1"/>
                      <line x1="30" y1="80" x2="30" y2="15" stroke="#475569" strokeWidth="1"/>

                      {/* CTC weight line */}
                      <path d="M30 35 L80 35 L230 50" fill="none" stroke="#f59e0b" strokeWidth="2"/>
                      <text x="235" y="50" fill="#f59e0b" fontSize="8">0.3</text>

                      {/* CE weight line */}
                      <path d="M30 50 L80 50 L230 35" fill="none" stroke="#ec4899" strokeWidth="2"/>
                      <text x="235" y="35" fill="#ec4899" fontSize="8">0.7</text>

                      {/* Warmup region */}
                      <rect x="30" y="15" width="50" height="65" fill="#334155" opacity="0.3"/>
                      <text x="55" y="12" textAnchor="middle" fill="#64748b" fontSize="7">Warmup</text>

                      {/* Labels */}
                      <text x="80" y="92" fill="#64748b" fontSize="7">5K steps</text>
                      <text x="230" y="92" fill="#64748b" fontSize="7">End</text>
                      <text x="15" y="35" fill="#f59e0b" fontSize="7">0.4</text>
                      <text x="15" y="50" fill="#ec4899" fontSize="7">0.6</text>
                    </svg>
                    <div className="text-xs text-slate-500 text-center mt-2">
                      More CTC early → More Attention later
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="ctc" className="flex-1 mt-4">
            <div className="grid grid-cols-2 gap-6">
              <Card className="bg-slate-800/30 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-lg text-white">CTC Forward-Backward</CardTitle>
                </CardHeader>
                <CardContent>
                  <svg viewBox="0 0 400 250" className="w-full">
                    {/* Grid for alignment paths */}
                    <g transform="translate(50, 30)">
                      {/* Labels */}
                      <text x="-10" y="-10" fill="#94a3b8" fontSize="10">Time →</text>
                      <text x="-40" y="100" fill="#94a3b8" fontSize="10" transform="rotate(-90, -40, 100)">Labels ↓</text>

                      {/* Time frames */}
                      {[0, 1, 2, 3, 4, 5, 6, 7].map((t) => (
                        <text key={t} x={t * 40 + 20} y="0" textAnchor="middle" fill="#64748b" fontSize="8">t{t + 1}</text>
                      ))}

                      {/* Labels (including blanks) */}
                      {["ε", "H", "ε", "E", "ε", "L", "ε"].map((l, i) => (
                        <text key={i} x="-15" y={i * 28 + 25} textAnchor="middle" fill={l === "ε" ? "#64748b" : "#f59e0b"} fontSize="10">{l}</text>
                      ))}

                      {/* Grid cells */}
                      {[0, 1, 2, 3, 4, 5, 6].map((row) =>
                        [0, 1, 2, 3, 4, 5, 6, 7].map((col) => (
                          <rect
                            key={`${row}-${col}`}
                            x={col * 40}
                            y={row * 28 + 10}
                            width="38"
                            height="26"
                            rx="2"
                            fill="#1e293b"
                            stroke="#334155"
                            strokeWidth="1"
                          />
                        ))
                      )}

                      {/* Highlight valid path */}
                      {[
                        [0, 0], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
                      ].map(([row, col], i) => (
                        <rect
                          key={i}
                          x={col * 40 + 2}
                          y={row * 28 + 12}
                          width="34"
                          height="22"
                          rx="2"
                          fill="#f59e0b"
                          opacity="0.4"
                        />
                      ))}
                    </g>

                    {/* Legend */}
                    <g transform="translate(50, 230)">
                      <rect width="12" height="12" fill="#f59e0b" opacity="0.4" rx="2"/>
                      <text x="18" y="10" fill="#94a3b8" fontSize="9">Valid alignment path</text>
                      <text x="150" y="10" fill="#64748b" fontSize="9">ε = blank token</text>
                    </g>
                  </svg>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/30 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-sm text-slate-400">CTC Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm">
                    <div className="text-amber-400">CTCLoss(</div>
                    <div className="text-slate-300 ml-4">blank = 0,</div>
                    <div className="text-slate-300 ml-4">reduction = &quot;mean&quot;,</div>
                    <div className="text-slate-300 ml-4">zero_infinity = True</div>
                    <div className="text-amber-400">)</div>
                  </div>

                  <div className="space-y-2">
                    {[
                      { label: "Blank Token ID", value: "0", desc: "Separator between labels" },
                      { label: "Weight", value: "0.3", desc: "30% of total loss" },
                      { label: "Input", value: "Encoder logits", desc: "[B, T, vocab]" },
                    ].map((item) => (
                      <div key={item.label} className="p-2 bg-slate-900/30 rounded">
                        <div className="flex justify-between">
                          <span className="text-xs text-slate-400">{item.label}</span>
                          <span className="text-xs font-mono text-amber-400">{item.value}</span>
                        </div>
                        <div className="text-xs text-slate-500">{item.desc}</div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="attention" className="flex-1 mt-4">
            <div className="grid grid-cols-2 gap-6">
              <Card className="bg-slate-800/30 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-lg text-white">Cross-Entropy with Label Smoothing</CardTitle>
                </CardHeader>
                <CardContent>
                  <svg viewBox="0 0 400 200" className="w-full">
                    {/* Hard targets vs smooth targets */}
                    <g transform="translate(30, 30)">
                      <text x="80" y="0" textAnchor="middle" fill="#94a3b8" fontSize="11">Hard Target</text>
                      <text x="280" y="0" textAnchor="middle" fill="#94a3b8" fontSize="11">Smooth Target (0.1)</text>

                      {/* Hard target bars */}
                      {[0, 1, 2, 3, 4].map((i) => {
                        const height = i === 2 ? 100 : 5;
                        return (
                          <rect
                            key={i}
                            x={i * 30 + 10}
                            y={120 - height}
                            width="25"
                            height={height}
                            fill={i === 2 ? "#ec4899" : "#334155"}
                            rx="2"
                          />
                        );
                      })}

                      {/* Smooth target bars */}
                      {[0, 1, 2, 3, 4].map((i) => {
                        const height = i === 2 ? 82 : 14;
                        return (
                          <rect
                            key={i}
                            x={i * 30 + 210}
                            y={120 - height}
                            width="25"
                            height={height}
                            fill={i === 2 ? "#ec4899" : "#f472b6"}
                            opacity={i === 2 ? 1 : 0.5}
                            rx="2"
                          />
                        );
                      })}

                      {/* Labels */}
                      <text x="80" y="140" textAnchor="middle" fill="#64748b" fontSize="9">P(correct) = 1.0</text>
                      <text x="280" y="140" textAnchor="middle" fill="#64748b" fontSize="9">P(correct) = 0.9</text>
                    </g>

                    {/* Formula */}
                    <text x="200" y="180" textAnchor="middle" fill="#94a3b8" fontSize="10">
                      y&apos; = (1-ε)y + ε/K where ε=0.1, K=vocab_size
                    </text>
                  </svg>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/30 border-slate-700/50">
                <CardHeader>
                  <CardTitle className="text-sm text-slate-400">Attention Loss Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-slate-900/50 p-4 rounded-lg font-mono text-sm">
                    <div className="text-pink-400">CrossEntropyLoss(</div>
                    <div className="text-slate-300 ml-4">ignore_index = -1,</div>
                    <div className="text-slate-300 ml-4">label_smoothing = 0.1,</div>
                    <div className="text-slate-300 ml-4">reduction = &quot;mean&quot;</div>
                    <div className="text-pink-400">)</div>
                  </div>

                  <div className="space-y-2">
                    {[
                      { label: "Label Smoothing", value: "0.1", desc: "Prevents overconfidence" },
                      { label: "Weight", value: "0.7", desc: "70% of total loss" },
                      { label: "Ignore Index", value: "-1", desc: "Padding tokens" },
                      { label: "Input", value: "Decoder logits", desc: "[B, T, vocab]" },
                    ].map((item) => (
                      <div key={item.label} className="p-2 bg-slate-900/30 rounded">
                        <div className="flex justify-between">
                          <span className="text-xs text-slate-400">{item.label}</span>
                          <span className="text-xs font-mono text-pink-400">{item.value}</span>
                        </div>
                        <div className="text-xs text-slate-500">{item.desc}</div>
                      </div>
                    ))}
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
