"use client";

import { TechSlideWrapper } from "../tech-slides/TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface EmbeddingVectorSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function EmbeddingVectorSlide({ slideNumber, totalSlides }: EmbeddingVectorSlideProps) {
  const models = [
    { name: "all-MiniLM-L6-v2", dims: "384", strength: "Fast CPU inference", recommended: true },
    { name: "BGE-M3", dims: "1,024", strength: "Technical docs", recommended: false },
    { name: "text-embedding-3-small", dims: "1,536", strength: "OpenAI API", recommended: false },
    { name: "E5-Large", dims: "1,024", strength: "General", recommended: false }
  ];

  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Embedding & Vector Search">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Vector <span className="text-cyan-400">Embeddings</span> & HNSW Index
            </h2>
            <p className="text-slate-400">MiniLM dense embeddings with hierarchical graph search</p>
          </div>
          <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10">
            384 Dimensions
          </Badge>
        </div>

        <div className="grid grid-cols-2 gap-6 flex-1">
          {/* Left: Embedding Process */}
          <div className="space-y-4">
            {/* Embedding Visualization */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-3">Embedding Pipeline</h3>
                <svg viewBox="0 0 400 180" className="w-full">
                  <defs>
                    <linearGradient id="embedGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#06b6d4"/>
                      <stop offset="100%" stopColor="#3b82f6"/>
                    </linearGradient>
                  </defs>

                  {/* Document Input */}
                  <g transform="translate(10, 30)">
                    <rect width="80" height="120" rx="6" fill="#1e293b" stroke="#334155" strokeWidth="1"/>
                    <text x="40" y="20" textAnchor="middle" fill="#94a3b8" fontSize="10" fontWeight="bold">Document</text>
                    <rect x="8" y="30" width="64" height="8" rx="2" fill="#475569"/>
                    <rect x="8" y="42" width="55" height="8" rx="2" fill="#475569"/>
                    <rect x="8" y="54" width="60" height="8" rx="2" fill="#475569"/>
                    <rect x="8" y="66" width="45" height="8" rx="2" fill="#475569"/>
                    <rect x="8" y="78" width="58" height="8" rx="2" fill="#475569"/>
                    <rect x="8" y="90" width="50" height="8" rx="2" fill="#475569"/>
                  </g>

                  {/* Arrow */}
                  <path d="M100 90 L140 90" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>

                  {/* MiniLM Model */}
                  <g transform="translate(145, 50)">
                    <rect width="100" height="80" rx="8" fill="#06b6d4" fillOpacity="0.2" stroke="#06b6d4" strokeWidth="2"/>
                    <text x="50" y="25" textAnchor="middle" fill="#06b6d4" fontSize="12" fontWeight="bold">MiniLM-L6</text>
                    <text x="50" y="42" textAnchor="middle" fill="#a5f3fc" fontSize="9">Transformer</text>
                    <text x="50" y="55" textAnchor="middle" fill="#a5f3fc" fontSize="9">6 Layers, 22M</text>
                    <rect x="15" y="62" width="70" height="12" rx="3" fill="#0f172a" opacity="0.5"/>
                    <text x="50" y="71" textAnchor="middle" fill="#67e8f9" fontSize="7">Pooling + L2 Norm</text>
                  </g>

                  {/* Arrow */}
                  <path d="M255 90 L290 90" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrow)"/>

                  {/* Vector Output */}
                  <g transform="translate(295, 25)">
                    <rect width="95" height="130" rx="6" fill="#1e293b" stroke="#10b981" strokeWidth="1.5"/>
                    <text x="47" y="18" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="bold">Vector</text>
                    <text x="47" y="32" textAnchor="middle" fill="#64748b" fontSize="8">384 floats</text>

                    {/* Vector visualization */}
                    {[0,1,2,3,4,5,6,7].map((i) => (
                      <g key={i} transform={`translate(10, ${42 + i * 10})`}>
                        <rect width="75" height="8" rx="2" fill="url(#embedGrad)" opacity={0.2 + Math.sin(i) * 0.2}/>
                        <text x="72" y="7" textAnchor="end" fill="#a7f3d0" fontSize="6">{(Math.random() * 0.1 - 0.05).toFixed(4)}</text>
                      </g>
                    ))}
                    <text x="47" y="127" textAnchor="middle" fill="#475569" fontSize="7">...</text>
                  </g>
                </svg>
              </CardContent>
            </Card>

            {/* Model Comparison */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-sm font-semibold text-white mb-2">Embedding Models</h3>
                <Table>
                  <TableHeader>
                    <TableRow className="border-slate-700">
                      <TableHead className="text-slate-400 text-xs">Model</TableHead>
                      <TableHead className="text-slate-400 text-xs">Dims</TableHead>
                      <TableHead className="text-slate-400 text-xs">Best For</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {models.map((model) => (
                      <TableRow key={model.name} className="border-slate-700/50">
                        <TableCell className={`text-xs ${model.recommended ? 'text-cyan-400 font-bold' : 'text-slate-300'}`}>
                          {model.name} {model.recommended && <span className="text-emerald-400">*</span>}
                        </TableCell>
                        <TableCell className="text-xs text-slate-400">{model.dims}</TableCell>
                        <TableCell className="text-xs text-slate-500">{model.strength}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>

          {/* Right: HNSW Index */}
          <div className="space-y-4">
            {/* HNSW Visualization */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-3">HNSW Index Structure</h3>
                <svg viewBox="0 0 350 200" className="w-full">
                  <defs>
                    <filter id="glowNode">
                      <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                      <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                      </feMerge>
                    </filter>
                  </defs>

                  {/* Layer labels */}
                  <text x="15" y="30" fill="#64748b" fontSize="9">Layer 2</text>
                  <text x="15" y="90" fill="#64748b" fontSize="9">Layer 1</text>
                  <text x="15" y="165" fill="#64748b" fontSize="9">Layer 0</text>

                  {/* Layer 2 (sparse) */}
                  <g transform="translate(70, 15)">
                    <line x1="0" y1="15" x2="250" y2="15" stroke="#334155" strokeWidth="1" strokeDasharray="4"/>
                    <circle cx="50" cy="15" r="10" fill="#a855f7" filter="url(#glowNode)"/>
                    <circle cx="180" cy="15" r="10" fill="#a855f7" filter="url(#glowNode)"/>
                    <line x1="60" y1="15" x2="170" y2="15" stroke="#a855f7" strokeWidth="1.5"/>
                  </g>

                  {/* Layer 1 (medium) */}
                  <g transform="translate(70, 75)">
                    <line x1="0" y1="15" x2="250" y2="15" stroke="#334155" strokeWidth="1" strokeDasharray="4"/>
                    <circle cx="30" cy="15" r="8" fill="#06b6d4" filter="url(#glowNode)"/>
                    <circle cx="90" cy="15" r="8" fill="#06b6d4" filter="url(#glowNode)"/>
                    <circle cx="150" cy="15" r="8" fill="#06b6d4" filter="url(#glowNode)"/>
                    <circle cx="210" cy="15" r="8" fill="#06b6d4" filter="url(#glowNode)"/>
                    <line x1="38" y1="15" x2="82" y2="15" stroke="#06b6d4" strokeWidth="1"/>
                    <line x1="98" y1="15" x2="142" y2="15" stroke="#06b6d4" strokeWidth="1"/>
                    <line x1="158" y1="15" x2="202" y2="15" stroke="#06b6d4" strokeWidth="1"/>
                  </g>

                  {/* Layer 0 (dense) */}
                  <g transform="translate(70, 150)">
                    <line x1="0" y1="15" x2="250" y2="15" stroke="#334155" strokeWidth="1" strokeDasharray="4"/>
                    {[0,1,2,3,4,5,6,7,8,9,10,11].map((i) => (
                      <circle key={i} cx={20 + i * 21} cy="15" r="6" fill="#10b981" opacity={0.6 + (i % 3) * 0.1}/>
                    ))}
                    {/* Some connections */}
                    <line x1="26" y1="15" x2="39" y2="15" stroke="#10b981" strokeWidth="0.5" opacity="0.5"/>
                    <line x1="47" y1="15" x2="60" y2="15" stroke="#10b981" strokeWidth="0.5" opacity="0.5"/>
                    <line x1="89" y1="15" x2="102" y2="15" stroke="#10b981" strokeWidth="0.5" opacity="0.5"/>
                  </g>

                  {/* Vertical connections */}
                  <line x1="120" y1="30" x2="100" y2="82" stroke="#64748b" strokeWidth="1" strokeDasharray="2"/>
                  <line x1="120" y1="30" x2="160" y2="82" stroke="#64748b" strokeWidth="1" strokeDasharray="2"/>
                  <line x1="250" y1="30" x2="220" y2="82" stroke="#64748b" strokeWidth="1" strokeDasharray="2"/>
                  <line x1="100" y1="98" x2="90" y2="157" stroke="#64748b" strokeWidth="1" strokeDasharray="2"/>
                  <line x1="160" y1="98" x2="153" y2="157" stroke="#64748b" strokeWidth="1" strokeDasharray="2"/>
                  <line x1="220" y1="98" x2="216" y2="157" stroke="#64748b" strokeWidth="1" strokeDasharray="2"/>

                  {/* Search path indicator */}
                  <g transform="translate(280, 10)">
                    <rect width="60" height="50" rx="4" fill="#f59e0b" fillOpacity="0.1" stroke="#f59e0b" strokeWidth="1"/>
                    <text x="30" y="18" textAnchor="middle" fill="#f59e0b" fontSize="8" fontWeight="bold">Search</text>
                    <text x="30" y="30" textAnchor="middle" fill="#fbbf24" fontSize="7">O(log N)</text>
                    <text x="30" y="42" textAnchor="middle" fill="#64748b" fontSize="7">~10 hops</text>
                  </g>
                </svg>
              </CardContent>
            </Card>

            {/* Cosine Similarity */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-sm font-semibold text-white mb-3">Cosine Similarity Search</h3>
                <div className="bg-slate-900/50 rounded-lg p-3 font-mono text-xs">
                  <div className="text-purple-400 mb-2">-- pgvector query</div>
                  <div className="text-slate-300">
                    <span className="text-cyan-400">SELECT</span> content,
                  </div>
                  <div className="text-slate-300 pl-4">
                    1 - (embedding <span className="text-amber-400">&lt;=&gt;</span> query_vec) <span className="text-cyan-400">AS</span> similarity
                  </div>
                  <div className="text-slate-300">
                    <span className="text-cyan-400">FROM</span> documents
                  </div>
                  <div className="text-slate-300">
                    <span className="text-cyan-400">ORDER BY</span> similarity <span className="text-cyan-400">DESC</span>
                  </div>
                  <div className="text-slate-300">
                    <span className="text-cyan-400">LIMIT</span> <span className="text-emerald-400">100</span>;
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Metrics */}
            <div className="grid grid-cols-3 gap-2">
              {[
                { value: "50ms", label: "Search Latency", color: "text-cyan-400" },
                { value: "99%", label: "Recall@100", color: "text-emerald-400" },
                { value: "16", label: "HNSW M-param", color: "text-purple-400" }
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
      </div>
    </TechSlideWrapper>
  );
}
