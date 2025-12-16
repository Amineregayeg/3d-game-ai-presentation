"use client";

import { MCPSlideWrapper } from "./MCPSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface MCPProtocolSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function MCPProtocolSlide({ slideNumber, totalSlides }: MCPProtocolSlideProps) {
  const primitives = [
    { name: "Prompts", desc: "Prepared instruction templates", example: "Asset creation strategy", icon: "M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" },
    { name: "Resources", desc: "Structured data (documents, code)", example: "Scene information", icon: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" },
    { name: "Tools", desc: "Executable functions", example: "create_mesh, apply_material", icon: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" },
    { name: "Roots", desc: "File system entry points", example: "Blender project directories", icon: "M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" },
    { name: "Sampling", desc: "Request AI completion", example: "Iterative refinement", icon: "M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" }
  ];

  return (
    <MCPSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="MCP Protocol">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Model Context <span className="text-orange-400">Protocol</span>
            </h2>
            <p className="text-slate-400">Anthropic&apos;s open standard for AI-to-tool communication</p>
          </div>
          <div className="flex gap-2">
            <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/50">
              JSON-RPC 2.0
            </Badge>
            <Badge variant="outline" className="border-amber-500/50 text-amber-400 bg-amber-500/10">
              Nov 2024
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 flex-1">
          {/* Left: Protocol Overview */}
          <div className="col-span-1 space-y-4">
            <Card className="bg-slate-800/50 border-orange-500/30">
              <CardContent className="p-4">
                <h3 className="text-sm font-bold text-orange-400 mb-3">What is MCP?</h3>
                <div className="space-y-2 text-xs text-slate-400">
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-orange-400 mt-1.5" />
                    <span><strong className="text-white">Universal Standard</strong> - Single protocol replaces fragmented integrations</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5" />
                    <span><strong className="text-white">Modular</strong> - Servers and clients operate independently</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-yellow-400 mt-1.5" />
                    <span><strong className="text-white">Tool Invocation</strong> - LLMs call functions, receive structured results</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-lime-400 mt-1.5" />
                    <span><strong className="text-white">Context Sharing</strong> - Bidirectional data exchange</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800/50 border-amber-500/30">
              <CardContent className="p-4">
                <h3 className="text-sm font-bold text-amber-400 mb-3">Transport Mechanisms</h3>
                <div className="space-y-2">
                  {[
                    { name: "stdio", desc: "Process-based I/O", active: false },
                    { name: "HTTP", desc: "REST + SSE", active: false },
                    { name: "TCP Sockets", desc: "BlenderMCP uses this", active: true }
                  ].map((t, i) => (
                    <div key={i} className={`flex items-center justify-between p-2 rounded ${t.active ? 'bg-amber-500/20 border border-amber-500/50' : 'bg-slate-900/50'}`}>
                      <span className={`text-xs font-mono ${t.active ? 'text-amber-400' : 'text-slate-500'}`}>{t.name}</span>
                      <span className="text-xs text-slate-600">{t.desc}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-orange-500/10 to-amber-500/10 border-orange-500/30">
              <CardContent className="p-4">
                <h3 className="text-sm font-bold text-orange-400 mb-2">BlenderMCP Config</h3>
                <div className="bg-slate-900/80 rounded p-2 font-mono text-xs">
                  <div className="text-slate-500"># Socket settings</div>
                  <div><span className="text-amber-400">Host:</span> <span className="text-slate-300">localhost</span></div>
                  <div><span className="text-amber-400">Port:</span> <span className="text-slate-300">9876</span></div>
                  <div><span className="text-amber-400">Timeout:</span> <span className="text-slate-300">180s</span></div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right: Core Primitives */}
          <div className="col-span-2">
            <Card className="bg-slate-800/30 border-slate-700/50 h-full">
              <CardContent className="p-4">
                <h3 className="text-sm font-bold text-white mb-4">Core MCP Primitives</h3>
                <div className="grid grid-cols-1 gap-3">
                  {primitives.map((p, i) => (
                    <div key={p.name} className="flex items-center gap-4 p-3 bg-slate-900/50 rounded-lg border border-slate-700/50 hover:border-orange-500/30 transition-colors">
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                        i === 0 ? 'bg-orange-500/20' :
                        i === 1 ? 'bg-amber-500/20' :
                        i === 2 ? 'bg-yellow-500/20' :
                        i === 3 ? 'bg-lime-500/20' :
                        'bg-green-500/20'
                      }`}>
                        <svg className={`w-5 h-5 ${
                          i === 0 ? 'text-orange-400' :
                          i === 1 ? 'text-amber-400' :
                          i === 2 ? 'text-yellow-400' :
                          i === 3 ? 'text-lime-400' :
                          'text-green-400'
                        }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={p.icon} />
                        </svg>
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className={`font-bold text-sm ${
                            i === 0 ? 'text-orange-400' :
                            i === 1 ? 'text-amber-400' :
                            i === 2 ? 'text-yellow-400' :
                            i === 3 ? 'text-lime-400' :
                            'text-green-400'
                          }`}>{p.name}</span>
                          <span className="text-xs text-slate-500">—</span>
                          <span className="text-xs text-slate-400">{p.desc}</span>
                        </div>
                        <div className="text-xs text-slate-600 mt-0.5 font-mono">Example: {p.example}</div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Architecture Diagram */}
                <div className="mt-4 pt-4 border-t border-slate-700/50">
                  <svg viewBox="0 0 700 120" className="w-full h-auto">
                    <defs>
                      <linearGradient id="mcpFlowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#f97316" />
                        <stop offset="100%" stopColor="#fbbf24" />
                      </linearGradient>
                    </defs>

                    {/* MCP Client */}
                    <rect x="20" y="20" width="150" height="80" rx="8" fill="#f97316" fillOpacity="0.15" stroke="#f97316" strokeWidth="1.5"/>
                    <text x="95" y="45" textAnchor="middle" fill="#f97316" fontSize="11" fontWeight="bold">MCP Client</text>
                    <text x="95" y="62" textAnchor="middle" fill="#fdba74" fontSize="9">Claude Desktop</text>
                    <text x="95" y="75" textAnchor="middle" fill="#fdba74" fontSize="9">Cursor IDE</text>
                    <text x="95" y="88" textAnchor="middle" fill="#fdba74" fontSize="9">VS Code</text>

                    {/* Bidirectional Arrow */}
                    <path d="M175,60 L285,60" stroke="url(#mcpFlowGrad)" strokeWidth="2"/>
                    <polygon points="280,55 290,60 280,65" fill="#fbbf24"/>
                    <polygon points="180,55 170,60 180,65" fill="#f97316"/>
                    <text x="230" y="50" textAnchor="middle" fill="#64748b" fontSize="8">JSON-RPC</text>

                    {/* MCP Server */}
                    <rect x="290" y="20" width="150" height="80" rx="8" fill="#fbbf24" fillOpacity="0.15" stroke="#fbbf24" strokeWidth="1.5"/>
                    <text x="365" y="45" textAnchor="middle" fill="#fbbf24" fontSize="11" fontWeight="bold">MCP Server</text>
                    <text x="365" y="62" textAnchor="middle" fill="#fde68a" fontSize="9">BlenderMCP</text>
                    <text x="365" y="75" textAnchor="middle" fill="#fde68a" fontSize="9">GitHub MCP</text>
                    <text x="365" y="88" textAnchor="middle" fill="#fde68a" fontSize="9">Postgres MCP</text>

                    {/* Arrow to External */}
                    <path d="M445,60 L525,60" stroke="#64748b" strokeWidth="1.5"/>
                    <polygon points="520,55 530,60 520,65" fill="#64748b"/>

                    {/* External System */}
                    <rect x="530" y="20" width="150" height="80" rx="8" fill="#1e293b" stroke="#64748b" strokeWidth="1.5"/>
                    <text x="605" y="45" textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="bold">External System</text>
                    <text x="605" y="62" textAnchor="middle" fill="#64748b" fontSize="9">Blender 3D</text>
                    <text x="605" y="75" textAnchor="middle" fill="#64748b" fontSize="9">Databases</text>
                    <text x="605" y="88" textAnchor="middle" fill="#64748b" fontSize="9">APIs</text>
                  </svg>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </MCPSlideWrapper>
  );
}
