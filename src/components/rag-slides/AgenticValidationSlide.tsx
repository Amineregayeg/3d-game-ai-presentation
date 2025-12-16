"use client";

import { TechSlideWrapper } from "../tech-slides/TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface AgenticValidationSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function AgenticValidationSlide({ slideNumber, totalSlides }: AgenticValidationSlideProps) {
  const validationChecks = [
    {
      name: "Syntax Check",
      desc: "Python compile validation",
      icon: "{ }",
      color: "cyan",
      example: "compile(code, '<answer>', 'exec')"
    },
    {
      name: "API Verify",
      desc: "Blender API existence",
      icon: "API",
      color: "blue",
      example: "bpy.ops.* → API spec lookup"
    },
    {
      name: "Version Match",
      desc: "Blender version compat",
      icon: "v4.2",
      color: "purple",
      example: "check_version_compatibility()"
    },
    {
      name: "Hallucination",
      desc: "Context grounding check",
      icon: "!",
      color: "rose",
      example: "answer_facts ⊆ context_facts"
    }
  ];

  const colorMap: Record<string, { bg: string; border: string; text: string }> = {
    cyan: { bg: "bg-cyan-500/20", border: "border-cyan-500/50", text: "text-cyan-400" },
    blue: { bg: "bg-blue-500/20", border: "border-blue-500/50", text: "text-blue-400" },
    purple: { bg: "bg-purple-500/20", border: "border-purple-500/50", text: "text-purple-400" },
    rose: { bg: "bg-rose-500/20", border: "border-rose-500/50", text: "text-rose-400" }
  };

  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Agentic Validation">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Agentic <span className="text-rose-400">Validation</span> Loop
            </h2>
            <p className="text-slate-400">Self-correcting RAG with multi-stage answer validation</p>
          </div>
          <Badge variant="outline" className="border-rose-500/50 text-rose-400 bg-rose-500/10">
            Max 3 Retries
          </Badge>
        </div>

        <div className="grid grid-cols-5 gap-4 flex-1">
          {/* Left: Validation Loop Diagram */}
          <div className="col-span-3">
            <Card className="bg-slate-800/50 border-slate-700/50 h-full">
              <CardContent className="p-4 h-full">
                <h3 className="text-lg font-semibold text-white mb-3">Validation Flow</h3>
                <svg viewBox="0 0 520 280" className="w-full h-full">
                  <defs>
                    <marker id="flowArrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                      <polygon points="0 0, 8 3, 0 6" fill="#64748b"/>
                    </marker>
                    <marker id="greenArrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                      <polygon points="0 0, 8 3, 0 6" fill="#10b981"/>
                    </marker>
                    <marker id="redArrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
                      <polygon points="0 0, 8 3, 0 6" fill="#f43f5e"/>
                    </marker>
                  </defs>

                  {/* Generate Answer */}
                  <g transform="translate(20, 100)">
                    <rect width="90" height="60" rx="8" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="2"/>
                    <text x="45" y="28" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold">Generate</text>
                    <text x="45" y="43" textAnchor="middle" fill="#93c5fd" fontSize="9">Answer</text>
                  </g>

                  {/* Arrow to Validate */}
                  <path d="M115 130 L155 130" stroke="#64748b" strokeWidth="2" markerEnd="url(#flowArrow)"/>

                  {/* Validation Box */}
                  <g transform="translate(160, 70)">
                    <rect width="140" height="120" rx="10" fill="#f43f5e" fillOpacity="0.1" stroke="#f43f5e" strokeWidth="2"/>
                    <text x="70" y="22" textAnchor="middle" fill="#f43f5e" fontSize="11" fontWeight="bold">Validation</text>

                    {/* Check items */}
                    {['Syntax', 'API', 'Version', 'Grounding'].map((check, i) => (
                      <g key={check} transform={`translate(15, ${35 + i * 20})`}>
                        <rect width="110" height="16" rx="3" fill="#0f172a" opacity="0.5"/>
                        <circle cx="12" cy="8" r="4" fill="#f43f5e" opacity="0.6"/>
                        <text x="25" y="12" fill="#fecdd3" fontSize="8">{check} Check</text>
                      </g>
                    ))}
                  </g>

                  {/* Decision Diamond */}
                  <g transform="translate(340, 100)">
                    <polygon points="40,0 80,30 40,60 0,30" fill="#1e293b" stroke="#f59e0b" strokeWidth="2"/>
                    <text x="40" y="28" textAnchor="middle" fill="#f59e0b" fontSize="9" fontWeight="bold">Valid?</text>
                    <text x="40" y="40" textAnchor="middle" fill="#fbbf24" fontSize="8">Pass all?</text>
                  </g>

                  {/* Arrow to Decision */}
                  <path d="M305 130 L338 130" stroke="#64748b" strokeWidth="2" markerEnd="url(#flowArrow)"/>

                  {/* YES path - Success */}
                  <path d="M420 130 L480 130" stroke="#10b981" strokeWidth="2" markerEnd="url(#greenArrow)"/>
                  <text x="450" y="120" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold">YES</text>

                  {/* Success Output */}
                  <g transform="translate(455, 105)">
                    <rect width="60" height="50" rx="8" fill="#10b981" fillOpacity="0.2" stroke="#10b981" strokeWidth="2"/>
                    <text x="30" y="22" textAnchor="middle" fill="#10b981" fontSize="9" fontWeight="bold">Return</text>
                    <text x="30" y="36" textAnchor="middle" fill="#a7f3d0" fontSize="8">Answer</text>
                  </g>

                  {/* NO path - Loop back */}
                  <path d="M380 160 L380 230 L45 230 L45 165" stroke="#f43f5e" strokeWidth="2" strokeDasharray="4" markerEnd="url(#redArrow)"/>
                  <text x="210" y="245" textAnchor="middle" fill="#f43f5e" fontSize="9" fontWeight="bold">NO - Retry (max 3)</text>

                  {/* Re-retrieve box on loop */}
                  <g transform="translate(170, 210)">
                    <rect width="100" height="30" rx="4" fill="#f43f5e" fillOpacity="0.1" stroke="#f43f5e" strokeWidth="1" strokeDasharray="3"/>
                    <text x="50" y="19" textAnchor="middle" fill="#fb7185" fontSize="8">Query Rewrite + Re-retrieve</text>
                  </g>

                  {/* Attempt counter */}
                  <g transform="translate(20, 40)">
                    <rect width="80" height="35" rx="4" fill="#1e293b" stroke="#334155"/>
                    <text x="40" y="18" textAnchor="middle" fill="#64748b" fontSize="8">Attempt</text>
                    <text x="40" y="32" textAnchor="middle" fill="#f59e0b" fontSize="12" fontWeight="bold">1 / 3</text>
                  </g>
                </svg>
              </CardContent>
            </Card>
          </div>

          {/* Right: Validation Checks */}
          <div className="col-span-2 space-y-3">
            {validationChecks.map((check) => (
              <Card key={check.name} className={`${colorMap[check.color].bg} ${colorMap[check.color].border} border`}>
                <CardContent className="p-3">
                  <div className="flex items-start gap-3">
                    <div className={`w-10 h-10 rounded-lg ${colorMap[check.color].bg} border ${colorMap[check.color].border} flex items-center justify-center`}>
                      <span className={`${colorMap[check.color].text} font-mono text-xs font-bold`}>{check.icon}</span>
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <h4 className={`text-sm font-semibold ${colorMap[check.color].text}`}>{check.name}</h4>
                      </div>
                      <p className="text-xs text-slate-400">{check.desc}</p>
                      <code className="text-xs text-slate-500 font-mono mt-1 block">{check.example}</code>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Bottom: Validation Logic */}
        <div className="mt-3 grid grid-cols-2 gap-4">
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-3">
              <h4 className="text-sm font-semibold text-white mb-2">Faithfulness Score</h4>
              <div className="bg-slate-900/50 rounded p-2 font-mono text-xs">
                <div className="text-purple-400">def check_grounding(context, answer):</div>
                <div className="text-slate-400 pl-4">answer_facts = extract_claims(answer)</div>
                <div className="text-slate-400 pl-4">context_facts = extract_facts(context)</div>
                <div className="text-emerald-400 pl-4">return overlap(answer_facts, context_facts)</div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardContent className="p-3">
              <h4 className="text-sm font-semibold text-white mb-2">Failure Recovery</h4>
              <div className="space-y-1 text-xs">
                <div className="flex items-center gap-2">
                  <span className="text-rose-400">Syntax Error</span>
                  <span className="text-slate-500">→</span>
                  <span className="text-slate-300">Re-retrieve Python syntax docs</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-rose-400">API Invalid</span>
                  <span className="text-slate-500">→</span>
                  <span className="text-slate-300">Re-retrieve official API reference</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-rose-400">Hallucination</span>
                  <span className="text-slate-500">→</span>
                  <span className="text-slate-300">Targeted retrieval for missing facts</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
